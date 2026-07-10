from contextlib import contextmanager
from typing import Any, Callable

import nanomoe_ops
import stk
import stk.ops
import torch
import torch.nn.functional as F
from einops import rearrange
from megablocks import ops


@contextmanager
def cuda_device_context(tensor: torch.Tensor):
    """Context manager to ensure CUDA ops run on the correct device.

    Megablocks CUDA kernels use getCurrentCUDAStream() without setting device,
    which fails on multi-GPU when tensors are on cuda:1 but context is cuda:0.

    We use set_device() to switch the active CUDA device before running kernels.
    """
    if tensor.is_cuda:
        old_device = torch.cuda.current_device()
        new_device = tensor.device.index
        if old_device != new_device:
            torch.cuda.set_device(new_device)
            try:
                yield
            finally:
                torch.cuda.set_device(old_device)
        else:
            yield
    else:
        yield


# Wrapper functions for megablocks ops that ensure correct device context
def mb_sort(x: torch.Tensor, end_bit: int):
    """Device-aware wrapper for ops.sort"""
    with cuda_device_context(x):
        return ops.sort(x, end_bit)


def mb_histogram(x: torch.Tensor, max_val: int):
    """Device-aware wrapper for ops.histogram"""
    with cuda_device_context(x):
        return ops.histogram(x, max_val)


def mb_inclusive_cumsum(x: torch.Tensor, dim: int):
    """Device-aware wrapper for ops.inclusive_cumsum"""
    with cuda_device_context(x):
        return ops.inclusive_cumsum(x, dim)


def mb_round_up(x: torch.Tensor, val: int):
    """Device-aware wrapper for ops.round_up"""
    with cuda_device_context(x):
        return ops.round_up(x, val)


def mb_padded_gather(x, indices, bin_ids, bins, padded_bins, num_active):
    """Device-aware wrapper for ops.padded_gather"""
    device = x.device
    # Ensure all tensors are on the same device
    indices = indices.to(device)
    bin_ids = bin_ids.to(device)
    bins = bins.to(device)
    padded_bins = padded_bins.to(device)
    with cuda_device_context(x):
        return ops.padded_gather(x, indices, bin_ids, bins, padded_bins, num_active)


def mb_padded_scatter(x, indices, bin_ids, weights, bins, padded_bins, num_active):
    """Device-aware wrapper for ops.padded_scatter"""
    device = x.device
    # Ensure all tensors are on the same device
    indices = indices.to(device)
    bin_ids = bin_ids.to(device)
    weights = weights.to(device)
    bins = bins.to(device)
    padded_bins = padded_bins.to(device)
    with cuda_device_context(x):
        return ops.padded_scatter(x, indices, bin_ids, weights, bins, padded_bins, num_active)


# Wrappers for stk.ops that also need device context for triton kernels
def stk_row_indices(shape, data, offsets, column_indices):
    """Device-aware wrapper for stk.ops.row_indices"""
    with cuda_device_context(data):
        return stk.ops.row_indices(shape, data, offsets, column_indices)


def stk_sdd(x, w, topology):
    """Device-aware wrapper for stk.ops.sdd (sparse-dense-dense)"""
    with cuda_device_context(x):
        return stk.ops.sdd(x, w, topology)


def stk_dsd(x, w):
    """Device-aware wrapper for stk.ops.dsd (dense-sparse-dense)"""
    with cuda_device_context(x.data):
        return stk.ops.dsd(x, w)


def stk_mul(a, b):
    """Device-aware wrapper for stk.ops.mul"""
    with cuda_device_context(a.data):
        return stk.ops.mul(a, b)


class TopologyVarOp(torch.autograd.Function):
    """Topology construction for variable-size experts.

    Unlike the standard topology op which assumes uniform expert sizes,
    this version accepts per-expert block counts and offsets.
    """

    @staticmethod
    def forward(
        ctx: Any,
        padded_bins: torch.Tensor,
        expert_block_counts: torch.Tensor,  # [num_experts] - blocks per expert
        expert_block_offsets: torch.Tensor,  # [num_experts+1] - cumulative offsets
        block_size: int,
        output_block_rows: int,
    ):
        # Calculate total output size and cumulative offsets using TENSOR OPS
        # Avoids Python loop with .item() calls that cause GPU syncs

        # Compute token blocks per expert: diff of padded_bins / block_size
        prepend = padded_bins.new_zeros(1)
        bin_sizes = torch.diff(padded_bins, prepend=prepend)
        expert_token_blocks = (bin_sizes // block_size).to(torch.int32)

        # Compute output size per expert (token_blocks * expert_blocks)
        expert_output_sizes = expert_token_blocks * expert_block_counts.to(torch.int32)

        # Single sync to get total (unavoidable for allocation)
        total_nnz = expert_output_sizes.sum().item()

        if total_nnz == 0:
            return torch.empty(0, dtype=torch.int16, device=padded_bins.device)

        # Compute offsets on GPU - cumsum gives [a, a+b, a+b+c, ...], we need [0, a, a+b, ...]
        cumsum = expert_output_sizes.cumsum(0)
        output_offsets_tensor = torch.cat([
            torch.zeros(1, dtype=torch.int32, device=padded_bins.device),
            cumsum[:-1]
        ]).to(torch.int32)

        out = torch.empty(
            total_nnz,
            dtype=torch.int16,
            device=padded_bins.device,
        )

        nanomoe_ops.indices_variable(
            padded_bins,
            expert_block_counts,
            expert_block_offsets,
            output_offsets_tensor,
            block_size,
            output_block_rows,
            out,
        )

        return out


def topology_var(
    padded_bins: torch.Tensor,
    expert_block_counts: torch.Tensor,
    expert_block_offsets: torch.Tensor,
    block_size: int,
    output_block_rows: int,
) -> torch.Tensor:
    """Construct variable-size expert topology."""
    return TopologyVarOp.apply(
        padded_bins,
        expert_block_counts,
        expert_block_offsets,
        block_size,
        output_block_rows,
    )


def sort_tokens_by_expert(top_k_index_flat, sort_end_bit, num_experts):
    bin_ids, indices = mb_sort(top_k_index_flat, sort_end_bit)
    tokens_per_expert = mb_histogram(top_k_index_flat, num_experts)
    return bin_ids, indices, tokens_per_expert


def sparse_transpose(
    size,
    row_indices,
    column_indices,
    total_expert_width,
    block_size,
    transpose_sort_end_bit,
):
    device = row_indices.device
    block_columns = total_expert_width // block_size

    _, gather_indices = mb_sort(
        column_indices.int(),
        transpose_sort_end_bit,
    )

    column_indices_t = row_indices.gather(0, gather_indices.long())
    block_offsets_t = gather_indices.int()

    zero = torch.zeros((1,), dtype=torch.int32, device=device)
    nnz_per_column = mb_histogram(column_indices, block_columns)
    nnz_per_column = mb_inclusive_cumsum(nnz_per_column, 0)
    if nnz_per_column.dim() == 0:
        nnz_per_column = nnz_per_column.unsqueeze(0)
    offsets_t = torch.cat([zero, nnz_per_column])
    return column_indices_t.to(device), offsets_t.to(device), block_offsets_t.to(device)


def create_topology(
    x,
    tokens_per_expert,
    block_size,
    expert_size_blocks,
    expert_block_offsets,
    total_expert_width,
    transpose_sort_end_bit,
):
    device = x.device

    padded_tokens_per_expert = mb_round_up(tokens_per_expert, block_size)
    padded_bins = mb_inclusive_cumsum(padded_tokens_per_expert, 0)
    padded_bins = padded_bins.contiguous()

    padded_tokens = padded_bins[-1].clamp_min(block_size).item()  # Convert to Python int

    block_rows = padded_tokens // block_size

    column_indices = topology_var(
        padded_bins,
        expert_size_blocks,
        expert_block_offsets,
        block_size,
        block_rows,
    )

    prepend = padded_bins.new_zeros(1)
    bin_sizes = torch.diff(padded_bins, prepend=prepend)
    expert_token_blocks = bin_sizes // block_size

    # Ensure everything is on device before repeat_interleave
    expert_size_blocks = expert_size_blocks.to(device)
    expert_token_blocks = expert_token_blocks.to(device)

    # Passing output_size makes repeat_interleave sync-free (otherwise it syncs to
    # discover the output length before allocating). block_rows == sum(expert_token_blocks).
    repeated_sizes = torch.repeat_interleave(
        expert_size_blocks, expert_token_blocks, output_size=block_rows
    )
    repeated_sizes = repeated_sizes.to(device)  # Force device

    offsets = torch.cat([torch.zeros(1, dtype=repeated_sizes.dtype, device=device), repeated_sizes.cumsum(0)])

    column_indices = column_indices.to(device=device, dtype=torch.int32)
    offsets = offsets.to(device=device, dtype=torch.int32)

    shape = (padded_tokens, total_expert_width)

    num_blocks = column_indices.numel()
    data_placeholder = torch.empty(
        num_blocks,
        block_size,
        block_size,
        dtype=x.dtype,
        device=device,
    )

    row_indices = stk_row_indices(shape, data_placeholder, offsets, column_indices)
    row_indices = row_indices.to(device=device, dtype=torch.int32)

    column_indices_t, offsets_t, block_offsets_t = sparse_transpose(
        shape,
        row_indices,
        column_indices,
        total_expert_width,
        block_size,
        transpose_sort_end_bit,
    )
    column_indices_t = column_indices_t.to(torch.int32)
    offsets_t = offsets_t.to(torch.int32)
    block_offsets_t = block_offsets_t.to(torch.int32)

    topology = stk.Matrix(
        shape,
        data_placeholder,
        row_indices,
        column_indices,
        offsets,
        column_indices_t,
        offsets_t,
        block_offsets_t,
    )

    return padded_bins, topology


def gather_tokens(
    hidden_states, indices, bin_ids, tokens_per_expert, padded_bins, num_active_experts
):
    bins = mb_inclusive_cumsum(tokens_per_expert, 0)
    bins = bins.contiguous()
    return mb_padded_gather(
        hidden_states, indices, bin_ids, bins, padded_bins, num_active_experts
    )


def scatter_tokens(
    hidden_states,
    indices,
    bin_ids,
    weights,
    tokens_per_expert,
    padded_bins,
    num_active_experts,
):
    bins = mb_inclusive_cumsum(tokens_per_expert, 0)
    bins = bins.contiguous()
    return mb_padded_scatter(
        hidden_states, indices, bin_ids, weights, bins, padded_bins, num_active_experts
    )


def _should_use_loop(num_experts: int, num_tokens: int) -> bool:
    """Decide whether to use loop or sparse implementation.

    Based on benchmarks:
    - 2 experts: loop always wins or ties
    - 7 experts: loop wins for < 1024 tokens
    - 8+ experts: sparse wins quickly
    """
    if num_experts <= 2:
        return True
    if num_experts <= 7:
        return num_tokens < 1024
    return num_tokens < 16


def _moe_forward_2x2_impl(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_up: torch.Tensor | None,
    e0_end: int,
    e1_start: int,
    e1_end: int,
) -> torch.Tensor:
    """Core 2x2 MoE computation - compile-friendly (no .tolist())."""
    # Remap weights from top-k order to expert order without allocation
    expert_0_first = (top_k_index[:, 0] == 0).unsqueeze(-1)  # [N, 1]
    w0 = torch.where(expert_0_first, top_k_weights[:, 0:1], top_k_weights[:, 1:2])
    w1_w = torch.where(expert_0_first, top_k_weights[:, 1:2], top_k_weights[:, 0:1])

    # Expert 0: all tokens, dense matmuls
    if w1_up is not None:
        gate0 = hidden_states @ w1[:, :e0_end]
        up0 = hidden_states @ w1_up[:, :e0_end]
        out0 = F.silu(gate0) * up0 @ w2[:e0_end, :]
    else:
        out0 = F.silu(hidden_states @ w1[:, :e0_end]) @ w2[:e0_end, :]

    # Expert 1: all tokens, dense matmuls
    if w1_up is not None:
        gate1 = hidden_states @ w1[:, e1_start:e1_end]
        up1 = hidden_states @ w1_up[:, e1_start:e1_end]
        out1 = F.silu(gate1) * up1 @ w2[e1_start:e1_end, :]
    else:
        out1 = F.silu(hidden_states @ w1[:, e1_start:e1_end]) @ w2[e1_start:e1_end, :]

    return w0 * out0 + w1_w * out1


# Compiled version for when torch.compile is beneficial
_moe_forward_2x2_compiled = torch.compile(_moe_forward_2x2_impl)


def _moe_forward_2x2(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    expert_block_offsets: torch.Tensor,
    w1_up: torch.Tensor | None = None,
    block_size: int = 128,
    use_compile: bool = True,
    precomputed_offsets: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """Optimized MoE forward for exactly 2 experts with top-k=2.

    Every token goes to both experts, so we eliminate:
    - Loop iteration
    - Mask computation and nonzero() calls
    - Dynamic token indexing
    - index_add_() accumulation

    Args:
        precomputed_offsets: Optional (e0_end, e1_start, e1_end) to avoid recomputing each forward.
    """
    if precomputed_offsets is not None:
        e0_end, e1_start, e1_end = precomputed_offsets
    else:
        # Compute expert boundaries (Python ints for slicing)
        expert_offsets = (expert_block_offsets * block_size).tolist()
        e0_end = expert_offsets[1]
        e1_start, e1_end = expert_offsets[1], expert_offsets[2]

    impl = _moe_forward_2x2_compiled if use_compile else _moe_forward_2x2_impl
    return impl(hidden_states, top_k_index, top_k_weights, w1, w2, w1_up, e0_end, e1_start, e1_end)


def _moe_forward_loop(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    activation_fn: Callable,
    num_experts: int,
    expert_block_offsets: torch.Tensor,
    w1_up: torch.Tensor | None = None,
    block_size: int = 128,
) -> torch.Tensor:
    """Loop-based MoE forward - fast for small expert counts."""
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    output = torch.zeros(num_tokens, hidden_size, dtype=hidden_states.dtype, device=hidden_states.device)

    # Convert block offsets to actual indices
    expert_offsets = (expert_block_offsets * block_size).tolist()

    for expert_idx in range(num_experts):
        # Find which (token, k) pairs route to this expert
        mask = (top_k_index == expert_idx)
        if not mask.any():
            continue

        token_indices, k_indices = mask.nonzero(as_tuple=True)
        weights = top_k_weights[token_indices, k_indices]

        # Slice expert weights (handles variable sizes)
        start, end = expert_offsets[expert_idx], expert_offsets[expert_idx + 1]
        w1_expert = w1[:, start:end]  # [hidden, expert_ffn]
        w2_expert = w2[start:end, :]  # [expert_ffn, hidden]

        # Expert forward
        x = hidden_states[token_indices]  # [num_routed, hidden]

        if w1_up is not None:
            # SwiGLU: silu(gate) * up
            w1_up_expert = w1_up[:, start:end]
            gate = x @ w1_expert
            up = x @ w1_up_expert
            x = F.silu(gate) * up
        else:
            x = x @ w1_expert
            x = F.silu(x)

        x = x @ w2_expert  # [num_routed, hidden]

        # Weighted accumulate
        output.index_add_(0, token_indices, weights.unsqueeze(-1) * x)

    return output


def _moe_forward_sparse(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    activation_fn: Callable[[stk.Matrix], stk.Matrix],
    num_experts: int,
    num_active_experts: int,
    sort_end_bit: int,
    transpose_sort_end_bit: int,
    expert_size_blocks: torch.Tensor,
    expert_block_offsets: torch.Tensor,
    total_expert_width: int,
    w1_up: torch.Tensor | None = None,
    block_size: int = 128,
) -> torch.Tensor:
    """Sparse MoE forward using stk ops - fast for many experts or large batches."""
    top_k_weights_flat = rearrange(top_k_weights, "... -> (...)")
    top_k_index_flat = rearrange(top_k_index, "... -> (...)")

    bin_ids, indices, tokens_per_expert = sort_tokens_by_expert(
        top_k_index_flat, sort_end_bit, num_experts
    )
    padded_bins, topology = create_topology(
        hidden_states,
        tokens_per_expert,
        block_size,
        expert_size_blocks,
        expert_block_offsets,
        total_expert_width,
        transpose_sort_end_bit,
    )
    hidden_states_permuted = gather_tokens(
        hidden_states,
        indices,
        bin_ids,
        tokens_per_expert,
        padded_bins,
        num_active_experts,
    )

    if w1_up is not None:
        gate = stk_sdd(hidden_states_permuted, w1, topology)
        up = stk_sdd(hidden_states_permuted, w1_up, topology)
        hidden_states_permuted = stk_mul(activation_fn(gate), up)
    else:
        hidden_states_permuted = stk_sdd(hidden_states_permuted, w1, topology)
        hidden_states_permuted = activation_fn(hidden_states_permuted)

    hidden_states_permuted = stk_dsd(hidden_states_permuted, w2)
    hidden_states_permuted = scatter_tokens(
        hidden_states_permuted,
        indices,
        bin_ids,
        top_k_weights_flat,
        tokens_per_expert,
        padded_bins,
        num_active_experts,
    )
    return hidden_states_permuted


@torch.compiler.disable  # this doesn't play well with compilation
def moe_forward(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    activation_fn: Callable[[stk.Matrix], stk.Matrix],
    num_experts: int,
    num_active_experts: int,
    sort_end_bit: int,
    transpose_sort_end_bit: int,
    expert_size_blocks: torch.Tensor,
    expert_block_offsets: torch.Tensor,
    total_expert_width: int,
    w1_up: torch.Tensor | None = None,
    block_size: int = 128,
    precomputed_offsets_2x2: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """Hybrid MoE forward - dispatches to optimal implementation based on config."""
    # Set device context at the top level for all megablocks/triton ops
    device = hidden_states.device
    if device.type == "cuda":
        torch.cuda.set_device(device)

    num_tokens = hidden_states.shape[0]

    # Special case: 2 experts, both active = every token to both experts
    # Use hardcoded path that eliminates all dynamic indexing
    if num_experts == 2 and num_active_experts == 2:
        return _moe_forward_2x2(
            hidden_states=hidden_states,
            top_k_index=top_k_index,
            top_k_weights=top_k_weights,
            w1=w1,
            w2=w2,
            expert_block_offsets=expert_block_offsets,
            w1_up=w1_up,
            block_size=block_size,
            precomputed_offsets=precomputed_offsets_2x2,
        )

    if _should_use_loop(num_experts, num_tokens):
        return _moe_forward_loop(
            hidden_states=hidden_states,
            top_k_index=top_k_index,
            top_k_weights=top_k_weights,
            w1=w1,
            w2=w2,
            activation_fn=activation_fn,
            num_experts=num_experts,
            expert_block_offsets=expert_block_offsets,
            w1_up=w1_up,
            block_size=block_size,
        )
    else:
        return _moe_forward_sparse(
            hidden_states=hidden_states,
            top_k_index=top_k_index,
            top_k_weights=top_k_weights,
            w1=w1,
            w2=w2,
            activation_fn=activation_fn,
            num_experts=num_experts,
            num_active_experts=num_active_experts,
            sort_end_bit=sort_end_bit,
            transpose_sort_end_bit=transpose_sort_end_bit,
            expert_size_blocks=expert_size_blocks,
            expert_block_offsets=expert_block_offsets,
            total_expert_width=total_expert_width,
            w1_up=w1_up,
            block_size=block_size,
        )
