"""Distilling training loop with resume support and full state saving.

Expects a pruned model (from prune_flex_olmo.py) where:
- Expert 0 is frozen (requires_grad=False)
- Expert 1 is trainable (pruned to target width)

Only expert 1's parameters are passed to the optimizer.
"""

import argparse
import os
import random
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from liger_kernel.transformers import LigerKLDIVLoss
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

import wandb
from prune_flex_olmo import load_pruned_model, save_pruned_model

# old apis for torch.compile compatibility
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Model config
WIDTHS = [8192, 5504, 2048]

# Data config
LOGPROBS_DATASET = "hbfreed/flexolmo-math-logprobs"

# Training config
DEFAULT_BATCH_SIZE = 2
SEQ_LEN = 4096
TARGET_TOKENS_PER_STEP = 262144  # 256k tokens per optimizer step
n_epochs = 1
val_split = 0.02  # fraction of data for validation
wandb_project = "flex-olmo-distillation"

torch.manual_seed(1223)

# DDP setup - detect if launched with torchrun via RANK env var
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    init_process_group(backend="nccl", device_id=torch.device(device))
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    device = "cuda"
    master_process = True


class SkipShuffleSampler(Sampler):
    """Shuffled sampler that can skip the first N samples (for fast resume).

    With DDP, shards data across GPUs using rank/world_size.
    """

    def __init__(
        self,
        data_source,
        skip: int = 0,
        seed: int = 1223,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.data_source = data_source
        self.skip = skip
        self.seed = seed
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        indices = torch.randperm(len(self.data_source), generator=g).tolist()
        # Shard across GPUs
        indices = indices[self.rank :: self.world_size]
        return iter(indices[self.skip :])

    def __len__(self):
        return len(self.data_source) // self.world_size - self.skip


def load_logprobs_dataset():
    """Load logprobs from HuggingFace, keeping as memory-mapped Dataset."""
    from datasets import concatenate_datasets

    if master_process:
        print(f"Loading dataset from {LOGPROBS_DATASET}...")

    # Load all splits (train_0, train_1, etc.)
    ds = load_dataset(LOGPROBS_DATASET)

    # Concatenate all splits into single Dataset (stays memory-mapped)
    splits = []
    for split_name in sorted(ds.keys()):
        if master_process:
            print(f"  Loading {split_name}: {len(ds[split_name])} sequences")
        splits.append(ds[split_name])

    combined = concatenate_datasets(splits)
    if master_process:
        print(f"Total sequences: {len(combined)}")
    return combined


class LogprobsDataset(Dataset):
    """PyTorch Dataset wrapper for HF dataset with prefetching support."""

    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "topk_indices": torch.tensor(item["topk_indices"]),
            "topk_logprobs": torch.tensor(item["topk_logprobs"]),
        }


def collate_fn(batch):
    """Collate batch items into training tensors."""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    topk_indices = torch.stack([b["topk_indices"] for b in batch])
    topk_logprobs = torch.stack([b["topk_logprobs"] for b in batch])

    bs, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(bs, -1).contiguous()

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "teacher_indices": topk_indices,
        "teacher_logits": topk_logprobs,
    }


def move_batch_to_device(batch, device):
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    batch["input_ids"] = batch["input_ids"].long()
    batch["position_ids"] = batch["position_ids"].long()
    batch["teacher_indices"] = batch["teacher_indices"].long()
    return batch


def eval_model(student, val_loader, kld_loss, device):
    student.eval()
    val_loss = 0
    n_batches = 0
    for batch in tqdm(val_loader, desc="Validating"):
        batch = move_batch_to_device(batch, device)
        with torch.inference_mode():
            student_logprobs = F.log_softmax(
                student(batch["input_ids"], position_ids=batch["position_ids"]).logits,
                dim=-1,
            )
            student_logprobs_at_topk = student_logprobs.gather(
                -1, batch["teacher_indices"]
            )
            teacher_logprobs = F.log_softmax(batch["teacher_logits"].float(), dim=-1).to(
                batch["teacher_logits"].dtype
            )
            val_loss += kld_loss(
                student_logprobs_at_topk.float(), teacher_logprobs.float()
            ).item()
            n_batches += 1
    student.train()
    return val_loss / n_batches if n_batches > 0 else 0


def save_checkpoint(
    student,
    optimizer,
    scheduler,
    global_step,
    checkpoint_dir,
    importance_type,
    expert_1_width,
    push_to_hub=False,
    hub_repo=None,
):
    """Save model + training state locally, optionally push to HF."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save model weights
    save_pruned_model(student, checkpoint_dir, importance_type, expert_1_width)

    # Save training state
    training_state = {
        "step": global_step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
        "python_rng_state": random.getstate(),
    }
    torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pt"))
    print(f"Saved checkpoint at step {global_step} to {checkpoint_dir}")

    if push_to_hub and hub_repo:
        api = HfApi()
        api.upload_folder(
            folder_path=checkpoint_dir,
            repo_id=hub_repo,
            repo_type="model",
            commit_message=f"step_{global_step}",
        )
        print(f"Pushed to {hub_repo}")


def load_checkpoint(checkpoint_dir, hub_repo=None):
    """Load training state from local checkpoint or HF Hub. Returns (step, state) or (0, None)."""
    state_path = os.path.join(checkpoint_dir, "training_state.pt")

    # Try downloading from HF if local files missing
    if hub_repo and not os.path.exists(state_path):
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            hf_hub_download(hub_repo, "training_state.pt", local_dir=checkpoint_dir)
            if master_process:
                print(f"Downloaded training state from {hub_repo}")
        except Exception as e:
            if master_process:
                print(f"No training state on HF Hub: {e}")

    if not os.path.exists(state_path):
        return 0, None

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if master_process:
        print(f"Found checkpoint at step {state['step']}")
    return state["step"], state


def parse_args():
    parser = argparse.ArgumentParser(description="Distill pruned FlexOlmo")
    parser.add_argument(
        "--width",
        "-w",
        type=int,
        choices=WIDTHS,
        required=True,
        help=f"Expert 1 width. Choices: {WIDTHS}",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push checkpoints to HuggingFace Hub (default: local only)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint if available",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    batch_size = args.batch_size
    # Derive grad_accum_steps to hit target tokens per step
    grad_accum_steps = TARGET_TOKENS_PER_STEP // (batch_size * SEQ_LEN * ddp_world_size)

    model_path = f"pruned_models/flex_math_{args.width}"
    checkpoint_dir = f"checkpoints/flex_math_{args.width}"
    hub_repo = f"hbfreed/flex-math-{args.width}"

    # Check for existing checkpoint
    if args.resume:
        resume_step, resume_state = load_checkpoint(checkpoint_dir, hub_repo)
    else:
        resume_step, resume_state = 0, None

    # Load data from HuggingFace (memory-mapped)
    dataset = load_logprobs_dataset()

    # Split into train/val using HF Dataset's method
    split = dataset.train_test_split(test_size=val_split, seed=1223)
    train_data = split["train"]
    val_data = split["test"]

    if master_process:
        print(f"Train sequences: {len(train_data)}")
        print(f"Val sequences: {len(val_data)}")

    # Load model (from checkpoint if resuming, otherwise from pruned model)
    if resume_step > 0:
        if master_process:
            print(f"Resuming from checkpoint at step {resume_step}...")
        student = load_pruned_model(checkpoint_dir, device=device)
    else:
        if master_process:
            print(f"Loading student model from {model_path}...")
        student = load_pruned_model(model_path, device=device)

    # Wrap with DDP for multi-GPU training
    if ddp:
        student = DDP(student, device_ids=[ddp_local_rank])

    # Create DataLoaders with prefetching
    # Use SkipShuffleSampler for fast resume (skips at sampler level, not iteration level)
    # Skip based on micro-batches: resume_step * grad_accum_steps * batch_size
    train_dataset = LogprobsDataset(train_data)
    skip_samples = resume_step * grad_accum_steps * batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=SkipShuffleSampler(
            train_dataset, skip=skip_samples, rank=ddp_rank, world_size=ddp_world_size
        ),
        num_workers=8
        pin_memory=True,
        prefetch_factor=2,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        LogprobsDataset(val_data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=True,
        collate_fn=collate_fn,
    )
    if master_process:
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Calculate training steps (optimizer steps, not micro-batches)
    # With DDP, each GPU processes micro_batches_per_epoch / world_size batches
    micro_batches_per_epoch = len(train_loader)
    steps_per_epoch = micro_batches_per_epoch // grad_accum_steps
    total_steps = steps_per_epoch * n_epochs
    eval_every = max(1, int(total_steps * 0.1))
    save_every = max(1, min(1000, int(total_steps * 0.05)))

    tokens_per_step = batch_size * SEQ_LEN * grad_accum_steps * ddp_world_size
    if master_process:
        print(f"Gradient accumulation steps: {grad_accum_steps}")
        print(f"Micro-batches per epoch: {micro_batches_per_epoch}")
        print(f"Optimizer steps per epoch: {steps_per_epoch}")
        print(f"Total optimizer steps: {total_steps}")
        print(f"Tokens per optimizer step: {tokens_per_step:,}")
        print(f"Eval every: {eval_every} steps")
        print(f"Save every: {save_every} steps")
        if ddp:
            print(f"DDP: {ddp_world_size} GPUs")

    # Only optimize trainable params (expert 1)
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    if master_process:
        print(
            f"Trainable parameters: {sum(p.numel() for p in trainable_params) / 1e9:.2f}B"
        )

    optimizer = torch.optim.AdamW(trainable_params, lr=2e-4, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=4.5e-7
    )

    # Restore optimizer/scheduler state if resuming
    if resume_state is not None:
        if master_process:
            print("Restoring optimizer and scheduler state...")
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        torch.set_rng_state(resume_state["rng_state"])
        torch.cuda.set_rng_state(resume_state["cuda_rng_state"])
        random.setstate(resume_state["python_rng_state"])

    kld_loss = LigerKLDIVLoss(log_target=True)

    if master_process:
        wandb.init(
            project=wandb_project,
            name=hub_repo,
            mode="online",
            config={
                "width": args.width,
                "model_path": model_path,
                "seq_len": SEQ_LEN,
                "batch_size": batch_size,
                "grad_accum_steps": grad_accum_steps,
                "effective_batch_size": batch_size * grad_accum_steps * ddp_world_size,
                "tokens_per_step": tokens_per_step,
                "steps_per_epoch": steps_per_epoch,
                "n_epochs": n_epochs,
                "total_steps": total_steps,
                "resume_step": resume_step,
                "lr_start": 2e-4,
                "lr_end": 4.5e-7,
                "ddp_world_size": ddp_world_size,
            },
            resume="allow",
        )

        # Create HF repo if pushing
        if args.push_to_hub:
            api = HfApi()
            try:
                api.create_repo(hub_repo, exist_ok=True)
            except Exception as e:
                print(f"Repo creation note: {e}")

    # Training loop (skip is handled at sampler level, so we start at resume_step)
    global_step = resume_step
    micro_step = 0
    accumulated_loss = 0.0
    accumulated_tokens = 0
    step_start_time = time.time()

    for epoch in range(n_epochs):
        pbar = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{n_epochs}",
            disable=not master_process,
        )
        for batch in pbar:
            batch = move_batch_to_device(batch, device)

            student_logprobs = F.log_softmax(
                student(batch["input_ids"], position_ids=batch["position_ids"]).logits,
                dim=-1,
            )

            student_logprobs_at_topk = student_logprobs.gather(
                -1, batch["teacher_indices"]
            )
            teacher_logprobs = F.log_softmax(batch["teacher_logits"].float(), dim=-1).to(
                batch["teacher_logits"].dtype
            )

            # Scale loss for gradient accumulation
            loss = kld_loss(student_logprobs_at_topk.float(), teacher_logprobs.float())
            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

            accumulated_loss += loss.item()
            accumulated_tokens += batch["input_ids"].numel()
            micro_step += 1

            # Only update weights after accumulating enough gradients
            if micro_step % grad_accum_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params, max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                step_time = time.time() - step_start_time
                avg_loss = accumulated_loss / grad_accum_steps
                throughput = accumulated_tokens / step_time

                if master_process:
                    wandb.log(
                        {
                            "train/loss": avg_loss,
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/grad_norm": grad_norm.item(),
                            "train/throughput": throughput,
                            "train/step": global_step,
                        }
                    )
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", toks=f"{throughput:.0f}")

                if master_process and (global_step + 1) % save_every == 0:
                    # Save unwrapped model for DDP
                    raw_model = student.module if ddp else student
                    save_checkpoint(
                        raw_model,
                        optimizer,
                        scheduler,
                        global_step,
                        checkpoint_dir,
                        args.width,
                        args.push_to_hub,
                        hub_repo,
                    )

                if master_process and (global_step + 1) % eval_every == 0:
                    raw_model = student.module if ddp else student
                    val_loss = eval_model(raw_model, val_loader, kld_loss, device)
                    wandb.log({"eval/loss": val_loss, "train/step": global_step})
                    print(f"\nStep {global_step}: val_loss={val_loss:.4f}")

                global_step += 1

                # Reset accumulators for next step
                accumulated_loss = 0.0
                accumulated_tokens = 0
                step_start_time = time.time()

    # Final save (only on master)
    if master_process:
        raw_model = student.module if ddp else student
        save_checkpoint(
            raw_model,
            optimizer,
            scheduler,
            global_step,
            checkpoint_dir,
            args.importance,
            args.width,
            args.push_to_hub,
            hub_repo,
        )
        wandb.finish()
        print("Done!")

    # Cleanup DDP
    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
