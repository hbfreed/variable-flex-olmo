"""Distilling training loop with resume support and full state saving.

Expects a pruned model (from prune_flex_olmo.py) where:
- Expert 0 is frozen (requires_grad=False)
- Expert 1 is trainable (pruned to target width)

Only expert 1's parameters are passed to the optimizer.
"""

import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from liger_kernel.transformers import LigerKLDIVLoss
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import wandb
from prune_flex_olmo import load_pruned_model, save_pruned_model

# old apis for torch.compile compatibility
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Model config
WIDTHS = [8192, 5504, 2048]
IMPORTANCE_TYPES = ["math", "general"]

# Data config
LOGPROBS_DATASET = "hbfreed/flexolmo-math-logprobs"
TOP_K = 128  # from generate_logprobs.py

# Training config
DEFAULT_BATCH_SIZE = 2
n_epochs = 1
val_split = 0.02  # fraction of data for validation
wandb_project = "flex-olmo-distillation"

device = "cuda"
torch.manual_seed(1223)


def load_logprobs_dataset():
    """Load logprobs from HuggingFace, keeping as memory-mapped Dataset."""
    from datasets import concatenate_datasets

    print(f"Loading dataset from {LOGPROBS_DATASET}...")

    # Load all splits (train_0, train_1, etc.)
    ds = load_dataset(LOGPROBS_DATASET)

    # Concatenate all splits into single Dataset (stays memory-mapped)
    splits = []
    for split_name in sorted(ds.keys()):
        print(f"  Loading {split_name}: {len(ds[split_name])} sequences")
        splits.append(ds[split_name])

    combined = concatenate_datasets(splits)
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
    pad_mask = torch.ones(bs, seq_len, dtype=torch.bool)

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "teacher_indices": topk_indices,
        "teacher_logits": topk_logprobs,
        "pad_mask": pad_mask,
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
    for batch in val_loader:
        batch = move_batch_to_device(batch, device)
        with torch.inference_mode():
            student_logprobs = F.log_softmax(
                student(batch["input_ids"], position_ids=batch["position_ids"]).logits,
                dim=-1,
            )
            pad_mask = batch["pad_mask"]
            student_logprobs_at_topk = student_logprobs.gather(
                -1, batch["teacher_indices"]
            )
            student_logprobs_at_topk = student_logprobs_at_topk[pad_mask]
            teacher_logits = batch["teacher_logits"][pad_mask]
            teacher_logprobs = F.log_softmax(teacher_logits.float(), dim=-1).to(
                teacher_logits.dtype
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

    with open(os.path.join(checkpoint_dir, "checkpoint_info.json"), "w") as f:
        json.dump({"step": global_step}, f)

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
    info_path = os.path.join(checkpoint_dir, "checkpoint_info.json")
    state_path = os.path.join(checkpoint_dir, "training_state.pt")

    # Try downloading from HF if local files missing
    if hub_repo and not os.path.exists(state_path):
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            hf_hub_download(hub_repo, "training_state.pt", local_dir=checkpoint_dir)
            hf_hub_download(hub_repo, "checkpoint_info.json", local_dir=checkpoint_dir)
            print(f"Downloaded training state from {hub_repo}")
        except Exception as e:
            print(f"No training state on HF Hub: {e}")

    if not os.path.exists(info_path) or not os.path.exists(state_path):
        return 0, None

    with open(info_path) as f:
        info = json.load(f)

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    print(f"Found checkpoint at step {info['step']}")
    return info["step"], state


def parse_args():
    parser = argparse.ArgumentParser(description="Distill pruned FlexOlmo")
    parser.add_argument(
        "--importance",
        "-i",
        choices=IMPORTANCE_TYPES,
        required=True,
        help="Which importance scores were used for pruning",
    )
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
        "--debug", action="store_true", help="Debug mode: overfit on 10 batches"
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers for prefetching (default: 4)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=4096,
        help=f"Sequence length (default: {4096})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    batch_size = args.batch_size

    model_path = f"pruned_models/flex_{args.importance}_{args.width}"
    checkpoint_dir = f"checkpoints/flex_{args.importance}_{args.width}"
    hub_repo = f"hbfreed/flex-{args.importance}-{args.width}"

    # Check for existing checkpoint
    resume_step, resume_state = load_checkpoint(checkpoint_dir, hub_repo)

    # Load data from HuggingFace (memory-mapped)
    dataset = load_logprobs_dataset()

    # Split into train/val using HF Dataset's method
    split = dataset.train_test_split(test_size=val_split, seed=1223)
    train_data = split["train"]
    val_data = split["test"]

    print(f"Train sequences: {len(train_data)}")
    print(f"Val sequences: {len(val_data)}")

    # Load model (from checkpoint if resuming, otherwise from pruned model)
    if resume_step > 0:
        print(f"Resuming from checkpoint at step {resume_step}...")
        student = load_pruned_model(checkpoint_dir, device=device)
    else:
        print(f"Loading student model from {model_path}...")
        student = load_pruned_model(model_path, device=device)

    student.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    print(f"Using batch size: {batch_size}")

    # Debug mode: overfit on 10 batches
    if args.debug:
        print("\n*** DEBUG MODE: overfitting on 10 batches ***\n")
        train_data = train_data.select(range(min(10 * batch_size, len(train_data))))
        val_data = val_data.select(range(min(batch_size, len(val_data))))

    # Create DataLoaders with prefetching
    train_loader = DataLoader(
        LogprobsDataset(train_data),
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        LogprobsDataset(val_data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=True,
        collate_fn=collate_fn,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Calculate training steps
    steps_per_epoch = len(train_loader)
    if args.debug:
        n_epochs_actual = 10  # 10 passes over 10 batches = 100 steps
        total_steps = steps_per_epoch * n_epochs_actual
    else:
        n_epochs_actual = n_epochs
        total_steps = steps_per_epoch * n_epochs_actual
    eval_every = max(1, int(total_steps * 0.1))
    save_every = (
        max(1, min(500, int(total_steps * 0.5)))
        if args.debug
        else max(1, min(1000, int(total_steps * 0.05)))
    )

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")
    print(f"Resume step: {resume_step}")
    print(f"Tokens per step: {batch_size * args.seq_len:,}")
    print(f"Eval every: {eval_every} steps")
    print(f"Save every: {save_every} steps")

    # Only optimize trainable params (expert 1)
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    print(
        f"Trainable parameters: {sum(p.numel() for p in trainable_params) / 1e9:.2f}B"
    )

    optimizer = torch.optim.AdamW(trainable_params, lr=2e-4, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=4.5e-7
    )

    # Restore optimizer/scheduler state if resuming
    if resume_state is not None:
        print("Restoring optimizer and scheduler state...")
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        torch.set_rng_state(resume_state["rng_state"])
        torch.cuda.set_rng_state(resume_state["cuda_rng_state"])
        random.setstate(resume_state["python_rng_state"])

    kld_loss = LigerKLDIVLoss(log_target=True)

    wandb.init(
        project=wandb_project,
        mode="offline" if args.debug else "online",
        config={
            "importance": args.importance,
            "width": args.width,
            "model_path": model_path,
            "seq_len": args.seq_len,
            "batch_size": batch_size,
            "steps_per_epoch": steps_per_epoch,
            "n_epochs": n_epochs_actual,
            "total_steps": total_steps,
            "resume_step": resume_step,
            "lr_start": 2e-4,
            "lr_end": 4.5e-7,
            "debug": args.debug,
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

    # Training loop
    global_step = 0
    for epoch in range(n_epochs_actual):
        pbar = tqdm(
            train_loader,
            total=steps_per_epoch,
            desc=f"Epoch {epoch + 1}/{n_epochs_actual}",
        )
        for batch in pbar:
            # Skip to resume point
            if global_step < resume_step:
                global_step += 1
                pbar.set_postfix(skipping=f"{global_step}/{resume_step}")
                continue

            step_start_time = time.time()
            batch = move_batch_to_device(batch, device)

            student_logprobs = F.log_softmax(
                student(batch["input_ids"], position_ids=batch["position_ids"]).logits,
                dim=-1,
            )

            pad_mask = batch["pad_mask"]
            student_logprobs_at_topk = student_logprobs.gather(
                -1, batch["teacher_indices"]
            )
            student_logprobs_at_topk = student_logprobs_at_topk[pad_mask]
            teacher_logits = batch["teacher_logits"][pad_mask]
            teacher_logprobs = F.log_softmax(teacher_logits.float(), dim=-1).to(
                teacher_logits.dtype
            )

            loss = kld_loss(student_logprobs_at_topk.float(), teacher_logprobs.float())
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            step_time = time.time() - step_start_time
            num_tokens = pad_mask.sum().item()
            throughput = num_tokens / step_time

            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/grad_norm": grad_norm.item(),
                    "train/throughput": throughput,
                    "train/step": global_step,
                }
            )
            pbar.set_postfix(loss=f"{loss.item():.4f}", toks=f"{throughput:.0f}")

            if (global_step + 1) % save_every == 0:
                save_checkpoint(
                    student,
                    optimizer,
                    scheduler,
                    global_step,
                    checkpoint_dir,
                    args.importance,
                    args.width,
                    args.push_to_hub,
                    hub_repo,
                )

            if (global_step + 1) % eval_every == 0:
                val_loss = eval_model(student, val_loader, kld_loss, device)
                wandb.log({"eval/loss": val_loss, "train/step": global_step})
                print(f"\nStep {global_step}: val_loss={val_loss:.4f}")

            global_step += 1

    # Final save
    save_checkpoint(
        student,
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


if __name__ == "__main__":
    main()
