"""Distilling training loop with resume support and full state saving.

Expects a model with SplitFlexOlmoExperts (from prune_flex_olmo.py) where:
- Expert 0 is frozen (requires_grad=False)
- Expert 1 is trainable (pruned to target width)

Only expert 1's parameters are passed to the optimizer, saving ~25GB
(no gradients/optimizer states for the 4.3B frozen expert 0).
"""

import json
import os
import random
import shutil
import time

import torch
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from liger_kernel.transformers import LigerKLDIVLoss
from transformers import AutoModelForCausalLM
from tqdm import tqdm
from transformers import AutoTokenizer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Model config
width = 2048
TEACHER = "allenai/Flex-math-2x7B-1T"
BASE_MODEL = f"hbfreed/flex-math-pruned-{width}"
HUB_REPO = f"hbfreed/flex-math-pruned-{width}"

# Data config
LOGPROBS_DATASET = "hbfreed/flexolmo-math-logprobs"
SEQ_LEN = 4096  # from generate_logprobs.py chunk_size
TOP_K = 128  # from generate_logprobs.py

# Training config
batch_size = 8
n_epochs = 1
val_split = 0.02  # fraction of data for validation
wandb_project = "flex-olmo-distillation"
TMP_CHECKPOINT = "/dev/shm/checkpoint_tmp"

device = "cuda"
torch.manual_seed(1223)


def load_logprobs_dataset():
    """Load logprobs from HuggingFace and convert to training format."""
    print(f"Loading dataset from {LOGPROBS_DATASET}...")

    # Load all splits (train_0, train_1, etc.)
    ds = load_dataset(LOGPROBS_DATASET)

    # Concatenate all splits
    all_data = []
    for split_name in sorted(ds.keys()):
        print(f"  Loading {split_name}: {len(ds[split_name])} sequences")
        all_data.extend(ds[split_name])

    print(f"Total sequences: {len(all_data)}")
    return all_data


def convert_to_training_format(examples):
    """Convert HF dataset format to training batch format.

    From: input_ids, topk_indices, topk_logprobs
    To: input_ids, position_ids, teacher_indices, teacher_logits, pad_mask
    """
    batch_size = len(examples)
    seq_len = len(examples[0]["input_ids"])

    input_ids = torch.tensor([ex["input_ids"] for ex in examples])
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    teacher_indices = torch.tensor([ex["topk_indices"] for ex in examples])
    teacher_logits = torch.tensor([ex["topk_logprobs"] for ex in examples])
    pad_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)  # no padding

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "teacher_indices": teacher_indices,
        "teacher_logits": teacher_logits,
        "pad_mask": pad_mask,
    }


def create_batches(data, batch_size, shuffle=False, seed=None):
    """Create batches from list of examples."""
    if shuffle:
        rng = random.Random(seed)
        data = data.copy()
        rng.shuffle(data)

    batches = []
    for i in range(0, len(data), batch_size):
        batch_examples = data[i:i + batch_size]
        if len(batch_examples) == batch_size:  # drop incomplete batch
            batches.append(convert_to_training_format(batch_examples))
    return batches


def move_batch_to_device(batch, device):
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    batch["input_ids"] = batch["input_ids"].long()
    batch["position_ids"] = batch["position_ids"].long()
    batch["teacher_indices"] = batch["teacher_indices"].long()
    return batch


def eval_model(student, val_batches, kld_loss, device):
    student.eval()
    val_loss = 0
    for batch in val_batches:
        batch = move_batch_to_device(batch, device)
        with torch.inference_mode():
            student_logprobs = F.log_softmax(
                student(batch["input_ids"], position_ids=batch["position_ids"]).logits,
                dim=-1,
            )
            pad_mask = batch["pad_mask"]
            student_logprobs_at_topk = student_logprobs.gather(-1, batch["teacher_indices"])
            student_logprobs_at_topk = student_logprobs_at_topk[pad_mask]
            teacher_logits = batch["teacher_logits"][pad_mask]
            teacher_logprobs = F.log_softmax(teacher_logits.float(), dim=-1).to(teacher_logits.dtype)
            val_loss += kld_loss(student_logprobs_at_topk.float(), teacher_logprobs.float()).item()
    student.train()
    return val_loss / len(val_batches)


def push_checkpoint(student, tokenizer, optimizer, scheduler, global_step, repo_id):
    """Save model + training state to RAM, upload to HF, cleanup."""
    if os.path.exists(TMP_CHECKPOINT):
        shutil.rmtree(TMP_CHECKPOINT)
    os.makedirs(TMP_CHECKPOINT)

    student.save_pretrained(TMP_CHECKPOINT)
    tokenizer.save_pretrained(TMP_CHECKPOINT)

    training_state = {
        "step": global_step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
        "python_rng_state": random.getstate(),
    }
    torch.save(training_state, os.path.join(TMP_CHECKPOINT, "training_state.pt"))

    with open(os.path.join(TMP_CHECKPOINT, "checkpoint_info.json"), "w") as f:
        json.dump({"step": global_step}, f)

    api = HfApi()
    api.upload_folder(
        folder_path=TMP_CHECKPOINT,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"step_{global_step}",
    )
    shutil.rmtree(TMP_CHECKPOINT)


def load_resume_state(repo_id):
    """Try to load training state from HF repo. Returns (step, state_dict) or (0, None)."""
    try:
        info_path = hf_hub_download(repo_id, "checkpoint_info.json")
        with open(info_path) as f:
            info = json.load(f)

        state_path = hf_hub_download(repo_id, "training_state.pt")
        state = torch.load(state_path, map_location="cpu", weights_only=False)

        print(f"Found checkpoint at step {info['step']}")
        return info["step"], state
    except Exception as e:
        print(f"No checkpoint found or error loading: {e}")
        return 0, None


def main():
    # Load data from HuggingFace
    all_data = load_logprobs_dataset()

    # Split into train/val
    random.seed(1223)
    random.shuffle(all_data)
    val_size = int(len(all_data) * val_split)
    val_data = all_data[:val_size]
    train_data = all_data[val_size:]

    print(f"Train sequences: {len(train_data)}")
    print(f"Val sequences: {len(val_data)}")

    # Create validation batches (fixed)
    val_batches = create_batches(val_data, batch_size)
    print(f"Val batches: {len(val_batches)}")

    # Calculate training steps
    steps_per_epoch = len(train_data) // batch_size
    total_steps = steps_per_epoch * n_epochs
    eval_every = max(1, int(total_steps * 0.1))
    save_every = max(1, min(500, int(total_steps * 0.02)))

    # Check for existing checkpoint
    resume_step, resume_state = load_resume_state(HUB_REPO)

    if resume_step > 0:
        print(f"Resuming from step {resume_step}")
        STUDENT = HUB_REPO
    else:
        print("Starting fresh")
        STUDENT = BASE_MODEL

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")
    print(f"Resume step: {resume_step}")
    print(f"Tokens per step: {batch_size * SEQ_LEN:,}")
    print(f"Eval every: {eval_every} steps")
    print(f"Save every: {save_every} steps")

    # Load model
    print(f"Loading student model from {STUDENT}...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT)
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device)

    # Only optimize trainable params (expert 1)
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params) / 1e9:.2f}B")

    optimizer = torch.optim.AdamW(trainable_params, lr=2e-4, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=4.5e-7
    )

    # Restore state if resuming
    if resume_state is not None:
        print("Restoring optimizer and scheduler state...")
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        torch.set_rng_state(resume_state["rng_state"])
        torch.cuda.set_rng_state(resume_state["cuda_rng_state"])
        random.setstate(resume_state["python_rng_state"])
    else:
        for _ in range(resume_step):
            scheduler.step()

    kld_loss = LigerKLDIVLoss(log_target=True)

    wandb.init(
        project=wandb_project,
        config={
            "teacher": TEACHER,
            "student": STUDENT,
            "seq_len": SEQ_LEN,
            "batch_size": batch_size,
            "steps_per_epoch": steps_per_epoch,
            "n_epochs": n_epochs,
            "total_steps": total_steps,
            "resume_step": resume_step,
            "lr_start": 2e-4,
            "lr_end": 4.5e-7,
        },
        resume="allow",
    )

    # Create HF repo if needed
    api = HfApi()
    try:
        api.create_repo(HUB_REPO, exist_ok=True)
    except Exception as e:
        print(f"Repo creation note: {e}")

    # Training loop
    global_step = 0
    for epoch in range(n_epochs):
        # Shuffle and batch training data each epoch
        train_batches = create_batches(train_data, batch_size, shuffle=True, seed=1223 + epoch)

        pbar = tqdm(train_batches, desc=f"Epoch {epoch + 1}/{n_epochs}")
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
            student_logprobs_at_topk = student_logprobs.gather(-1, batch["teacher_indices"])
            student_logprobs_at_topk = student_logprobs_at_topk[pad_mask]
            teacher_logits = batch["teacher_logits"][pad_mask]
            teacher_logprobs = F.log_softmax(teacher_logits.float(), dim=-1).to(teacher_logits.dtype)

            loss = kld_loss(student_logprobs_at_topk.float(), teacher_logprobs.float())
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            step_time = time.time() - step_start_time
            num_tokens = pad_mask.sum().item()
            throughput = num_tokens / step_time

            wandb.log({
                "train/loss": loss.item(),
                "train/lr": scheduler.get_last_lr()[0],
                "train/grad_norm": grad_norm.item(),
                "train/throughput": throughput,
                "train/step": global_step,
            })
            pbar.set_postfix(loss=f"{loss.item():.4f}", toks=f"{throughput:.0f}")

            if (global_step + 1) % save_every == 0:
                print(f"\nPushing checkpoint at step {global_step} to {HUB_REPO}...")
                push_checkpoint(student, tokenizer, optimizer, scheduler, global_step, HUB_REPO)

            if (global_step + 1) % eval_every == 0:
                val_loss = eval_model(student, val_batches, kld_loss, device)
                wandb.log({"eval/loss": val_loss, "train/step": global_step})
                print(f"\nStep {global_step}: val_loss={val_loss:.4f}")

            global_step += 1

    # Final push
    print(f"\nPushing final model to {HUB_REPO}...")
    push_checkpoint(student, tokenizer, optimizer, scheduler, global_step, HUB_REPO)

    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
