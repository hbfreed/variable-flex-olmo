"""Distilling training loop with resume support and full state saving."""

import glob
import json
import os
import random
import shutil
import time

import torch
import torch.nn.functional as F
import wandb
from huggingface_hub import HfApi, hf_hub_download
from liger_kernel.transformers import AutoLigerKernelForCausalLM, LigerKLDIVLoss
from tqdm import tqdm
from transformers import AutoTokenizer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

TEACHER = "allenai/Olmo-3-7B-Instruct"
BASE_MODEL = "hbfreed/pruned_olmo3_4096_16_29"
HUB_REPO = "hbfreed/pruned_olmo3_4096_16_29_distilled"
wandb_project = "olmo-distillation"

batch_size = 8
n_epochs = 1
CHUNKS_DIR = "cache/chunks"
TMP_CHECKPOINT = "/dev/shm/checkpoint_tmp"

device = "cuda"
torch.manual_seed(1223)


def move_batch_to_device(batch, device):
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    if batch["input_ids"].dtype != torch.long:
        batch["input_ids"] = batch["input_ids"].long()
    if batch["position_ids"].dtype != torch.long:
        batch["position_ids"] = batch["position_ids"].long()
    if batch["teacher_indices"].dtype != torch.long:
        batch["teacher_indices"] = batch["teacher_indices"].long()
    return batch


def stack_sequences(batch_group, pin_memory=False):
    stacked = {
        key: torch.cat([seq[key] for seq in batch_group], dim=0)
        for key in batch_group[0].keys()
    }
    if pin_memory:
        stacked = {k: v.pin_memory() for k, v in stacked.items()}
    return stacked


def iter_packed_batches_from_file(path, pin_memory=False):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, list):
        for batch in data:
            if pin_memory:
                batch = {k: v.pin_memory() for k, v in batch.items()}
            yield batch
    else:
        if pin_memory:
            data = {k: v.pin_memory() for k, v in data.items()}
        yield data


def iter_batched_sequences_from_files(chunk_files, batch_size, pin_memory=False):
    buffer = []
    for path in chunk_files:
        for seq in iter_packed_batches_from_file(path, pin_memory=False):
            buffer.append(seq)
            if len(buffer) == batch_size:
                yield stack_sequences(buffer, pin_memory=pin_memory)
                buffer = []
    if buffer:
        yield stack_sequences(buffer, pin_memory=pin_memory)


def batch_packed_sequences(packed_sequences, batch_size, pin_memory=True):
    batched = []
    for batch_start in range(0, len(packed_sequences), batch_size):
        batch_group = packed_sequences[batch_start : batch_start + batch_size]
        stacked = {
            key: torch.cat([seq[key] for seq in batch_group], dim=0)
            for key in batch_group[0].keys()
        }
        if pin_memory:
            stacked = {k: v.pin_memory() for k, v in stacked.items()}
        batched.append(stacked)
    return batched


def eval(student, val_batches, kld_loss, device):
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
    student.train()
    return val_loss / len(val_batches)


def push_checkpoint(student, tokenizer, optimizer, scheduler, global_step, repo_id):
    """Save model + training state to RAM, upload to HF, cleanup."""
    if os.path.exists(TMP_CHECKPOINT):
        shutil.rmtree(TMP_CHECKPOINT)
    os.makedirs(TMP_CHECKPOINT)

    # Save model and tokenizer
    student.save_pretrained(TMP_CHECKPOINT)
    tokenizer.save_pretrained(TMP_CHECKPOINT)

    # Save training state
    training_state = {
        "step": global_step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
        "python_rng_state": random.getstate(),
    }
    torch.save(training_state, os.path.join(TMP_CHECKPOINT, "training_state.pt"))

    # Save step number as JSON for easy reading
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
        # Try to download checkpoint info
        info_path = hf_hub_download(repo_id, "checkpoint_info.json")
        with open(info_path) as f:
            info = json.load(f)

        # Download training state
        state_path = hf_hub_download(repo_id, "training_state.pt")
        state = torch.load(state_path, map_location="cpu", weights_only=False)

        print(f"Found checkpoint at step {info['step']}")
        return info["step"], state
    except Exception as e:
        print(f"No checkpoint found or error loading: {e}")
        return 0, None


# Load data
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
pack_length = 6144

train_chunk_files = sorted(glob.glob(f"{CHUNKS_DIR}/train_*.pt"))
val_path = f"{CHUNKS_DIR}/val.pt"

if not train_chunk_files or not os.path.exists(val_path):
    print(f"ERROR: Pre-packed data not found in {CHUNKS_DIR}/")
    import sys

    sys.exit(1)

print(f"Loading pre-packed data from {CHUNKS_DIR}/...")
val_packed = torch.load(val_path, weights_only=False)
print(f"Found {len(train_chunk_files)} train chunks, {len(val_packed)} val batches")

# Count total batches
total_train_batches = 0
for path in train_chunk_files:
    batches = torch.load(path, weights_only=False)
    total_train_batches += len(batches)
    del batches

steps_per_epoch = total_train_batches // batch_size
total_steps = steps_per_epoch * n_epochs
eval_every = max(1, int(total_steps * 0.1))
save_every = max(1, min(500, int(total_steps * 0.02)))

# Check for existing checkpoint
resume_step, resume_state = load_resume_state(HUB_REPO)

# if resume_step > 0:
#    print(f"Resuming from step {resume_step}")
#    STUDENT = HUB_REPO  # Load from checkpoint
# else:
#    print("Starting fresh")
#    STUDENT = BASE_MODEL

# CHANGE THIS ONCE WE GET RUNNING AGAIN
# HARDCODED FOR THIS RESUME - auto-detect will work after first checkpoint
resume_step = 1391
resume_state = None  # No optimizer state saved from previous run
STUDENT = HUB_REPO  # Load from checkpoint

remaining_steps = total_steps - resume_step

print(f"Total train sequences: {total_train_batches}")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Total steps: {total_steps}")
print(f"Resume step: {resume_step}")
print(f"Remaining steps: {remaining_steps}")
print(f"Tokens per step: {batch_size * pack_length:,}")
print(f"Eval every: {eval_every} steps")
print(f"Save every: {save_every} steps (pushing to HF)")

val_batches = batch_packed_sequences(val_packed, batch_size, pin_memory=False)
print(f"Created {len(val_batches)} validation batches")

# Load model
print(f"Loading student model from {STUDENT}...")
student = AutoLigerKernelForCausalLM.from_pretrained(
    STUDENT, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
).to(device)

optimizer = torch.optim.AdamW(student.parameters(), lr=2e-4, fused=True)
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
else:
    # Fast-forward scheduler if we have a checkpoint but no state (shouldn't happen now)
    for _ in range(resume_step):
        scheduler.step()

kld_loss = LigerKLDIVLoss(log_target=True)

wandb.init(
    project=wandb_project,
    config={
        "teacher": TEACHER,
        "student": STUDENT,
        "pack_length": pack_length,
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

global_step = 0
for epoch in range(n_epochs):
    epoch_chunks = list(train_chunk_files)
    rng = random.Random(1223 + epoch)
    rng.shuffle(epoch_chunks)

    train_iter = iter_batched_sequences_from_files(
        epoch_chunks, batch_size, pin_memory=False
    )

    pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}/{n_epochs}")
    for _ in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            print("Warning: ran out of batches before steps_per_epoch")
            break

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
        teacher_logprobs = F.log_softmax(teacher_logits.float(), dim=-1).to(
            teacher_logits.dtype
        )

        loss = kld_loss(student_logprobs_at_topk.float(), teacher_logprobs.float())
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
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

        # Push checkpoint to HF
        if (global_step + 1) % save_every == 0:
            print(f"\nPushing checkpoint at step {global_step} to {HUB_REPO}...")
            push_checkpoint(
                student, tokenizer, optimizer, scheduler, global_step, HUB_REPO
            )

        # Eval
        if (global_step + 1) % eval_every == 0:
            val_loss = eval(student, val_batches, kld_loss, device)
            wandb.log({"eval/loss": val_loss, "train/step": global_step})
            print(f"\nStep {global_step}: val_loss={val_loss:.4f}")

        global_step += 1

# Final push
print(f"\nPushing final model to {HUB_REPO}...")
push_checkpoint(student, tokenizer, optimizer, scheduler, global_step, HUB_REPO)

wandb.finish()
print("Done!")
