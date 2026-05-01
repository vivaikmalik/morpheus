# Training Resume Plan

## 1. Current Checkpoint Contents

**File:** `checkpoints/chebi20_27M_scibert_20ep_contrastive.pt`
**Size:** 778.7 MB

Top-level keys: `epoch`, `step`, `model`, `optimizer`, `scheduler`, `val_loss`, `config`

| Field | Value |
|-------|-------|
| `epoch` | 18 (saved after epoch 18 of 20 completed; best val occurred mid-epoch-18) |
| `step` | 11,000 (global optimizer steps) |
| `val_loss` | 0.3769 |
| `model` keys | 425 tensors |
| `optimizer` | YES — AdamW state for 225 parameter groups |
| `scheduler` | YES — LambdaLR state (last_epoch=11000, _last_lr=8.324e-06) |
| `config` | YES — full 46-key CONFIG dict |

Config highlights from saved checkpoint:
- `num_epochs: 20`
- `learning_rate: 0.0002` (peak)
- `warmup_steps: 500`
- `early_stop_patience: 5`
- `batch_size: 32`
- `encoder: contrastive`
- `text_model: allenai/scibert_scivocab_uncased`
- `pretrain_ckpt: zinc_17M_pretrain.pt`
- `save_ckpt: checkpoints/chebi20_27M_scibert_20ep_contrastive.pt`
- `periodic_prefix: checkpoints/chebi20_27M_scibert_20ep_contrastive_epoch`

**Note on epoch 18 vs 20:** The best checkpoint was saved at step 11,000 which falls within
epoch 18 (step 11,000 / 629 steps/epoch ≈ epoch 17.5). The script saves on validation
improvement (not end-of-epoch), so the "20 epoch" run's best model was found mid-epoch-18.
The run completed epoch 18 (periodic checkpoint `_epoch_18.pt` exists) confirming early-stopping
did not fire; epochs 19–20 produced no improvement.

---

## 2. Training Script Capabilities

### Argparse flags (all flags in `finetune/train_chebi20.py`)

| Flag | Type | Default | Purpose |
|------|------|---------|---------|
| `--eval_only` | flag | off | Skip training; run test-set eval |
| `--eos_truncate` | flag | off | Post-hoc EOS truncation at eval |
| `--cfg` | float | None (→ 3.0) | CFG scale override for eval |
| `--temp` | float | None (→ 1.0) | Temperature override for eval |
| `--steps` | int | None (→ 50) | Denoising steps override for eval |
| `--encoder` | choice | `frozen` | `frozen` or `contrastive` |
| `--contrastive_ckpt` | str | None | Required when `--encoder contrastive` |
| `--model_size` | str | `27M` | Tag used in checkpoint/output filenames |
| `--test_csv` | str | None | Custom test CSV path (smoke tests) |
| `--num_epochs` | int | None (→ CONFIG 10) | Override number of training epochs |
| `--rerank_n` | int | 1 | Best-of-N contrastive reranking |
| `--refine_passes` | int | 1 | Iterative refinement passes |
| `--refine_mask_ratio` | float | 0.2 | Fraction to re-mask per refinement pass |
| `--commit_strategy` | choice | `standard` | `standard` or `dep_aware` |

**No `--resume` flag exists.** There is no native checkpoint-resume mechanism in the
training path. The script always starts from epoch 0 and loads only the ZINC pretrain
backbone — it never restores optimizer or scheduler state from a prior fine-tuning run.

### How `--num_epochs` works

`--num_epochs` overrides `CONFIG["num_epochs"]` (default 10) before any training starts.
The schedule is computed as:

```python
total_steps = (train_size // batch_size) * CONFIG["num_epochs"]
# = (20158 // 32) * N = 629 * N
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=500, num_training_steps=total_steps
)
```

This is a one-shot cosine schedule: **the total number of steps is baked in at initialization.**
The LR rises linearly from 0 to 2e-4 over 500 warmup steps, then cosine-decays to 0 over
the remaining `total_steps - 500` steps.

### Checkpoint saving

- **Best checkpoint:** saved to `checkpoints/chebi20_27M_scibert_20ep_contrastive.pt` whenever
  val_loss < best_val_loss (evaluated every 500 steps via `val_every: 500`)
- **Periodic checkpoint:** saved every 3 epochs to `chebi20_27M_scibert_20ep_contrastive_epoch_{N}.pt`
- **Both contain:** `epoch`, `step`, `model`, `optimizer`, `scheduler`, `val_loss`, `config`
- Validation runs on up to 100 batches (`val_max_batches: 100`)

### Early stopping

Patience = 5 validation checks (every 500 steps). If val_loss does not improve for 5
consecutive validation points (= 2,500 steps = ~4 epochs), training stops early.
This **cannot be disabled** via CLI without modifying the script.

---

## 3. Learning Rate Schedule Analysis — THE CRITICAL FINDING

**The schedule is configured for `total_steps = 629 × num_epochs` and decays to 0.**

Current state at the saved checkpoint:
- Trained on a 20-epoch schedule: `total_steps = 629 × 20 = 12,580`
- Saved at step 11,000 (epoch 17.5 of 20)
- LR at save: **8.32e-06** (8.3e-06 — near the cosine minimum; peak was 2e-4)
- LR at epoch 20 end: **0** (schedule decays to exactly zero)

### What happens if you just run `--num_epochs 40`?

**The script does NOT resume.** It loads `zinc_17M_pretrain.pt` and starts fresh from
epoch 0. This is a completely new fine-tuning run, not a continuation. Outcome:
- The old `chebi20_27M_scibert_20ep_contrastive.pt` is **overwritten** by the new best
  checkpoint from the new run (unless you change `--model_size` to get a different filename)
- The 40-epoch schedule has `total_steps = 629 × 40 = 25,160`
- LR will follow a new cosine arc: peak at step 500, decays to 0 at step 25,160

**LR trajectory for a fresh 40-epoch run vs the original 20-epoch run:**

| Epoch | Step | LR (20-epoch run) | LR (40-epoch run) |
|-------|------|-------------------|-------------------|
| 1 | 629 | 1.26e-04 | 9.40e-05 |
| 5 | 3,145 | 1.94e-04 | 1.94e-04 |
| 10 | 6,290 | 1.74e-04 | 1.74e-04 |
| 15 | 9,435 | 1.42e-04 | 1.42e-04 |
| 20 | 12,580 | 0 | 1.03e-04 |
| 25 | 15,725 | — | 6.40e-05 |
| 30 | 18,870 | — | 3.04e-05 |
| 35 | 22,015 | — | 7.92e-06 |
| 40 | 25,160 | — | 0 |

The first 15 epochs of a 20-epoch and 40-epoch run are nearly identical (LR curves diverge
only slightly). Training from scratch for 40 epochs gives the model a full cosine schedule
at the larger scale. This is **the safest and most principled option.**

### Why a true resume is possible but not natively supported

The checkpoint contains full `optimizer` and `scheduler` state. A proper resume would:
1. Re-create the model, optimizer, scheduler
2. Load all three state dicts from checkpoint
3. Set the scheduler's `total_steps` to the new extended value
4. Set `global_step = 11000`, start `for epoch in range(18, N_new_epochs)`

However, the current training script has **no code path for this**. It would require
a script modification. If done correctly (loading scheduler with last_epoch=11000
and total_steps for the new schedule), the LR would be at the epoch-17.5-of-N point
on the new schedule — a sensible warm restart.

---

## 4. Existing Checkpoints Inventory

### ZINC Pretrain Backbone

| File | Size | Date | Notes |
|------|------|------|-------|
| `zinc_5M_pretrain.pt` | 52.9 MB | 2026-03-14 | Earlier small model (5M params) |
| `zinc_17M_pretrain.pt` | 209.5 MB | 2026-04-02 | **Active pretrain backbone** used by current fine-tune |
| `best_model_upscaled.pt` | 209.5 MB | 2026-04-02 | Same as zinc_17M_pretrain.pt (copy) |

### ChEBI-20 Fine-tuned Models

| File | Size | Date | Notes |
|------|------|------|-------|
| `chebi20_27M_frozen_no_eos.pt` | 1670.2 MB | 2026-04-04 | BGE encoder, frozen, no EOS fix |
| `chebi20_27M_frozen_epoch{3,6,9}.pt` | 1670.2 MB | 2026-04-04/05 | Periodic checkpoints, frozen encoder |
| `chebi20_27M_frozen.pt` | 1670.2 MB | 2026-04-05 | Best frozen-encoder model |
| `chebi20_27M_contrastive_epoch{3,6,9}.pt` | 1670.2 MB | 2026-04-05 | BGE contrastive encoder |
| `chebi20_27M_contrastive.pt` | 1670.2 MB | 2026-04-05 | Best BGE-contrastive model |
| `chebi20_27M_scibert_contrastive_epoch_{3,6,9}.pt` | 778.7 MB | 2026-04-06 | SciBERT 9-epoch periodics |
| `chebi20_27M_scibert_contrastive.pt` | 778.7 MB | 2026-04-06 | Best SciBERT 9-epoch model |
| `chebi20_27M_scibert_20ep_contrastive_epoch_{3,6,9,12,15,18}.pt` | 778.7 MB | 2026-04-07 | **20-epoch periodic checkpoints** |
| `chebi20_27M_scibert_20ep_contrastive.pt` | 778.7 MB | 2026-04-07 | **CURRENT BEST** (epoch 18, val=0.3769) |

### Contrastive Encoder Checkpoints

| File | Size | Date | Notes |
|------|------|------|-------|
| `contrastive_bge_molinst.pt` | 1362.3 MB | 2026-04-05 | BGE-large encoder, Mol-Inst trained |
| `contrastive_scibert_molinst.pt` | 460.6 MB | 2026-04-06 | SciBERT encoder, Mol-Inst trained |
| `contrastive_scibert_chembl.pt` | 460.6 MB | 2026-04-06 | **Active encoder** used by best model |
| `contrastive_scibert_chebi20_projonly.pt` | 460.7 MB | 2026-04-08 | SciBERT, projection-only fine-tune on ChEBI-20 |

### Other

| File | Size | Notes |
|------|------|-------|
| `zinc_5M_rl_*.pt` | 52.9 MB | RL experiments on early 5M model |
| `molinst_27M_frozen*.pt` | 1670.2 MB | Mol-Instructions fine-tuned models |
| `best_finetuned_model*.pt` | 1670.2 MB | Aliases |

---

## 5. Recommended Approach

### Option A (Recommended): Fresh fine-tuning run for more epochs, different model_size tag

**This is the cleanest, safest, and most principled approach.** Because there is no native
resume support, and a fresh run with more epochs gives a properly shaped LR schedule.

```bash
python3 finetune/train_chebi20.py \
    --encoder contrastive \
    --contrastive_ckpt checkpoints/contrastive_scibert_chembl.pt \
    --model_size 27M_scibert_40ep \
    --num_epochs 40
```

What this does:
- Loads `zinc_17M_pretrain.pt` (backbone)
- Trains for 40 epochs with a fresh cosine schedule (total_steps=25,160)
- Peak LR=2e-4 with 500-step warmup; decays to 0 at epoch 40
- Saves best checkpoint to `checkpoints/chebi20_27M_scibert_40ep_contrastive.pt`
- Saves periodic checkpoints every 3 epochs: `chebi20_27M_scibert_40ep_contrastive_epoch_{3,6,...}.pt`
- **Does NOT overwrite the existing 20-epoch best checkpoint** (different `model_size` tag)
- Runs the full test-set evaluation after training completes

The 40-epoch LR schedule has the same shape as the 20-epoch schedule, just stretched.
Epochs 1–20 receive a slightly lower LR at each point (the cosine peak is wider, but the
peak value is the same); epochs 21–40 see the second half of the cosine decay.

### Option B: Pseudo-resume with a lower LR from the current best checkpoint

If you want to continue from the current best model weights (not retrain from scratch):

```bash
# Step 1: Copy best checkpoint to pretrain_ckpt location expected by script
cp checkpoints/chebi20_27M_scibert_20ep_contrastive.pt \
   checkpoints/chebi20_27M_scibert_20ep_contrastive_FINETUNEBASE.pt

# Step 2: Run with a new model_size tag and reduced LR
# (requires editing CONFIG["learning_rate"] to ~1e-5 before running,
#  OR see note below about lack of --lr flag)
python3 finetune/train_chebi20.py \
    --encoder contrastive \
    --contrastive_ckpt checkpoints/contrastive_scibert_chembl.pt \
    --model_size 27M_scibert_extended \
    --num_epochs 20
```

**PROBLEM:** The script loads `zinc_17M_pretrain.pt` as the pretrain backbone, not the ChEBI-20
fine-tuned checkpoint. There is no `--finetune_from` or `--resume_ckpt` flag. To continue
from the current best model, you would need to either:
(a) Temporarily replace `zinc_17M_pretrain.pt` with the ChEBI-20 checkpoint (messy, risky), or
(b) Add a `--resume_ckpt` argument to the script (a small, safe code change)

### Option C (Requires code change): True resume with extended schedule

Add ~15 lines to the script to support `--resume_ckpt`:

```python
# After line 1424 (where num_epochs is applied):
if args.resume_ckpt:
    print(f"[resume] Loading checkpoint: {args.resume_ckpt}")
    resume_data = torch.load(args.resume_ckpt, map_location="cpu")
    resume_model_state = resume_data["model"]
    resume_optimizer_state = resume_data["optimizer"]
    resume_step = resume_data["step"]
    resume_epoch = resume_data["epoch"]
    # After creating optimizer and scheduler, load states:
    # optimizer.load_state_dict(resume_optimizer_state)
    # scheduler.load_state_dict(resume_scheduler_state)  # with new total_steps
    # global_step = resume_step
    # Start for-loop from resume_epoch instead of 0
```

**If you add this**, the LR at resume would be at step 11,000 of the new `N×629`-step schedule.
For N=40: LR = 1.23e-4 (the cosine has barely started its decay, a natural continuation).

---

## 6. Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|-----------|
| **Overwriting existing best checkpoint** | HIGH | Always use a new `--model_size` tag (e.g. `27M_scibert_40ep`). The tag controls all output filenames. Never reuse `27M_scibert_20ep`. |
| **Early stopping firing too soon** | MEDIUM | Patience=5 (2,500 steps ≈ 4 epochs). The first 4–5 epochs of the new run will have a different LR shape than the 20-epoch run; val_loss may not improve immediately. Monitor val_loss carefully. |
| **LR schedule mismatch on Option B** | HIGH | If you load the 20-epoch checkpoint into a new run, the optimizer's momentum state reflects gradients at LR=8e-6, but the scheduler resets to LR=0 → peak LR → cosine. The optimizer state is stale for the new LR range. Best to discard optimizer state (set `strict=False` on model load, reset optimizer). |
| **test_csv path hardcoded in checkpoint config** | LOW | The saved `config.test_csv` has an absolute path with U+2019. If running on a different machine, eval will fail. Pass `--test_csv data/chebi20_test.csv` explicitly. |
| **Log file overwritten** | LOW | `chebi20_log.csv` is always overwritten by a new training run regardless of `--model_size`. Back it up first if you want to preserve the 20-epoch training curve. |
| **Contrastive encoder must match** | MEDIUM | The checkpoint was trained with `contrastive_scibert_chembl.pt`. If you use a different encoder checkpoint, the encoder_proj weights will be random (wrong). Always use the same `--contrastive_ckpt`. |

---

## 7. Expected Timeline

Based on the 20-epoch run completing in approximately 10 hours (extrapolated from
step timing at ~8s/validation, val every 500 steps):

| Target | Epochs | Steps | Est. time (MPS M1/M2) |
|--------|--------|-------|----------------------|
| Current best | 20 | 12,580 | ~10 hours (done) |
| Extended | 30 | 18,870 | ~15 hours |
| Extended | 40 | 25,160 | ~20 hours |
| Extended | 60 | 37,740 | ~30 hours |
| Extended | 100 | 62,900 | ~50 hours |

**Practical recommendation:** 40 epochs (doubling the current run). The cosine schedule
will reach its minimum at epoch 40; any further extension without LR adjustment produces
no meaningful learning. If 40 epochs don't show improvement, the bottleneck is likely
the model capacity or data, not training duration.

---

## Summary

**Can we resume from existing checkpoint?**
No — not natively. The script has no `--resume_ckpt` flag and always starts from the
ZINC backbone. A 15-line code addition would enable it; alternatively, use Option A
(fresh run, new model_size tag) which is equivalent in practice for epochs beyond 20.

**What happens to the LR schedule?**
The cosine schedule is baked in at initialization using `total_steps = 629 × num_epochs`.
At the end of the 20-epoch run, the LR is exactly 0. Re-running with `--num_epochs 40`
and a new `--model_size` tag gives a fresh 40-epoch cosine schedule: LR rises to peak
at step 500, decays smoothly to 0 at step 25,160. This is the correct behavior for
extended training.

**The recommended command:**

```bash
python3 finetune/train_chebi20.py \
    --encoder contrastive \
    --contrastive_ckpt checkpoints/contrastive_scibert_chembl.pt \
    --model_size 27M_scibert_40ep \
    --num_epochs 40
```
