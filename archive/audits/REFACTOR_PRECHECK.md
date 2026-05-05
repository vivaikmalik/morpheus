# REFACTOR_PRECHECK.md

Pre-refactor context for the sampling-loop deduplication task.
Read-only investigation — no code was modified.

---

## Section 1: Best-Config Eval Command

### Shell scripts and runbooks

No `run_*.sh`, `eval_*.sh`, or `RUNBOOK.md` files exist anywhere in the repo.  
The only documented eval commands are in `README.md`.

### Canonical entrypoint

**`finetune/train_chebi20.py`** is the canonical eval entrypoint.  
Every eval CSV in `outputs/` that matches the pattern
`{dataset}_{model_size}_{encoder}_eval_cfg*` was produced by this script.  
`inference/evaluate_prompted.py` is a secondary script (targets Mol-Instructions
by default, lacks `--contrastive_ckpt`, `--rerank_n`, `--refine_passes`).

Evidence: the best-config CSV
`outputs/chebi20_27M_scibert_20ep_contrastive_eval_cfg1.5_t0.6_s50.csv`
was added in commit `1903fe5` ("SciBERT contrastive support + encoder_proj + 20ep
training + HP sweeps"), which also modified `finetune/train_chebi20.py`.
Timestamp: 2026-04-07 12:38.

### Best-config command — WITHOUT eos_truncate (matches existing CSV)

```bash
python3 finetune/train_chebi20.py --eval_only \
    --encoder contrastive \
    --contrastive_ckpt checkpoints/contrastive_scibert_chembl.pt \
    --model_size 27M_scibert_20ep \
    --cfg 1.5 --temp 0.6 --steps 50
```

Produces: `outputs/chebi20_27M_scibert_20ep_contrastive_eval_cfg1.5_t0.6_s50.csv`  
This file **already exists** (2543 rows, 1.14 MB).

### Best-config command — WITH eos_truncate (the intended "best config" as described)

```bash
python3 finetune/train_chebi20.py --eval_only \
    --encoder contrastive \
    --contrastive_ckpt checkpoints/contrastive_scibert_chembl.pt \
    --model_size 27M_scibert_20ep \
    --cfg 1.5 --temp 0.6 --steps 50 \
    --eos_truncate
```

Produces: `outputs/chebi20_27M_scibert_20ep_contrastive_eval_cfg1.5_t0.6_s50_eost.csv`  
This file **does NOT yet exist** — it has never been run.

Note: `--model_size 27M_scibert_20ep --encoder contrastive` causes `_apply_run_config()`
(line 135) to set `CONFIG["save_ckpt"] = "checkpoints/chebi20_27M_scibert_20ep_contrastive.pt"`.

### All CLI flags — `finetune/train_chebi20.py` (argparse lines 1287–1322)

| Flag | Type | Default | Help |
|------|------|---------|------|
| `--eval_only` | store_true | False | Skip training; load checkpoint and run evaluate_test_set |
| `--eos_truncate` | store_true | False | Post-hoc EOS truncation during evaluation |
| `--cfg` | float | None → CONFIG `eval_cfg_scale=3.0` | CFG guidance scale |
| `--temp` | float | None → CONFIG `eval_temperature=1.0` | Sampling temperature |
| `--steps` | int | None → CONFIG `eval_steps=50` | Number of denoising steps |
| `--encoder` | choices: frozen/contrastive | frozen | Text encoder variant |
| `--contrastive_ckpt` | str | None | Path to contrastive aligner .pt (required when --encoder contrastive) |
| `--model_size` | str | 27M | Model size tag used in checkpoint/output filenames |
| `--num_epochs` | int | None → CONFIG `num_epochs=10` | Override training epochs |
| `--rerank_n` | int | 1 | Best-of-N reranking; 1=disabled (requires --contrastive_ckpt if >1) |
| `--refine_passes` | int | 1 | Iterative refinement passes; 1=disabled |
| `--refine_mask_ratio` | float | 0.2 | Fraction to re-mask per refinement pass |

### All CLI flags — `inference/evaluate_prompted.py` (argparse lines 488–507)

| Flag | Type | Default | Help |
|------|------|---------|------|
| `--cfg` | float | 3.0 (from GEN_CFG) | CFG guidance scale |
| `--temp` | float | 1.2 (from GEN_CFG) | Sampling temperature |
| `--steps` | int | 32 (from GEN_CFG) | MaskGIT denoising steps |
| `--checkpoint` | str | None → `checkpoints/molinst_27M_frozen.pt` | Path to checkpoint .pt |
| `--test_csv` | str | None → `data/test.csv` | Path to test CSV |
| `--eos_truncate` | store_true | False | Post-hoc EOS truncation |
| `--encoder` | str | frozen | Encoder tag for output filenames only (no functional effect) |
| `--model_size` | str | 27M | Model size tag for output filenames only |

Missing vs train_chebi20.py: no `--contrastive_ckpt`, `--rerank_n`, `--refine_passes`,
`--refine_mask_ratio`, `--num_epochs`.

### Environment variable dependencies

Neither `finetune/train_chebi20.py` nor `inference/evaluate_prompted.py` reads
`CUDA_VISIBLE_DEVICES`, `PYTHONPATH`, or any other environment variable.
No conda env is referenced in any script.

---

## Section 2: Seed Handling

### grep output (exact command requested)

```
grep -rn "manual_seed|seed_everything|np.random.seed|random.seed|set_seed" \
    finetune/ inference/ model/ scripts/
```

Results:

```
finetune/train_chebi20.py:1036:    random.seed(CONFIG["seed"])
finetune/train_chebi20.py:1037:    np.random.seed(CONFIG["seed"])
finetune/train_chebi20.py:1038:    torch.manual_seed(CONFIG["seed"])
finetune/train_chebi20.py:1341:        random.seed(CONFIG["seed"])
finetune/train_chebi20.py:1342:        np.random.seed(CONFIG["seed"])
finetune/train_chebi20.py:1343:        torch.manual_seed(CONFIG["seed"])
finetune/train_text_condition.py:425:    random.seed(CONFIG["seed"])
finetune/train_text_condition.py:426:    np.random.seed(CONFIG["seed"])
finetune/train_text_condition.py:427:    torch.manual_seed(CONFIG["seed"])
finetune/resume_eos_fix.py:419:    random.seed(CONFIG["seed"])
finetune/resume_eos_fix.py:420:    np.random.seed(CONFIG["seed"])
finetune/resume_eos_fix.py:421:    torch.manual_seed(CONFIG["seed"])
inference/diversity_check.py:23:random.seed(SEED)
inference/diversity_check.py:24:np.random.seed(SEED)
inference/reconstruction_accuracy.py:97:        generator=torch.Generator().manual_seed(SEED),
inference/reconstruction_accuracy.py:141:    rng     = torch.Generator().manual_seed(SEED)
scripts/kaggle_full_finetune.py:686:    random.seed(CONFIG["seed"])
scripts/kaggle_full_finetune.py:687:    np.random.seed(CONFIG["seed"])
scripts/kaggle_full_finetune.py:688:    torch.manual_seed(CONFIG["seed"])
scripts/kaggle_eos_fix.py:662:    random.seed(CONFIG["seed"])
scripts/kaggle_eos_fix.py:663:    np.random.seed(CONFIG["seed"])
scripts/kaggle_eos_fix.py:664:    torch.manual_seed(CONFIG["seed"])
```

### Per-match analysis

| File:line | Exact code | Scope | CLI-controlled? |
|-----------|-----------|-------|-----------------|
| `finetune/train_chebi20.py:1036` | `random.seed(CONFIG["seed"])` | Inside `train()` function | No — hardcoded `CONFIG["seed"] = 42` (line 125) |
| `finetune/train_chebi20.py:1037` | `np.random.seed(CONFIG["seed"])` | Inside `train()` function | No |
| `finetune/train_chebi20.py:1038` | `torch.manual_seed(CONFIG["seed"])` | Inside `train()` function | No |
| `finetune/train_chebi20.py:1341` | `random.seed(CONFIG["seed"])` | Inside `main() → if args.eval_only:` block | No — same hardcoded 42 |
| `finetune/train_chebi20.py:1342` | `np.random.seed(CONFIG["seed"])` | Inside `main() → if args.eval_only:` block | No |
| `finetune/train_chebi20.py:1343` | `torch.manual_seed(CONFIG["seed"])` | Inside `main() → if args.eval_only:` block | No |
| `finetune/train_text_condition.py:425` | `random.seed(CONFIG["seed"])` | Inside `train()` function | No |
| `finetune/train_text_condition.py:426` | `np.random.seed(CONFIG["seed"])` | Inside `train()` function | No |
| `finetune/train_text_condition.py:427` | `torch.manual_seed(CONFIG["seed"])` | Inside `train()` function | No |
| `finetune/resume_eos_fix.py:419` | `random.seed(CONFIG["seed"])` | Inside `train()` function | No |
| `finetune/resume_eos_fix.py:420` | `np.random.seed(CONFIG["seed"])` | Inside `train()` function | No |
| `finetune/resume_eos_fix.py:421` | `torch.manual_seed(CONFIG["seed"])` | Inside `train()` function | No |
| `inference/diversity_check.py:23` | `random.seed(SEED)` | Module-level (top of file, `SEED = 42` line 20) | No — hardcoded |
| `inference/diversity_check.py:24` | `np.random.seed(SEED)` | Module-level | No — hardcoded |
| `inference/reconstruction_accuracy.py:97` | `generator=torch.Generator().manual_seed(SEED)` | Inside `build_val_dataset()` function | No — hardcoded `SEED` constant |
| `inference/reconstruction_accuracy.py:141` | `rng = torch.Generator().manual_seed(SEED)` | Inside module-level `__main__` block | No — hardcoded `SEED` constant |
| `scripts/kaggle_full_finetune.py:686-688` | same triple seed | Inside `train()` function | No |
| `scripts/kaggle_eos_fix.py:662-664` | same triple seed | Inside `train()` function | No |

**`inference/evaluate_prompted.py`**: NO seed-setting calls found anywhere in the file.

This means eval runs via `evaluate_prompted.py` are **non-deterministic** — no seed is set
for PyTorch sampling (including MPS random number generation). Runs via `train_chebi20.py`
with `--eval_only` ARE seeded (lines 1341–1343, seed=42, hardcoded, not a CLI flag).

### Deterministic algorithms

```
grep -rn "use_deterministic_algorithms|cudnn.deterministic" \
    finetune/ inference/ model/ scripts/
```

**No matches found.** `torch.use_deterministic_algorithms` and
`torch.backends.cudnn.deterministic` are not set anywhere in the codebase.

---

## Section 3: Device

### Exact command output

```
python3 -c "import torch; print('mps:', torch.backends.mps.is_available(), 'cuda:', torch.cuda.is_available(), 'device_count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"
```

**Output:**
```
mps: True cuda: False device_count: 0
```

### Device-selection logic

Both `finetune/train_chebi20.py` (lines 1042–1046 in `train()`, lines 1346–1350 in
`eval_only` block) and `inference/evaluate_prompted.py` (lines 542–546) use identical
auto-detection logic:

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"[device] {device}")
```

- **No `--device` CLI flag** in either entrypoint.
- **No `CUDA_VISIBLE_DEVICES` or other env var** is read.
- Priority: CUDA > MPS > CPU.
- **On this machine**: will always select `mps` (Apple M-series, MPS available, no CUDA).

### Impact for refactor

The sampling loop uses standard PyTorch tensor ops that run on whatever `device` is
selected. No device-specific code paths exist inside `generate_cfg_batch` or
`_generate_cfg_batch`. The refactored version will inherit the same device automatically.
