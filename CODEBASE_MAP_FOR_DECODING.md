# CODEBASE_MAP_FOR_DECODING.md

**Purpose**: Complete context for implementing swappable commit strategies in MaskGIT-style
sampling for the Morpheus discrete-diffusion molecule generator.  
**Audience**: A downstream Claude session that will add `commit_strategy` logic without
having read any other file in the repo.

---

## 1. Repo Layout

```
molgen/
├── chemical_tokenizer.json        # 110-token SELFIES vocab (PAD/MASK/EOS + 107 chemical)
├── model/
│   ├── molecularDiffusionModel.py # Top-level model class; forward() → (B,L,110) logits
│   ├── transformerBlock.py        # TransformerBlock: SelfAttn → CrossAttn → FFN
│   └── embeddings.py              # TimestepEmbedding (sinusoidal→MLP), PositionalEmbedding (learned)
├── tokenizer/
│   └── chemicalTokenizer.py       # ChemicalTokenizer: encode(selfies_str), decode(ids)
├── inference/
│   ├── evaluate_prompted.py       # PRIMARY inference CLI for ChEBI-20 (text-conditioned)
│   └── evaluate.py                # Unconditional generation from pretrain checkpoint
├── finetune/
│   └── train_chebi20.py           # Training + CANONICAL evaluation pipeline
├── contrastive/                   # Contrastive aligner (used for reranking)
├── scripts/
│   ├── build_summary_table.py     # Aggregates per-run eval CSVs → summary table
│   └── rl_distribution_analysis.py# RL config property distribution analysis
├── data/                          # ChEBI-20 splits (train/val/test .csv), ZINC val set
├── outputs/
│   ├── evaluations/               # Per-run eval CSVs (one row per molecule)
│   └── slides/                    # Presentation figures
└── checkpoints/                   # Model weights (not tracked in git)
```

---

## 2. Sampling Pipeline Call Graph

### ChEBI-20 inference (text-conditioned)
```
inference/evaluate_prompted.py :: main()
  └─ build_model_and_tokenizer()           # loads checkpoint, sets text encoder
  └─ load_text_encoder()                   # BGE or SciBERT frozen
  └─ evaluate_test_set()
       ├─ [if rerank_n > 1] _rerank_candidates(...)
       │    └─ generate_cfg_batch() × N   # N independent candidate sets
       │    └─ contrastive_score()         # cosine sim in embedding space → pick best
       └─ generate_cfg_batch(             # single-pass generation
              model, tokenizer, text_embeds, null_embeds,
              num_steps, cfg_scale, temperature, max_length)
            └─ model.forward(input_ids, t, text_embeds, mask)  # ×num_steps
            └─ apply_eos_truncation()      # post-hoc EOS fix (separate forward pass)
```

### Training-time evaluation (canonical, most feature-complete)
```
finetune/train_chebi20.py :: evaluate_test_set()   # line 2242
  └─ _rerank_candidates() or _generate_cfg_batch()
       └─ _generate_cfg_batch(            # CANONICAL VERSION — line 1928
              model, tokenizer, text_embeds, null_embeds,
              num_steps, cfg_scale, temperature, max_length,
              refine_passes=0, refine_mask_ratio=0.1)
```

The two `generate_cfg_batch` implementations are nearly identical.
`_generate_cfg_batch` in `train_chebi20.py` is canonical because it additionally
supports `refine_passes` / `refine_mask_ratio` for iterative refinement.

---

## 3. MaskGIT Commit Step — Exact Code with Line Numbers

### Location 1 (production inference): `inference/evaluate_prompted.py`, lines 222–243

```python
for step_idx, t_val in enumerate(t_vals):
    step_t        = t_val.repeat(num_mols).unsqueeze(-1)          # (B,1)
    cond_logits   = model(input_ids, step_t, text_embeds,  text_padding_mask)
    uncond_logits = model(input_ids, step_t, null_embeds,  text_padding_mask)
    logits        = uncond_logits + cfg_scale * (cond_logits - uncond_logits)  # CFG

    # Block MASK token on final step so output is fully committed
    if step_idx < num_steps - 1:
        logits[:, :, tokenizer.mask_token_id] = float("-inf")

    probs      = torch.softmax(logits / temperature, dim=-1)
    sampled    = torch.distributions.Categorical(probs=probs).sample()  # (B,L)
    confidence = torch.gather(probs, 2, sampled.unsqueeze(-1)).squeeze(-1)  # (B,L)

    # Cosine noise schedule: alpha_t = cos(t * pi/2)^2
    alpha_t     = (torch.cos(t_val * math.pi / 2) ** 2).item()
    num_to_mask = int((1.0 - alpha_t) * max_length)

    # Re-mask lowest-confidence tokens (commit the rest)
    if num_to_mask > 0 and step_idx < num_steps - 1:
        _, mask_indices = torch.topk(confidence, num_to_mask, dim=-1, largest=False)
        sampled.scatter_(1, mask_indices, tokenizer.mask_token_id)

    input_ids = sampled
```

### Location 2 (canonical): `finetune/train_chebi20.py`, lines 1961–1965

Identical commit logic. Additional context: the `_generate_cfg_batch` function in
`train_chebi20.py` also handles `refine_passes` (iterative refinement after main loop):
after the main loop, it re-masks the bottom `refine_mask_ratio` fraction and runs a
short re-denoising schedule for `refine_passes` iterations (default 0 = disabled).

### Key invariants to preserve when adding commit strategies:
- `t_vals` runs from **t=1.0 → t=0.0** (denoising direction; cosine schedule)
- `input_ids` starts as all-MASK (`tokenizer.mask_token_id = 1`)
- On the **final step** (`step_idx == num_steps - 1`), MASK is blocked from logits so
  every position commits
- The number of positions to re-mask is determined by `alpha_t`, which decreases each step

---

## 4. Tokenizer Vocab — Special and Structural Token IDs

**File**: `chemical_tokenizer.json`  
**Vocab size**: 110  
**Chemical tokens start at ID**: 3

### Special tokens
| Token   | ID |
|---------|----|
| [PAD]   | 0  |
| [MASK]  | 1  |
| [EOS]   | 2  |

### Ring tokens (all ring-forming tokens in vocab)
| Token      | ID | Notes                         |
|------------|----|-------------------------------|
| [Ring1]    | 5  | single ring closure 1         |
| [Branch1]  | 6  | opens branch depth 1          |
| [=Branch1] | 7  | double-bond branch depth 1    |
| [Branch2]  | 11 | opens branch depth 2          |
| [Ring2]    | 12 | ring closure 2                |
| [#Branch1] | 15 | triple-bond branch depth 1    |
| [=Branch2] | 18 | double-bond branch depth 2    |
| [#Branch2] | 20 | triple-bond branch depth 2    |
| [=Ring1]   | 25 | double-bond ring closure 1    |
| [=Ring2]   | 36 | double-bond ring closure 2    |
| [-\Ring1]  | 85 | stereo ring closure           |
| [-/Ring1]  | 97 | stereo ring closure           |
| [-/Ring2]  | 102| stereo ring closure           |

**IMPORTANT**: There is NO [Ring3], [Branch3], [-/Ring3], or [=Branch3] in the vocab.
SELFIES encodes all structures ≤2 ring-depth using the tokens above.

### Identifying structural tokens programmatically
The `ChemicalTokenizer` class has **no** `is_special_token()` or `is_chemical_token()`
helper methods. To detect ring/branch tokens in new code, use:
```python
RING_TOKEN_IDS   = {5, 12, 25, 36, 85, 97, 102}
BRANCH_TOKEN_IDS = {6, 7, 11, 15, 18, 20}
SPECIAL_IDS      = {0, 1, 2}  # PAD, MASK, EOS
# Chemical token = ID >= 3
```

---

## 5. Model Forward Signature and CFG / Conditioning Wiring

### Forward signature (`model/molecularDiffusionModel.py`)
```python
model.forward(
    input_ids:        Tensor[B, L],           # SELFIES token IDs
    timesteps:        Tensor[B, 1],           # t in [0,1]; 1.0=fully masked, 0.0=committed
    text_embeds:      Tensor[B, T_text, H],   # encoder output (projected); H=hidden_size=512
    text_padding_mask: Tensor[B, T_text],     # True = padding position
) -> logits: Tensor[B, L, vocab_size=110]
```

### CONFIG (from `finetune/train_chebi20.py`)
```python
vocab_size  = 110
hidden_size = 512
num_heads   = 8
ffn_dim     = 1024
num_layers  = 8
max_length  = 74      # SELFIES sequence length (padded/truncated to this)
```

### Conditioning pathway
1. `model.get_text_embeddings(input_ids, attention_mask)` calls frozen `text_encoder`
2. If `encoder_proj` exists (SciBERT hidden=768 ≠ BGE hidden=1024), applies `nn.Linear(768→1024)`
3. Then applies `text_proj` MLP to reach `hidden_size=512`
4. `null_token`: learned `nn.Parameter(1,1,hidden_size)`, expanded to `(B,T_text,H)` for
   unconditional pass (replaces text_embeds entirely)

### CFG is applied OUTSIDE the model in the sampler
```python
logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
```
Requires **two forward passes per step**: one with `text_embeds`, one with `null_embeds`.

### Cross-attention architecture (per `model/transformerBlock.py`)
Each `TransformerBlock`: `norm1 → SelfAttention → norm_cross → CrossAttention → norm2 → FFN`  
`CrossAttention.out_proj` is **zero-initialized** to preserve unconditional pre-training.

---

## 6. Audit Script Locations and CSV Schema

### Evaluation CSV (per-molecule, produced by `evaluate_test_set`)
Filename: `{dataset}_{model_size}_{encoder}_eval_cfg{X}_t{Y}_s{Z}[_eost][_rerank{N}][_refine{P}x{R}pct].csv`

| Column       | Type    | Description                                   |
|--------------|---------|-----------------------------------------------|
| prompt       | str     | ChEBI-20 text description                     |
| gt_smiles    | str     | Ground-truth SMILES                           |
| gen_smiles   | str     | Generated SMILES (EOS-truncated if _eost)     |
| valid        | bool    | RDKit-parseable                               |
| exact_match  | bool    | gen_smiles == gt_smiles                       |
| bleu2/4      | float   | Character BLEU (2-gram / 4-gram)              |
| atom_bleu2/4 | float   | Atom-level BLEU (regex-tokenized SMILES)      |
| lev_sim      | float   | Normalized Levenshtein similarity             |
| morgan_sim   | float   | Morgan FP Tanimoto (radius=2, 2048-bit)       |
| maccs_sim    | float   | MACCS keys Tanimoto (167-bit)                 |
| rdk_sim      | float   | RDKit FP Tanimoto                             |

### Fingerprint computation: `finetune/train_chebi20.py`, line 2226 (`_fp_tanimoto`)
```python
def _fp_tanimoto(smi_a, smi_b, fp_type='morgan'):
    # returns float in [0,1]; 0.0 on any parse error
```

### Summary aggregation: `scripts/build_summary_table.py`
- Parses canonical filename to extract: encoder, cfg_scale, temperature, num_steps, flags
- Handles legacy filename formats (old HP format, loose `_eval` prefix)
- Encoder name normalization: `frozen→bge_frozen`, `contrastive→bge_contrastive`
- `pre_eos_fix` flag set for files named `eval_27M_*` and `chebi20_eval`
- Computes `atom_bleu2/4` on-the-fly if column absent but `gt_smiles/gen_smiles` present

---

## 7. Filename Convention with Examples

### Pattern
```
{dataset}_{model_size}_{encoder}_eval_cfg{X}_t{Y}_s{Z}[_eost][_rerank{N}][_refine{P}x{R}pct].csv
```

| Field        | Example values         | Notes                                 |
|--------------|------------------------|---------------------------------------|
| dataset      | `chebi20`              | always chebi20 for ChEBI-20           |
| model_size   | `27M`                  | parameter count suffix                |
| encoder      | `scibert20ep`, `bge_contrastive`, `bge_frozen` |             |
| cfg{X}       | `cfg1.5`, `cfg2.0`     | classifier-free guidance scale        |
| t{Y}         | `t0.4`, `t0.6`         | sampling temperature                  |
| s{Z}         | `s50`, `s100`          | number of denoising steps             |
| _eost        | present/absent         | EOS truncation applied                |
| _rerank{N}   | `_rerank10`            | N=10 candidates reranked              |
| _refine{P}x{R}pct | `_refine2x10pct` | 2 refinement passes, 10% remask ratio |

### Concrete examples (from `outputs/evaluations/`)
```
chebi20_27M_scibert20ep_eval_cfg1.5_t0.4_s50_eost_rerank10.csv   ← best config
chebi20_27M_scibert20ep_eval_cfg1.5_t0.4_s50_eost.csv
chebi20_27M_bge_contrastive_eval_cfg1.5_t0.4_s50_eost.csv
chebi20_27M_bge_frozen_eval_cfg1.0_t0.6_s50.csv
```

---

## 8. Git State

**Current branch**: `feature/dep-aware-decoding`  
**Base branch**: `main`  
**Working tree**: Clean except untracked `outputs/slide10_data_inventory.md`

### Recent commit history (newest first)
```
e3cb187  Switch slide 10 ablation charts from Morgan to MACCS
34c71bb  Stratification analysis: identify best config and metric for slide 11
e7b9ee3  RL distribution analysis: characterize concentration and dispersion
148d5d1  Audit: branch count prediction accuracy and complexity stratification
b0f46a8  Audit: tokenizer, truncation, and index handling for SELFIES ring tokens
236b73b  Add RDK fingerprint comparison figure (supplementary)
90751c2  Fix spacing and overlapping text in slide 12 comparison figures
bbc75e7  Split slide 12 comparison into three separate figures
fdb609e  Three-panel comparison with verified TGM-DLM paper numbers
9f38cd2  Add efficiency-focused comparison figure for slide 12
f49acfc  Regenerate RL and comparison figures for presentation
3282dfd  Add presentation figures for slide deck
```

---

## 9. Open Questions / Gotchas

### 1. Two nearly-duplicate sampling implementations
`inference/evaluate_prompted.py::generate_cfg_batch` and
`finetune/train_chebi20.py::_generate_cfg_batch` are almost identical.
Any commit-strategy change must be applied to **both** (or the inference version must be
refactored to call the train version). The train version is canonical (has refinement support).

### 2. No `is_special_token` helper on tokenizer
`ChemicalTokenizer` exposes only `encode`, `decode`, and direct attribute access
(`mask_token_id`, `pad_token_id`, `eos_token_id`). There is no `is_chemical_token(id)`
or `is_ring_token(id)` method. Any strategy that needs to distinguish structural tokens
must implement its own ID sets (see Section 4).

### 3. [Ring3] / [Branch3] absent from vocab
SELFIES molecules with three ring closures use [Ring1]+[Ring2] combinations.
Do not expect ID 3 or any dedicated [Ring3] token — they don't exist in this 110-token vocab.

### 4. OOV tokens silently become [PAD]
`ChemicalTokenizer.encode` maps unknown tokens to `pad_token_id` (ID=0) without error.
This can silently corrupt ring structure in molecules that use tokens outside the 110-word vocab
(shouldn't happen with valid SELFIES, but worth knowing for debugging).

### 5. iCloud Drive path encoding (environment-specific)
The repo lives under a path containing a Unicode RIGHT SINGLE QUOTATION MARK (U+2019) in the
directory name. The shell CWD uses a straight apostrophe (U+0027) which doesn't exist on disk.
All file I/O in scripts must use `os.scandir` to discover the real path, or pass an absolute
path obtained that way. `git` commands must be run with `cwd=molgen` via `subprocess.run`.

### 6. CFG null_embeds are model parameters, not zeros
The unconditional forward pass uses `model.null_token` (a learned `nn.Parameter`), NOT a
zero tensor or empty string embedding. This is retrieved via:
```python
null_embeds = model.null_token.expand(batch_size, seq_len, hidden_size)
```
Any new commit strategy that calls `model.forward` must pass `null_embeds` identically.

### 7. Temperature is applied before sampling, not after
`probs = torch.softmax(logits / temperature, dim=-1)` — temperature rescales logits before
the softmax. Commit confidence is computed from these temperature-scaled probabilities.
A strategy based on absolute confidence thresholds will be temperature-dependent.

### 8. Confidence is the sampled-token probability, not the max probability
```python
sampled    = Categorical(probs).sample()
confidence = probs.gather(dim=2, index=sampled.unsqueeze(-1)).squeeze(-1)
```
Confidence is the probability of the **actually sampled** token, which may not be the argmax.
This is correct for MaskGIT but differs from greedy-decode confidence. A strategy that wants
"certainty" should use `probs.max(dim=-1).values` instead.
