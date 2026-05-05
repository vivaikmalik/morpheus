# Audit: Source of "~40% ring-count accuracy at generation time" claim

**Generated:** 2026-05-03
**Claim audited:** main.tex §5.4 / Table 4 / abstract:
> "...ring-count tokens correctly 79.1% of the time given ground-truth context. At generation time, when those same tokens are predicted jointly with the rest of the sequence under confidence-based parallel commitment, accuracy falls to around 40%. The 37pp gap is the size of the expression loss attributable to MaskGIT''s irreversible commitment under partial context."

**Bottom line:** No source in the repo computes "ring-count token accuracy at generation time" as a single number. The "~40%" is most plausibly a paraphrase of the **39.7% molecule-level ring-count match** from `outputs/branch_audit_summary.md`, but that is **molecule-level**, not **token-level** — different units, not directly comparable to the 79.1% probe figure that creates the "37pp gap" framing.

---

## Search Coverage

I searched exhaustively for the source:

1. ✅ All `.md` files under `outputs/` (including `outputs/slides/`, `outputs/rl/`)
2. ✅ All `.csv` files under `outputs/` and `outputs/evaluations/`
3. ✅ Git history for deleted files matching `*token*`, `*generation*`, `*audit*`, `*ring*`, `*count*`, `*align*`, `*probe*`
4. ✅ All `.py` and `.ipynb` files under `scripts/`, `inference/`, `training/`, `finetune/`, `contrastive/`, `tokenizer/`, `model/`, `dataExtractor/`
5. ✅ W&B run directories under `wandb/`
6. ✅ `outputs/probe_token_knowledge_raw.csv` and `outputs/probe_token_knowledge_raw_40ep.csv` (the per-position probe outputs)

**Deleted files found in git history:** none related to ring-count or token-alignment analysis. The only relevant deletion is `scripts/precompute_embeddings.py` (unrelated). All deletions otherwise are stale checkpoints (`checkpoints/checkpoint_step*.pt`).

**Scripts that produced the existing audit MDs:** Not present in the repo. `bug_audit_summary.md`, `generated_count_audit.md`, `branch_audit_summary.md`, and `dep_aware_investigation.md` were written but the generating scripts were never committed.

---

## Candidate Sources

Five different numbers in five different places, none of them token-level on the full test set, all on the **20-epoch checkpoint** (the paper claims 40ep — see "Caveat" below).

### Candidate 1: `outputs/branch_audit_summary.md` line 12

> "Ring count match (exact): **39.7%** of all molecules"

- **Number: 39.7%**
- **Granularity: molecule-level** — counts the number of rings in `gt_smiles` and the number of rings in `gen_smiles` (presumably via `Chem.GetSSSR(mol)` or similar) and reports the fraction of molecules where the two ring counts match exactly
- **Test set: 2,542 molecules** from `chebi20_27M_scibert_20ep_contrastive_eval_cfg1.5_t0.6_s50.csv`
- **Closest to "~40%"** of any candidate. Most likely the source.
- **NOT token-level.** This measures whether a generated molecule has the right *number* of rings, not whether the model predicted the right *token value* at a ring-count token position.

Per-bin breakdown from same file:

| Bin | n | Match % |
|-----|---|---------|
| 0 rings | 609 | 77.7% |
| 1 ring | 484 | 53.9% |
| 2 rings | 336 | 26.5% |
| 3 rings | 200 | 24.5% |
| 4 rings | 172 | 19.8% |
| 5+ rings | 741 | varies |

Note: bin sizes (609 acyclic) differ from main.tex Table 3 (762 acyclic) — different counting methods or different versions of the eval.

### Candidate 2: `outputs/dep_aware_investigation.md` Section 5 (fallback metric)

> "Fraction of molecules with exact opener count match: **0.3505**"

- **Number: 35.05%**
- **Granularity: molecule-level** — counts ring opener tokens (`[Ring1]`, `[Ring2]`, `[=Ring1]`, etc.) in the SELFIES of `gen_smiles` and `gt_smiles` and reports fraction with exact-match opener-token count
- **Test set: 2,542 molecules** (standard 20ep run, same eval CSV as Candidate 1)
- Closest token-counting analog, but still molecule-level.
- Stratified:

| ring_bin | n | std exact_count |
|---------|---|-----------------|
| 0 | 762 | 0.7415 |
| 1–2 | 924 | 0.2608 |
| 3+ | 856 | 0.0993 |

### Candidate 3: `outputs/dep_aware_investigation.md` Section 5 (position-aligned)

> "Position-aligned match rate: **0.2857** (60/210)"

- **Number: 28.57%**
- **Granularity: TRUE TOKEN-LEVEL** — at each ring opener position, does the generated SELFIES token equal the ground-truth SELFIES token at that exact position?
- **Test set restricted: only 4.8% of molecules (122 / 2,542) where generated SELFIES has the same length as ground-truth SELFIES.** 210 ring-opener positions total.
- This is the **only token-level position-aligned ring-count accuracy at generation time** in the repo, but the sample is too small to be reliable (the audit itself flags this).

### Candidate 4: `outputs/generated_count_audit.md` lines 39–62

> "Overall exact ring count match: 1428/2542 = **56.2%**"

- **Number: 56.2%**
- **Granularity: molecule-level** — same as Candidate 1 in spirit, but different counting method (re-encodes generated SMILES → SELFIES via `sf.encoder()` then counts ring openers)
- **Test set: 2,542 molecules**
- Per-bin (also molecule-level):

| GT rings | n | Accuracy |
|----------|---|----------|
| 0 | 762 | 96% |
| 1 | 521 | 57% |
| 2 | 403 | 44% |
| 3 | 421 | 32% |
| 4 | 263 | 27% |
| 5 | 137 | 17% |
| 6+ | 35 | 0% |

The **44% at "2 rings"** could also be a near-match for "~40%", but it is per-bin, not overall.

### Candidate 5: `outputs/bug_audit_summary.md` lines 36–38

> "96% accuracy for 0-ring molecules / 57% accuracy for 1-ring molecules / 0% accuracy for 6+ ring molecules"

This is a re-statement of Candidate 4''s per-bin numbers. Not a separate measurement; same source data, same numbers.

### What the probe raw CSVs do NOT contain

- `outputs/probe_token_knowledge_raw.csv` (20ep) and `outputs/probe_token_knowledge_raw_40ep.csv` (40ep) contain probe-time predictions only (single-mask reconstruction with full ground-truth context). They do **not** contain generation-time predictions.

### What the eval CSVs do NOT contain

- The 113 eval CSVs in `outputs/` contain per-molecule fingerprint similarity (Morgan, MACCS, RDK), exact match, BLEU, validity. They do **not** contain per-token alignment data. There is no column for "ring-count token correct at position k".

### W&B logs

- 14 W&B run directories exist under `wandb/`. None contain summary metrics for ring-count accuracy at generation time. They are pretraining/contrastive training runs, not generation analyses.

---

## Side-by-side Summary

| # | Source | Value | Granularity | Test set | Comparable to 79.1% probe? |
|---|--------|-------|-------------|----------|---------------------------|
| 1 | `branch_audit_summary.md` | **39.7%** | Molecule-level | 2,542 mols | NO (different unit) |
| 2 | `dep_aware_investigation.md` §5 fallback | 35.05% | Molecule-level (token-count match) | 2,542 mols | NO (different unit) |
| 3 | `dep_aware_investigation.md` §5 position-aligned | 28.57% | **Token-level position-aligned** | 122 mols / 210 positions | YES (but tiny sample) |
| 4 | `generated_count_audit.md` | 56.2% | Molecule-level | 2,542 mols | NO (different unit) |
| 5 | `bug_audit_summary.md` | restates #4 | Molecule-level | 2,542 mols | NO |

**Probe value cited in paper:** 79.1% (top-1 single-token prediction accuracy under ground-truth context, n=7,482 ring-count token positions, 40ep checkpoint).

The probe is genuinely token-level (one position masked at a time, all other positions show ground-truth context, ask "what is the top-1 prediction at this masked position?"). The closest token-level generation-time analog in the repo is Candidate 3 at 28.57% — but on only 4.8% of the test set, and on the 20ep checkpoint.

---

## Caveat: All Candidates Are 20ep, Paper Claims 40ep

Every audit MD that produced a number was generated on **2026-04-11**, against the **20-epoch checkpoint** (`chebi20_27M_scibert_20ep_contrastive_eval_cfg1.5_t0.6_s50.csv`). The paper''s §5.4 / Table 4 reports the **40-epoch checkpoint** (probe 79.1%, "around 40%" at generation).

For the 40ep checkpoint, there is **no audit at all** of generation-time ring-count accuracy. So the "~40%" in the paper:
- Cannot be the **39.7%** number directly (that''s 20ep)
- Could be an extrapolation: "the 20ep model was 39.7%; the 40ep model is slightly better; round to ~40%"
- Could be the **44%** at "2 rings" bin from Candidate 4 (also 20ep)
- Could be a hand-wave / ballpark

The "37pp gap" (79.1 − 40 ≈ 39pp, rounded to 37) further suggests the "~40%" was chosen *post hoc* to produce a clean-looking gap, not measured directly.

---

## What the Paper''s Probe-vs-Generation Comparison ACTUALLY Says

The paper''s claim, parsed strictly:
- **79.1%** = probe top-1 accuracy: "given ground-truth context at every other position, how often does the model predict the correct ring-count token?"
- **~40%** = generation-time ring-count "accuracy"

These are **different units** unless "~40%" is the position-aligned token-level accuracy (Candidate 3 = 28.57%, on 4.8% of mols). The most likely actual source is the molecule-level **39.7%** from `branch_audit_summary.md`, which asks "does the generated molecule have the right *number* of rings?" — a much weaker question than "did the model predict the right *token* at the ring-count position?"

The "37pp gap" is therefore an **apples-to-oranges comparison**: token-level capability under ideal context vs. molecule-level outcome under realistic generation. Both numbers are honest, but their juxtaposition implies a stricter comparison than the data supports.

---

## Recommendations

### Option A — Cite molecular-level explicitly (preferred for honesty)

Rewrite §5.4 to acknowledge the unit mismatch:

> "...ring-count tokens correctly **79.1%** of the time given ground-truth context. At generation time, only **39.7%** of generated molecules have the correct *number* of rings (a molecule-level measurement; per `outputs/branch_audit_summary.md`). These numbers measure different things — token-level prediction accuracy vs. whole-molecule ring-count match — but the gap between them characterizes the cumulative cost of decoding errors."

This drops the "~40%" euphemism and the "37pp gap" claim, in exchange for honest reporting.

### Option B — Compute the missing 40ep token-level number

Run a token-level alignment script on the 40ep `cfg=1.5_t=0.7` eval CSV that:

1. For each row, re-encode `gen_smiles` to SELFIES via `sf.encoder()` and compare to `response` (gt SELFIES).
2. For molecules with same SELFIES length, count token matches at ring-count positions.
3. For molecules with different lengths, fall back to molecule-level ring-count match.
4. Report the combined statistic (or both stats separately).

This would produce the exact number to cite, on the 40ep checkpoint, with a defensible methodology. Estimated effort: 1 small script, 30 minutes.

### Option C — Drop the "~40%" and the "37pp gap"

Replace §5.4 paragraph with: "Ring-count tokens are predicted correctly 79.1% of the time given ground-truth context; generation-time accuracy is substantially lower (per outputs/branch_audit_summary.md, only 39.7% of generated molecules even have the correct ring count). The expression gap between probe capability and generation outcome is the central diagnostic of this paper, and we make no claim about its precise magnitude in token-level terms."

### Option D — If the "~40%" stays as-is

At minimum, footnote it in the paper: "Approximate: derived from molecule-level ring-count match (39.7%, branch_audit_summary.md) on the 20-epoch checkpoint; we do not have a token-level position-aligned measurement on the 40-epoch checkpoint."

This is the bare minimum for academic honesty.

---

## Files Referenced

- `outputs/branch_audit_summary.md` (5,509 B, 2026-04-11) — primary candidate, molecule-level 39.7%
- `outputs/dep_aware_investigation.md` (11,735 B, 2026-04-13) — Section 5, candidates 35.05% and 28.57%
- `outputs/generated_count_audit.md` (2,626 B, 2026-04-11) — molecule-level 56.2% (different counting method)
- `outputs/bug_audit_summary.md` (2,981 B, 2026-04-11) — restates generated_count_audit numbers
- `outputs/probe_token_knowledge.md` (3,221 B, 2026-04-13) — 20ep probe, ring-count 77.6%
- `outputs/probe_token_knowledge_40ep.md` (3,221 B, 2026-05-03) — 40ep probe, ring-count 79.1% (the only 40ep number)
- `outputs/probe_comparison_20ep_vs_40ep.md` (5,849 B, 2026-05-03) — uses 20ep Morgan 0.2944 → 40ep Morgan 0.3164, +7.5%

Eval CSVs used by all the audits: `chebi20_27M_scibert_20ep_contrastive_eval_cfg1.5_t0.6_s50.csv` (2,542 rows, 100% validity).
