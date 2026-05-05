# Repository Cleanup Plan — feature/dep-aware-decoding

**Status:** Investigation only. No files will be modified by this plan; review and approve before execution.
**Generated:** 2026-05-04

## 1. Branch context

| Field | Value |
|-------|-------|
| Current branch | `feature/dep-aware-decoding` |
| Branch base (merge-base with main) | `9ee8478b16715048a8118e3e4df582b97783b2e4` |
| Commits on this branch | 53 (per `git log main..HEAD --oneline`) |
| Files added on branch | ~80 (mostly outputs CSV/MD, scripts, docs) |
| Files modified on branch | 4 (model/molecularDiffusionModel.py, model/transformerBlock.py, inference/evaluate.py, dataExtractor/conditionalSELFIESDataset.py) |
| Files deleted on branch | 9 (old `checkpoints/checkpoint_step*.pt` deletions) |

The branch contains: the 40-epoch SciBERT contrastive run + HP sweep + dep-aware decoding patch + token capability probe + replication experiment + a large set of investigation docs and audit reports.

## 2. Inventory of files at project root

| Filename | Category | Originated on this branch? | Recommended action |
|----------|----------|---------------------------|--------------------|
| `.gitignore` | config | yes | keep at root |
| `README.md` | doc | yes (added by this branch) | keep at root, refresh content (see §7) |
| `main.tex` | paper source | yes | keep at root (paper deliverable) |
| `IFT6759_Final_Report (8).pdf` | paper deliverable | yes (untracked) | rename to `IFT6759_Final_Report.pdf`, keep at root |
| `chemical_tokenizer.json` | config (vocab) | inherited | keep at root |
| `40pct_source_audit.md` | audit | yes | move to `archive/audits/` |
| `audit_report.md` | audit | yes | move to `archive/audits/` |
| `technical_audit.md` | audit | yes | move to `archive/audits/` |
| `REFACTOR_PRECHECK.md` | investigation | yes | move to `archive/investigations/` |
| `TRAINING_RESUME_PLAN.md` | investigation | yes | move to `archive/investigations/` |
| `CODEBASE_MAP_FOR_DECODING.md` | doc-for-AI | yes (committed) | move to `archive/investigations/` AND delete the "downstream Claude session" sentence (see §3) |
| `repo_cleanup_plan.md` | this file | yes (created now) | move to `archive/audits/` after execution |

Subdirectories (no top-level changes proposed):
- `model/`, `tokenizer/`, `training/`, `finetune/`, `contrastive/`, `inference/`, `scripts/`, `data/`, `checkpoints/`, `outputs/`, `logs/`, `wandb/`, `dataExtractor/`

## 3. AI-trace findings

No literal "Co-authored-by: Claude" or "Generated with Claude Code" strings appear in any markdown or python file on this branch (good). The traces that DO exist are stylistic:

| File | Trace type | Severity | Suggested action |
|------|-----------|----------|------------------|
| `CODEBASE_MAP_FOR_DECODING.md` line 4 | Literal text: "Audience: A downstream Claude session that will add `commit_strategy` logic without having read any other file in the repo." | **HIGH** | Either delete the line or replace with: "Audience: A future maintainer adding new commit strategies without prior context in this codebase." This file is COMMITTED (commit 74ae103). |
| `audit_report.md` | 65 em-dashes used as paragraph separators / parenthetical breaks (high density for ~1000 lines) | medium | If kept in archive: leave. If kept at root: rewrite to use commas / colons / semicolons. |
| `technical_audit.md` | 28 em-dashes; 63 verdict-label uppercase words (MATCH / MISMATCH / NOT_FOUND / UNVERIFIABLE) which is recognizable AI table-output style | medium | Archive as-is; this is private investigation output. |
| `40pct_source_audit.md` | 22 em-dashes; "Bottom line:" framing at the top | medium | Archive as-is. |
| `REFACTOR_PRECHECK.md` | 13 em-dashes | low | Archive as-is. |
| `TRAINING_RESUME_PLAN.md` | 12 em-dashes | low | Archive as-is. |
| `checkpoints/README.md` | 9 em-dashes | low | This file IS at root of `checkpoints/` and appears in git history. Consider light edit to replace em-dashes with commas in the prose paragraphs (table separators are fine). Optional. |
| `outputs/slide10_data_inventory.md` | 18 em-dashes | low | Move to `archive/` (already in outputs/). |
| `git log` commit `ace04b2` | Commit message is verbatim: "Your current version of Antigravity is out of date. Please visit https://antigravity.google/download to download and install the latest version." (an AI tool''s error message accidentally used as a commit message) | **HIGH** in absolute terms; **CANNOT FIX** without rewriting history (which the user has excluded). | Flag for awareness only. The commit is from an early ZINC pretraining commit. Consider: leave as-is and accept the trace; or `git rebase -i` to reword (NOT proposed here per scope). |
| `outputs/dep_aware_investigation.md` and similar audit MDs | 11 such files in `outputs/` were authored as audit outputs with em-dash styling | low | All already in `outputs/` (not root). Consider: move audit-style ones (the *_audit.md and *_investigation.md and *_verification.md files) into `outputs/audits/` for organization. |

### What I checked and did NOT find:
- "Co-authored-by:" lines in any committed file (clean)
- "Generated with [Claude Code]" or "claude.ai" URLs (clean)
- "Welcome to the cutting-edge..." / "Let''s dive into..." / "comprehensive" / "leverage" / "seamlessly" — none in README.md or any other root .md (clean)
- Emoji decoration in headers (✓, ✨, 🚀) — none (clean)
- "═══" decorative bars in committed .md files — none (clean)

The README.md is well-written and reads as human-authored. It does not need a tone rewrite, only a content refresh (§7).

## 4. Proposed folder structure

```
molgen/                                    # repo root
├── README.md                              # refreshed (see §7)
├── main.tex                               # paper source
├── IFT6759_Final_Report.pdf               # renamed, keep at root
├── chemical_tokenizer.json                # vocab
├── .gitignore
│
├── archive/                               # NEW. Investigation/audit outputs.
│   ├── audits/
│   │   ├── 40pct_source_audit.md
│   │   ├── audit_report.md
│   │   ├── technical_audit.md
│   │   └── repo_cleanup_plan.md           # this file, after execution
│   └── investigations/
│       ├── REFACTOR_PRECHECK.md
│       ├── TRAINING_RESUME_PLAN.md
│       └── CODEBASE_MAP_FOR_DECODING.md   # after the "Claude session" line is edited
│
├── model/                                 # source — unchanged
├── tokenizer/
├── training/
├── finetune/
├── contrastive/
├── inference/
├── scripts/
├── data/
├── checkpoints/
├── outputs/                               # see optional sub-cleanup below
│   ├── audits/                            # OPTIONAL. Move *_audit.md *_investigation.md *_verification.md here.
│   ├── plots/                             # already exists
│   ├── slides/                            # already exists
│   ├── rl/                                # already exists
│   └── (eval CSVs and summary .txt remain at outputs/ top level)
├── logs/
├── wandb/
└── dataExtractor/
```

Rationale: a single `archive/` directory makes it obvious which files are "permanent project record" vs "investigation scratch from build-out." The two subdirectories (`audits/` and `investigations/`) preserve the fact that these were two distinct categories of work — formal verification reports vs. exploratory pre-implementation context.

The `outputs/audits/` reorganization is optional. The `outputs/` directory already has structure (`plots/`, `slides/`, `rl/`); adding `audits/` for the ~13 audit/investigation/verification MDs would be consistent. Skip if the eval CSVs and audit MDs being intermixed is not bothering you.

## 5. Source file documentation gaps

For each .py file added or modified on this branch:

| File | Module docstring | Functions missing docstrings | Priority |
|------|-----------------|------------------------------|----------|
| `model/molecularDiffusionModel.py` | NO | 1 (the `MolecularDiffusionModel` class itself; the methods have docstrings already) | **HIGH** — this is the central model class |
| `model/transformerBlock.py` | NO | 4 (`SelfAttention`, `CrossAttention`, `FeedForward`, `TransformerBlock` — all class-level) | **HIGH** — central architecture |
| `model/embeddings.py` (NOT in diff list, inherited) | — | — | — |
| `tokenizer/chemicalTokenizer.py` (NOT in diff list, inherited) | — | — | — |
| `finetune/conditionalSELFIESDataset.py` | NO | 1 (the dataset class) | **HIGH** — load path for the headline experiments |
| `finetune/diffusionCollatorPrompt.py` | NO | 1 (collator class) | medium |
| `finetune/train_chebi20.py` | YES (one-liner header) | 6 of 21 functions | medium — main entry; some of the 6 are private helpers |
| `finetune/train_text_condition.py` | NO | 4 of 7 | low — superseded by train_chebi20.py |
| `finetune/resume_eos_fix.py` | YES | 3 of 6 | low — one-off resume script |
| `contrastive/dataset.py` | NO | 2 of 3 | medium |
| `contrastive/model.py` | NO | 4 of 4 (`MoleculeEncoder`, `ContrastiveAligner`, `contrastive_loss`, `retrieval_metrics`) | **HIGH** — public API, paper §3.3 references this |
| `contrastive/utils.py` | NO | 0 of 1 | low (single trivial function) |
| `contrastive/train_contrastive.py` | YES | 2 of 5 | medium |
| `training/diffusionCollator.py` | NO | 1 of 1 | medium |
| `training/diffusionCollatorPrompt.py` | NO | 1 of 1 | low — annotated as unused in the file itself |
| `training/rl_reinforce.py` | YES | 1 of 5 | low — superseded by v2 |
| `training/rl_reinforce_v2.py` | YES | 1 of 8 | medium — paper §3.4 references this |
| `training/train_text_condition.py` | NO | 4 of 4 | low — superseded |
| `training/train_upscaled.py` | YES | 4 of 9 | medium |
| `inference/evaluate.py` | NO | 3 of 5 | low — superseded by `evaluate_prompted.py` and `train_chebi20.py --eval_only` |
| `inference/evaluate_prompted.py` | YES | 3 of 13 | medium |
| `inference/distribution_analysis.py` | YES | 0 of 2 | OK |
| `inference/diversity_check.py` | YES | 0 of 0 | OK |
| `inference/novelty_check.py` | YES | 0 of 0 | OK |
| `inference/plot_comparison.py` | YES | 1 of 1 | low |
| `inference/plot_rl_training.py` | YES | 0 of 0 | OK |
| `inference/reconstruction_accuracy.py` | YES | 1 of 6 | low |
| `scripts/build_combined_pretrain.py` | YES | 6 of 11 | medium — exposed via README workflow |
| `scripts/build_smoke_slice.py` | YES | 3 of 3 | low — internal helper |
| `scripts/build_smoke_slice_v2.py` | YES | 3 of 3 | low |
| `scripts/build_summary_table.py` | YES | 4 of 9 | low |
| `scripts/compare_replication_results.py` | YES | 7 of 7 | low — internal helpers |
| `scripts/compare_smoke_results.py` | YES | 1 of 1 | low |
| `scripts/fix_molinst_extraction.py` | YES | 5 of 10 | low |
| `scripts/probe_token_knowledge.py` | YES | 3 of 6 | medium — paper §5.4 references this |
| `scripts/recompute_atom_bleu.py` | YES | 3 of 6 | low |
| `scripts/rl_distribution_analysis.py` | YES | 0 of 1 | OK |
| `scripts/kaggle_eos_fix.py` | YES | 16 of 16 | low — one-off Kaggle notebook export |
| `scripts/kaggle_full_finetune.py` | YES | 15 of 17 | low — one-off Kaggle notebook export |
| `data/prepare_data.py` | YES | 2 of 5 | low |
| `dataExtractor/conditionalSELFIESDataset.py` | NO | 1 of 1 | low — duplicate of finetune/ version |

**Summary:** 12 files lack a module docstring (HIGH or MEDIUM priority), and ~30 individual class/function docstrings are missing. **Recommendation:** add docstrings to the HIGH priority items only (model/molecularDiffusionModel.py module + class, model/transformerBlock.py 4 classes, finetune/conditionalSELFIESDataset.py, contrastive/model.py 4 items). The rest can be left as-is since they''re internal helpers or superseded code.

## 6. Code quality issues found

**Categorized list:**

### Hardcoded paths (1 instance)
- `scripts/rl_distribution_analysis.py`: contains 1 absolute path string (likely a `/Users/...` reference). Replace with `Path(__file__).parent.parent / ...` or argparse arg.

### Duplicated logic (3 instances)
- `finetune/conditionalSELFIESDataset.py` and `dataExtractor/conditionalSELFIESDataset.py`: near-identical dataset class. The `dataExtractor/` version is the older one; `finetune/` is the active one. Recommendation: move dataExtractor''s version to `archive/old_code/` and remove the directory, OR add a comment in dataExtractor pointing to the canonical location. Do NOT delete (per scope).
- `finetune/diffusionCollatorPrompt.py` and `training/diffusionCollatorPrompt.py`: near-identical. The training/ version even has an inline note "NOTE: Not imported by any training/ script." Move training/ version to archive.
- `training/train_text_condition.py` and `finetune/train_text_condition.py`: both exist. The finetune/ version is canonical; training/ version is superseded. Move training/ to archive.

### Dead / superseded code (4 instances)
- `training/rl_reinforce.py` (v1) — superseded by `rl_reinforce_v2.py`. Per the paper §3.4 only v2 is the canonical implementation. Move v1 to `archive/old_code/`.
- `inference/evaluate.py` — superseded by `inference/evaluate_prompted.py` and `train_chebi20.py --eval_only`. Move to archive.
- `training/train_text_condition.py` and `finetune/train_text_condition.py` — superseded by `finetune/train_chebi20.py`. Both should move.
- `finetune/resume_eos_fix.py` — one-off script for the EOS bug fix; the fix is in `conditionalSELFIESDataset.py` now. Move to archive.

### Broken imports / missing __init__.py
- No `__init__.py` files in any of `model/`, `training/`, `finetune/`, `contrastive/`, `tokenizer/`, `dataExtractor/`, `inference/`, `scripts/`. The codebase uses `sys.path.insert` instead (e.g., `model/molecularDiffusionModel.py` line 6: `sys.path.insert(0, str(Path(__file__).parent.parent))`). This is a working pattern but considered poor practice. **Risk: if a future maintainer tries `import` from one of these dirs, it''ll fail without the path-hack.** Severity: low (intentional, working).
- No broken imports detected in any file.

### TODOs / FIXMEs
- **None found.** All scanned files have 0 TODO/FIXME/XXX/HACK comments. Either the project is unusually disciplined or the comments were stripped before commit.

### Stale data in committed docs
- `README.md` data table reports row counts of 20,159 / 2,505 / 2,543 for ChEBI-20 train/val/test. The actual CSV row counts are 20,158 / 2,504 / 2,542. Off-by-one in each — likely originally counted including a header line that pandas later excludes. Trivial fix.
- `README.md` Results table only includes the 20-epoch model line and rerank@N — missing the 40-epoch best (Morgan 0.318, MACCS 0.661). The paper''s final headline number is not in the README.
- `checkpoints/README.md` does not mention the 40-epoch checkpoint (`chebi20_27M_scibert_40ep_contrastive.pt`) or the 9 corresponding HP sweep CSVs. Stale.

### Inconsistent naming
- `scripts/kaggle_eos_fix.py` (37KB) and `scripts/kaggle_full_finetune.py` (40KB) are exported Jupyter notebooks committed as .py. They''re not part of the canonical pipeline and have many functions without docstrings. **Recommendation:** move to `archive/old_code/kaggle/` or rename with a `notebook_export_` prefix to signal status.
- `dataExtractor/` directory uses camelCase, all other Python directories use lowercase. Minor consistency issue, low priority.

### Stray content in git history
- Commit `ace04b2` has the message "Your current version of Antigravity is out of date. Please visit https://antigravity.google/download to download and install the latest version." This was a tool error message that became a commit message. **Cannot fix without history rewrite (excluded per scope).** Flag for awareness only — graders looking at `git log` will see it.

## 7. README assessment

**Current state:** adequate. It is well-structured (overview, model pipeline diagram, quick-start commands, data table, directory layout, results table). Tone is professional, no AI-marketing tics. Length is appropriate (~80 lines).

**What''s stale or missing:**

1. **Data row counts off by 1**: 20,159 / 2,505 / 2,543 → should be 20,158 / 2,504 / 2,542.
2. **Results table missing the 40-epoch headline.** Add a row for `Morpheus 27M_scibert_40ep` with Morgan 0.318, MACCS 0.661, atom-BLEU-2 0.614 (cfg=1.5, t=0.7, 50 steps).
3. **No reproduction recipe** for the canonical headline number. The "Quick Start" shows commands for the 20-epoch model. Add: "To reproduce the 40-epoch headline result, run training with `--num_epochs 40 --model_size 27M_scibert_40ep`, then eval with the canonical command from above adapted to model_size 27M_scibert_40ep."
4. **No setup instructions**. Missing: Python version, conda env, `pip install -r requirements.txt` step (and the `requirements.txt` itself doesn''t appear to be committed). Add a "Setup" section.
5. **No mention of paper deliverable** (`main.tex` and the PDF). Add a one-line link/reference.
6. **Directory layout is missing several directories**: `logs/`, `wandb/`, `dataExtractor/`, and the proposed `archive/`. Add them with one-line descriptions.

**Proposed sections to add (outline only — don''t write content yet):**
- `## Setup` — Python version, env, requirements install
- (within Quick Start) — add "Reproduce the 40-epoch headline result"
- `## Paper` — link to `main.tex` and PDF
- `## Repository structure` — replace existing "Directory Layout" with an updated one that includes `archive/`, `logs/`, `wandb/`
- (within Results) — add the 40ep row and update existing data row counts

## 8. Recommended execution plan

Numbered steps in execution order. Each step is constrained to: (a) inside this branch only, (b) no deletion (only `git mv`), (c) no source-code logic changes.

| # | Step | Why this order | Risk | Touches files outside branch? |
|---|------|---------------|------|-------------------------------|
| 1 | Create `archive/audits/`, `archive/investigations/`, and (optionally) `archive/old_code/` directory tree by adding a placeholder `.gitkeep` in each. | Foundation for all subsequent moves. No risk. | none | No |
| 2 | `git mv 40pct_source_audit.md archive/audits/40pct_source_audit.md` | Move the most recent audit first; smallest dependency footprint. | low — git tracks the rename, no other file references it. | No |
| 3 | `git mv audit_report.md archive/audits/audit_report.md` | Same category. | low | No |
| 4 | `git mv technical_audit.md archive/audits/technical_audit.md` | Same category. | low | No |
| 5 | `git mv REFACTOR_PRECHECK.md archive/investigations/REFACTOR_PRECHECK.md` | Investigation, not audit. | low | No |
| 6 | `git mv TRAINING_RESUME_PLAN.md archive/investigations/TRAINING_RESUME_PLAN.md` | Same. | low | No |
| 7 | EDIT `CODEBASE_MAP_FOR_DECODING.md` line 4: replace "A downstream Claude session that will add `commit_strategy` logic" with "A future maintainer adding new commit strategies". This is the single literal AI attribution in the repo. | Should happen BEFORE moving so the file is clean when archived. | medium — file is committed (commit 74ae103); the edit goes in a new commit on this branch. | No |
| 8 | `git mv CODEBASE_MAP_FOR_DECODING.md archive/investigations/CODEBASE_MAP_FOR_DECODING.md` | Move post-edit. | low | No |
| 9 | (OPTIONAL) Move audit-style files inside `outputs/` to `outputs/audits/`: `branch_audit_summary.md`, `branch_roundtrip.md`, `branch_token_inventory.md`, `bug_audit_summary.md`, `dep_aware_investigation.md`, `dep_aware_verification.md`, `generated_count_audit.md`, `index_verification.md`, `probe_comparison_20ep_vs_40ep.md`, `probe_token_knowledge.md`, `probe_token_knowledge_40ep.md`, `replication_results.md`, `rl_distribution_stats.md`, `rl_distribution_summary.md`, `slide10_data_inventory.md`, `slide11_metric_recommendation.md`, `tokenizer_audit.md`, `truncation_analysis.md`, `version_consistency.md`. | Cosmetic; the eval CSVs remain at `outputs/` top level which is the canonical content. | low — but worth checking that no script references these by exact path (`grep -r "outputs/branch_audit"` etc.) | No |
| 10 | (OPTIONAL) Move superseded code: `git mv training/rl_reinforce.py archive/old_code/rl_reinforce_v1.py`; `git mv inference/evaluate.py archive/old_code/inference_evaluate_v1.py`; `git mv training/train_text_condition.py archive/old_code/training_train_text_condition.py`; `git mv finetune/train_text_condition.py archive/old_code/finetune_train_text_condition.py`; `git mv finetune/resume_eos_fix.py archive/old_code/resume_eos_fix.py`; `git mv training/diffusionCollatorPrompt.py archive/old_code/training_diffusionCollatorPrompt.py`. | Done after the doc moves so any breakage is easier to attribute. | medium — verify nothing imports these (grep `import rl_reinforce`, `import evaluate`, etc.) before moving. | No |
| 11 | Add module docstrings to the 5 HIGH-priority files listed in §5: `model/molecularDiffusionModel.py`, `model/transformerBlock.py`, `finetune/conditionalSELFIESDataset.py`, `contrastive/model.py`, and class-level docstrings for the same. | After file moves stabilize so docstrings reference final paths. | low — pure additions, no logic change. | No |
| 12 | Fix hardcoded path in `scripts/rl_distribution_analysis.py`. | Independent of moves. | low | No |
| 13 | Update `README.md` per §7: fix the off-by-1 row counts, add the 40-epoch row to the results table, add a Setup section, add a Paper section, update Directory Layout to include `archive/`, `logs/`, `wandb/`. | After all moves — README references the final structure. | medium — content edits to a top-level user-facing doc; review carefully. | No |
| 14 | Update `checkpoints/README.md` to mention the 40-epoch checkpoint. Also consider replacing em-dashes in prose paragraphs with commas / semicolons. | Independent. | low | No |
| 15 | Rename the PDF: `git mv "IFT6759_Final_Report (8).pdf" IFT6759_Final_Report.pdf`. | Cosmetic. | low — verify no scripts hard-reference the old name. | No |
| 16 | Move `repo_cleanup_plan.md` (this file) to `archive/audits/repo_cleanup_plan.md`. | Last step; this plan archives itself once executed. | low | No |
| 17 | Single commit: "Repo cleanup: archive audit reports, fix README staleness, add docstrings to core model classes." Or split into 3 commits: (a) folder structure + moves, (b) edits to CODEBASE_MAP, README, checkpoints/README, (c) docstring additions. | Final commit after review. | low — review the diff before committing. | No |

**Total estimated effort:** 1.5–2 hours including a careful review of the edits at steps 7, 11, 13, 14.

## 9. What I will NOT do

Explicit out-of-scope list:

- Will not delete any files. Every file is moved (`git mv`) or edited in place; nothing is removed from disk or from git tracking.
- Will not modify any file outside the working tree of `feature/dep-aware-decoding`. No `main` files, no other branches.
- Will not rewrite git history. Commit `ace04b2`''s "Antigravity is out of date" message stays. No `git rebase -i`, no `git filter-branch`, no `git commit --amend` on already-pushed commits.
- Will not change source code logic. Step 11 adds docstrings (pure additions). Step 12 fixes one hardcoded path (one-line edit). All other source files are untouched.
- Will not modify `main.tex` or the submitted PDF. The paper deliverable is frozen.
- Will not modify any `.pt` checkpoint, `.csv` eval output, `.txt` summary file, or `.png/.pdf` figure. Only documentation, structure, and the one hardcoded path.
- Will not push to remote. After committing, the user reviews and pushes manually.
- Will not modify `chemical_tokenizer.json`, `.gitignore`, `wandb/` contents, or `logs/` contents.
- Will not run training, evaluation, or any compute-intensive task as part of this cleanup.
- Will not contact W&B, GitHub, or any external service.
- Will not edit the `outputs/` evaluation CSVs / summary .txt files even if they have stale numbers (those are run records, not editable artifacts).

INVESTIGATION COMPLETE
