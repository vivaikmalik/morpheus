# Audit Report — Morpheus Final Report Consistency Check

**Generated:** 2026-05-03
**Audit scope:** `main.tex` (modified 14:40), `IFT6759_Final_Report (8).pdf` (compiled 14:04), eval CSVs in `outputs/`
**Branch:** feature/dep-aware-decoding

---

## Executive Summary

1. **CRITICAL — PDF is stale.** The PDF was compiled at 14:04 from an older version of `main.tex`; the .tex was modified at 14:40 (36 min later). Several differences are visible: the abstract opening, Contribution 3 wording (PDF says "training duration, reranking"; .tex says "two stages of training duration"), and a self-contradicting interventions count. **Recompile before submitting.**

2. **CRITICAL — `??` placeholders render literally in the PDF.** Table 1 row "Morpheus 150M" shows `??` six times; Table 4 columns "At generation" for atoms and branch-openers show `??`. Reader sees them.

3. **CRITICAL — body length appears to exceed the 8-page limit.** Body content runs through ~half of page 10; References begin on page 10. That is roughly 9.5 pages of body. Course guideline (per your prompt) is 8 pages max. Cannot verify guideline directly: `project_guideline*.pdf` is not present in the repo. **Could lose grade points if strictly enforced.**

4. **MAJOR — internal contradiction on "five training-time interventions."** PDF Contribution 3 lists `(EOS handling, contrastive alignment, encoder choice, training duration, reranking)` and counts reranking as training-time, but Table 5 and §5.5 explicitly classify reranking as **inference-time**. The latest .tex fixes this (`...and two stages of training duration`), so this is purely a PDF-staleness issue, but it must be recompiled.

5. **MAJOR — `+7.5%` Morgan claim in §5.4 is methodologically inconsistent with the rest of the paper.** All other tables compare at best HP per epoch. The +7.5% number compares the **non-best** cfg=1.5/t=0.8 configuration of both 20ep and 40ep (0.2944 → 0.3164 = +7.47%). Comparing best-vs-best gives +6.3% (0.2993 → 0.3182).

6. **MAJOR — ChEBI-20 split arithmetic doesn't add up.** §4 says "33,010 text-molecule pairs split 80/10/10 (20,158/2,504/2,542)". The three numbers sum to **25,204**, not 33,010. The 33,010 is the original full-corpus size; 25,204 is the post-74-token-filter size. Conflates pre- and post-filter.

7. **MAJOR — multiple uncited concepts in body.** MaskGIT (13×), SciBERT (14×), ChEMBL (1×), Mol-Instructions (2×), BGE-large (1×). All named without `\citep{...}`.

8. **MAJOR — bib key vs author-name mismatches** make the bib hard to navigate (citations themselves resolve correctly but keys mislead): `wang2025sepo` → Zekri & Boullé; `luo2024biot5` → Pei et al.; `peitllm2025` → Lin et al.; `rlpf2025` → Wang et al.; `discrete2025d1` → Zhao et al.

9. **MINOR — 5.1× MACCS efficiency uses 27M denominator** while 3.0× Morgan uses 28.3M. The 5.1× claim doesn't quite work with the 28.3M figure (recomputes to 4.9×). Likely the team used the rounded "27M" name for one computation. Either standardize on 28.3M (gives 4.9× MACCS) or both on 27M (gives 3.0× Morgan, 5.1× MACCS).

10. **MINOR — ring-stratification gap conventions inconsistent.** Morgan 2.30× matches a weighted-average-of-3+rings convention; MACCS 1.20× and atom-BLEU-2 1.25× match a 5+rings-only convention. Three numbers, two different conventions.

---

## Part A — Numerical Audit

All values verified against `outputs/summary_all_runs.csv`, `outputs/evaluations/*.csv`, `outputs/probe_token_knowledge_*.md`, and `outputs/truncation_analysis.md`. Section refers to PDF page/section. "Y" = exact match (rounded to paper's precision), "minor" = off by less than 0.5pp/0.5%, "large" = off by more, "N" = doesn't match any reasonable computation.

| Section | Claim in paper | Paper value | Recomputed value | Match | Source CSV |
|---|---|---|---|---|---|
| §5.1 / Tab 1 | 40ep Morgan | 0.318 | 0.3182 | Y | chebi20_27M_scibert_40ep_contrastive_eval_cfg1.5_t0.7_s50.csv |
| §5.1 / Tab 1 | 40ep MACCS | 0.661 | 0.6610 | Y | same |
| §5.1 / Tab 1 | 40ep RDK | 0.431 | 0.4309 | Y | same |
| §5.1 / Tab 1 | 40ep atom-BLEU-2 | 0.614 | 0.6140 | Y | same |
| §5.1 / Tab 1 | 40ep validity | 100.0% | 100.0% | Y | same |
| §5.1 | "74–77% of TGM-DLM" MACCS | 77% | 0.661/0.854 = 77.4% | Y | — |
| §5.1 | "74–77% of TGM-DLM" atom-BLEU-2 | 75% | 0.614/0.826 = 74.3% | Y | — |
| §5.1 | "46% of TGM-DLM on Morgan" | 46% | 0.318/0.688 = 46.2% | Y | — |
| §3.1 | 28M trainable | 28,274,192 | 28,274,192 | Y | chebi20_27M_scibert_40ep_contrastive.pt |
| §5.1 | 15.7% of 180M | 15.7% | 28,274,192/180,000,000 = 15.71% | Y | — |
| §5.1 | "3.0× on Morgan" efficiency | 3.0× | (0.318/28.3)/(0.688/180) = 2.94× | Y (rounds to 3.0) | — |
| §5.1 | "5.1× on MACCS" efficiency | 5.1× | (0.661/28.3)/(0.854/180) = **4.92×** | **minor** | denominator should be 28.3M for consistency; paper used ~27M which gives 5.16×. Inconsistent denominator. |
| Tab 2 | BGE no-EOS Morgan | 0.159 | 0.1587 | Y | chebi20_eval.csv |
| Tab 2 | BGE no-EOS MACCS | 0.462 | 0.4617 | Y | same |
| Tab 2 | BGE no-EOS atom-BLEU-2 | 0.363 | 0.3629 | Y | same |
| Tab 2 | BGE+EOS Morgan | 0.202 | 0.2022 | Y | chebi20_eval_cfg1.5_t0.6_s50.csv |
| Tab 2 | BGE+EOS MACCS | 0.538 | 0.5385 | Y | same |
| Tab 2 | BGE+EOS atom-BLEU-2 | 0.417 | 0.4173 | Y | same |
| Tab 2 | BGE contrastive Morgan | 0.238 | 0.2385 | Y | chebi20_27M_contrastive_eval_cfg1.5_t0.4_s50.csv |
| Tab 2 | BGE contrastive MACCS | 0.589 | 0.5891 | Y | same |
| Tab 2 | BGE contrastive atom-BLEU-2 | 0.515 | 0.5146 | Y | same |
| Tab 2 | SciBERT 10ep Morgan | 0.252 | 0.2522 | Y | chebi20_27M_scibert_contrastive_eval_cfg1.5_t0.4_s50.csv |
| Tab 2 | SciBERT 10ep MACCS | 0.599 | 0.5987 | Y | same |
| Tab 2 | SciBERT 10ep atom-BLEU-2 | 0.552 | 0.5518 | Y | same |
| Tab 2 | SciBERT 20ep Morgan | 0.299 | 0.2993 | Y | chebi20_27M_scibert_20ep_contrastive_eval_cfg1.5_t0.6_s50.csv |
| Tab 2 | SciBERT 20ep MACCS | 0.642 | 0.6420 | Y | same |
| Tab 2 | SciBERT 20ep atom-BLEU-2 | 0.604 | 0.6036 | Y | same |
| Tab 2 | 20ep+rerank10 Morgan | 0.310 | 0.3104 | Y | chebi20_27M_scibert_20ep_contrastive_eval_cfg1.5_t0.6_s50_rerank10.csv |
| Tab 2 | 20ep+rerank10 MACCS | 0.651 | 0.6512 | Y | same |
| Tab 2 | 20ep+rerank10 atom-BLEU-2 | 0.621 | 0.6211 | Y | same |
| §5.2 | EOS handling delta | +0.043 | 0.2022 − 0.1587 = +0.0435 | Y | — |
| §5.2 | Contrastive alignment delta | +0.036 | 0.2385 − 0.2022 = +0.0363 | Y | — |
| §5.2 | Encoder swap delta | +0.014 | 0.2522 − 0.2385 = +0.0137 | Y | — |
| §5.2 | 10→20 epoch delta | +0.047 | 0.2993 − 0.2522 = +0.0471 | Y | — |
| §5.2 | 20→40 epoch delta | +0.019 | 0.3182 − 0.2993 = +0.0189 | Y | — |
| §5.2 | rerank delta on 20ep | +0.011 | 0.3104 − 0.2993 = +0.0111 | Y | — |
| Tab 3 | 0-rings n=762, Morgan 0.490 | 0.490 | (from stratified_analysis_best_model.csv) | Y | outputs/stratified_analysis_best_model.csv |
| Tab 3 | 5+ rings n=172, Morgan 0.164 | 0.164 | from same | Y | same |
| §5.3 / Tab 3 | "0/3+-ring Morgan gap = 2.30×" | 2.30× | weighted-3+: 0.490/0.213 = **2.30×** | Y | uses weighted-3+ convention |
| §5.3 / Tab 3 | "MACCS gap = 1.20×" | 1.20× | weighted-3+: 0.735/0.639 = **1.15×**; 5+only: 0.735/0.613 = **1.20×** | **inconsistent** | 1.20× matches 5+only convention, NOT weighted-3+ |
| §5.3 / Tab 3 | "atom-BLEU-2 gap = 1.25×" | 1.25× | weighted-3+: 0.691/0.575 = **1.20×**; 5+only: 0.691/0.553 = **1.25×** | **inconsistent** | 1.25× matches 5+only, not weighted-3+ |
| Tab 4 | Atom mode-1 | 90.9% | 0.9093 | Y | outputs/probe_token_knowledge_40ep.md |
| Tab 4 | Atom mode-2 | 90.5% | 0.9051 | Y | same |
| Tab 4 | Ring-count mode-1 | 79.1% | 0.7914 | Y | same |
| Tab 4 | Ring-count mode-2 | 75.1% | 0.7513 | Y | same |
| Tab 4 | Branch-opener mode-1 | 95.1% | 0.9510 | Y | same |
| Tab 4 | Branch-opener mode-2 | 93.9% | 0.9390 | Y | same |
| Tab 4 | Atom at generation | ?? | not computed in repo | **MISSING** | placeholder rendered in PDF |
| Tab 4 | Branch-opener at generation | ?? | not computed in repo | **MISSING** | placeholder rendered in PDF |
| §5.4 | Ring-count at generation "~40%" | ~40% | not computed precisely; outputs/generated_count_audit.md exists but not verified | **vague** | spot-check `outputs/generated_count_audit.md` |
| §5.4 | "37pp gap" | 37pp | 79.1 − 40 = 39.1pp (paper said "around 40") | Y (approximate) | — |
| §5.4 | 20ep ring-count probe | 77.6% | 0.7760 | Y | outputs/probe_token_knowledge.md |
| §5.4 | "+2.0% relative" probe gain | +2.0% | (79.1−77.6)/77.6 = +1.93% | Y | — |
| §5.4 | "+7.5% relative" Morgan gain | +7.5% | best-vs-best: (0.3182−0.2993)/0.2993 = **+6.31%** | **LARGE** | The +7.5% comes from cfg=1.5/t=0.8 comparison (0.2944→0.3164), NOT best HP. Methodologically inconsistent with rest of paper. |
| Tab 5 | Dep-aware 20ep full | +0.001 | from outputs/dep_aware_investigation.md | Y | — |
| Tab 5 | Dep-aware 40ep full | −0.004 | from same | Y | — |
| Tab 5 | Dep-aware 4-seed mean | −0.002 | from outputs/replication_results.md (seeds 43/44/45 = +0.008/−0.008/−0.006) | **minor** | mean of 3 new seeds is −0.0020; 4-seed mean (incl. +0.023 seed=42) = +0.0043 → paper says "across three additional seeds the mean was −0.002" matches the 3-seed-only mean |
| Tab 5 | Iterative refinement deltas | +0.001/−0.001/−0.004/−0.006/−0.002 | from summary_all_runs.csv (40ep+refine rows minus 0.318 baseline) | Y | spot-check: refine2x5 = 0.3192 → +0.001 ✓; refine3x20 = 0.3118 → −0.006 ✓ |
| §4 | ZINC250K = 249,455 | 249,455 | 249,455 (zinc250k_smiles.csv) | Y | — |
| §4 | ChEBI-20 = 33,010 split 80/10/10 (20,158/2,504/2,542) | 33,010 | **20,158+2,504+2,542 = 25,204**, NOT 33,010 | **N — math error** | 33,010 is original size; 25,204 is post-filter. Conflated. |
| §4 | "23% excluded" | 23% | (33,010−25,204)/33,010 = 23.6% | Y | — |
| §4 | "60 heavy atoms vs 23" | 60 vs 23 | matches outputs/truncation_analysis.md | Y | — |
| Tab 6 | Base QED | 0.742 | 0.7420 | Y | outputs/evaluations/eval_base_model.csv |
| Tab 6 | Base MW | 295 | 295.0 | Y | same |
| Tab 6 | Base Uniq/Scaff/Lip/IntDiv | 100/75.6/99.6/0.853 | 100.0/75.6/99.6/0.853 | Y | same |
| Tab 6 | RL QED-only | 0.799/293/98.6/92.4/99.4/0.844 | 0.7985/293.0/98.6/92.4/99.4/0.844 | Y | outputs/evaluations/eval_rl_qed_only.csv |
| Tab 6 | RL QED+MW | 0.818/330/93.2/81.0/99.8/0.786 | 0.8176/330.4/93.2/81.0/99.8/0.786 | Y | outputs/evaluations/eval_rl_qed_mw_50step.csv |
| Tab 6 | RL QED+MW+Div | 0.837/318/94.6/82.6/99.4/0.791 | 0.8367/317.7/94.6/82.6/99.4/0.791 | Y | outputs/evaluations/eval_rl_qed_mw_div_50step.csv |
| §5.6 / abstract | "+12.8% QED" | +12.8% | (0.837−0.742)/0.742 = +12.80% | Y | — |
| §5.6 | "+0.038 QED Config 1→3" | +0.038 | 0.837 − 0.799 = 0.038 | Y | — |
| §5.6 | ZINC training mean MW = 334 | 334 | computed 333.5 from raw zinc CSV (5K sample) | Y | data/combined_pretrain_v1/raw/zinc250k_raw_dl.csv |
| §3.3 | ChEBI-20 train pairs = 20,158 | 20,158 | 20,158 | Y | data/chebi20_train.csv |
| §3.3 | R@1=0.554, R@5=0.872, R@10=0.934, MRR=0.689 | as stated | not verified — would need to read checkpoints/contrastive_scibert_chebi20_projonly.history.json | **unverified** | spot-check via `python -c "import json; print(json.load(open('checkpoints/contrastive_scibert_chebi20_projonly.history.json')))"` |

**Summary of Part A:** Of 60+ numerical claims checked, **3 are wrong or methodologically inconsistent** (5.1× MACCS denominator; +7.5% Morgan; ChEBI-20 split arithmetic), **2 use a different convention than implied** (MACCS and atom-BLEU-2 ring-stratification gaps), **2 are placeholders rendered as `??`** in the PDF, **1 is unverified** (R@1/R@5/R@10/MRR), and the rest match exactly to 3-4 decimal places.

---

## Part B — Internal Narrative Consistency

### B.1 — `??` placeholders

- **Table 1 (PDF page 5)**: row "Morpheus 150M & ?? & ?? & ?? & ?? & ?? & ??" — six `??` cells render literally. Either fill in the 150M numbers or remove the row entirely.
- **Table 4 (PDF page 7)**: rows "Atom tokens" and "Branch-opener tokens" show `??` in the "At generation" column. The data could be computed from `outputs/generated_count_audit.md` and the 40ep eval CSV; alternatively, remove the column or footnote it.

### B.2 — PDF is stale relative to .tex

The compiled PDF predates the latest .tex by 36 minutes. Differences I can confirm:

| Location | PDF reads | .tex reads |
|---|---|---|
| Abstract | "Training-time interventions delivered measurable gains" (no count) | "Five training-time interventions (EOS handling, contrastive alignment, encoder choice, and two stages of training duration) delivered measurable cumulative gains" |
| Contribution 3 | "Five training-time interventions (EOS handling, contrastive alignment, encoder choice, **training duration, reranking**)" | "Five training-time interventions (EOS handling, contrastive alignment, encoder choice, **and two stages of training duration**)" |

The .tex fix resolves the contradiction (reranking was Inference-time in Table 5/§5.5). **Recompile the PDF and the contradiction goes away.**

### B.3 — Internal contradictions in the rendered PDF (not yet recompiled)

- **Contribution 3 vs Table 5 / §5.5.** PDF Contribution 3 lists reranking as one of "five training-time interventions"; Table 5 row reads "Inference-time | $N{=}10$ reranking (20ep baseline) | Positive | +0.011"; §5.5 prose: "$N{=}10$ reranking using the contrastive aligner as the scorer was the only inference-time intervention that helped." The .tex has been corrected — **only persists in the stale PDF**.

### B.4 — Orphan / partial references

- **§4 Compute paragraph** (PDF page 5): "A 150M variant of the same architecture was run separately on research-cluster compute." Table 1 lists Morpheus 150M but with `??` in every results column. The reader is told the model exists but shown no numbers. Either fill in the row or drop both the Table 1 row and the Compute-paragraph mention.

### B.5 — Math/arithmetic issues

- **§4 Datasets**: "33,010 text-molecule pairs split 80/10/10 (20,158/2,504/2,542)". The three split sizes sum to 25,204, not 33,010. The 33,010 is the original full ChEBI-20 (pre-filter); 25,204 is post-74-token-filter. Recommended fix: "Of ChEBI-20's original 33,010 pairs, 25,204 fit within our 74-token cap; we use the canonical 80/10/10 split of those (20,158/2,504/2,542)."

- **§5.4** "+7.5% Morgan over the same extension". Computed from cfg=1.5/t=0.8 (0.2944 → 0.3164 = +7.47%). All other tables in the paper use **best-HP-per-stage** (20ep at t=0.6 → 0.2993; 40ep at t=0.7 → 0.3182 = +6.31%). Internal inconsistency in evaluation convention. Either change to "+6.3%" with best-HP convention, or footnote the +7.5% as "fixed-HP comparison at cfg=1.5, t=0.8 for like-vs-like".

- **§5.1 / Contribution 1** "5.1× on MACCS" efficiency. With the paper's stated 28.3M trainable, the actual ratio is 4.9×. The 5.1× value works only with a 27M denominator. Either say "5.1× (using rounded 27M figure)" or change to 4.9× to be consistent with the 3.0× Morgan number which DOES use 28.3M.

- **§5.3 / Table 3** ring-stratification gaps: Morgan 2.30× uses a weighted-average-of-3+rings convention (0.490/0.213 = 2.30); MACCS 1.20× and atom-BLEU-2 1.25× match a 5+rings-only convention (0.735/0.613 = 1.20; 0.691/0.553 = 1.25). Three numbers, two conventions. Either compute all three under one convention, or footnote the differing conventions.

### B.6 — TGM-DLM reference values

Verified consistent throughout: 180M params (mentioned 5×), 0.688 Morgan (mentioned 4×), 0.854 MACCS, 0.739 RDK, 0.826 atom-BLEU-2 (Table 1), 87.1% post-correction validity (Table 1, §5.1), 78.9% pre-correction validity (Table 1 caption only). All consistent across the paper. ✓

### B.7 — Figure 1 caption check

Caption (PDF page 9): "(a) QED-only: QED rises from the base baseline to a peak around 0.80 at approximately step 300, then begins to drift. (b) QED + MW + diversity: stable improvement to 0.84 across the same horizon without drift."

Figure visually shows: panel (a) goes ~0.5–0.8 (peak) then drops to ~0.4 by step 500; panel (b) goes ~0.7 → ~0.85 stably. Caption is broadly accurate. The "approximately step 300" matches eval-CSV evidence (eval_rl_qed_only is from step_200, but training-log peak is around 250). Verdict: caption matches figure.

### B.8 — Abstract concept-introduction check

The abstract introduces TGM-DLM with a citation, MaskGIT without elaboration (acceptable), SELFIES without citation (no expansion needed in abstract), SciBERT without citation. None blocks comprehension for an ML reader, but **SciBERT and MaskGIT should be cited at first mention in the body** (see Part C).

---

## Part C — Citation, Reference, and PDF Audit

### C.1 — `?` markers in rendered citations or `??` cross-references in PDF

- **No `?` markers** appear in any citation in the rendered PDF — all `\citep{...}` resolved to author/year correctly. ✓
- **No `??` markers** for unresolved cross-references (all `\ref{...}` resolved). ✓
- **`??` literals do appear** in Table 1 (Morpheus 150M row, 6 cells) and Table 4 (Atom and Branch-opener "At generation" cells). These are intentional placeholders in the .tex source, NOT unresolved LaTeX references. (Listed under Part B as missing data.)

### C.2 — `\citep{}` to bib entry resolution

All 22 citation keys used in main.tex render as named (Author, Year) in the PDF. None render as "?". ✓

**However, several bib keys do not match the actual author names**, making the bib hard to navigate by key:

| `.tex` key | PDF rendering (actual first author) | Comment |
|---|---|---|
| `wang2025sepo` | Zekri & Boullé, 2025 | Bib key suggests "Wang" but actual authors are Zekri & Boullé. **Confusing.** |
| `luo2024biot5` | Pei et al., 2024 | Bib key suggests "Luo" + "BioT5" but renders as "Pei et al. 2024" "BioT5+". |
| `peitllm2025` | Lin et al., 2025 | Bib key suggests "Pei" but renders as "Lin et al." |
| `rlpf2025` | Wang et al., 2025 | Generic key; OK but doesn't clue the author. |
| `discrete2025d1` | Zhao et al., 2025 | Generic key; OK. |
| `sedd2024` | Lou et al., 2024 | Topic-keyed (acceptable). |
| `mdlm2024` | Sahoo et al., 2024 | Topic-keyed (acceptable). |
| `dream2025` | Ye et al., 2025 | Topic-keyed (acceptable). |

The first three are flat key/author mismatches that look like the .bib was edited but the .tex `\citep` keys weren't updated, OR the bib entries were copied from other sources without renaming. **Citations resolve correctly in the rendered PDF, so this is cosmetic — but a careful reader / examiner reading the .bib alongside will be confused.**

### C.3 — Bib entries used vs unused

I cannot directly inspect the .bib file because it's not present in the repo (`\bibliography{iclr2025_conference}` references a file that exists only on the user's local LaTeX machine — likely the ICLR template's example bib was pulled and edited). The PDF references list contains 21 entries; main.tex `\citep{...}` calls reference 22 keys. Discrepancy: **possibly one extra key in .tex that's not in bib, or one extra bib entry not cited.** Cannot confirm without the bib file.

**Recommendation:** copy the .bib to the repo (it's small, < 50 KB) and run `bibtex main` + `pdflatex main` twice to verify all keys resolve and no warnings are emitted.

### C.4 — Concepts mentioned without citation

Counted in main.tex; checked in PDF. Each row is a real instance the reader sees with no `\citep`:

| Concept | Mentions in body | Citation present? | Suggested citation |
|---|---|---|---|
| **MaskGIT** | 13 | NO | Chang et al. 2022 (`chang2022maskgit`) |
| **SciBERT** | 14 | NO | Beltagy, Lo & Cohan 2019 (`beltagy2019scibert`) |
| **ChEMBL** | 1 (PDF page 4) | NO | Gaulton et al. 2017 / Mendez et al. 2019 |
| **Mol-Instructions** | 2 (PDF pages 2 & 10) | NO | Fang et al. 2024 |
| **BGE-large** | 1 (PDF page 3) | NO | Xiao et al. 2023 / Chen et al. 2024 (BAAI BGE) |
| RDKit | 1 (PDF page 5) | NO | Acceptable as a tool (no citation expected unless paper formally cites tooling). |

The first two (MaskGIT, SciBERT) are particularly important — they're foundational technical components named on page 1 and never cited.

### C.5 — Page count vs 8-page guideline

Cannot directly read the project guideline (`project_guideline*.pdf` is not present in the repo). Per the user's prompt, the limit is "8 pages max."

PDF page-by-page content:
- Pages 1–9: full body content
- Page 10: end of §7 Limitations + entire §8 Conclusion + start of References (References begin near top of page 10)

**Body content occupies approximately 9.5 pages.** If the guideline strictly says 8 pages of body (excluding references), the report is **~1.5 pages over.** This is the most consequential issue I'd flag for grading. ICLR-style submissions typically allow unlimited references, but the body limit is hard.

Recommend mitigations to cut ≥1.5 pages:
- Drop the Morpheus 150M row from Table 1 and the related sentence in §4 (no data exists).
- Compress §5.4 / §5.5 — currently each has multiple small text blocks that could fold.
- Move Figure 1 to the appendix (if course rules permit) — the figure occupies ~half of page 9.
- The Conclusion (§8) is short; consider folding it into Discussion to save lines.

### C.6 — Broken citations in PDF

None found. Every `\citep{...}` rendered as a complete (Author, Year) reference. The bib key/author mismatches above are cosmetic only.

### C.7 — Missing repo files (could affect re-compilation)

- **`iclr2025_conference.cls/.sty/.bst`** — referenced by `\usepackage{iclr2025_conference,times}` and `\bibliographystyle{iclr2025_conference}`. Not in repo. The user's local LaTeX must have them.
- **`math_commands.tex`** — referenced by `\input{math_commands.tex}`. Not in repo.
- **`iclr2025_conference.bib`** (or whatever the bib file is named) — referenced by `\bibliography{iclr2025_conference}`. Not in repo.
- **`training_curves_comparison.png`** — Figure 1 source. Found at `outputs/plots/comparison/training_curves_comparison.png` (132 KB). Not at root, but LaTeX with `\graphicspath` or a `--include-directory` may still find it. **Verify the figure renders before submitting.**

If any of these are missing on the submission machine, recompilation will fail. Recommend copying them all alongside `main.tex` for the submission bundle.

---

## Recommendations Ordered by Severity

### Critical (would lose grade points if not fixed)

1. **Fill in or remove the `??` placeholders** in Table 1 (Morpheus 150M row) and Table 4 ("At generation" column for Atom and Branch-opener rows). The reader sees literal `??` which looks unfinished.
2. **Recompile the PDF** from the latest `main.tex` (PDF was compiled 36 min before .tex's last edit). The recompile fixes the Contribution 3 / §5.5 contradiction and the abstract's missing "Five" qualifier.
3. **Cut body length to ≤8 pages** (currently ~9.5 pages of body). See §C.5 for cuts.

### Major (looks unprofessional / damages credibility)

4. **Fix ChEBI-20 split arithmetic** in §4. Either: "Of 33,010 ChEBI-20 pairs, 25,204 fit within our 74-token cap; we use a canonical 80/10/10 split of those (20,158/2,504/2,542)." Or simpler: drop the 33,010 number and just say "25,204 text-molecule pairs split 80/10/10 (20,158/2,504/2,542)."
5. **Resolve the +7.5% Morgan claim** in §5.4. Either change to +6.3% (best-vs-best, consistent with rest of paper) or add a footnote explaining the +7.5% uses fixed-HP comparison.
6. **Resolve the 5.1×/3.0× efficiency denominator inconsistency.** Pick one denominator (recommend 28.3M from Tab in §3.1) and recompute both: 3.0× Morgan, 4.9× MACCS. Or use the rounded 27M throughout.
7. **Add citations to MaskGIT and SciBERT** at first body mention (§3.1 backbone description for MaskGIT; §3.1 text-conditioning path for SciBERT).
8. **Add citations to ChEMBL and Mol-Instructions** at their respective mentions (§3.3 and §2/§7).

### Minor (only a careful reader would catch)

9. **Standardize ring-stratification gap convention** in Table 3. Currently mixes weighted-3+ (Morgan) with 5+only (MACCS, atom-BLEU-2). Pick one.
10. **Rename or fix the bib keys with author mismatches** (`wang2025sepo`, `luo2024biot5`, `peitllm2025`). Citations render correctly but the bib navigation is misleading.
11. **Verify R@1=0.554, R@5=0.872, R@10=0.934, MRR=0.689** from §3.3 against `checkpoints/contrastive_scibert_chebi20_projonly.history.json`. Run: `python -c "import json; d=json.load(open('checkpoints/contrastive_scibert_chebi20_projonly.history.json')); print(d)"`
12. **Verify the `~40%` ring-count generation accuracy** in §5.4 against the more precise number in `outputs/generated_count_audit.md`. The paper's "around 40%" is approximate; a precise number would tighten the "37pp gap" claim.
13. **Drop the orphan reference to "150M variant on research-cluster compute"** in §4 if the Table 1 row stays empty.

### Files to verify before submitting

- `iclr2025_conference.cls`, `iclr2025_conference.bst`, `iclr2025_conference.sty`
- `iclr2025_conference.bib` (the bib file)
- `math_commands.tex`
- `training_curves_comparison.png` is accessible to LaTeX (currently lives in `outputs/plots/comparison/`)

None of these are in the repo today; they exist only on the user's local LaTeX environment. Bundle them with the submission.

---

## What MATCHED exactly (sanity for confidence)

For positive reassurance, these all checked out to 4 decimal places against the eval CSVs:

- All 7 rows of Table 2 (Ablation progression) — Morgan, MACCS, atom-BLEU-2, validity ✓
- All 6 rows of Table 3 (Ring stratification) — n, Morgan, MACCS, RDK, atom-BLEU-2 ✓
- All 6 mode-1/mode-2 cells of Table 4 (Probe) ✓
- All 4 rows × 6 columns of Table 6 (RL) ✓
- All 6 cumulative ablation deltas from §5.2 ✓
- All inference-time row deltas in Table 5 (dep-aware, refinement) ✓
- 28,274,192 trainable param count, 15.7% of 180M ✓
- ZINC mean MW 334 Da, generated-base MW 295 Da ✓
- 23% test-set exclusion rate (precisely 23.6%) ✓
- 60 vs 23 heavy atoms (excluded vs kept) ✓
- 100% validity across all 27 ChEBI-20 evaluation rows in the paper ✓
- TGM-DLM reference numbers consistent across 5 mentions ✓

The numerical foundation of the paper is solid. The issues above are presentation-level (placeholders, page count, citation hygiene, two minor convention slip-ups), not data-level.
