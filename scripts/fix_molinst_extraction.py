#!/usr/bin/env python3
"""
Fix Mol-Instructions SMILES extraction: the original script used a regex
that extracted almost nothing because the dataset uses SELFIES format, not SMILES.

This script re-processes the already-extracted JSON files using selfies.decoder(),
then re-runs canonicalization, contamination check, combined assembly, and stats.

Re-uses ZINC and ChEBI outputs already produced by build_combined_pretrain.py.
"""
import os, sys, re, json, time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

SCRIPT_DIR = Path(__file__).resolve().parent
REPO       = SCRIPT_DIR.parent
DATA_DIR   = REPO / "data"
OUT_DIR    = REPO / "data" / "combined_pretrain_v1"
RAW_DIR    = OUT_DIR / "raw"
EXTRACT_DIR = RAW_DIR / "molinst_extracted" / "Molecule-oriented_Instructions"

MOLINST_SMILES = OUT_DIR / "molinst_smiles.csv"
MOLINST_CAN    = OUT_DIR / "molinst_canonical.csv"
MOLINST_CLEAN  = OUT_DIR / "molinst_canonical_clean.csv"
CHEBI_TEST_SMILES = OUT_DIR / "chebi20_test_smiles.csv"
COMBINED          = OUT_DIR / "combined_pretrain_smiles.csv"
README            = OUT_DIR / "README.md"

ZINC_CAN   = OUT_DIR / "zinc250k_canonical.csv"
CHEBI_CAN  = OUT_DIR / "chebi20_train_canonical.csv"

def log(msg): print(msg, flush=True)
def banner(t):
    log(f"\n{'='*60}")
    log(f"  {t}")
    log('='*60)

def canonicalize(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None

def selfies_to_canonical(sel_str):
    """Convert a single SELFIES string to canonical SMILES or None."""
    try:
        smi = sf.decoder(sel_str)
        if not smi: return None
        return canonicalize(smi)
    except Exception:
        return None

def extract_selfies_parts(text):
    """
    Split text into SELFIES fragments.
    Handles:
      - Single SELFIES:  [C][C@H1]...
      - Multi-fragment:  [C][C].[Cl].[BH4-1]   (dot between ] and [ )
      - Reaction arrow:  A>>B  or  A.B>>C
    Returns list of SELFIES strings.
    """
    # Split on >> first (reaction arrow)
    parts = re.split(r'>>', text)
    fragments = []
    for part in parts:
        # Split on dot-between-fragments: ]. [ pattern
        # Use regex: look for literal . between ] and [
        sub_parts = re.split(r'(?<=\])\.(?=\[)', part)
        fragments.extend([s.strip() for s in sub_parts if s.strip()])
    return fragments

# Decide which fields in each task contain SELFIES
# Based on examination of data structure:
TASK_FIELDS = {
    "description_guided_molecule_design": ["output"],          # input = description text
    "molecular_description_generation":   ["input"],           # output = description text
    "forward_reaction_prediction":        ["input", "output"],  # both are SELFIES
    "property_prediction":                ["input"],            # output = numeric
    "retrosynthesis":                     ["input", "output"],  # both SELFIES
    "reagent_prediction":                 ["input", "output"],  # both SELFIES
}

# ── Re-extract Mol-Instructions ───────────────────────────────────────────────
banner("Task 3 FIX: Re-extracting Mol-Instructions (SELFIES format)")

if not EXTRACT_DIR.exists():
    log(f"ERROR: {EXTRACT_DIR} not found. Run build_combined_pretrain.py first.")
    sys.exit(1)

json_files = sorted(EXTRACT_DIR.glob("*.json"))
log(f"Found {len(json_files)} JSON files")

molinst_smiles_set = set()
task_stats = {}

for jf in json_files:
    task_name = jf.stem
    fields_to_check = TASK_FIELDS.get(task_name, ["input", "output"])
    log(f"\n  Processing: {task_name}  (fields: {fields_to_check})")
    
    with open(jf, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    n_rows = len(data)
    n_extracted = 0
    n_failed = 0
    count_before = len(molinst_smiles_set)
    
    t0 = time.time()
    for i, row in enumerate(data):
        for field in fields_to_check:
            val = row.get(field, "")
            if not val or not isinstance(val, str): continue
            # Only try SELFIES extraction if it looks like SELFIES
            if not val.strip().startswith('['):
                continue
            # Extract and convert each fragment
            fragments = extract_selfies_parts(val.strip())
            for frag in fragments:
                if not frag.startswith('['):
                    continue
                can = selfies_to_canonical(frag)
                if can:
                    molinst_smiles_set.add(can)
                    n_extracted += 1
                else:
                    n_failed += 1
        
        if (i+1) % 50000 == 0:
            elapsed = time.time() - t0
            log(f"    {i+1:,}/{n_rows:,}  unique_so_far={len(molinst_smiles_set):,}  ({elapsed:.1f}s)")
    
    new_added = len(molinst_smiles_set) - count_before
    task_stats[task_name] = {"rows": n_rows, "extracted": n_extracted, "failed": n_failed, "new_unique": new_added}
    log(f"    Done: rows={n_rows:,}  extracted={n_extracted:,}  failed={n_failed:,}  "
        f"new_unique={new_added:,}  total_unique={len(molinst_smiles_set):,}  "
        f"({time.time()-t0:.1f}s)")

log(f"\n  Total unique valid SMILES from Mol-Instructions: {len(molinst_smiles_set):,}")
molinst_smiles_list = sorted(molinst_smiles_set)
pd.DataFrame({"smiles": molinst_smiles_list}).to_csv(MOLINST_SMILES, index=False)
log(f"  Saved: {MOLINST_SMILES.name}")

# ── Canonicalize molinst ──────────────────────────────────────────────────────
banner("Task 4: Canonicalize Mol-Instructions")
# Already canonical (canonicalize was called during extraction), just deduplicate
unique_can = sorted(molinst_smiles_set)
pd.DataFrame({"smiles": unique_can}).to_csv(MOLINST_CAN, index=False)
log(f"  molinst_canonical.csv: {len(unique_can):,} rows (already canonical)")

# ── Test set contamination ────────────────────────────────────────────────────
banner("Task 5: Test set contamination check")

test_df = pd.read_csv(CHEBI_TEST_SMILES)
test_can = set()
for s in test_df["smiles"].dropna().astype(str):
    c = canonicalize(s)
    if c: test_can.add(c)
log(f"  Test set: {len(test_can):,} canonical unique molecules")

molinst_set = set(unique_can)
overlap = molinst_set & test_can
pct_src  = 100 * len(overlap) / max(len(molinst_set), 1)
pct_test = 100 * len(overlap) / max(len(test_can), 1)
log(f"  Mol-Instructions overlap: {len(overlap):,} / {len(molinst_set):,} "
    f"({pct_src:.3f}% of source, {pct_test:.2f}% of test)")

if pct_src < 1.0:
    mi_verdict = "SAFE (< 1% overlap, include as-is)"
elif pct_src < 5.0:
    mi_verdict = "FLAGGED (1-5% overlap, include but note)"
elif pct_src < 15.0:
    mi_verdict = "FILTERED (5-15% overlap, removing overlapping molecules)"
else:
    mi_verdict = "EXCLUDED (> 15% overlap, contamination concern)"

molinst_clean = sorted(molinst_set - test_can)
pd.DataFrame({"smiles": molinst_clean}).to_csv(MOLINST_CLEAN, index=False)
log(f"  molinst_canonical_clean.csv: {len(molinst_clean):,} molecules")
log(f"  Verdict: {mi_verdict}")

# ── Load ZINC and ChEBI canonical ─────────────────────────────────────────────
banner("Task 6: Re-combine all sources")

zinc_can  = pd.read_csv(ZINC_CAN)["smiles"].tolist()
chebi_can = pd.read_csv(CHEBI_CAN)["smiles"].tolist()

seen = {}
multi_source_count = 0
for label, smi_list in [("zinc", zinc_can), ("chebi", chebi_can), ("molinst", molinst_clean)]:
    new_count = 0
    for s in smi_list:
        if s not in seen:
            seen[s] = label
            new_count += 1
        else:
            multi_source_count += 1
    log(f"  Added {new_count:,} new from {label}  (total: {len(seen):,})")

log(f"\n  Total unique: {len(seen):,}  (multi-source: {multi_source_count:,})")
source_counts = Counter(seen.values())
for src, cnt in sorted(source_counts.items()):
    log(f"  {src}: {cnt:,}  ({100*cnt/len(seen):.1f}%)")

rows = [{"smiles": s, "source": src} for s, src in seen.items()]
combined_df = pd.DataFrame(rows)
combined_df.to_csv(COMBINED, index=False)
log(f"  Saved: {COMBINED.name}")

# ── Descriptive statistics ─────────────────────────────────────────────────────
banner("Task 7: Recompute statistics")

def mol_props(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None
        ring_info = mol.GetRingInfo()
        aromatic_rings = sum(1 for ring in ring_info.AtomRings()
                             if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring))
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smi = Chem.MolToSmiles(scaffold) if scaffold else ""
        except Exception:
            scaffold_smi = ""
        return {
            "heavy_atoms":    mol.GetNumHeavyAtoms(),
            "mol_weight":     Descriptors.MolWt(mol),
            "ring_count":     ring_info.NumRings(),
            "branch_count":   smi.count("("),
            "aromatic_rings": aromatic_rings,
            "stereocenters":  len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
            "scaffold":       scaffold_smi,
        }
    except Exception:
        return None

MAX_STATS = 100_000
rng = np.random.default_rng(42)
all_smiles = combined_df["smiles"].tolist()
stat_smiles = all_smiles if len(all_smiles) <= MAX_STATS else \
              rng.choice(all_smiles, size=MAX_STATS, replace=False).tolist()

props_list = []
t0 = time.time()
for i, smi in enumerate(stat_smiles):
    p = mol_props(smi)
    if p: props_list.append(p)
    if (i+1) % 20000 == 0:
        log(f"    {i+1:,}/{len(stat_smiles):,}  ({time.time()-t0:.1f}s)")

props_df = pd.DataFrame(props_list)
log(f"  Computed props for {len(props_df):,} molecules in {time.time()-t0:.1f}s")

source_props = {}
for src in ("zinc", "chebi", "molinst"):
    src_smiles = combined_df[combined_df["source"] == src]["smiles"].tolist()
    sample_n = min(10_000, len(src_smiles))
    src_sample = rng.choice(src_smiles, size=sample_n, replace=False).tolist() if len(src_smiles) > sample_n else src_smiles
    sp = [mol_props(s) for s in src_sample]
    sp = [p for p in sp if p]
    source_props[src] = pd.DataFrame(sp) if sp else pd.DataFrame()
    if not source_props[src].empty:
        log(f"  {src}: n={len(sp):,}  mean_heavy={source_props[src]['heavy_atoms'].mean():.1f}  "
            f"mean_rings={source_props[src]['ring_count'].mean():.2f}")

scaffold_counts = props_df["scaffold"].value_counts()
n_unique_scaffolds = scaffold_counts.nunique()
top10_scaffolds = scaffold_counts.head(10)
ring_hist = props_df["ring_count"].clip(upper=6).value_counts().sort_index()
ring_hist.index = [str(i) if i < 6 else "6+" for i in ring_hist.index]

# ── Rewrite README ─────────────────────────────────────────────────────────────
banner("Task 8: Update README.md")

def pct_str(n, total): return f"{100*n/max(total,1):.1f}%"
def perc_str(series):
    p = series.quantile([0.25, 0.50, 0.75, 0.95])
    return f"p25={p[0.25]:.1f}  p50={p[0.50]:.1f}  p75={p[0.75]:.1f}  p95={p[0.95]:.1f}"
def fmt(v, d=4): return "n/a" if (v != v) else f"{v:.{d}f}"
def fmt_delta(v, d=4):
    if v != v: return "n/a"
    return f"+{v:.{d}f}" if v >= 0 else f"{v:.{d}f}"

lines = []
A = lines.append
A("# Combined Pretraining Dataset v1")
A("")
A("## Overview")
A("")
A(f"This folder contains a cleaned, canonicalized, deduplicated pretraining corpus "
  f"assembled from three sources: ZINC250K (drug-like small molecules), ChEBI-20 "
  f"(biochemically annotated molecules), and Mol-Instructions (instruction-tuning molecules). "
  f"All SMILES were canonicalized with RDKit and deduplicated globally. ChEBI-20 test-set "
  f"molecules were identified and removed from all sources to prevent evaluation contamination. "
  f"Mol-Instructions uses SELFIES format internally; molecules were converted to SMILES via "
  f"`selfies.decoder()` + RDKit canonicalization. "
  f"Total unique molecules: **{len(seen):,}**.")
A("")

A("## Sources")
A("")
zinc_raw = len(pd.read_csv(OUT_DIR / "zinc250k_smiles.csv"))
chebi_raw = len(pd.read_csv(OUT_DIR / "chebi20_train_smiles.csv"))
A("| Source | Raw SMILES | After canonicalize/dedup | In combined |")
A("|--------|-----------|--------------------------|-------------|")
A(f"| ZINC250K | {zinc_raw:,} | {len(zinc_can):,} | {source_counts['zinc']:,} |")
A(f"| ChEBI-20 train | {chebi_raw:,} | {len(chebi_can):,} | {source_counts['chebi']:,} |")
A(f"| Mol-Instructions | {len(molinst_smiles_list):,} (from SELFIES) | {len(unique_can):,} | {source_counts.get('molinst',0):,} |")
A(f"| **Combined** | — | — | **{len(seen):,}** |")
A("")
A("Mol-Instructions tasks and extraction:")
A("")
A("| Task | Rows | Extracted | New Unique |")
A("|------|------|-----------|------------|")
for task, st in task_stats.items():
    A(f"| {task} | {st['rows']:,} | {st['extracted']:,} | {st['new_unique']:,} |")
A("")

A("## Test Set Contamination Check")
A("")
A(f"ChEBI-20 test set: {len(test_can):,} canonical unique molecules (contamination reference).")
A("")
zinc_set = set(zinc_can)
chebi_set = set(chebi_can)
contam_data = {
    "ZINC250K": (zinc_set, zinc_set & test_can),
    "ChEBI-20 train": (chebi_set, chebi_set & test_can),
    "Mol-Instructions": (molinst_set, overlap),
}
A("| Source | Source size | Test overlap | % of source | % of test covered | Verdict |")
A("|--------|-------------|-------------|-------------|-------------------|---------|")
for label, (src_set, ovlp) in contam_data.items():
    p_src  = 100 * len(ovlp) / max(len(src_set), 1)
    p_test = 100 * len(ovlp) / max(len(test_can), 1)
    if label == "Mol-Instructions":
        verdict_short = mi_verdict.split("(")[0].strip()
    elif label == "ChEBI-20 train":
        verdict_short = "Expected ~0 (same split)"
    else:
        verdict_short = "Safe" if p_src < 1 else "Flagged"
    A(f"| {label} | {len(src_set):,} | {len(ovlp):,} | {p_src:.2f}% | {p_test:.2f}% | {verdict_short} |")
A("")
A(f"**Mol-Instructions verdict: {mi_verdict}**")
A(f"After removing test overlaps, `molinst_canonical_clean.csv` has {len(molinst_clean):,} molecules.")
A("")

A("## Combined Dataset Statistics")
A("")
total_mols = len(props_df)
A(f"Computed on {total_mols:,} molecules {'(full corpus)' if len(all_smiles) <= MAX_STATS else f'(sample of {MAX_STATS:,})'}.")
A("")
ha = props_df["heavy_atoms"]
mw = props_df["mol_weight"]
A("### Heavy Atom Count")
A(f"- Mean: {ha.mean():.1f}  Median: {ha.median():.0f}  Min: {ha.min()}  Max: {ha.max()}")
A(f"- {perc_str(ha)}")
A("")
A("### Molecular Weight")
A(f"- Mean: {mw.mean():.1f}  Median: {mw.median():.1f}")
A(f"- {perc_str(mw)}")
A("")
A("### Ring Count Distribution")
A("| Rings | Count | % |")
A("|-------|-------|---|")
for r_idx, cnt in ring_hist.items():
    A(f"| {r_idx} | {cnt:,} | {pct_str(cnt, total_mols)} |")
A("")
A("### Branch Count Distribution")
br_hist = props_df["branch_count"].clip(upper=10).value_counts().sort_index()
A("| Branches | Count | % |")
A("|----------|-------|---|")
for b, cnt in br_hist.items():
    A(f"| {'10+' if b>=10 else int(b)} | {cnt:,} | {pct_str(cnt, total_mols)} |")
A("")
ar = props_df["aromatic_rings"]
A("### Aromatic Rings")
A(f"- Mean: {ar.mean():.2f}  Median: {ar.median():.0f}")
A("")
sc = props_df["stereocenters"]
A("### Stereocenters")
A(f"- Mean: {sc.mean():.2f}  p75: {sc.quantile(0.75):.0f}  p95: {sc.quantile(0.95):.0f}")
A("")
A("### Murcko Scaffold Diversity")
A(f"- Unique scaffolds (in sample): {n_unique_scaffolds:,} / {total_mols:,} ({pct_str(n_unique_scaffolds, total_mols)})")
A("")
A("Top 10 most common scaffolds:")
A("| Rank | Scaffold SMILES | Count |")
A("|------|----------------|-------|")
for rank, (sca, cnt) in enumerate(top10_scaffolds.items(), 1):
    A(f"| {rank} | `{(sca or '(no scaffold / acyclic)')[:80]}` | {cnt:,} |")
A("")

A("## Source Comparison")
A("")
A("### Heavy Atom Count by Source")
A("| Source | n | Mean | Median | p25 | p75 | p95 |")
A("|--------|---|------|--------|-----|-----|-----|")
for src, sp_df in source_props.items():
    if sp_df.empty: continue
    h = sp_df["heavy_atoms"]
    p = h.quantile([0.25, 0.50, 0.75, 0.95])
    A(f"| {src} | {len(sp_df):,} | {h.mean():.1f} | {p[0.50]:.0f} | {p[0.25]:.0f} | {p[0.75]:.0f} | {p[0.95]:.0f} |")
A("")
A("### Ring Count by Source")
A("| Source | 0 rings | 1 | 2 | 3 | 4 | 5 | 6+ |")
A("|--------|---------|---|---|---|---|---|-----|")
for src, sp_df in source_props.items():
    if sp_df.empty: continue
    n = len(sp_df)
    rc = sp_df["ring_count"].clip(upper=6)
    vals = [pct_str((rc == r).sum(), n) for r in range(6)]
    vals.append(pct_str((rc == 6).sum(), n))
    A(f"| {src} | " + " | ".join(vals) + " |")
A("")
A("### Molecular Weight by Source")
A("| Source | Mean | Median | p25 | p75 | p95 |")
A("|--------|------|--------|-----|-----|-----|")
for src, sp_df in source_props.items():
    if sp_df.empty: continue
    mw2 = sp_df["mol_weight"]
    p = mw2.quantile([0.25, 0.50, 0.75, 0.95])
    A(f"| {src} | {mw2.mean():.1f} | {p[0.50]:.1f} | {p[0.25]:.1f} | {p[0.75]:.1f} | {p[0.95]:.1f} |")
A("")

A("## Files in This Folder")
A("")
files_desc = [
    ("zinc250k_smiles.csv",           "ZINC250K SMILES (raw valid, pre-canonicalization)"),
    ("chebi20_train_smiles.csv",       "ChEBI-20 train SELFIES→SMILES conversion"),
    ("chebi20_test_smiles.csv",        "ChEBI-20 test SELFIES→SMILES (contamination reference only)"),
    ("molinst_smiles.csv",             "Mol-Instructions: all SELFIES decoded to SMILES, before dedup"),
    ("zinc250k_canonical.csv",         "ZINC250K: canonical, deduplicated"),
    ("chebi20_train_canonical.csv",    "ChEBI-20 train: canonical, deduplicated"),
    ("molinst_canonical.csv",          "Mol-Instructions: canonical, deduplicated (with test overlap)"),
    ("molinst_canonical_clean.csv",    "Mol-Instructions: canonical, deduplicated, test overlap removed"),
    ("combined_pretrain_smiles.csv",   "Final combined corpus (smiles, source columns)"),
    ("raw/zinc250k_raw.csv",           "ZINC250K raw download (has logP/qed/SAS columns)"),
    ("raw/Molecule-oriented_Instructions.zip", "Mol-Instructions raw zip (73MB)"),
    ("raw/molinst_extracted/",         "Mol-Instructions extracted JSON files (~812MB)"),
]
A("| File | Description |")
A("|------|-------------|")
for fname, desc in files_desc:
    A(f"| `{fname}` | {desc} |")
A("")

A("## Notes")
A("")
A(f"- **ZINC250K**: Downloaded from aspuru-guzik-group GitHub (22.6 MB CSV). "
  f"All {len(zinc_can):,} SMILES were valid.")
A(f"- **ChEBI-20**: Response column contains SELFIES. Converted to SMILES via "
  f"`selfies.decoder()` + RDKit. All {chebi_raw:,} rows converted successfully.")
A(f"- **Mol-Instructions**: Uses SELFIES format throughout (not SMILES). "
  f"Extracted SELFIES from designated fields per-task (see table above), "
  f"split on `>>` and inter-fragment `.` separators, converted via `selfies.decoder()`. "
  f"Note: description_guided and molecular_description_generation use the same molecule set "
  f"(input/output swapped), so deduplication eliminates duplicates.")
A(f"- **Test contamination**: Mol-Instructions overlap = {len(overlap):,} molecules "
  f"({pct_src:.3f}% of source). Verdict: {mi_verdict}.")
A(f"- **Priority in combined**: zinc > chebi > molinst for duplicate molecules.")
A(f"- **Not done**: tokenization, length filtering, model training.")
A(f"- **Statistics**: Computed on sample of {len(props_df):,} molecules.")
A("")

README.write_text("\n".join(lines))
log(f"  Saved: {README}")

# ── Final summary ──────────────────────────────────────────────────────────────
banner("COMPLETE")
log(f"")
log(f"  README: {README}")
log(f"  Combined: {COMBINED}")
log(f"  Total unique molecules: {len(seen):,}")
log(f"")
log(f"  Contamination verdicts:")
for label, (src_set, ovlp) in contam_data.items():
    p_src = 100 * len(ovlp) / max(len(src_set), 1)
    p_test = 100 * len(ovlp) / max(len(test_can), 1)
    log(f"    {label}: overlap={len(ovlp):,}  ({p_src:.3f}% source, {p_test:.2f}% test)")
log(f"  Mol-Instructions: {mi_verdict}")
log(f"")
log(f"  Source breakdown:")
for src, cnt in sorted(source_counts.items()):
    log(f"    {src}: {cnt:,}  ({100*cnt/len(seen):.1f}%)")
log(f"  Mol-Instructions task stats:")
for task, st in task_stats.items():
    log(f"    {task}: {st['rows']:,} rows → {st['new_unique']:,} new unique")
