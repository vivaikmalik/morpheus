#!/usr/bin/env python3
"""
Data acquisition and analysis pipeline for combined_pretrain_v1.
Produces data/combined_pretrain_v1/ with processed CSVs and README.md.
"""
import os, sys, re, json, time, shutil, zipfile
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO       = SCRIPT_DIR.parent
DATA_DIR   = REPO / "data"
OUT_DIR    = REPO / "data" / "combined_pretrain_v1"
RAW_DIR    = OUT_DIR / "raw"

OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

def log(msg): print(msg, flush=True)
def banner(title):
    log(f"\n{'='*60}")
    log(f"  {title}")
    log('='*60)

# ── RDKit helpers ─────────────────────────────────────────────────────────────
def canonicalize(smi):
    """Return canonical SMILES or None if invalid."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None

def is_valid(smi, min_heavy=1):
    """Check SMILES is valid with >= min_heavy atoms."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return False
        return mol.GetNumHeavyAtoms() >= min_heavy
    except Exception:
        return False

# ── TASK 1: ZINC250K ──────────────────────────────────────────────────────────
banner("Task 1: ZINC250K")

ZINC_RAW   = RAW_DIR / "zinc250k_raw.csv"
ZINC_SMILES = OUT_DIR / "zinc250k_smiles.csv"

# Check for existing ZINC file in project
zinc_source = None
for candidate in [DATA_DIR / "zinc250k.csv", DATA_DIR / "zinc_250k.csv",
                  DATA_DIR / "zinc250k.csv.gz", DATA_DIR / "zinc_250k.csv.gz"]:
    if candidate.exists():
        zinc_source = candidate
        log(f"Found existing ZINC file: {zinc_source}")
        break

if zinc_source is None:
    log("ZINC250K not found locally. Downloading...")
    import urllib.request
    # ZINC250K from aspuru-guzik-group GitHub mirror (reliable)
    url = "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
    gz_path = RAW_DIR / "zinc250k_raw_dl.csv"
    
    log(f"  URL: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk = 65536
        with open(gz_path, "wb") as out_f:
            while True:
                data = resp.read(chunk)
                if not data: break
                out_f.write(data)
                downloaded += len(data)
                if total > 0:
                    pct = downloaded * 100 / total
                    print(f"\r  {downloaded/1e6:.1f}/{total/1e6:.1f} MB ({pct:.0f}%)", end="", flush=True)
            print()
    log(f"\n  Downloaded {gz_path.stat().st_size/1e6:.1f} MB")
    zinc_source = gz_path

# Load ZINC
log("Loading ZINC250K...")
zinc_df = pd.read_csv(zinc_source)

log(f"  Raw rows: {len(zinc_df):,}")
log(f"  Columns: {list(zinc_df.columns)}")

# Find SMILES column
smi_col = None
for c in zinc_df.columns:
    if c.lower() in ('smiles', 'smi', 'canonical_smiles'):
        smi_col = c; break
if smi_col is None:
    # Use first column that looks like SMILES
    for c in zinc_df.columns:
        sample = str(zinc_df[c].iloc[0])
        if any(ch in sample for ch in ['C','c','N','O','[',']']):
            smi_col = c; break
log(f"  Using SMILES column: '{smi_col}'")

zinc_raw_smiles = zinc_df[smi_col].dropna().astype(str).tolist()
zinc_valid = [s for s in zinc_raw_smiles if is_valid(s)]
log(f"  Valid SMILES: {len(zinc_valid):,} / {len(zinc_raw_smiles):,}  "
    f"(dropped {len(zinc_raw_smiles)-len(zinc_valid):,})")

# Save raw copy
zinc_df.to_csv(ZINC_RAW, index=False)

# Save smiles-only file
pd.DataFrame({"smiles": zinc_valid}).to_csv(ZINC_SMILES, index=False)
log(f"  Saved: {ZINC_SMILES.name}  ({len(zinc_valid):,} rows)")

zinc_stats = {"raw": len(zinc_raw_smiles), "valid": len(zinc_valid),
              "dropped": len(zinc_raw_smiles) - len(zinc_valid)}

# ── TASK 2: ChEBI-20 ──────────────────────────────────────────────────────────
banner("Task 2: ChEBI-20 SELFIES → SMILES")

CHEBI_TRAIN_SMILES = OUT_DIR / "chebi20_train_smiles.csv"
CHEBI_TEST_SMILES  = OUT_DIR / "chebi20_test_smiles.csv"

def selfies_to_smiles(selfies_str):
    """Convert SELFIES to canonical SMILES, return None on failure."""
    try:
        smi = sf.decoder(selfies_str)
        if smi is None or smi == '': return None
        return canonicalize(smi)
    except Exception:
        return None

def process_chebi(csv_path, label):
    df = pd.read_csv(csv_path)
    log(f"  {label}: {len(df):,} rows, columns={list(df.columns)}")
    selfies_list = df["response"].dropna().astype(str).tolist()
    smiles_list = []
    failed = 0
    for s in selfies_list:
        smi = selfies_to_smiles(s)
        if smi is None:
            failed += 1
        else:
            smiles_list.append(smi)
    log(f"  {label}: {len(smiles_list):,} valid SMILES, {failed:,} failed conversions")
    return smiles_list

train_smiles = process_chebi(DATA_DIR / "chebi20_train.csv", "train")
test_smiles  = process_chebi(DATA_DIR / "chebi20_test.csv",  "test")

pd.DataFrame({"smiles": train_smiles}).to_csv(CHEBI_TRAIN_SMILES, index=False)
pd.DataFrame({"smiles": test_smiles}).to_csv(CHEBI_TEST_SMILES, index=False)
log(f"  Saved: {CHEBI_TRAIN_SMILES.name}  ({len(train_smiles):,} rows)")
log(f"  Saved: {CHEBI_TEST_SMILES.name}  ({len(test_smiles):,} rows)")

chebi_stats = {
    "train_raw": len(pd.read_csv(DATA_DIR / "chebi20_train.csv")),
    "train_valid": len(train_smiles),
    "test_raw": len(pd.read_csv(DATA_DIR / "chebi20_test.csv")),
    "test_valid": len(test_smiles),
}

# ── TASK 3: Mol-Instructions ──────────────────────────────────────────────────
banner("Task 3: Mol-Instructions")

MOLINST_SMILES = OUT_DIR / "molinst_smiles.csv"
MOLINST_ZIP    = RAW_DIR / "Molecule-oriented_Instructions.zip"
MOLINST_EXTRACT = RAW_DIR / "molinst_extracted"

# Download zip if not already present
if not MOLINST_ZIP.exists():
    log("Downloading Mol-Instructions (molecule zip, ~73MB)...")
    from huggingface_hub import hf_hub_download
    hf_path = hf_hub_download(
        repo_id="zjunlp/Mol-Instructions",
        filename="data/Molecule-oriented_Instructions.zip",
        repo_type="dataset",
        local_dir=str(RAW_DIR),
        local_dir_use_symlinks=False,
    )
    # The file may be saved in a subfolder; find it
    if not MOLINST_ZIP.exists():
        for r, dirs, files in os.walk(RAW_DIR):
            for f in files:
                if f == "Molecule-oriented_Instructions.zip":
                    shutil.move(os.path.join(r, f), str(MOLINST_ZIP))
                    break
    log(f"  Downloaded: {MOLINST_ZIP.stat().st_size/1e6:.1f} MB")
else:
    log(f"  Already present: {MOLINST_ZIP.name}  ({MOLINST_ZIP.stat().st_size/1e6:.1f} MB)")

# Extract
log("Extracting zip...")
MOLINST_EXTRACT.mkdir(exist_ok=True)
with zipfile.ZipFile(MOLINST_ZIP, 'r') as zf:
    names = zf.namelist()
    log(f"  {len(names)} files in zip")
    zf.extractall(MOLINST_EXTRACT)

# Find all JSON files
json_files = list(MOLINST_EXTRACT.rglob("*.json"))
log(f"  Found {len(json_files)} JSON files")
for jf in sorted(json_files):
    size = jf.stat().st_size
    log(f"    {jf.relative_to(MOLINST_EXTRACT)}  ({size/1e6:.1f} MB)")

# SMILES regex — broad pattern
SMILES_RE = re.compile(r'[A-Za-z0-9@+\-\[\]\(\)=#\$\/\\%\.]{6,}')
MIN_HEAVY = 3

def extract_smiles_from_text(text):
    """Extract SMILES candidates from a text field and validate with RDKit."""
    candidates = SMILES_RE.findall(str(text))
    valid = []
    for c in candidates:
        mol = Chem.MolFromSmiles(c)
        if mol is not None and mol.GetNumHeavyAtoms() >= MIN_HEAVY:
            valid.append(Chem.MolToSmiles(mol))
    return valid

molinst_smiles_set = set()
task_counts = Counter()
total_candidates = 0

for jf in sorted(json_files):
    task_name = jf.stem
    log(f"  Processing: {task_name}")
    with open(jf, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    count_before = len(molinst_smiles_set)
    local_cands = 0
    for row in data:
        for field in ("instruction", "input", "output"):
            val = row.get(field, "")
            if val:
                found = extract_smiles_from_text(val)
                local_cands += len(found)
                molinst_smiles_set.update(found)
    
    total_candidates += local_cands
    new_added = len(molinst_smiles_set) - count_before
    task_counts[task_name] = len(data)
    log(f"    rows={len(data):,}  candidates={local_cands:,}  new_unique={new_added:,}  total_unique={len(molinst_smiles_set):,}")

log(f"\n  Total tasks: {len(task_counts)}")
log(f"  Total candidate strings: {total_candidates:,}")
log(f"  Unique valid SMILES (>=3 heavy atoms): {len(molinst_smiles_set):,}")

molinst_smiles_list = sorted(molinst_smiles_set)
pd.DataFrame({"smiles": molinst_smiles_list}).to_csv(MOLINST_SMILES, index=False)
log(f"  Saved: {MOLINST_SMILES.name}  ({len(molinst_smiles_list):,} rows)")

molinst_stats = {
    "n_tasks": len(task_counts),
    "total_candidates": total_candidates,
    "unique_valid": len(molinst_smiles_list),
    "task_counts": dict(task_counts),
}

# ── TASK 4: Canonicalize ──────────────────────────────────────────────────────
banner("Task 4: Canonicalize and deduplicate each source")

def canonicalize_df(smiles_csv, out_csv, label):
    df = pd.read_csv(smiles_csv)
    raw_smi = df["smiles"].dropna().astype(str).tolist()
    canonical = []
    for s in raw_smi:
        c = canonicalize(s)
        if c: canonical.append(c)
    unique = sorted(set(canonical))
    pd.DataFrame({"smiles": unique}).to_csv(out_csv, index=False)
    log(f"  {label}: {len(raw_smi):,} raw → {len(canonical):,} valid → {len(unique):,} unique")
    return unique

ZINC_CAN     = OUT_DIR / "zinc250k_canonical.csv"
CHEBI_CAN    = OUT_DIR / "chebi20_train_canonical.csv"
MOLINST_CAN  = OUT_DIR / "molinst_canonical.csv"

zinc_can   = canonicalize_df(ZINC_SMILES,         ZINC_CAN,    "ZINC250K")
chebi_can  = canonicalize_df(CHEBI_TRAIN_SMILES,  CHEBI_CAN,   "ChEBI-20 train")
molinst_can= canonicalize_df(MOLINST_SMILES,      MOLINST_CAN, "Mol-Instructions")

# ── TASK 5: Test set contamination ────────────────────────────────────────────
banner("Task 5: Test set contamination check")

test_df = pd.read_csv(CHEBI_TEST_SMILES)
test_can = set()
for s in test_df["smiles"].dropna().astype(str):
    c = canonicalize(s)
    if c: test_can.add(c)
log(f"  ChEBI-20 test set: {len(test_df):,} raw → {len(test_can):,} canonical unique")

zinc_set    = set(zinc_can)
chebi_set   = set(chebi_can)
molinst_set = set(molinst_can)

contam_results = {}
for label, src_set in [("ZINC250K", zinc_set), ("ChEBI-20 train", chebi_set), ("Mol-Instructions", molinst_set)]:
    overlap = src_set & test_can
    pct_src  = 100 * len(overlap) / max(len(src_set), 1)
    pct_test = 100 * len(overlap) / max(len(test_can), 1)
    contam_results[label] = {
        "source_size": len(src_set),
        "overlap": len(overlap),
        "pct_source": pct_src,
        "pct_test": pct_test,
        "overlap_set": overlap,
    }
    log(f"  {label}:  size={len(src_set):,}  overlap={len(overlap):,}  "
        f"(src {pct_src:.2f}%  test {pct_test:.2f}%)")

# Verdict for Mol-Instructions
mi_overlap_pct = contam_results["Mol-Instructions"]["pct_source"]
if mi_overlap_pct < 1.0:
    mi_verdict = "SAFE (< 1% overlap, include as-is)"
elif mi_overlap_pct < 5.0:
    mi_verdict = "FLAGGED (1–5% overlap, include but note)"
elif mi_overlap_pct < 15.0:
    mi_verdict = "FILTERED (5–15% overlap, removing overlapping molecules)"
else:
    mi_verdict = "EXCLUDED (> 15% overlap, contamination concern)"
log(f"\n  Mol-Instructions verdict: {mi_verdict}")

# Produce molinst_canonical_clean.csv (overlap removed)
MOLINST_CLEAN = OUT_DIR / "molinst_canonical_clean.csv"
molinst_clean = sorted(molinst_set - test_can)
pd.DataFrame({"smiles": molinst_clean}).to_csv(MOLINST_CLEAN, index=False)
log(f"  molinst_canonical_clean.csv: {len(molinst_clean):,} molecules "
    f"(removed {len(molinst_set)-len(molinst_clean):,} overlapping)")

# ── TASK 6: Combine ───────────────────────────────────────────────────────────
banner("Task 6: Combine and deduplicate")

COMBINED = OUT_DIR / "combined_pretrain_smiles.csv"

seen = {}
multi_source = []

for label, smi_list in [("zinc", zinc_can), ("chebi", chebi_can), ("molinst", molinst_clean)]:
    new_count = 0
    for s in smi_list:
        if s not in seen:
            seen[s] = label
            new_count += 1
        else:
            if s not in [m[0] for m in multi_source]:  # track first duplicate
                multi_source.append((s, seen[s], label))
    log(f"  Added {new_count:,} new from {label}  (total so far: {len(seen):,})")

log(f"\n  Total unique molecules: {len(seen):,}")

source_counts = Counter(seen.values())
for src, cnt in sorted(source_counts.items()):
    log(f"  {src}: {cnt:,}")
log(f"  Molecules in multiple sources: {len(multi_source):,}")

rows = [{"smiles": s, "source": src} for s, src in seen.items()]
combined_df = pd.DataFrame(rows)
combined_df.to_csv(COMBINED, index=False)
log(f"\n  Saved: {COMBINED.name}  ({len(combined_df):,} rows)")

# ── TASK 7: Descriptive statistics ────────────────────────────────────────────
banner("Task 7: Descriptive statistics")

log("  Computing molecular properties (this may take 1-2 minutes)...")

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
            "heavy_atoms":  mol.GetNumHeavyAtoms(),
            "mol_weight":   Descriptors.MolWt(mol),
            "ring_count":   ring_info.NumRings(),
            "branch_count": smi.count("("),
            "aromatic_rings": aromatic_rings,
            "stereocenters": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
            "scaffold": scaffold_smi,
        }
    except Exception:
        return None

# Sample at most 100K for speed (or all if smaller)
MAX_STATS = 100_000
rng_stat = np.random.default_rng(42)
all_smiles = combined_df["smiles"].tolist()
stat_smiles = all_smiles if len(all_smiles) <= MAX_STATS else \
              rng_stat.choice(all_smiles, size=MAX_STATS, replace=False).tolist()

props_list = []
t0 = time.time()
for i, smi in enumerate(stat_smiles):
    p = mol_props(smi)
    if p: props_list.append(p)
    if (i+1) % 10000 == 0:
        log(f"    {i+1:,}/{len(stat_smiles):,}  ({time.time()-t0:.1f}s)")

props_df = pd.DataFrame(props_list)
log(f"  Computed props for {len(props_df):,} molecules in {time.time()-t0:.1f}s")

# Compute per-source stats for comparison
log("  Computing per-source stats...")
source_props = {}
for src in ("zinc", "chebi", "molinst"):
    src_smiles = combined_df[combined_df["source"] == src]["smiles"].tolist()
    src_sample = src_smiles if len(src_smiles) <= 10_000 else \
                 rng_stat.choice(src_smiles, size=10_000, replace=False).tolist()
    sp = [mol_props(s) for s in src_sample]
    sp = [p for p in sp if p]
    source_props[src] = pd.DataFrame(sp) if sp else pd.DataFrame()
    if not source_props[src].empty:
        log(f"    {src}: n={len(sp):,}  mean_heavy={source_props[src]['heavy_atoms'].mean():.1f}  "
            f"mean_rings={source_props[src]['ring_count'].mean():.2f}")

# Ring distribution
ring_hist = props_df["ring_count"].clip(upper=6).value_counts().sort_index()
ring_hist.index = [str(i) if i < 6 else "6+" for i in ring_hist.index]

# Scaffold analysis
scaffold_counts = props_df["scaffold"].value_counts()
n_unique_scaffolds = scaffold_counts.nunique()
top10_scaffolds = scaffold_counts.head(10)

log(f"  Unique Murcko scaffolds: {n_unique_scaffolds:,}")
log(f"  Top scaffold: '{top10_scaffolds.index[0]}' ({top10_scaffolds.iloc[0]:,} molecules)")

# ── TASK 8: Markdown report ───────────────────────────────────────────────────
banner("Task 8: Writing README.md")

def pct_str(n, total):
    return f"{100*n/max(total,1):.1f}%"

def perc_str(series):
    p = series.quantile([0.25, 0.50, 0.75, 0.95])
    return f"p25={p[0.25]:.1f}  p50={p[0.50]:.1f}  p75={p[0.75]:.1f}  p95={p[0.95]:.1f}"

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
  f"No tokenization, length filtering, or training was performed. "
  f"Total unique molecules: **{len(seen):,}**.")
A("")

A("## Sources")
A("")
A("| Source | Raw rows | Valid SMILES | After canonicalize/dedup | In combined |")
A("|--------|----------|-------------|--------------------------|-------------|")
A(f"| ZINC250K | {zinc_stats['raw']:,} | {zinc_stats['valid']:,} | "
  f"{len(zinc_can):,} | {source_counts['zinc']:,} |")
A(f"| ChEBI-20 train | {chebi_stats['train_raw']:,} | {chebi_stats['train_valid']:,} | "
  f"{len(chebi_can):,} | {source_counts['chebi']:,} |")
A(f"| Mol-Instructions | {molinst_stats['unique_valid']:,} (after SMILES extraction) | — | "
  f"{len(molinst_can):,} | {source_counts.get('molinst',0):,} |")
A(f"| **Combined** | — | — | — | **{len(seen):,}** |")
A("")
A(f"Mol-Instructions tasks processed: {molinst_stats['n_tasks']}")
A("")
A("| Task | Rows |")
A("|------|------|")
for task, cnt in sorted(molinst_stats['task_counts'].items()):
    A(f"| {task} | {cnt:,} |")
A("")

A("## Test Set Contamination Check")
A("")
A(f"ChEBI-20 test set: {len(test_can):,} canonical unique molecules used as contamination reference.")
A("")
A("| Source | Source size | Test overlap | % of source | % of test covered | Verdict |")
A("|--------|-------------|-------------|-------------|-------------------|---------|")
for label, res in contam_results.items():
    if label == "Mol-Instructions":
        verdict = mi_verdict.split("(")[0].strip()
    elif label == "ChEBI-20 train":
        verdict = "Expected ~0 (same split)"
    else:
        v = res['pct_source']
        verdict = "Safe" if v < 1 else "Flagged" if v < 5 else "Filtered"
    A(f"| {label} | {res['source_size']:,} | {res['overlap']:,} | "
      f"{res['pct_source']:.2f}% | {res['pct_test']:.2f}% | {verdict} |")
A("")
A(f"**Mol-Instructions verdict: {mi_verdict}**")
A(f"After removing test-set overlaps, `molinst_canonical_clean.csv` has {len(molinst_clean):,} molecules.")
A("")

A("## Combined Dataset Statistics")
A("")
A(f"Statistics below are computed on a sample of {len(props_df):,} molecules "
  f"{'(sampled from full corpus)' if len(all_smiles) > MAX_STATS else '(full corpus)'}.")
A("")
A("### Heavy Atom Count")
A("")
ha = props_df["heavy_atoms"]
A(f"- Mean: {ha.mean():.1f}  Median: {ha.median():.0f}  Min: {ha.min()}  Max: {ha.max()}")
A(f"- {perc_str(ha)}")
A("")
A("### Molecular Weight")
mw = props_df["mol_weight"]
A(f"- Mean: {mw.mean():.1f}  Median: {mw.median():.1f}")
A(f"- {perc_str(mw)}")
A("")
A("### Ring Count Distribution")
A("")
A("| Rings | Count | % |")
A("|-------|-------|---|")
total_mols = len(props_df)
for r_idx, cnt in ring_hist.items():
    A(f"| {r_idx} | {cnt:,} | {pct_str(cnt, total_mols)} |")
A("")
A("### Branch Count Distribution (open parens in SMILES)")
br_hist = props_df["branch_count"].clip(upper=10).value_counts().sort_index()
A("")
A("| Branches | Count | % |")
A("|----------|-------|---|")
for b, cnt in br_hist.items():
    label_b = str(int(b)) if b < 10 else "10+"
    A(f"| {label_b} | {cnt:,} | {pct_str(cnt, total_mols)} |")
A("")
A("### Aromatic Rings")
ar = props_df["aromatic_rings"]
A(f"- Mean: {ar.mean():.2f}  Median: {ar.median():.0f}")
ar_hist = ar.clip(upper=5).value_counts().sort_index()
ar_hist_str = "  ".join(f"{int(k)}={'10+' if k>=5 else cnt}" for k,cnt in ar_hist.items())
A(f"- Distribution: {ar_hist_str}")
A("")
A("### Stereocenters")
sc = props_df["stereocenters"]
A(f"- Mean: {sc.mean():.2f}  Median: {sc.median():.0f}  p75: {sc.quantile(0.75):.0f}  p95: {sc.quantile(0.95):.0f}")
A("")
A("### Murcko Scaffold Diversity")
A("")
A(f"- Unique scaffolds (in sample): {n_unique_scaffolds:,} / {len(props_df):,} molecules")
A(f"- Scaffold coverage: {pct_str(n_unique_scaffolds, len(props_df))}")
A("")
A("Top 10 most common scaffolds:")
A("")
A("| Rank | Scaffold SMILES | Count |")
A("|------|----------------|-------|")
for rank, (sca, cnt) in enumerate(top10_scaffolds.items(), 1):
    sca_display = sca if sca else "(no scaffold / acyclic)"
    A(f"| {rank} | `{sca_display[:80]}` | {cnt:,} |")
A("")

A("## Source Comparison")
A("")
A("### Heavy Atom Count by Source")
A("")
A("| Source | n | Mean | Median | p25 | p75 | p95 |")
A("|--------|---|------|--------|-----|-----|-----|")
for src, sp_df in source_props.items():
    if sp_df.empty: continue
    h = sp_df["heavy_atoms"]
    p = h.quantile([0.25, 0.50, 0.75, 0.95])
    A(f"| {src} | {len(sp_df):,} | {h.mean():.1f} | {p[0.50]:.0f} "
      f"| {p[0.25]:.0f} | {p[0.75]:.0f} | {p[0.95]:.0f} |")
A("")
A("### Ring Count by Source")
A("")
ring_labels = list(range(7))
# Build header
A("| Source | 0 rings | 1 | 2 | 3 | 4 | 5 | 6+ |")
A("|--------|---------|---|---|---|---|---|-----|")
for src, sp_df in source_props.items():
    if sp_df.empty: continue
    n = len(sp_df)
    rc = sp_df["ring_count"].clip(upper=6)
    row_vals = []
    for r in range(6):
        cnt = (rc == r).sum()
        row_vals.append(f"{pct_str(cnt, n)}")
    cnt6 = (rc == 6).sum()
    row_vals.append(f"{pct_str(cnt6, n)}")
    A(f"| {src} | " + " | ".join(row_vals) + " |")
A("")
A("### Molecular Weight by Source")
A("")
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
    ("zinc250k_smiles.csv",          "ZINC250K SMILES, raw valid (pre-canonicalization)"),
    ("chebi20_train_smiles.csv",      "ChEBI-20 train SELFIES→SMILES conversion"),
    ("chebi20_test_smiles.csv",       "ChEBI-20 test SELFIES→SMILES (contamination reference only)"),
    ("molinst_smiles.csv",            "Mol-Instructions SMILES extracted from all tasks"),
    ("zinc250k_canonical.csv",        "ZINC250K: canonical, deduplicated"),
    ("chebi20_train_canonical.csv",   "ChEBI-20 train: canonical, deduplicated"),
    ("molinst_canonical.csv",         "Mol-Instructions: canonical, deduplicated (with test overlap)"),
    ("molinst_canonical_clean.csv",   "Mol-Instructions: canonical, deduplicated, test overlap removed"),
    ("combined_pretrain_smiles.csv",  "Final combined corpus (smiles, source columns)"),
    ("raw/zinc250k_raw.csv",          "ZINC250K raw download"),
    ("raw/Molecule-oriented_Instructions.zip", "Mol-Instructions raw zip"),
    ("raw/molinst_extracted/",        "Mol-Instructions extracted JSON files"),
]
A("| File | Description |")
A("|------|-------------|")
for fname, desc in files_desc:
    A(f"| `{fname}` | {desc} |")
A("")

A("## Notes")
A("")
zinc_note = "found locally" if zinc_source and 'raw' not in str(zinc_source) else "downloaded from DeepChem S3"
A(f"- **ZINC250K**: {zinc_note}.")
A(f"- **ChEBI-20**: Response column contains SELFIES strings. Converted to SMILES via "
  f"`selfies.decoder()` then RDKit canonicalization. "
  f"{chebi_stats['train_raw'] - chebi_stats['train_valid']:,} train rows failed conversion.")
A(f"- **Mol-Instructions**: SMILES extracted using a broad regex from `instruction`, `input`, and "
  f"`output` fields, validated with RDKit (≥3 heavy atoms). This approach may miss some valid SMILES "
  f"(false negatives) and occasionally extract non-SMILES strings that happen to parse (false positives), "
  f"but is conservative due to the RDKit validation gate.")
A(f"- **Test contamination**: ChEBI-20 train ↔ test overlap = "
  f"{contam_results['ChEBI-20 train']['overlap']} molecules "
  f"({contam_results['ChEBI-20 train']['pct_source']:.2f}% of train). "
  f"Mol-Instructions overlap = {contam_results['Mol-Instructions']['overlap']} molecules "
  f"({contam_results['Mol-Instructions']['pct_source']:.2f}%). Verdict: {mi_verdict}.")
A(f"- **Statistics**: Computed on a random sample of {len(props_df):,} molecules for speed.")
A(f"- **Priority in combined file**: zinc > chebi > molinst (for molecules in multiple sources).")
A(f"- **Not done**: tokenization, length filtering, model training. This is raw data only.")
A("")

README = OUT_DIR / "README.md"
README.write_text("\n".join(lines))
log(f"  Saved: {README}")

# ── Final summary ──────────────────────────────────────────────────────────────
banner("COMPLETE")
log(f"")
log(f"  Path (README):     {README}")
log(f"  Path (combined):   {COMBINED}")
log(f"  Total unique mols: {len(seen):,}")
log(f"")
log(f"  Contamination verdicts:")
for label, res in contam_results.items():
    log(f"    {label}: overlap={res['overlap']:,}  "
        f"({res['pct_source']:.2f}% of source,  {res['pct_test']:.2f}% of test)")
log(f"  Mol-Instructions: {mi_verdict}")
log(f"")
log(f"  Source breakdown in combined:")
for src, cnt in sorted(source_counts.items()):
    log(f"    {src}: {cnt:,}  ({100*cnt/len(seen):.1f}%)")

if __name__ == "__main__":
    pass
