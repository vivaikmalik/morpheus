"""
scripts/build_summary_table.py
------------------------------
Scan all *_eval_*.csv files in outputs/, parse metadata from filenames,
compute per-run averages, and write outputs/summary_all_runs.csv.

CANONICAL filename convention:
  {dataset}_{model_size}_{encoder}_eval_cfg{cfg}_t{temp}_s{steps}.csv
  e.g. chebi20_27M_scibert_contrastive_eval_cfg1.5_t0.6_s50.csv
       chebi20_27M_bge_contrastive_eval_cfg2.0_t0.8_s50.csv
       chebi20_27M_bge_frozen_eval_cfg1.5_t0.6_s50.csv

LEGACY FILENAME PATTERNS (also parsed, with inferred metadata):

  chebi20_eval_cfg{N}_t{T}_s{S}.csv
  chebi20_eval_eost.csv / chebi20_eval.csv
    — Pre-standardisation ChEBI-20 runs without model_size in filename.
      Inferred as: dataset=chebi20, model_size=27M, encoder=bge_frozen.
      Same column schema as canonical ChEBI-20 files.

  molinst_eval_cfg{N}_t{T}_s{S}.csv
    — Pre-standardisation Mol-Instructions runs, new HP format.
      Inferred as: dataset=molinst, model_size=27M, encoder=bge_frozen.
      Different schema: bleu→bleu2, norm_levenshtein→lev_sim, tanimoto→morgan_sim.

  eval_27M_cfg{N}_temp{T}_steps{S}.csv / eval_27M_frozen.csv
    — Very early Mol-Instructions runs, old HP format (temp/steps not t/s).
      Inferred as: dataset=molinst, model_size=27M, encoder=bge_frozen.
      Same different schema as molinst_eval_*.

ENCODER NAMES (normalised in output):
  frozen           → bge_frozen
  contrastive      → bge_contrastive
  scibert_contrastive stays as-is
"""

import math, re, sys
from collections import Counter
from pathlib import Path
import pandas as pd

OUTPUTS = Path("outputs")

# ── atom tokenizer ─────────────────────────────────────────────────────────
_ATOM_RE = re.compile(
    r'(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p'
    r'|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])'
)

def _atom_tok(smi):
    return _ATOM_RE.findall(smi) if isinstance(smi, str) else []

def _bleu_n(ref, hyp, n):
    if not ref or not hyp or len(hyp) < n:
        return 0.0
    ref_ng = Counter(tuple(ref[i:i+n]) for i in range(len(ref)-n+1))
    hyp_ng = Counter(tuple(hyp[i:i+n]) for i in range(len(hyp)-n+1))
    clipped = sum(min(c, ref_ng[g]) for g, c in hyp_ng.items())
    return clipped / sum(hyp_ng.values())

def _sentence_bleu(ref, hyp, max_n=4):
    if not ref or not hyp:
        return 0.0
    precs = [_bleu_n(ref, hyp, n) for n in range(1, max_n+1)]
    if all(p == 0 for p in precs):
        return 0.0
    log_avg = sum(math.log(p) if p > 0 else -1e9 for p in precs) / max_n
    bp = min(1.0, math.exp(1 - len(ref) / max(len(hyp), 1)))
    return bp * math.exp(log_avg)

# ── filename parser ─────────────────────────────────────────────────────────
# Canonical HP suffix: _eval_cfg{N}_t{T}_s{S}
_HP_NEW = re.compile(
    r'^(?P<prefix>.+)_eval_cfg(?P<cfg>[0-9.]+)_t(?P<temp>[0-9.]+)_s(?P<steps>\d+)'
    r'(?P<suffix>.*)$'
)
# Old HP suffix used in eval_27M_* files: _cfg{N}_temp{T}_steps{S}
_HP_OLD = re.compile(
    r'^(?P<prefix>.+)_cfg(?P<cfg>[0-9.]+)_temp(?P<temp>[0-9.]+)_steps(?P<steps>\d+)'
    r'(?P<suffix>.*)$'
)
# Loose match for files with _eval but no HP grid (e.g. chebi20_eval_eost, eval_27M_frozen)
_EVAL_LOOSE = re.compile(r'^(?P<prefix>.+?)_eval(?P<suffix>.*)$')

# Encoder name normalisation
_ENCODER_MAP = {
    "frozen":     "bge_frozen",
    "contrastive": "bge_contrastive",
}

def _normalise_encoder(raw: str) -> str:
    return _ENCODER_MAP.get(raw, raw) if raw else "bge_frozen"

def _split_prefix(prefix: str):
    """Return (dataset, model_size, encoder_raw) from a prefix string."""
    parts = prefix.split("_")

    # eval_27M_* style: starts with 'eval', model_size follows
    if parts[0] == "eval":
        size_idx = next((i for i, p in enumerate(parts) if re.match(r'^\d+M$', p)), None)
        dataset = "molinst"
        if size_idx is not None:
            model_size = parts[size_idx]
            enc_raw = "_".join(parts[size_idx+1:]) or "frozen"
        else:
            model_size = "27M"
            enc_raw = "frozen"
        return dataset, model_size, enc_raw

    # Standard: first token is dataset
    dataset = parts[0]
    size_idx = next((i for i, p in enumerate(parts[1:], 1) if re.match(r'^\d+M$', p)), None)
    if size_idx is not None:
        model_size = parts[size_idx]
        enc_raw = "_".join(parts[size_idx+1:]) or "frozen"
    else:
        # No model_size token — infer 27M, encoder from remainder
        model_size = "27M"
        enc_raw = "_".join(parts[1:]) or "frozen"
    return dataset, model_size, enc_raw

def parse_filename(stem: str) -> dict:
    suffix = ""
    # Try canonical HP format
    m = _HP_NEW.match(stem)
    if m:
        prefix, suffix = m.group("prefix"), m.group("suffix")
        hp = dict(cfg=float(m.group("cfg")), temp=float(m.group("temp")),
                  steps=int(m.group("steps")))
    else:
        # Try old HP format (temp/steps)
        m2 = _HP_OLD.match(stem)
        if m2:
            prefix, suffix = m2.group("prefix"), m2.group("suffix")
            hp = dict(cfg=float(m2.group("cfg")), temp=float(m2.group("temp")),
                      steps=int(m2.group("steps")))
        else:
            # No HP grid — try to extract prefix before _eval
            m3 = _EVAL_LOOSE.match(stem)
            prefix = m3.group("prefix") if m3 else stem
            suffix = m3.group("suffix") if m3 else ""
            hp = dict(cfg=float("nan"), temp=float("nan"), steps=-1)

    dataset, model_size, enc_raw = _split_prefix(prefix)
    encoder = _normalise_encoder(enc_raw)

    # Fill default HP for files that had no HP in their filename
    if hp["steps"] == -1:
        hp = dict(cfg=3.0, temp=1.0, steps=50)

    # pre_eos_fix: eval_27M_* series and chebi20_eval.csv are pre-EOS-fix checkpoints
    pre_eos_fix = (
        stem.startswith("eval_27M_") or
        stem == "chebi20_eval"
    )

    return dict(dataset=dataset, model_size=model_size, encoder=encoder,
                eos_truncate="_eost" in suffix, pre_eos_fix=pre_eos_fix, **hp)

# ── schema normalisation ────────────────────────────────────────────────────
# Map legacy molinst column names → canonical names
_COL_MAP = {
    "bleu":             "bleu2",
    "norm_levenshtein": "lev_sim",
    "tanimoto":         "morgan_sim",
}

def _normalise_df(df: pd.DataFrame) -> pd.DataFrame:
    renames = {old: new for old, new in _COL_MAP.items()
               if old in df.columns and new not in df.columns}
    return df.rename(columns=renames) if renames else df

# ── metric extraction ───────────────────────────────────────────────────────
def extract_metrics(df: pd.DataFrame) -> dict:
    df = _normalise_df(df)
    metrics = {}
    n = len(df)
    n_valid = int(df["valid"].sum()) if "valid" in df.columns else n
    metrics["n"] = n
    metrics["validity"]    = round(100.0 * n_valid / n, 2) if n else float("nan")
    metrics["exact_match"] = round(100.0 * df["exact_match"].mean(), 2) \
        if "exact_match" in df.columns else float("nan")

    for col in ["bleu2", "bleu4", "lev_sim", "morgan_sim", "maccs_sim", "rdk_sim"]:
        if col in df.columns:
            metrics[col] = round(float(df[col].mean()), 4)

    # atom BLEU: use existing cols or compute on the fly from gt_smiles/gen_smiles
    if "atom_bleu2" in df.columns and "atom_bleu4" in df.columns:
        metrics["atom_bleu2"] = round(float(df["atom_bleu2"].mean()), 4)
        metrics["atom_bleu4"] = round(float(df["atom_bleu4"].mean()), 4)
    elif "gt_smiles" in df.columns and "gen_smiles" in df.columns:
        ab2, ab4 = [], []
        for _, row in df.iterrows():
            ref = _atom_tok(str(row["gt_smiles"]))
            hyp = _atom_tok(str(row["gen_smiles"]))
            ab2.append(_sentence_bleu(ref, hyp, max_n=2))
            ab4.append(_sentence_bleu(ref, hyp, max_n=4))
        metrics["atom_bleu2"] = round(sum(ab2) / len(ab2), 4)
        metrics["atom_bleu4"] = round(sum(ab4) / len(ab4), 4)

    return metrics

# ── main ────────────────────────────────────────────────────────────────────
def main():
    csvs = sorted(OUTPUTS.glob("*eval*.csv"))
    csvs = [p for p in csvs if p.name != "summary_all_runs.csv"]
    if not csvs:
        print("No eval CSVs found in outputs/")
        sys.exit(1)

    rows = []
    for path in csvs:
        meta = parse_filename(path.stem)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"  SKIP  {path.name}  ({type(e).__name__}: {e})")
            continue
        m = extract_metrics(df)
        rows.append({"filename": path.name, **meta, **m})
        print(f"  parsed: {path.name}  [{meta['dataset']} / {meta['model_size']} / {meta['encoder']}  cfg={meta['cfg']} t={meta['temp']}]")

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        ["dataset", "encoder", "cfg", "temp"]
    ).reset_index(drop=True)

    out = OUTPUTS / "summary_all_runs.csv"
    summary.to_csv(out, index=False)
    print(f"\nWritten: {out}  ({len(summary)} rows)\n")

    # Print formatted table
    display_cols = ["dataset", "model_size", "encoder", "cfg", "temp",
                    "pre_eos_fix", "eos_truncate", "n", "validity",
                    "bleu2", "bleu4", "atom_bleu2", "atom_bleu4",
                    "lev_sim", "morgan_sim", "maccs_sim", "rdk_sim"]
    show = [c for c in display_cols if c in summary.columns]
    disp = summary[show].copy()

    col_w = {c: max(len(c), disp[c].astype(str).str.len().max()) for c in show}
    header = "  ".join(f"{c:<{col_w[c]}}" for c in show)
    sep    = "  ".join("-" * col_w[c] for c in show)
    print(header)
    print(sep)
    for _, row in disp.iterrows():
        print("  ".join(f"{str(row[c]):<{col_w[c]}}" for c in show))
    print()

if __name__ == "__main__":
    main()
