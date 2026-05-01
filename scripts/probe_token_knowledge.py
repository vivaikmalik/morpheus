"""
Probe: What does Morpheus 27M know about structural tokens?

Measures per-token-category top-1 accuracy when masking tokens one-at-a-time
(mode 1) and all-at-once (mode 2). Covers atom tokens (control), ring count
tokens, and branch opener tokens. No generation is run.
"""
import sys, os, time, random, math, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import selfies as sf

# ── Repo root and sys.path ────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tokenizer.chemicalTokenizer import ChemicalTokenizer
from model.molecularDiffusionModel import MolecularDiffusionModel
from transformers import AutoTokenizer, AutoModel

# ── Constants ─────────────────────────────────────────────────────────────────
RING_OPENER_IDS   = frozenset({5, 12, 25, 36, 85, 97, 102})
BRANCH_OPENER_IDS = frozenset({6, 7, 11, 15, 18, 20})
SPECIAL_IDS       = frozenset({0, 1, 2})          # PAD, MASK, EOS
STRUCTURAL_IDS    = RING_OPENER_IDS | BRANCH_OPENER_IDS
PAD_ID  = 0
MASK_ID = 1
EOS_ID  = 2
MAX_LEN = 74
BATCH_SIZE = 32
MAX_ATOM_PER_MOL = 3   # subsample atom positions

# ── CLI arguments ─────────────────────────────────────────────────────────────
import argparse
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--checkpoint", type=str,
    default=str(REPO_ROOT / "checkpoints" / "chebi20_27M_scibert_20ep_contrastive.pt"))
_parser.add_argument("--output_suffix", type=str, default="")
_parser.add_argument("--sanity", action="store_true")
_cli_args, _ = _parser.parse_known_args()

MODEL_CKPT       = Path(_cli_args.checkpoint)
CONTRASTIVE_CKPT = REPO_ROOT / "checkpoints" / "contrastive_scibert_chembl.pt"
TOKENIZER_PATH   = REPO_ROOT / "chemical_tokenizer.json"
TEST_CSV         = REPO_ROOT / "data" / "chebi20_test.csv"
_suffix          = f"_{_cli_args.output_suffix}" if _cli_args.output_suffix else ""
OUT_MD           = REPO_ROOT / "outputs" / f"probe_token_knowledge{_suffix}.md"
OUT_CSV          = REPO_ROOT / "outputs" / f"probe_token_knowledge_raw{_suffix}.csv"

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"[device] {device}")

# ── Load tokenizer ─────────────────────────────────────────────────────────────
tokenizer = ChemicalTokenizer(TOKENIZER_PATH)
print(f"[tokenizer] vocab_size={tokenizer.vocab_size}")

# ── Load contrastive encoder ──────────────────────────────────────────────────
print(f"[encoder] loading from {CONTRASTIVE_CKPT.name} ...")
ckpt_enc = torch.load(CONTRASTIVE_CKPT, map_location="cpu", weights_only=False)
enc_cfg   = ckpt_enc.get("config", {})
enc_model_name = enc_cfg.get("text_model")
if enc_model_name is None:
    raise ValueError("No text_model in contrastive checkpoint config")
text_state = {k[len("text_model."):]: v
              for k, v in ckpt_enc["model_state"].items()
              if k.startswith("text_model.")}
assert text_state, "No text_model.* keys in contrastive checkpoint"
encoder = AutoModel.from_pretrained(enc_model_name)
encoder.load_state_dict(text_state, strict=True)
encoder = encoder.to(device).eval()
hf_tokenizer = AutoTokenizer.from_pretrained(enc_model_name)
print(f"[encoder] loaded: {enc_model_name}")

# ── Load diffusion model ──────────────────────────────────────────────────────
print(f"[model] loading MolecularDiffusionModel ...")
model = MolecularDiffusionModel(
    vocab_size      = 110,
    hidden_size     = 512,
    num_heads       = 8,
    ffn_dim         = 1024,
    num_layers      = 8,
    max_length      = MAX_LEN,
    pad_token_id    = PAD_ID,
    text_model_name = "BAAI/bge-large-en-v1.5",  # placeholder; overwritten by set_text_encoder
    uncond_prob     = 0.0,
    dropout         = 0.0,
).to(device)
model.set_text_encoder(encoder)   # set_text_encoder BEFORE load_state_dict

print(f"[model] loading checkpoint {MODEL_CKPT.name} ...")
ckpt = torch.load(MODEL_CKPT, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"], strict=True)
model.eval()
print(f"[model] loaded  val_loss={ckpt.get('val_loss', float('nan')):.4f}  "
      f"step={ckpt.get('step','?')}")

# ── Load test data ────────────────────────────────────────────────────────────
test_df = pd.read_csv(TEST_CSV)
print(f"[data] {len(test_df)} test molecules")


# ── Helper: get ring count of molecule via RDKit ──────────────────────────────
def get_ring_count(selfies_str):
    try:
        smiles = sf.decoder(selfies_str)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return -1
        return mol.GetRingInfo().NumRings()
    except Exception:
        return -1


# ── Helper: tokenize selfies and classify positions ───────────────────────────
def tokenize_and_classify(selfies_str, rng):
    """Returns (ids_full, atom_pos, ring_count_pos, branch_pos) or None on failure."""
    try:
        ids = tokenizer.encode(selfies_str)
    except Exception:
        return None
    # Remove trailing PADs
    ids_trimmed = ids
    n = len(ids_trimmed)
    if n == 0:
        return None

    atom_pos        = []
    ring_count_pos  = []
    branch_pos      = []

    for i in range(n):
        tid = ids_trimmed[i]
        # atom/non-structural: id >= 3 and not ring opener or branch opener
        if tid >= 3 and tid not in RING_OPENER_IDS and tid not in BRANCH_OPENER_IDS:
            atom_pos.append(i)
        # ring count: position i is count token if i-1 is a ring opener
        if i >= 1 and ids_trimmed[i-1] in RING_OPENER_IDS:
            ring_count_pos.append(i)
        # branch opener
        if tid in BRANCH_OPENER_IDS:
            branch_pos.append(i)

    # Subsample atom positions
    if len(atom_pos) > MAX_ATOM_PER_MOL:
        atom_pos = list(rng.choice(atom_pos, size=MAX_ATOM_PER_MOL, replace=False))
        atom_pos.sort()

    return ids_trimmed, n, atom_pos, ring_count_pos, branch_pos


# ── Helper: pad sequence to MAX_LEN ──────────────────────────────────────────
def pad_seq(ids, mask_positions=None):
    """Return a MAX_LEN list of token ids with given positions set to MASK_ID."""
    out = list(ids[:MAX_LEN])
    out += [PAD_ID] * (MAX_LEN - len(out))
    if mask_positions:
        for p in mask_positions:
            if p < MAX_LEN:
                out[p] = MASK_ID
    return out


# ── Helper: run batched forward passes and collect per-position stats ─────────
@torch.no_grad()
def run_probe_batch(batch_input_ids, batch_timesteps, batch_text_embeds, batch_text_masks,
                    batch_probe_positions, batch_gt_tokens):
    """
    batch_input_ids:   [B, L] long tensor
    batch_timesteps:   [B, 1] float tensor
    batch_text_embeds: [B, T, H] float tensor
    batch_text_masks:  [B, T] bool tensor
    batch_probe_positions: list of ints (one per batch element)
    batch_gt_tokens:   list of ints

    Returns list of dicts with keys: top1_pred_id, top1_correct, p_gt, entropy, gt_token_id
    """
    logits = model(batch_input_ids, batch_timesteps, batch_text_embeds, batch_text_masks)
    # logits: [B, L, vocab_size]
    results = []
    for b, (pos, gt_tok) in enumerate(zip(batch_probe_positions, batch_gt_tokens)):
        pos_logits = logits[b, pos, :]   # [vocab_size]
        probs = torch.softmax(pos_logits.float(), dim=-1)
        top1_id = probs.argmax().item()
        p_gt = probs[gt_tok].item()
        # Entropy in nats
        ent = -(probs * torch.log(probs + 1e-12)).sum().item()
        # Top-3
        top3 = probs.topk(3).indices.tolist()
        results.append({
            "top1_pred_id": top1_id,
            "top1_correct": int(top1_id == gt_tok),
            "p_gt": p_gt,
            "entropy": ent,
            "gt_token_id": gt_tok,
            "top3_ids": top3,
            "top3_probs": [probs[i].item() for i in top3],
        })
    return results


# ── Pre-compute text embeddings for all prompts ───────────────────────────────
print(f"\n[probe] Pre-computing text embeddings for {len(test_df)} prompts ...")
t0 = time.time()
all_text_embeds = []   # list of [1, T, H] tensors
all_text_masks  = []   # list of [1, T] bool tensors

HF_BATCH = 64
prompts = test_df["prompt"].tolist()
for start in range(0, len(prompts), HF_BATCH):
    batch_prompts = prompts[start:start+HF_BATCH]
    enc_inputs = hf_tokenizer(
        batch_prompts, padding=True, truncation=True,
        max_length=128, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        text_emb = model.get_text_embeddings(
            enc_inputs["input_ids"], enc_inputs["attention_mask"]
        )
    text_pad_mask = (enc_inputs["attention_mask"] == 0)
    # Store per-molecule tensors (not whole batch) so mol_idx indexing works
    for _i in range(len(batch_prompts)):
        all_text_embeds.append(text_emb[_i:_i+1].cpu())    # [1, T, H]
        all_text_masks.append(text_pad_mask[_i:_i+1].cpu()) # [1, T]
    if (start // HF_BATCH + 1) % 5 == 0:
        print(f"  {start+len(batch_prompts)}/{len(prompts)} embeddings computed")

print(f"  Done in {time.time()-t0:.1f}s")


# ── Build probe instances ──────────────────────────────────────────────────────
# Each instance: (mol_idx, probe_pos, gt_token_id, input_ids_list, t_frac, category, mode)
print("\n[probe] Building probe instances from test molecules ...")
rng = np.random.default_rng(42)

instances = []          # all probes
mol_ring_counts = {}    # mol_idx -> ring_count

sanity_only = _cli_args.sanity
max_mol = 10 if sanity_only else len(test_df)

skip_count = 0
for mol_idx in range(min(max_mol, len(test_df))):
    row = test_df.iloc[mol_idx]
    selfies_str = str(row["response"])

    parsed = tokenize_and_classify(selfies_str, rng)
    if parsed is None:
        skip_count += 1
        continue
    ids_trimmed, n, atom_pos, ring_count_pos, branch_pos = parsed

    # Ring count for stratification
    rc = get_ring_count(selfies_str)
    mol_ring_counts[mol_idx] = rc

    # ── MODE 1: one-at-a-time ─────────────────────────────────────────────────
    t_frac_one = 1.0 / max(n, 1)

    for pos in atom_pos:
        gt_tok = ids_trimmed[pos]
        inp = pad_seq(ids_trimmed, mask_positions=[pos])
        instances.append((mol_idx, pos, gt_tok, inp, t_frac_one, "atom", "mode1"))

    for pos in ring_count_pos:
        gt_tok = ids_trimmed[pos]
        inp = pad_seq(ids_trimmed, mask_positions=[pos])
        instances.append((mol_idx, pos, gt_tok, inp, t_frac_one, "ring_count", "mode1"))

    for pos in branch_pos:
        gt_tok = ids_trimmed[pos]
        inp = pad_seq(ids_trimmed, mask_positions=[pos])
        instances.append((mol_idx, pos, gt_tok, inp, t_frac_one, "branch_opener", "mode1"))

    # ── MODE 2: all-at-once per category ──────────────────────────────────────
    if atom_pos:
        n_masked = len(atom_pos)
        t_frac_all_atom = n_masked / max(n, 1)
        inp = pad_seq(ids_trimmed, mask_positions=atom_pos)
        for pos in atom_pos:
            gt_tok = ids_trimmed[pos]
            instances.append((mol_idx, pos, gt_tok, inp, t_frac_all_atom, "atom", "mode2"))

    if ring_count_pos:
        n_masked = len(ring_count_pos)
        t_frac_all_rc = n_masked / max(n, 1)
        inp = pad_seq(ids_trimmed, mask_positions=ring_count_pos)
        for pos in ring_count_pos:
            gt_tok = ids_trimmed[pos]
            instances.append((mol_idx, pos, gt_tok, inp, t_frac_all_rc, "ring_count", "mode2"))

    if branch_pos:
        n_masked = len(branch_pos)
        t_frac_all_br = n_masked / max(n, 1)
        inp = pad_seq(ids_trimmed, mask_positions=branch_pos)
        for pos in branch_pos:
            gt_tok = ids_trimmed[pos]
            instances.append((mol_idx, pos, gt_tok, inp, t_frac_all_br, "branch_opener", "mode2"))

print(f"  Built {len(instances)} probe instances from {max_mol - skip_count} molecules "
      f"({skip_count} skipped)")

# Sanity check output
if sanity_only:
    from collections import Counter
    cat_mode_counts = Counter((c, m) for _, _, _, _, _, c, m in instances)
    print("\n[sanity] Probe instances per (category, mode):")
    for k, v in sorted(cat_mode_counts.items()):
        print(f"  {k}: {v}")
    print("\n[sanity] First 3 instances per category (mode1):")
    for cat in ["atom", "ring_count", "branch_opener"]:
        shown = 0
        for inst in instances:
            mol_idx, pos, gt_tok, inp, t_frac, c, m = inst
            if c == cat and m == "mode1" and shown < 3:
                print(f"  cat={cat} mol={mol_idx} pos={pos} gt_tok={gt_tok} t={t_frac:.4f}")
                shown += 1
    print()


# ── Run all probes in batches ──────────────────────────────────────────────────
print(f"\n[probe] Running {len(instances)} forward passes in batches of {BATCH_SIZE} ...")
t_run = time.time()
raw_rows = []

# Group instances into batches
total_batches = math.ceil(len(instances) / BATCH_SIZE)
n_fwd = 0

for b_start in range(0, len(instances), BATCH_SIZE):
    batch = instances[b_start:b_start+BATCH_SIZE]
    B = len(batch)

    # Build tensors
    inp_ids = torch.tensor([inst[3] for inst in batch], dtype=torch.long, device=device)
    t_fracs = torch.tensor([[inst[4]] for inst in batch], dtype=torch.float32, device=device)

    # Text embeds: stack from pre-computed
    text_emb_list = []
    text_mask_list = []
    for inst in batch:
        mol_idx = inst[0]
        text_emb_list.append(all_text_embeds[mol_idx])    # [1, T, H]
        text_mask_list.append(all_text_masks[mol_idx])    # [1, T]

    # Pad to same T dimension within this batch
    max_T = max(e.size(1) for e in text_emb_list)
    H = text_emb_list[0].size(2)
    text_emb_padded  = torch.zeros(B, max_T, H, dtype=torch.float32)
    text_mask_padded = torch.ones(B, max_T, dtype=torch.bool)   # True = padding
    for i, (te, tm) in enumerate(zip(text_emb_list, text_mask_list)):
        t_len = te.size(1)
        text_emb_padded[i, :t_len, :] = te[0]
        text_mask_padded[i, :t_len]   = tm[0]
    text_emb_padded  = text_emb_padded.to(device)
    text_mask_padded = text_mask_padded.to(device)

    probe_positions = [inst[1] for inst in batch]
    gt_tokens       = [inst[2] for inst in batch]

    results = run_probe_batch(
        inp_ids, t_fracs, text_emb_padded, text_mask_padded,
        probe_positions, gt_tokens
    )
    n_fwd += 1

    for inst, res in zip(batch, results):
        mol_idx, pos, gt_tok, _, t_frac, cat, mode = inst
        raw_rows.append({
            "molecule_idx":            mol_idx,
            "position":                pos,
            "category":                cat,
            "mode":                    mode,
            "gt_token_id":             gt_tok,
            "top1_pred_id":            res["top1_pred_id"],
            "top1_correct":            res["top1_correct"],
            "p_gt":                    res["p_gt"],
            "entropy":                 res["entropy"],
            "ring_count_of_molecule":  mol_ring_counts.get(mol_idx, -1),
        })

    if n_fwd % 50 == 0:
        elapsed = time.time() - t_run
        pct = 100 * b_start / len(instances)
        print(f"  batch {n_fwd}/{total_batches} ({pct:.0f}%)  elapsed={elapsed:.1f}s")

total_time = time.time() - t_run
print(f"  Done: {n_fwd} batches in {total_time:.1f}s  ({len(raw_rows)} probe results)")

# Sanity print for first run
if sanity_only:
    print("\n[sanity] First 3 results per (category, mode1):")
    df_s = pd.DataFrame(raw_rows)
    for cat in ["atom", "ring_count", "branch_opener"]:
        sub = df_s[(df_s["category"] == cat) & (df_s["mode"] == "mode1")].head(3)
        if len(sub) == 0:
            print(f"  {cat}: NO RESULTS")
            continue
        for _, r in sub.iterrows():
            print(f"  cat={cat} pos={r.position} gt={r.gt_token_id} "
                  f"pred={r.top1_pred_id} correct={r.top1_correct} "
                  f"p_gt={r.p_gt:.4f} entropy={r.entropy:.4f}")
    print()
    if df_s["p_gt"].isna().any() or (df_s["p_gt"] == 0).all():
        print("ERROR: p_gt looks broken (all 0 or NaN). Check model loading.")
        sys.exit(1)
    print("[sanity] PASS — proceeding to full run would look correct")
    if "--sanity-only" in sys.argv:
        sys.exit(0)
    print()


# ── Aggregate results ──────────────────────────────────────────────────────────
df = pd.DataFrame(raw_rows)

def ring_bin(rc):
    if rc <= 0: return "0"
    if rc <= 2: return "1-2"
    return "3+"

df["ring_bin"] = df["ring_count_of_molecule"].apply(ring_bin)

def agg(sub):
    if len(sub) == 0:
        return dict(n=0, top1_acc=float("nan"), mean_p_gt=float("nan"),
                    mean_ent=float("nan"), median_rank=float("nan"),
                    mean_top1_frac=float("nan"), mean_top3_frac=float("nan"))
    n = len(sub)
    top1_acc = sub["top1_correct"].mean()
    mean_p_gt = sub["p_gt"].mean()
    mean_ent = sub["entropy"].mean()
    return dict(n=n, top1_acc=top1_acc, mean_p_gt=mean_p_gt, mean_ent=mean_ent)

# Headline table
print("\n=== HEADLINE RESULTS ===")
rows_md = []
for cat in ["atom", "ring_count", "branch_opener"]:
    for mode in ["mode1", "mode2"]:
        sub = df[(df["category"] == cat) & (df["mode"] == mode)]
        a = agg(sub)
        print(f"  {cat:15s} {mode}  n={a['n']:6d}  top1={a['top1_acc']:.4f}  "
              f"p_gt={a['mean_p_gt']:.4f}  ent={a['mean_ent']:.4f}")
        rows_md.append((cat, mode, a))

# Stratified by ring bin (for ring_count and branch_opener)
print("\n=== STRATIFIED RESULTS ===")
strat_rows = {}
for cat in ["ring_count", "branch_opener"]:
    strat_rows[cat] = {}
    for mode in ["mode1", "mode2"]:
        strat_rows[cat][mode] = {}
        for rbin in ["0", "1-2", "3+"]:
            sub = df[(df["category"] == cat) & (df["mode"] == mode) & (df["ring_bin"] == rbin)]
            a = agg(sub)
            strat_rows[cat][mode][rbin] = a
            print(f"  {cat:15s} {mode} rings={rbin}  n={a['n']:5d}  "
                  f"top1={a['top1_acc']:.4f}  p_gt={a['mean_p_gt']:.4f}")

# Distribution analysis
print("\n=== DISTRIBUTION ANALYSIS ===")
dist_stats = {}
for cat in ["atom", "ring_count", "branch_opener"]:
    sub = df[(df["category"] == cat) & (df["mode"] == "mode1")]
    if len(sub) == 0:
        continue
    # entropy stats
    mean_ent = sub["entropy"].mean()
    max_ent  = math.log(110)  # uniform entropy over 110 tokens
    norm_ent = mean_ent / max_ent
    dist_stats[cat] = {"mean_ent": mean_ent, "norm_ent": norm_ent, "mean_p_gt": sub["p_gt"].mean()}
    print(f"  {cat}: mean_entropy={mean_ent:.4f}/{max_ent:.4f} ({100*norm_ent:.1f}% of max)  "
          f"mean_p_gt={sub['p_gt'].mean():.4f}")


# ── Write CSV ──────────────────────────────────────────────────────────────────
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"\n[output] Raw CSV written: {OUT_CSV}")


# ── Write Markdown ─────────────────────────────────────────────────────────────
atom_m1  = agg(df[(df["category"] == "atom")         & (df["mode"] == "mode1")])
rc_m1    = agg(df[(df["category"] == "ring_count")   & (df["mode"] == "mode1")])
br_m1    = agg(df[(df["category"] == "branch_opener")& (df["mode"] == "mode1")])
atom_m2  = agg(df[(df["category"] == "atom")         & (df["mode"] == "mode2")])
rc_m2    = agg(df[(df["category"] == "ring_count")   & (df["mode"] == "mode2")])
br_m2    = agg(df[(df["category"] == "branch_opener")& (df["mode"] == "mode2")])

# verdict
atom_acc = atom_m1["top1_acc"]
rc_acc   = rc_m1["top1_acc"]
br_acc   = br_m1["top1_acc"]

THRESHOLD_HIGH = 0.50
THRESHOLD_LOW  = 0.25

if rc_acc >= THRESHOLD_HIGH and br_acc >= THRESHOLD_HIGH:
    verdict = ("MODEL KNOWS COUNT TOKENS: high accuracy on count tokens with full context. "
               "The dep_aware null result is not explained by capability gap; "
               "the failure must be elsewhere (decoding ordering interactions, encoder routing, etc).")
elif rc_acc < THRESHOLD_LOW and br_acc < THRESHOLD_LOW:
    verdict = ("MODEL DOES NOT KNOW COUNT TOKENS: low accuracy on count tokens even with full context. "
               "The aggregate generation failure is consistent with a training-time capability gap; "
               "no decoding-time fix can address this.")
else:
    verdict = (f"MIXED: model knows some structural tokens but not others. "
               f"Ring count top-1 accuracy = {rc_acc:.3f}, "
               f"branch opener top-1 accuracy = {br_acc:.3f}. "
               f"Atom (control) accuracy = {atom_acc:.3f}.")

print(f"\n[verdict] {verdict}")

md_content = f"""# Probe: What the Model Knows About Structural Tokens

## Methodology

For each of 2542 ChEBI-20 test molecules, probe positions were identified in three categories:
**atom** (non-structural tokens, vocab id ≥3 and not ring/branch opener; subsampled to 3 per molecule),
**ring_count** (position immediately following a ring opener token), and
**branch_opener** (positions carrying a branch opener token id).
Two masking modes were tested: **mode 1** masks one target position at a time (all other
ground-truth tokens visible); **mode 2** masks all positions of that category simultaneously.
For each masked position, one conditional forward pass was run (CFG=1.0, timestep proportional
to fraction masked) and the predicted probability distribution over the 110-token vocab was recorded.
Total forward passes: {n_fwd} batches × ≤{BATCH_SIZE} = {len(raw_rows)} probe instances.
Wall-clock time: {total_time:.1f}s on {device}.

---

## Headline Results

| Category | Mode | N positions | Top-1 acc | Mean p(gt) | Mean entropy |
|---|---|---|---|---|---|
| atom | one-at-a-time | {atom_m1['n']} | {atom_m1['top1_acc']:.4f} | {atom_m1['mean_p_gt']:.4f} | {atom_m1['mean_ent']:.4f} |
| atom | all-at-once | {atom_m2['n']} | {atom_m2['top1_acc']:.4f} | {atom_m2['mean_p_gt']:.4f} | {atom_m2['mean_ent']:.4f} |
| ring_count | one-at-a-time | {rc_m1['n']} | {rc_m1['top1_acc']:.4f} | {rc_m1['mean_p_gt']:.4f} | {rc_m1['mean_ent']:.4f} |
| ring_count | all-at-once | {rc_m2['n']} | {rc_m2['top1_acc']:.4f} | {rc_m2['mean_p_gt']:.4f} | {rc_m2['mean_ent']:.4f} |
| branch_opener | one-at-a-time | {br_m1['n']} | {br_m1['top1_acc']:.4f} | {br_m1['mean_p_gt']:.4f} | {br_m1['mean_ent']:.4f} |
| branch_opener | all-at-once | {br_m2['n']} | {br_m2['top1_acc']:.4f} | {br_m2['mean_p_gt']:.4f} | {br_m2['mean_ent']:.4f} |

---

## Stratified by Ring Complexity

### Ring Count Token Accuracy

| Ring bin | Mode | N | Top-1 acc | Mean p(gt) |
|---|---|---|---|---|
"""
for rbin in ["0", "1-2", "3+"]:
    for mode_label, mode_key in [("one-at-a-time", "mode1"), ("all-at-once", "mode2")]:
        a = strat_rows["ring_count"][mode_key][rbin]
        md_content += f"| {rbin} | {mode_label} | {a['n']} | {a['top1_acc']:.4f} | {a['mean_p_gt']:.4f} |\n"

md_content += "\n### Branch Opener Token Accuracy\n\n"
md_content += "| Ring bin | Mode | N | Top-1 acc | Mean p(gt) |\n"
md_content += "|---|---|---|---|---|\n"
for rbin in ["0", "1-2", "3+"]:
    for mode_label, mode_key in [("one-at-a-time", "mode1"), ("all-at-once", "mode2")]:
        a = strat_rows["branch_opener"][mode_key][rbin]
        md_content += f"| {rbin} | {mode_label} | {a['n']} | {a['top1_acc']:.4f} | {a['mean_p_gt']:.4f} |\n"

max_ent_str = f"{math.log(110):.4f}"
md_content += f"""
---

## Distribution Analysis

Maximum possible entropy (uniform over 110 tokens): {max_ent_str} nats.

| Category | Mean entropy | % of max entropy | Mean p(gt) mode1 |
|---|---|---|---|
| atom | {dist_stats.get('atom', {}).get('mean_ent', float('nan')):.4f} | {100*dist_stats.get('atom', {}).get('norm_ent', float('nan')):.1f}% | {dist_stats.get('atom', {}).get('mean_p_gt', float('nan')):.4f} |
| ring_count | {dist_stats.get('ring_count', {}).get('mean_ent', float('nan')):.4f} | {100*dist_stats.get('ring_count', {}).get('norm_ent', float('nan')):.1f}% | {dist_stats.get('ring_count', {}).get('mean_p_gt', float('nan')):.4f} |
| branch_opener | {dist_stats.get('branch_opener', {}).get('mean_ent', float('nan')):.4f} | {100*dist_stats.get('branch_opener', {}).get('norm_ent', float('nan')):.1f}% | {dist_stats.get('branch_opener', {}).get('mean_p_gt', float('nan')):.4f} |

A low entropy (small % of max) means the model is confident. A high entropy means
the predicted distribution is diffuse. Mean p(gt) shows how much probability mass
the model assigns to the correct token regardless of its rank.

---

## Key Numbers to Remember

- **Atom (control) top-1 accuracy (mode1):** {atom_m1['top1_acc']:.4f}  ({atom_m1['n']} positions)
- **Ring count top-1 accuracy (mode1):** {rc_m1['top1_acc']:.4f}  ({rc_m1['n']} positions)
- **Branch opener top-1 accuracy (mode1):** {br_m1['top1_acc']:.4f}  ({br_m1['n']} positions)

---

## Verdict

{verdict}

---
"""

OUT_MD.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write(md_content)
print(f"[output] Markdown written: {OUT_MD}")

print(f"\n{'='*60}")
print(f"Path (markdown): {OUT_MD}")
print(f"Path (raw CSV):  {OUT_CSV}")
print(f"Verdict: {verdict[:80]}...")
print(f"Wall-clock time: {total_time:.1f}s")
print(f"Top-1 accuracy — atom: {atom_acc:.4f}  ring_count: {rc_acc:.4f}  branch_opener: {br_acc:.4f}")
print(f"{'='*60}")

if __name__ == "__main__":
    pass
