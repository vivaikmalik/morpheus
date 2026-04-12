"""
inference/evaluate_chebi20.py — SMILES pipeline evaluation
"""

import argparse, sys, time
import numpy as np, pandas as pd, torch
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, QED, inchi
from rdkit.Chem.Scaffolds import MurckoScaffold
from transformers import AutoTokenizer
from nltk.translate.bleu_score import corpus_bleu
from Levenshtein import distance as lev_distance

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from tokenizer.smilesTokenizer import SmilesTokenizer
from model.molecularDiffusionModel import MolecularDiffusionModel


def maccs_tanimoto(m1, m2):
    return DataStructs.TanimotoSimilarity(MACCSkeys.GenMACCSKeys(m1), MACCSkeys.GenMACCSKeys(m2))

def morgan_unhashed(m1, m2, r=2):
    return DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(m1,r), AllChem.GetMorganFingerprint(m2,r))

def morgan_hashed(m1, m2, r=2, n=2048):
    return DataStructs.TanimotoSimilarity(
        AllChem.GetMorganFingerprintAsBitVect(m1,r,nBits=n),
        AllChem.GetMorganFingerprintAsBitVect(m2,r,nBits=n))

def rdk_tanimoto(m1, m2):
    return DataStructs.TanimotoSimilarity(Chem.RDKFingerprint(m1), Chem.RDKFingerprint(m2))

def get_scaffold(mol):
    try: return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except: return None


def generate_batch_cfg(model, mol_tok, text_tok, device, prompts,
                       cfg, steps, temp, max_len, max_text_len):
    B = len(prompts)
    if model.text_encoder is None:
        from transformers import AutoModel
        enc = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to(device).eval()
    else:
        enc = model.text_encoder
    text_enc = text_tok(prompts, padding=True, truncation=True,
                         max_length=max_text_len, return_tensors="pt").to(device)
    with torch.no_grad():
        states = enc(input_ids=text_enc["input_ids"],
                     attention_mask=text_enc["attention_mask"]).last_hidden_state
        text_embeds = model.project_text_embeddings(states)
    text_pad = (text_enc["attention_mask"] == 0)
    if model.text_encoder is None:
        del enc; torch.cuda.empty_cache()

    ids = torch.full((B, max_len), mol_tok.mask_token_id, dtype=torch.long, device=device)
    t_vals = torch.linspace(1.0, 0.0, steps, device=device)
    with torch.no_grad():
        for si, tv in enumerate(t_vals):
            st = tv.repeat(B).unsqueeze(-1)
            lc = model(ids, st, text_embeds, text_pad)
            lu = model(ids, st, None, None)
            lo = lu + cfg * (lc - lu)
            if si < steps-1: lo[:,:,mol_tok.mask_token_id] = float('-inf')
            p = torch.softmax(lo/temp, -1)
            s = torch.distributions.Categorical(p).sample()
            c = torch.gather(p, 2, s.unsqueeze(-1)).squeeze(-1)
            at = (torch.cos(tv * torch.pi / 2)**2).item()
            nm = int((1-at)*max_len)
            if nm > 0 and si < steps-1:
                _, mi = torch.topk(c, nm, -1, largest=False)
                s.scatter_(1, mi, mol_tok.mask_token_id)
            ids = s
    return ids


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mol_tok = SmilesTokenizer(str(ROOT / "smiles_vocab.json"))
    text_tok = AutoTokenizer.from_pretrained(args.text_model or "allenai/scibert_scivocab_uncased")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = MolecularDiffusionModel(
        vocab_size=config["vocab_size"], hidden_size=config["hidden_size"],
        num_heads=config["num_heads"], ffn_dim=config["ffn_dim"],
        num_layers=config["num_layers"], max_length=config["max_length"],
        pad_token_id=mol_tok.pad_token_id,
        text_model_name=config.get("text_model", "allenai/scibert_scivocab_uncased"),
        uncond_prob=0.0, dropout=0.0, load_text_encoder=True).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    print(f"Step {ckpt.get('step','?')} | Val {ckpt.get('val_loss','?')}")

    test_df = pd.read_csv(ROOT / "data" / "chebi20_test.csv")
    print(f"Test: {len(test_df):,}")

    max_len = config["max_length"]
    max_text_len = config.get("max_text_len", 256)

    start = time.time()
    all_results = []
    all_gt_smi, all_gen_smi = [], []

    for si in tqdm(range(0, len(test_df), args.batch_size), desc="Generating"):
        ei = min(si + args.batch_size, len(test_df))
        batch_df = test_df.iloc[si:ei]
        prompts = batch_df["description"].tolist()
        raw = generate_batch_cfg(model, mol_tok, text_tok, device, prompts,
                                  args.cfg, args.steps, args.temp, max_len, max_text_len)

        for i, row in enumerate(batch_df.itertuples()):
            ids = raw[i].cpu().tolist()
            if mol_tok.eos_token_id in ids:
                ids = ids[:ids.index(mol_tok.eos_token_id)]
            gen_smi = mol_tok.decode(ids)
            gen_mol = Chem.MolFromSmiles(gen_smi)
            gen_canonical = Chem.MolToSmiles(gen_mol) if gen_mol else None

            gt_smi = row.smiles
            gt_mol = Chem.MolFromSmiles(gt_smi) if gt_smi else None
            gt_canonical = Chem.MolToSmiles(gt_mol) if gt_mol else None

            valid = gen_mol is not None
            maccs = morgan_u = morgan_h = rdk = None
            exact_inchi = exact_smi = scaffold = False
            qed_v = logp_v = mw_v = None

            if valid and gt_mol:
                try:
                    maccs = maccs_tanimoto(gen_mol, gt_mol)
                    morgan_u = morgan_unhashed(gen_mol, gt_mol)
                    morgan_h = morgan_hashed(gen_mol, gt_mol)
                    rdk = rdk_tanimoto(gen_mol, gt_mol)
                except: pass
                try: exact_inchi = inchi.MolToInchi(gen_mol) == inchi.MolToInchi(gt_mol)
                except: pass
                exact_smi = gen_canonical == gt_canonical
                gs, gts = get_scaffold(gen_mol), get_scaffold(gt_mol)
                scaffold = gs == gts if gs and gts else False

            if valid:
                try:
                    qed_v = round(QED.qed(gen_mol), 4)
                    logp_v = round(Descriptors.MolLogP(gen_mol), 3)
                    mw_v = round(Descriptors.ExactMolWt(gen_mol), 2)
                except: pass

            all_gt_smi.append(gt_canonical)
            all_gen_smi.append(gen_canonical)
            all_results.append({
                "gt_smiles": gt_canonical, "gen_smiles": gen_canonical,
                "valid": valid, "exact_inchi": exact_inchi, "exact_smi": exact_smi,
                "scaffold": scaffold, "maccs": round(maccs,4) if maccs else None,
                "morgan_u": round(morgan_u,4) if morgan_u else None,
                "morgan_h": round(morgan_h,4) if morgan_h else None,
                "rdk": round(rdk,4) if rdk else None,
                "qed": qed_v, "logp": logp_v, "mw": mw_v,
            })

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s ({elapsed/len(test_df):.2f}s/mol)")

    df = pd.DataFrame(all_results)
    out_dir = ROOT / "outputs"; out_dir.mkdir(exist_ok=True)
    csv = out_dir / f"chebi20_eval_cfg{args.cfg}_temp{args.temp}_steps{args.steps}.csv"
    df.to_csv(csv, index=False)

    n = len(df); nv = df["valid"].sum(); vdf = df[df["valid"]]

    # TGM-DLM corpus BLEU (char-level SMILES)
    refs, hyps = [], []
    for gt, gen in zip(all_gt_smi, all_gen_smi):
        if gt and gen:
            refs.append([[c for c in gt]])
            hyps.append([c for c in gen])
    tgm_bleu = corpus_bleu(refs, hyps) if hyps else 0.0

    # TGM-DLM levenshtein
    levs = [lev_distance(gen, gt) for gt, gen in zip(all_gt_smi, all_gen_smi) if gt and gen]
    tgm_lev = np.mean(levs) if levs else 0.0

    fdf = vdf.dropna(subset=["maccs"])
    m_maccs = fdf["maccs"].mean() if len(fdf) else 0
    m_morgan_u = fdf["morgan_u"].mean() if len(fdf) else 0
    m_morgan_h = fdf["morgan_h"].mean() if len(fdf) else 0
    m_rdk = fdf["rdk"].mean() if len(fdf) else 0

    validity = nv / n
    exact_inchi = df["exact_inchi"].mean()
    exact_smi = df["exact_smi"].mean()
    scaffold = df["scaffold"].mean()
    uniq = vdf["gen_smiles"].nunique() / nv if nv else 0

    m_qed = vdf["qed"].mean() if len(vdf) else 0
    m_logp = vdf["logp"].mean() if len(vdf) else 0
    m_mw = vdf["mw"].mean() if len(vdf) else 0

    summary = f"""
{'='*65}
  CheBI-20 EVALUATION (SMILES pipeline)
  CFG={args.cfg}  temp={args.temp}  steps={args.steps}
{'='*65}

  TGM-DLM COMPATIBLE:
    BLEU (char corpus)    : {tgm_bleu:.4f}
    MACCS FTS             : {m_maccs:.4f}
    Morgan FTS (unhashed) : {m_morgan_u:.4f}
    RDK FTS               : {m_rdk:.4f}
    Exact Match (InChI)   : {exact_inchi:.4f}
    Levenshtein (raw)     : {tgm_lev:.2f}
    Validity              : {validity:.4f}

  ADDITIONAL:
    Morgan FTS (hashed)   : {m_morgan_h:.4f}
    Exact Match (SMILES)  : {exact_smi:.4f}
    Scaffold Match        : {scaffold:.4f}
    Uniqueness            : {uniq:.4f}

  PROPERTIES:
    QED: {m_qed:.4f}  LogP: {m_logp:.3f}  MW: {m_mw:.1f}

{'='*65}
  Model        | BLEU   | MACCS  | Morgan | RDK    | Exact  | Valid
  Morpheus     | {tgm_bleu:.3f}  | {m_maccs:.3f}  | {m_morgan_u:.3f}  | {m_rdk:.3f}  | {exact_inchi:.3f}  | {validity:.3f}
  TGM-DLM      | 0.828  | 0.874  | 0.609  | 0.677  | 0.082  | 0.789
{'='*65}"""

    print(summary)
    (out_dir / "chebi20_eval_summary.txt").write_text(summary)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(ROOT / "checkpoints/best_finetuned_model.pt"))
    p.add_argument("--cfg", type=float, default=3.0)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--text_model", type=str, default=None)
    evaluate(p.parse_args())