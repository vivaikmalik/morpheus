"""
Reward functions for RL post-training on prompt adherence.

Provides:
  - Tanimoto similarity (Morgan fingerprints)
  - BLEU-2 / BLEU-4 on SELFIES token sequences
  - Combined reward computation
"""

import numpy as np
from collections import Counter
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
import selfies as sf


# =============================================================================
# TANIMOTO SIMILARITY
# =============================================================================

def tanimoto_similarity(mol_gen, mol_ref):
    """Morgan-fingerprint Tanimoto similarity between two RDKit Mol objects."""
    if mol_gen is None or mol_ref is None:
        return 0.0
    fp_gen = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol_gen, radius=2, nBits=2048)
    fp_ref = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol_ref, radius=2, nBits=2048)
    return float(DataStructs.TanimotoSimilarity(fp_gen, fp_ref))


# =============================================================================
# BLEU SCORE (SELFIES tokens as "words")
# =============================================================================

def _ngrams(tokens, n):
    """Return a Counter of n-grams."""
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu_score(hypothesis_tokens, reference_tokens, max_n=4):
    """
    Compute corpus-style BLEU-n treating SELFIES tokens as words.

    Returns
    -------
    dict with "bleu2" and "bleu4"
    """
    if not hypothesis_tokens or not reference_tokens:
        return {"bleu2": 0.0, "bleu4": 0.0}

    bp = min(1.0, np.exp(1.0 - len(reference_tokens) / max(len(hypothesis_tokens), 1)))

    precisions = []
    for n in range(1, max_n + 1):
        hyp_ng = _ngrams(hypothesis_tokens, n)
        ref_ng = _ngrams(reference_tokens, n)
        if not hyp_ng:
            precisions.append(0.0)
            continue
        clipped = sum(min(c, ref_ng.get(ng, 0)) for ng, c in hyp_ng.items())
        precisions.append(clipped / max(sum(hyp_ng.values()), 1))

    results = {}
    for n_val, key in [(2, "bleu2"), (4, "bleu4")]:
        p_slice = precisions[:n_val]
        if all(p > 0 for p in p_slice):
            log_avg = sum(np.log(p) for p in p_slice) / n_val
            results[key] = float(bp * np.exp(log_avg))
        else:
            results[key] = 0.0
    return results


# =============================================================================
# COMBINED REWARD
# =============================================================================

def compute_rewards(gen_ids_list, ref_selfies_list, tokenizer,
                    w_tanimoto=1.0, w_bleu2=0.5, w_bleu4=0.5,
                    validity_bonus=0.2):
    """
    Compute prompt-adherence rewards for a batch of generated molecules.

    For each (generated, reference) pair:
      reward = w_tanimoto * tanimoto_sim
             + w_bleu2    * bleu2
             + w_bleu4    * bleu4
             + validity_bonus * is_valid

    Parameters
    ----------
    gen_ids_list     : list[list[int]]  — generated token IDs
    ref_selfies_list : list[str]        — ground-truth SELFIES strings
    tokenizer        : ChemicalTokenizer

    Returns
    -------
    rewards, tanimoto_scores, bleu2_scores, bleu4_scores, validity_flags
        Each is a list of floats with length == batch size.
    """
    from rl.utils import decode_molecule  # avoid circular import

    rewards, tan_scores, b2_scores, b4_scores, valid_flags = [], [], [], [], []

    for gen_ids, ref_selfies in zip(gen_ids_list, ref_selfies_list):
        # Decode generated molecule
        gen_selfies, gen_tokens, gen_smiles, gen_mol = decode_molecule(gen_ids, tokenizer)

        # Parse reference molecule
        ref_tokens = list(sf.split_selfies(ref_selfies)) if ref_selfies else []
        try:
            ref_smiles = sf.decoder(ref_selfies)
            ref_mol = Chem.MolFromSmiles(ref_smiles)
        except Exception:
            ref_mol = None

        # Tanimoto
        tan_sim = tanimoto_similarity(gen_mol, ref_mol)
        tan_scores.append(tan_sim)

        # BLEU
        bleu = bleu_score(gen_tokens, ref_tokens)
        b2_scores.append(bleu["bleu2"])
        b4_scores.append(bleu["bleu4"])

        # Validity
        is_valid = gen_mol is not None
        valid_flags.append(float(is_valid))

        # Combined
        r = (w_tanimoto * tan_sim
             + w_bleu2 * bleu["bleu2"]
             + w_bleu4 * bleu["bleu4"]
             + validity_bonus * float(is_valid))
        rewards.append(r)

    return rewards, tan_scores, b2_scores, b4_scores, valid_flags
