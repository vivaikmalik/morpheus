import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .utils import mean_pool_last_hidden

class MoleculeEncoder(nn.Module):
    def __init__(self, molecular_model):
        super().__init__()
        self.molecular_model = molecular_model

    def forward(self, input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        padding_mask = (input_ids == pad_token_id)
        attention_mask = (~padding_mask).long()

        # Token + timestep embeddings (no additive positional embedding;
        # position is injected via RoPE inside each self-attention block).
        x = self.molecular_model.token_embedding(input_ids)
        t = torch.zeros((batch_size, 1), device=device)
        x = x + self.molecular_model.timestep_embedding(t).unsqueeze(1)
        x = self.molecular_model.input_norm(x)

        # Compute RoPE cos/sin tables once.
        rope = self.molecular_model.rope(seq_len, device=device, dtype=x.dtype)

        for block in self.molecular_model.blocks:
            x = block(x, padding_mask, rope=rope)
        x = self.molecular_model.output_norm(x)
        return mean_pool_last_hidden(x, attention_mask)

class ContrastiveAligner(nn.Module):
    def __init__(self, text_model, molecule_encoder, text_hidden, mol_hidden, proj_dim, project_molecule=True):
        super().__init__()
        self.text_model = text_model
        self.molecule_encoder = molecule_encoder
        self.project_molecule = bool(project_molecule)
        if self.project_molecule:
            self.text_proj = nn.Linear(text_hidden, proj_dim)
            self.mol_proj = nn.Linear(mol_hidden, proj_dim)
            self.embedding_dim = proj_dim
        else:
            self.text_proj = nn.Linear(text_hidden, mol_hidden)
            self.mol_proj = nn.Identity()
            self.embedding_dim = mol_hidden

    def encode_text(self, text_inputs):
        out = self.text_model(**text_inputs)
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = mean_pool_last_hidden(out.last_hidden_state, text_inputs['attention_mask'])
        return F.normalize(self.text_proj(pooled), dim=-1)

    def encode_molecule(self, mol_ids, pad_token_id):
        pooled = self.molecule_encoder(mol_ids, pad_token_id)
        return F.normalize(self.mol_proj(pooled), dim=-1)

    def forward(self, text_inputs, mol_ids, pad_token_id):
        z_text = self.encode_text(text_inputs)
        z_mol = self.encode_molecule(mol_ids, pad_token_id)
        return z_text, z_mol

def contrastive_loss(z_text, z_mol, temperature=0.07):
    logits = (z_text @ z_mol.T) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

def retrieval_metrics(z_text, z_mol, ks=(1, 5, 10)):
    import numpy as np
    sims = z_text @ z_mol.T
    sorted_idx = torch.argsort(sims, dim=1, descending=True)
    targets = torch.arange(sims.shape[0], device=sims.device)
    m = {}
    for k in ks:
        topk = sorted_idx[:, :k]
        m[f'recall@{k}'] = (topk == targets.unsqueeze(1)).any(dim=1).float().mean().item()
    sorted_np = sorted_idx.detach().cpu().numpy()
    ranks = []
    rr = []
    for i in range(sorted_np.shape[0]):
        r = int(np.where(sorted_np[i] == i)[0][0]) + 1
        ranks.append(r)
        rr.append(1.0 / r)
    m['mean_rank'] = float(np.mean(ranks))
    m['mrr'] = float(np.mean(rr))
    return m
