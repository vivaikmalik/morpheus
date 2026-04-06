import torch
import torch.nn as nn
import sys
from pathlib import Path
from transformers import AutoModel
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.embeddings import TimestepEmbedding, PositionalEmbedding
from model.transformerBlock import TransformerBlock


class MolecularDiffusionModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_heads,
        ffn_dim,
        num_layers,
        max_length,
        pad_token_id,
        text_model_name="BAAI/bge-large-en-v1.5",
        uncond_prob=0.1,
        dropout=0.1,
    ):
        super().__init__()

        self.pad_token_id = pad_token_id
        self.vocab_size   = vocab_size
        self.hidden_size  = hidden_size
        self.uncond_prob  = uncond_prob

        # ------------------------------------------------------------------ #
        # TEXT ENCODER & PROJECTOR (FROZEN)                                 #
        # ------------------------------------------------------------------ #
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        # Freeze the text encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        text_hidden_size = self.text_encoder.config.hidden_size
        
        # MLP Projector 
        self.text_proj = nn.Sequential(
            nn.Linear(text_hidden_size, ffn_dim),
            nn.SiLU(),
            nn.Linear(ffn_dim, hidden_size)
        )
        
        # Learnable null token for CFG
        self.null_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        # ------------------------------------------------------------------ #
        # DIFFUSION EMBEDDINGS                                              #
        # ------------------------------------------------------------------ #
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.pos_embedding = PositionalEmbedding(max_length, hidden_size)
        self.timestep_embedding = TimestepEmbedding(hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)

        # ------------------------------------------------------------------ #
        # TRANSFORMER BLOCKS                                                #
        # ------------------------------------------------------------------ #
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(hidden_size)
        self.lm_head     = nn.Linear(hidden_size, vocab_size, bias=False)

        self._init_weights()
        self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self):
        for name, module in self.named_modules():
            if name.startswith('text_encoder'):
                continue
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

        
        for block in self.blocks:
            nn.init.zeros_(block.cross_attention.out_proj.weight)
            nn.init.zeros_(block.cross_attention.out_proj.bias)

    def get_text_embeddings(self, text_input_ids, text_attention_mask):
        """ Runs the text encoder and projects the embeddings.
        Gradients flow through the encoder only when it has trainable parameters. """
        # BERT has a hard 512-token limit; truncate silently if inputs exceed it
        max_bert_len = self.text_encoder.config.max_position_embeddings
        if text_input_ids.shape[1] > max_bert_len:
            text_input_ids    = text_input_ids[:, :max_bert_len]
            text_attention_mask = text_attention_mask[:, :max_bert_len]

        encoder_frozen = not any(p.requires_grad for p in self.text_encoder.parameters())
        ctx = torch.no_grad() if encoder_frozen else torch.enable_grad()
        with ctx:
            outputs = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask)
            hidden_states = outputs.last_hidden_state

        # Pass through the learnable projector
        return self.text_proj(hidden_states)

    def forward(self, input_ids, timesteps, text_embeds=None, text_padding_mask=None):
        device  = input_ids.device
        B, seq_len = input_ids.shape

        if text_embeds is None:
            text_embeds = self.null_token.expand(B, 1, -1)

        if self.training:
            # Drop text conditioning with probability `uncond_prob`
            keep_mask = torch.rand(B, device=device) > self.uncond_prob
            keep_mask = keep_mask.view(B, 1, 1)
            
            null_seq = self.null_token.expand(B, text_embeds.size(1), -1)
            text_embeds = torch.where(keep_mask, text_embeds, null_seq)

        padding_mask = (input_ids == self.pad_token_id)

        x = self.token_embedding(input_ids)
        pos = self.pos_embedding(seq_len, device)
        x = x + pos
        t_emb = self.timestep_embedding(timesteps)
        x = x + t_emb.unsqueeze(1)
        x = self.input_norm(x)

        for block in self.blocks:
            x = block(x, text_embeds, padding_mask, text_padding_mask)

        x      = self.output_norm(x)
        logits = self.lm_head(x)

        return logits

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable