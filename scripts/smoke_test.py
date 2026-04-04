"""
scripts/smoke_test.py
----------------------
Quick end-to-end test of all text-conditioning components.
Uses synthetic data — no CheBI-20 download, no pretrained checkpoint needed.

Tests:
  1. TextConditionedDataset
  2. TextDiffusionCollator
  3. MolecularDiffusionModel (with SciBERT + cross-attention)
  4. Forward pass + loss computation
  5. Backward pass (gradient flow)
  6. Generation with CFG
  7. Decode pipeline
  8. Pretrained checkpoint loading with strict=False (if available)

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --checkpoint checkpoints/best_model.pt
"""

import sys
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def test_separator(name):
    print(f"\n{'─'*60}")
    print(f"  TEST: {name}")
    print(f"{'─'*60}")


def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # ================================================================= #
    # TEST 1: Tokenizer                                                  #
    # ================================================================= #
    test_separator("ChemicalTokenizer")

    from tokenizer.chemicalTokenizer import ChemicalTokenizer
    mol_tokenizer = ChemicalTokenizer(str(ROOT / "chemical_tokenizer.json"))

    print(f"  Vocab size:    {mol_tokenizer.vocab_size}")
    print(f"  PAD token ID:  {mol_tokenizer.pad_token_id}")
    print(f"  MASK token ID: {mol_tokenizer.mask_token_id}")
    print(f"  EOS token ID:  {mol_tokenizer.eos_token_id}")

    test_selfies = "[C][C][=O]"
    ids = mol_tokenizer.encode(test_selfies)
    back = mol_tokenizer.decode(ids)
    assert back == test_selfies, f"Round-trip failed: {test_selfies} → {ids} → {back}"
    print(f"  Round-trip:    {test_selfies} → {ids} → {back}  ✅")

    # ================================================================= #
    # TEST 2: SciBERT tokenizer loads                                    #
    # ================================================================= #
    test_separator("SciBERT tokenizer")

    from transformers import AutoTokenizer
    text_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

    test_text = "The molecule contains a hydroxyl group."
    enc = text_tokenizer(test_text, return_tensors="pt")
    print(f"  Input:      \"{test_text}\"")
    print(f"  Token IDs:  {enc['input_ids'].shape}  ({enc['input_ids'][0,:5].tolist()}...)")
    print(f"  ✅")

    # ================================================================= #
    # TEST 3: TextConditionedDataset                                     #
    # ================================================================= #
    test_separator("TextConditionedDataset")

    from dataExtractor.textConditionedDataset import TextConditionedDataset

    # Create synthetic data
    fake_df = pd.DataFrame({
        "selfies":     ["[C][C][=O]", "[C][C][O]", "[C][=C][C]", "[C][C]"] * 5,
        "description": [
            "A simple ketone molecule.",
            "An alcohol with two carbons.",
            "A molecule with a double bond.",
            "Ethane, the simplest alkane with two carbons.",
        ] * 5,
    })

    dataset = TextConditionedDataset(fake_df, mol_tokenizer, max_length=74)
    sample = dataset[0]

    print(f"  Dataset size:      {len(dataset)}")
    print(f"  input_ids shape:   {sample['input_ids'].shape}")
    print(f"  input_ids[:10]:    {sample['input_ids'][:10].tolist()}")
    print(f"  prompt:            \"{sample['prompt']}\"")

    # Verify EOS is present
    ids_list = sample['input_ids'].tolist()
    assert mol_tokenizer.eos_token_id in ids_list, "EOS token not found"
    eos_pos = ids_list.index(mol_tokenizer.eos_token_id)
    print(f"  EOS at position:   {eos_pos}")

    # Verify padding after EOS
    after_eos = ids_list[eos_pos + 1:]
    assert all(t == mol_tokenizer.pad_token_id for t in after_eos), "Non-PAD after EOS"
    print(f"  PADs after EOS:    {len(after_eos)}  ✅")

    # ================================================================= #
    # TEST 4: TextDiffusionCollator                                      #
    # ================================================================= #
    test_separator("TextDiffusionCollator")

    from training.textDiffusionCollator import TextDiffusionCollator
    from torch.utils.data import DataLoader

    collator = TextDiffusionCollator(mol_tokenizer, text_tokenizer, max_text_len=64)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collator)
    batch = next(iter(loader))

    print(f"  Batch keys: {list(batch.keys())}")
    for k, v in batch.items():
        print(f"    {k:25s} → {v.shape}  dtype={v.dtype}")

    # Verify masking happened
    n_masked = (batch["input_ids"] == mol_tokenizer.mask_token_id).sum().item()
    n_active = (batch["labels"] != -100).sum().item()
    print(f"  Masked tokens:     {n_masked}")
    print(f"  Active labels:     {n_active} (should ≈ masked: {n_masked})")
    assert n_masked == n_active, "Masked count != active label count"
    print(f"  ✅")

    # ================================================================= #
    # TEST 5: Model construction                                         #
    # ================================================================= #
    test_separator("MolecularDiffusionModel (with SciBERT)")

    from model.molecularDiffusionModel import MolecularDiffusionModel

    model = MolecularDiffusionModel(
        vocab_size      = mol_tokenizer.vocab_size,
        hidden_size     = 256,
        num_heads       = 8,
        ffn_dim         = 512,
        num_layers      = 2,       # small for smoke test
        max_length      = 74,
        pad_token_id    = mol_tokenizer.pad_token_id,
        text_model_name = "allenai/scibert_scivocab_uncased",
        uncond_prob     = 0.1,
        dropout         = 0.1,
    ).to(device)

    total, trainable, frozen = model.count_parameters()
    print(f"  Total params:      {total:,}")
    print(f"  Trainable params:  {trainable:,}")
    print(f"  Frozen params:     {frozen:,} (SciBERT)")

    # Verify SciBERT is frozen
    scibert_grad = any(p.requires_grad for p in model.text_encoder.parameters())
    assert not scibert_grad, "SciBERT should be frozen"
    print(f"  SciBERT frozen:    ✅")

    # Verify cross-attention zero-init
    block0 = model.blocks[0]
    cross_w = block0.cross_attention.out_proj.weight
    cross_b = block0.cross_attention.out_proj.bias
    assert cross_w.abs().max().item() == 0.0, "Cross-attn weight not zero"
    assert cross_b.abs().max().item() == 0.0, "Cross-attn bias not zero"
    print(f"  Cross-attn zero:   ✅")

    # ================================================================= #
    # TEST 6: Forward pass                                               #
    # ================================================================= #
    test_separator("Forward pass")

    model.train()
    input_ids = batch["input_ids"].to(device)
    timesteps = batch["timesteps"].to(device)
    text_ids  = batch["text_input_ids"].to(device)
    text_mask = batch["text_attention_mask"].to(device)
    text_pad  = batch["text_padding_mask"].to(device)

    # Get text embeddings
    text_embeds = model.get_text_embeddings(text_ids, text_mask)
    print(f"  text_embeds shape: {text_embeds.shape}")

    # Conditioned forward pass
    logits = model(input_ids, timesteps, text_embeds, text_pad)
    print(f"  logits shape:      {logits.shape}")
    assert logits.shape == (4, 74, mol_tokenizer.vocab_size)
    print(f"  Shape correct:     ✅")

    # Unconditional forward pass (null token)
    logits_uncond = model(input_ids, timesteps, None, None)
    print(f"  uncond logits:     {logits_uncond.shape}")
    assert logits_uncond.shape == logits.shape
    print(f"  Null token path:   ✅")

    # ================================================================= #
    # TEST 7: Loss + backward pass                                       #
    # ================================================================= #
    test_separator("Loss + backward")

    from training.finetune_text import diffusion_loss

    labels = batch["labels"].to(device)
    loss = diffusion_loss(logits, labels, timesteps)
    print(f"  Loss value:        {loss.item():.4f}")

    loss.backward()

    # Check gradient flow through key components
    #
    # IMPORTANT: cross-attention out_proj is zero-initialized.
    # This means out_proj.weight GETS gradient (the bootstrap signal),
    # but everything UPSTREAM (q_proj, k_proj, v_proj, text_proj,
    # null_token) gets ZERO gradient because:
    #   d_loss/d_attended = out_proj.weight^T @ d_loss/d_output = 0^T @ ... = 0
    #
    # After the first optimizer step, out_proj.weight becomes non-zero
    # and gradients flow upstream from step 2 onward.

    # 1. cross-attention out_proj.weight — THE bootstrap signal
    cross_out_grad = model.blocks[0].cross_attention.out_proj.weight.grad
    assert cross_out_grad is not None, "No gradient on cross-attention out_proj"
    assert cross_out_grad.abs().max().item() > 0, "out_proj gradient is zero"
    print(f"  cross_attn out_proj grad: {cross_out_grad.abs().mean().item():.6f}  ✅ (bootstrap)")

    # 2. text_proj — zero gradient expected (blocked by zero-init out_proj)
    proj_grad = model.text_proj[0].weight.grad
    if proj_grad is not None:
        print(f"  text_proj grad:    {proj_grad.abs().mean().item():.6f}  "
              f"(zero expected, will flow after step 1)")
    else:
        print(f"  text_proj grad:    None  (zero-init blocks upstream — expected)")
    print(f"  text_proj:         ✅")

    # 3. null_token — zero or None gradient expected (same reason + rare dropout)
    null_grad = model.null_token.grad
    if null_grad is not None:
        print(f"  null_token grad:   {null_grad.abs().mean().item():.6f}  "
              f"(zero expected)")
    else:
        print(f"  null_token grad:   None  (zero-init blocks upstream — expected)")
    print(f"  null_token:        ✅")

    # 4. Self-attention — MUST have non-zero gradient (main pathway)
    self_attn_grad = model.blocks[0].attention.q_proj.weight.grad
    assert self_attn_grad is not None, "No gradient on self-attention"
    assert self_attn_grad.abs().max().item() > 0, "Self-attention gradient is zero"
    print(f"  self_attn grad:    {self_attn_grad.abs().mean().item():.6f}  ✅")

    # 5. SciBERT — MUST have NO gradient (frozen)
    scibert_grad_any = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.text_encoder.parameters()
    )
    assert not scibert_grad_any, "SciBERT got gradients (should be frozen)"
    print(f"  SciBERT no grad:   ✅")

    # 6. Simulate one optimizer step, then check upstream gradients flow
    opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    opt.step()    # out_proj.weight becomes non-zero
    opt.zero_grad()

    # Second forward + backward — now gradients should flow upstream
    # Must recompute text_embeds (old graph was freed after first backward)
    text_embeds2 = model.get_text_embeddings(text_ids, text_mask)
    logits2 = model(input_ids, timesteps, text_embeds2, text_pad)
    loss2 = diffusion_loss(logits2, labels, timesteps)
    loss2.backward()

    proj_grad2 = model.text_proj[0].weight.grad
    assert proj_grad2 is not None and proj_grad2.abs().max().item() > 0, \
        "text_proj still has no gradient after out_proj bootstrap"
    print(f"  text_proj grad (after bootstrap): {proj_grad2.abs().mean().item():.6f}  ✅")

    model.zero_grad()

    # ================================================================= #
    # TEST 8: Generation with CFG                                        #
    # ================================================================= #
    test_separator("Generation with CFG")

    from training.finetune_text import generate_with_cfg
    import selfies as sf
    from rdkit import Chem

    test_prompts = [
        "A simple alcohol molecule.",
        "A molecule with a carbonyl group.",
    ]

    raw_ids = generate_with_cfg(
        model, mol_tokenizer, text_tokenizer, device,
        test_prompts,
        cfg_scale=2.0,
        num_steps=8,       # very few steps for speed
        temperature=1.0,
    )

    print(f"  Generated shape:   {raw_ids.shape}")

    for i, prompt in enumerate(test_prompts):
        ids = raw_ids[i].cpu().tolist()
        if mol_tokenizer.eos_token_id in ids:
            ids = ids[:ids.index(mol_tokenizer.eos_token_id)]
        selfies_str = mol_tokenizer.decode(ids)
        try:
            smiles = sf.decoder(selfies_str)
            mol = Chem.MolFromSmiles(smiles)
            status = f"✅ {Chem.MolToSmiles(mol)}" if mol else f"❌ invalid"
        except Exception as e:
            status = f"❌ {e}"
        print(f"  [{i}] {status}")
        print(f"       Prompt: {prompt}")

    print(f"  (quality irrelevant — untrained model, 8 steps)")
    print(f"  CFG pipeline:      ✅")

    # ================================================================= #
    # TEST 9: Checkpoint loading with strict=False (optional)            #
    # ================================================================= #
    if args.checkpoint and Path(args.checkpoint).exists():
        test_separator("Checkpoint loading (strict=False)")

        from training.finetune_text import load_pretrained

        # Build fresh model to load into
        model2 = MolecularDiffusionModel(
            vocab_size      = mol_tokenizer.vocab_size,
            hidden_size     = 256,
            num_heads       = 8,
            ffn_dim         = 512,
            num_layers      = 8,     # must match checkpoint
            max_length      = 74,
            pad_token_id    = mol_tokenizer.pad_token_id,
            text_model_name = "allenai/scibert_scivocab_uncased",
            uncond_prob     = 0.1,
            dropout         = 0.0,
        ).to(device)

        load_pretrained(model2, args.checkpoint, device)

        # Verify cross-attention is still zero after loading
        cross_w = model2.blocks[0].cross_attention.out_proj.weight
        assert cross_w.abs().max().item() == 0.0, "Cross-attn got overwritten"
        print(f"  Cross-attn still zero after load: ✅")

        # Verify self-attention loaded non-zero weights
        self_w = model2.blocks[0].attention.q_proj.weight
        assert self_w.abs().max().item() > 0.01, "Self-attn weights look random"
        print(f"  Self-attn loaded real weights:    ✅")

    else:
        test_separator("Checkpoint loading (SKIPPED)")
        if args.checkpoint:
            print(f"  File not found: {args.checkpoint}")
        else:
            print(f"  No --checkpoint provided, skipping")
        print(f"  Run with: python scripts/smoke_test.py --checkpoint checkpoints/best_model.pt")

    # ================================================================= #
    # SUMMARY                                                            #
    # ================================================================= #
    print(f"\n{'='*60}")
    print(f"  ALL TESTS PASSED ✅")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test for text conditioning")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional: path to pretrained revealPad checkpoint")
    args = parser.parse_args()
    main(args)