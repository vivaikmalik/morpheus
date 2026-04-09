"""
Text-conditioned MaskGIT generation with CFG and per-step log-prob accumulation.

Gradients flow through the CFG-combined logits so the accumulated log-probs
are differentiable w.r.t. model parameters — suitable for REINFORCE.
"""

import torch
import torch.nn.functional as F


def generate_batch_cfg_with_log_probs(
    model, tokenizer, device, text_embeds, text_padding_mask,
    model_config, num_steps=15, temperature=0.9, cfg_scale=3.0,
):
    """
    MaskGIT decoding with classifier-free guidance and per-step log-prob
    accumulation.

    At each denoising step:
      1. Conditional forward pass   → cond_logits
      2. Unconditional forward pass → uncond_logits  (null token)
      3. CFG combination: uncond + cfg_scale * (cond - uncond)
      4. Sample from the CFG distribution
      5. Confidence-based re-masking (except last step)
      6. Accumulate log P(token) for positions locked in at this step

    Parameters
    ----------
    model           : MolecularDiffusionModel  (train mode, gradients enabled)
    tokenizer       : ChemicalTokenizer
    device          : torch.device
    text_embeds     : (B, T, D) pre-computed text embeddings
    text_padding_mask : (B, T) bool, True = padding
    model_config    : dict containing at least "max_length"
    num_steps       : int  — number of MaskGIT denoising steps
    temperature     : float — sampling temperature
    cfg_scale       : float — classifier-free guidance scale

    Returns
    -------
    input_ids       : (B, L) LongTensor  — final generated token IDs
    total_log_probs : (B,)   FloatTensor — accumulated log-prob **with gradients**
    """
    max_length = model_config["max_length"]
    B = text_embeds.shape[0]

    input_ids = torch.full(
        (B, max_length), tokenizer.mask_token_id,
        dtype=torch.long, device=device,
    )

    null_embeds = model.null_token.expand(B, text_embeds.size(1), -1)

    t_vals = torch.linspace(1.0, 0.0, num_steps, device=device)
    total_log_probs = torch.zeros(B, device=device)

    for step_idx, t_val in enumerate(t_vals):
        is_last_step = (step_idx == num_steps - 1)
        was_masked = (input_ids == tokenizer.mask_token_id)  # (B, L)

        step_t = t_val.repeat(B).unsqueeze(-1)  # (B, 1)

        # ── CFG: two forward passes ──────────────────────────────────
        cond_logits = model(input_ids, step_t, text_embeds, text_padding_mask)
        uncond_logits = model(input_ids, step_t, null_embeds, text_padding_mask)
        cfg_logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)

        # Never output the MASK token
        cfg_logits[:, :, tokenizer.mask_token_id] = float("-inf")

        scaled_logits = cfg_logits / temperature
        log_p = F.log_softmax(scaled_logits, dim=-1)  # (B, L, V) — has grad
        probs = torch.exp(log_p)

        # ── Sample ───────────────────────────────────────────────────
        sampled = torch.distributions.Categorical(probs=probs).sample()  # (B, L)

        if is_last_step:
            newly_locked = was_masked
            token_lp = torch.gather(
                log_p, 2, sampled.unsqueeze(-1)
            ).squeeze(-1)
            total_log_probs = total_log_probs + \
                token_lp.masked_fill(~newly_locked, 0.0).sum(dim=-1)
            input_ids = sampled
            break

        # ── Confidence-based re-masking ──────────────────────────────
        confidence = torch.gather(
            probs, 2, sampled.unsqueeze(-1)
        ).squeeze(-1)  # (B, L)

        alpha_t = (torch.cos(t_val * torch.pi / 2) ** 2).item()
        num_to_mask = int((1.0 - alpha_t) * max_length)

        if num_to_mask > 0:
            _, mask_indices = torch.topk(
                confidence,
                k=min(num_to_mask, max_length),
                dim=-1,
                largest=False,
            )
            sampled.scatter_(1, mask_indices, tokenizer.mask_token_id)

        # ── Accumulate log-probs for newly locked positions ──────────
        newly_locked = was_masked & (sampled != tokenizer.mask_token_id)
        token_lp = torch.gather(
            log_p, 2, sampled.unsqueeze(-1)
        ).squeeze(-1)
        total_log_probs = total_log_probs + \
            token_lp.masked_fill(~newly_locked, 0.0).sum(dim=-1)

        input_ids = sampled

    return input_ids, total_log_probs  # (B, L), (B,) with grad


def generate_batch_cfg_no_grad(
    model, tokenizer, device, text_embeds, text_padding_mask,
    model_config, num_steps=32, temperature=0.9, cfg_scale=3.0,
):
    """
    Same as above but wrapped in torch.no_grad() for evaluation.
    Returns only the generated token IDs.
    """
    max_length = model_config["max_length"]
    B = text_embeds.shape[0]

    input_ids = torch.full(
        (B, max_length), tokenizer.mask_token_id,
        dtype=torch.long, device=device,
    )

    null_embeds = model.null_token.expand(B, text_embeds.size(1), -1)
    t_vals = torch.linspace(1.0, 0.0, num_steps, device=device)

    with torch.no_grad():
        for step_idx, t_val in enumerate(t_vals):
            step_t = t_val.repeat(B).unsqueeze(-1)

            cond_logits = model(input_ids, step_t, text_embeds, text_padding_mask)
            uncond_logits = model(input_ids, step_t, null_embeds, text_padding_mask)
            logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
            logits[:, :, tokenizer.mask_token_id] = float("-inf")

            probs = torch.softmax(logits / temperature, dim=-1)
            sampled = torch.distributions.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, 2, sampled.unsqueeze(-1)).squeeze(-1)

            alpha_t = (torch.cos(t_val * torch.pi / 2) ** 2).item()
            num_to_mask = int((1.0 - alpha_t) * max_length)

            if num_to_mask > 0 and step_idx < num_steps - 1:
                _, mask_indices = torch.topk(
                    confidence, min(num_to_mask, max_length),
                    dim=-1, largest=False,
                )
                sampled.scatter_(1, mask_indices, tokenizer.mask_token_id)

            input_ids = sampled

    return input_ids  # (B, L)
