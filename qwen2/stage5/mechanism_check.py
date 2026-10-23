"""
Stage 5 mechanism check (H4).

target_token_jacobian_alignment: unlike Stage 2's jacobian_magnitude
(which measures sensitivity of the TOP predicted token on generic
background text), this measures the directional derivative of the
TARGET LABEL's logprob specifically, at the actual elicitation
prompt -- the precise thing training was trying to increase. Compared
before vs. after training via peft_model.disable_adapter(), which was
verified to correctly reflect the trained contribution (confirmed
against a real trained adapter before building this: enabled vs.
disabled outputs were identical pre-training, and measurably
different -- max diff 0.79 -- after 20 training steps).

ablation_test operationalizes "ablate the direction" as clamping its
component to exactly zero (apply_direction_editing(..., alpha=0.0)),
matching standard mechanistic-interpretability usage: an active null
intervention on that specific component, not simply skipping the
hook and leaving the activation's natural (unknown) value in place.
"ablate the input" replaces the input-content sentence with the bare
elicitation prompt -- no concept-bearing text at all -- while the
direction stays clamped normally.
"""

import torch
from typing import List, Optional

# --- make stage2/ and stage3/ importable from here ---
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage2"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage3"))

from elicitation import elicit_reports
from scoring import exact_match_score
from decorrelation import build_input_content

# --- make stage1/ importable (model_utils) ---
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage1"))
from model_utils import get_transformer_layers, format_prompt_for_chat


def target_token_jacobian_alignment(
    model, tokenizer, layer: int, direction: torch.Tensor, prompt: str, target_label: str, device: str,
    use_chat_template: bool = True,
) -> float:
    d_hat = (direction / direction.norm()).to(device)
    target_id = tokenizer.encode(" " + target_label.strip(), add_special_tokens=False)[0]

    formatted_prompt = format_prompt_for_chat(tokenizer, prompt, use_chat_template)
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    captured = {}

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h.requires_grad_(True)  # force grad-tracking from this point forward -- inside
        # peft_model.disable_adapter(), every parameter is frozen (base model frozen by
        # PEFT, LoRA contribution bypassed), so nothing upstream has requires_grad=True
        # and h would otherwise never be part of a tracked graph at all.
        h.retain_grad()
        captured["h"] = h
        return out

    handle = get_transformer_layers(model)[layer].register_forward_hook(hook)
    try:
        logits = model(**inputs).logits[0, -1]
        logprob = torch.log_softmax(logits, dim=-1)[target_id]
        model.zero_grad(set_to_none=True)
        logprob.backward()
        grad = captured["h"].grad[0, -1].detach().float()
        return torch.dot(grad, d_hat.float()).item()
    finally:
        handle.remove()


def jacobian_alignment_pre_post(peft_model, tokenizer, layer: int, direction: torch.Tensor,
                                 prompt: str, target_label: str, device: str) -> dict:
    after = target_token_jacobian_alignment(peft_model, tokenizer, layer, direction, prompt, target_label, device)
    with peft_model.disable_adapter():
        before = target_token_jacobian_alignment(peft_model, tokenizer, layer, direction, prompt, target_label, device)
    return {"before_training": before, "after_training": after, "delta": after - before}


def ablation_test(
    peft_model, tokenizer, layer: int, direction: torch.Tensor, alpha: float,
    x_concept: str, elicitation_prompt: str, device: str, n_trials: int = 3, max_new_tokens: int = 10,
) -> dict:
    normal_prompt = f"{build_input_content(x_concept)}\n{elicitation_prompt}"
    ablated_input_prompt = elicitation_prompt  # no concept-bearing input content at all

    def rate(direction_alpha, prompt):
        reports = elicit_reports(peft_model, tokenizer, layer, direction, direction_alpha, prompt, device,
                                  n_trials=n_trials, max_new_tokens=max_new_tokens)
        scores = [s for s in (exact_match_score(r, x_concept) for r in reports) if s is not None]
        return (sum(scores) / len(scores) if scores else None), reports

    baseline_rate, baseline_reports = rate(alpha, normal_prompt)
    ablate_direction_rate, ablate_direction_reports = rate(0.0, normal_prompt)     # direction zeroed, input intact
    ablate_input_rate, ablate_input_reports = rate(alpha, ablated_input_prompt)    # direction intact, input removed

    return {
        "x_concept": x_concept, "layer": layer, "alpha": alpha,
        "baseline": {"match_rate": baseline_rate, "reports": baseline_reports},
        "ablate_direction": {"match_rate": ablate_direction_rate, "reports": ablate_direction_reports},
        "ablate_input": {"match_rate": ablate_input_rate, "reports": ablate_input_reports},
        "direction_necessary": (baseline_rate is not None and ablate_direction_rate is not None
                                 and ablate_direction_rate < baseline_rate),
        "input_necessary": (baseline_rate is not None and ablate_input_rate is not None
                             and ablate_input_rate < baseline_rate),
    }
