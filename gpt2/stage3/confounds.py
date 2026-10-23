"""
Stage 3 confound controls.

(b) co-activated correlates: the bipolar direction d_XY might just BE
(or strongly overlap with) the input-token embeddings for X/Y, or
whatever SAE features happen to co-occur with it -- in which case any
apparent causal effect could be mediated through those, not d_XY
itself. Gram-Schmidt orthogonalizes d_XY against both before use.

(c) unembedding triviality: if a direction's causal effect on the
report is no better than what you'd get by just reading off its
closest vocabulary token via the unembedding matrix (skipping all
intermediate computation), that's evidence the "report" isn't
reflecting genuine processing -- just the direction's raw logit-lens
alignment. logit_lens_report() computes that trivial baseline
directly, for comparison against the real elicited reports.
"""

import torch
from typing import Dict, List, Optional

# --- make stage1/ importable from here ---
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage1"))

from model_utils import get_transformer_layers


def gram_schmidt_orthogonalize(direction: torch.Tensor, confounds: List[torch.Tensor]) -> torch.Tensor:
    """
    Project `direction` to be orthogonal to the ENTIRE SPAN of
    `confounds`, then re-normalize.

    Built via an orthonormal basis (QR) of the confound vectors
    rather than naive sequential subtraction: subtracting each raw
    confound's projection one at a time only guarantees exact
    orthogonality to the last one subtracted, since arbitrary
    confound vectors aren't orthogonal to each other -- subtracting
    the second can reintroduce a small component along the first.
    Projecting onto an orthonormal basis of their span avoids that.
    """
    d = direction.clone().float()
    if not confounds:
        return d / d.norm()

    device = d.device
    C = torch.stack([c.float().to(device) for c in confounds], dim=1)  # [d_model, n_confounds]
    Q, _ = torch.linalg.qr(C)                                 # orthonormal basis for span(confounds)
    d_orth = d - Q @ (Q.T @ d)

    norm = d_orth.norm()
    if norm < 1e-6:
        raise ValueError(
            "Direction collapsed to ~zero after orthogonalization -- it was almost "
            "entirely explained by the confound vectors. This is itself a finding "
            "worth reporting, not just an error to work around."
        )
    return d_orth / norm


def get_input_embedding_vectors(model, tokenizer, tokens: List[str]) -> List[torch.Tensor]:
    """Input-embedding (not unembedding) rows for the given tokens -- the confound-(b) 'is this
    direction just the word embedding' check."""
    W_E = model.get_input_embeddings().weight.detach().float()
    vecs = []
    for tok in tokens:
        tok_id = tokenizer.encode(tok, add_special_tokens=False)[0]
        vecs.append(W_E[tok_id])
    return vecs


def get_top_cofiring_sae_features(
    direction: torch.Tensor, sae_vectors: Dict[str, torch.Tensor], k: int = 5
) -> List[torch.Tensor]:
    """
    Top-k SAE decoder vectors (from this layer's slice of the Stage 1
    bank) by cosine similarity to `direction` -- a cheap static proxy
    for "features that tend to co-activate with this direction",
    since it doesn't require live activation data to compute.
    """
    d_hat = direction.float() / direction.float().norm()
    scored = []
    for key, vec in sae_vectors.items():
        v = vec.float().to(d_hat.device)
        cos = torch.dot(d_hat, v / v.norm().clamp_min(1e-8)).abs().item()
        scored.append((cos, vec))
    scored.sort(key=lambda x: -x[0])
    return [vec.to(d_hat.device) for _, vec in scored[:k]]


def logit_lens_report(model, tokenizer, direction: torch.Tensor, k: int = 3) -> List[str]:
    """
    Trivial baseline: project `direction` straight through the
    unembedding matrix (skipping all intermediate layers/computation)
    and return the top-k closest vocabulary tokens by cosine
    similarity. This is what confound (c) compares the real elicited
    reports against.
    """
    W_U = model.get_output_embeddings().weight.detach().float()
    d_hat = direction.float().to(W_U.device)
    d_hat = d_hat / d_hat.norm()
    row_norms = W_U.norm(dim=1).clamp_min(1e-8)
    cos_sims = (W_U @ d_hat) / row_norms
    top_ids = torch.topk(cos_sims, k=k).indices.tolist()
    return [tokenizer.decode([i]).strip() for i in top_ids]
