"""
Stage 1 configuration: model, SAE, and direction-extraction settings.

Edit `model_name` / `sae_release` to switch between the primary model
and the replication model called for in Section 5.1 of the proposal
(e.g. Llama-3.1-8B primary + Gemma-2-9B replication).
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # --- Model ---
    model_name: str = "meta-llama/Llama-3.1-8B"          # primary model
    replication_model_name: str = "google/gemma-2-9b"    # replication model
    device: str = "cuda"                                  # falls back to "cpu" automatically if unavailable
    dtype: str = "bfloat16"

    # --- SAE (via sae_lens; release/id naming follows SAELens conventions) ---
    sae_release: str = "llama_scope_lxr_32x"              # Llama-Scope, 32x expansion
    sae_layers: List[int] = field(default_factory=lambda: [8, 12, 16, 20, 24])

    # --- Direction extraction counts (per layer, unless noted) ---
    n_sae_features_per_layer: int = 40                    # stratified by activation-frequency decile
    n_probe_attributes: int = 20                           # number of CounterFact relations used as probes
    n_random_directions_per_layer: int = 40                # fully random
    n_random_in_span_per_layer: int = 40                   # random within SAE decoder span

    # --- J-lens ---
    jlens_n_contexts: int = 2000                           # contexts averaged over per (token, layer)
    jlens_target_tokens: List[str] = field(default_factory=lambda: [
        "Paris", "Rome", "dog", "happy", "sad", "France", "Italy",
        # extend with whatever concept vocabulary Stage 2/3 will actually test
    ])

    # --- Causal efficacy scoring (steering) ---
    steer_alphas: List[float] = field(default_factory=lambda: [4.0, 8.0, 12.0])
    causal_eval_n_contexts: int = 200

    # --- Output ---
    output_dir: str = "direction_bank"


CFG = Config()
