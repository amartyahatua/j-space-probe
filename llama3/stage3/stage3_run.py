"""
Stage 3 entry point: for each (X, Y) concept pair and each layer,
build the bipolar J-space direction, run the two conflict cells
(confound (a) handled structurally by decorrelation.build_cell_prompt
-- injection-then-report, no prior model-generated text to leak
from), then repeat with the direction Gram-Schmidt-orthogonalized
against input embeddings + top co-firing SAE features (confound (b)),
and compute a logit-lens baseline report (confound (c)) for
comparison. Saves everything to stage3_results.json.

Usage:
    python stage3_run.py --dry_run
    python stage3_run.py --bank_dir ../stage1/direction_bank

IMPORTANT CAVEAT carried over from Stage 2: H1's calibration work
found that free-text generation on GPT-2-small (and Qwen2-0.5B-
Instruct) doesn't reliably produce reports that literally name a
clamped concept, across an extensive alpha/prompt/model sweep. Since
this stage's direction-following index depends on the SAME exact-
match-on-generated-text mechanism, it's very possible DFI comes back
near-zero/uninformative for the same underlying reason, not because
H3 is false. Worth knowing going in -- run it and look at the actual
report text (same way we diagnosed H1) before concluding anything
about the decorrelation itself.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage1"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage2"))

from config import CFG
from build_direction_bank import load_real_model

from elicitation import elicit_reports
from decorrelation import build_bipolar_direction, build_cell_prompt, direction_following_index
from confounds import gram_schmidt_orthogonalize, get_input_embedding_vectors, get_top_cofiring_sae_features, logit_lens_report


DEFAULT_PAIRS = [("Paris", "Rome"), ("dog", "cat"), ("sad", "happy"), ("France", "Italy")]
# Every label appears in exactly ONE pair -- a real training run on Qwen2 showed that
# "happy" repeating across (dog,happy) AND (sad,happy) gave it 2x the training-gradient
# frequency of every other label, and the adapter mode-collapsed onto "happy" as a
# default output regardless of the actual clamped direction. "dog"/"cat" isn't a
# semantically "natural" pair any more than "dog"/"happy" was -- these bipolar pairs
# were never meant to be semantically related, just balanced in training frequency.


def load_bank(bank_dir: str):
    vectors = torch.load(os.path.join(bank_dir, "vectors.pt"))
    metadata = json.load(open(os.path.join(bank_dir, "metadata.json")))
    return vectors, metadata


def run_pair(model, tokenizer, vectors, metadata, layer, x, y, alpha, elicitation_prompt, n_trials, max_new_tokens, device):
    key_x = f"jspace::{layer}::{x}"
    key_y = f"jspace::{layer}::{y}"
    if key_x not in vectors or key_y not in vectors:
        return None  # this pair/layer combo isn't in the bank -- skip rather than error

    vec_x, vec_y = vectors[key_x], vectors[key_y]
    d_xy = build_bipolar_direction(vec_x, vec_y)

    sae_vectors = {k: v for k, v in vectors.items() if metadata[k]["class"] == "sae" and metadata[k]["layer"] == layer}
    input_embeds = get_input_embedding_vectors(model, tokenizer, [x, y])
    top_cofiring = get_top_cofiring_sae_features(d_xy, sae_vectors, k=5) if sae_vectors else []
    d_orth = gram_schmidt_orthogonalize(d_xy, input_embeds + top_cofiring)

    result = {"layer": layer, "x": x, "y": y, "alpha": alpha}

    for tag, direction in (("raw", d_xy), ("orthogonalized", d_orth)):
        prompt_input_y = build_cell_prompt(y, elicitation_prompt)  # direction=X, input=Y
        prompt_input_x = build_cell_prompt(x, elicitation_prompt)  # direction=Y, input=X

        reports_dirX_inputY = elicit_reports(model, tokenizer, layer, direction, alpha, prompt_input_y, device,
                                              n_trials=n_trials, max_new_tokens=max_new_tokens)
        reports_dirY_inputX = elicit_reports(model, tokenizer, layer, direction, -alpha, prompt_input_x, device,
                                              n_trials=n_trials, max_new_tokens=max_new_tokens)

        dfi = direction_following_index(reports_dirX_inputY, reports_dirY_inputX, x, y)
        result[tag] = {
            "reports_dirX_inputY": reports_dirX_inputY,
            "reports_dirY_inputX": reports_dirY_inputX,
            **dfi,
            "logit_lens_top_tokens": logit_lens_report(model, tokenizer, direction, k=3),
        }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--bank_dir", default="../stage1/direction_bank")
    parser.add_argument("--out_dir", default="stage3_results")
    parser.add_argument("--alpha", type=float, default=None,
                         help="Override: use this ONE alpha for every layer instead of the "
                              "calibrated per-layer CFG.stage2_alphas lookup. Only for quick "
                              "exploration -- a single alpha spanning layers 3-27 reproduces "
                              "exactly the miscalibration the Stage 2 alpha_sweep.py calibration "
                              "pass fixed (layers 3/7 need ~1.5, layers 19-27 need ~6-6.5; no "
                              "single value is coherent at one end without being inert at the "
                              "other). Leave unset for a real run.")
    parser.add_argument("--n_trials", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--pairs", type=str, default=None,
                         help='Override DEFAULT_PAIRS, e.g. \'[["Paris","Rome"],["dog","happy"]]\'')
    args = parser.parse_args()

    device = CFG.device if torch.cuda.is_available() else "cpu"
    pairs = json.loads(args.pairs) if args.pairs else DEFAULT_PAIRS
    pairs = [tuple(p) for p in pairs]

    if args.dry_run:
        from transformers import GPT2LMHeadModel, GPT2Config
        from tests.mock_model import MockGenerativeTokenizer

        gpt2_cfg = GPT2Config(n_layer=5, n_embd=32, n_head=2, vocab_size=200)
        model = GPT2LMHeadModel(gpt2_cfg).to(device)
        model.eval()
        tokenizer = MockGenerativeTokenizer(vocab_size=200)

        d = 32
        vectors = {}
        metadata = {}
        for layer in [2, 4]:
            for tok in ["Paris", "Rome", "dog", "happy", "sad", "France", "Italy"]:
                vectors[f"jspace::{layer}::{tok}"] = torch.randn(d)
                metadata[f"jspace::{layer}::{tok}"] = {"layer": layer, "class": "jspace"}
            for i in range(10):
                vectors[f"sae::{layer}::{i}"] = torch.randn(d)
                metadata[f"sae::{layer}::{i}"] = {"layer": layer, "class": "sae"}

        pairs = pairs[:2]
        out_dir = "stage3_results_dryrun"
        n_trials, max_new_tokens = 2, 6
        # Mock bank uses synthetic layers [2, 4], which aren't in CFG.stage2_alphas (that dict is
        # keyed by the REAL bank's layers 3/7/19/23/27) -- a per-layer lookup would just KeyError
        # here for no good reason, so the dry run always uses a fixed placeholder alpha regardless
        # of --alpha. This path never touches real weights/results, so it doesn't need to be
        # calibration-accurate, just fast and non-crashing.
        alpha_for_layer = lambda layer: 4.0
    else:
        model, tokenizer = load_real_model(CFG.model_name, device, CFG.dtype)
        vectors, metadata = load_bank(args.bank_dir)
        out_dir = args.out_dir
        n_trials, max_new_tokens = args.n_trials, args.max_new_tokens

        def alpha_for_layer(layer):
            if args.alpha is not None:
                return args.alpha
            try:
                return CFG.stage2_alphas[layer]
            except KeyError:
                raise KeyError(
                    f"No calibrated stage2_alphas entry for layer {layer!r}. If CFG.sae_layers "
                    f"gained a new layer, run alpha_sweep.py for it and add the entry to "
                    f"CFG.stage2_alphas before trusting a real Stage 3 run -- or pass --alpha to "
                    f"force a single value across all layers (not recommended, see --help)."
                )

    layers = sorted({m["layer"] for m in metadata.values() if m["class"] == "jspace"})
    results = []
    total = len(pairs) * len(layers)
    i = 0
    for (x, y) in pairs:
        for layer in layers:
            i += 1
            alpha = alpha_for_layer(layer)
            r = run_pair(model, tokenizer, vectors, metadata, layer, x, y, alpha,
                         CFG.stage2_prompt, n_trials, max_new_tokens, device)
            if r is None:
                print(f"  [{i}/{total}] ({x},{y}) layer={layer}: skipped (not in bank)")
                continue
            results.append(r)
            raw_dfi = r["raw"]["dfi_overall"]
            orth_dfi = r["orthogonalized"]["dfi_overall"]
            print(f"  [{i}/{total}] ({x},{y}) layer={layer} alpha={alpha}: "
                  f"raw_dfi={raw_dfi} orth_dfi={orth_dfi} "
                  f"logit_lens={r['raw']['logit_lens_top_tokens']}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "stage3_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved stage3_results.json to {out_dir}/")


if __name__ == "__main__":
    main()