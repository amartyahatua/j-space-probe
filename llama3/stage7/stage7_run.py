"""
Stage 7 entry point: builds the covert-cognition (prompt-injection)
direction from scratch, trains a FRESH adapter on it using Stage 4's
exact correlated-only recipe (on several varied phrasings per class,
not one fixed sentence -- see covert_cognition.py), then runs
decorrelation on HELD-OUT phrasings never seen during training, plus
the full mechanism check.

A fresh adapter (not a continuation of Stage 4's) is used deliberately
-- Paris/Rome/dog/happy/sad are semantically unrelated to injection
detection, so warm-starting from that adapter would confound whatever
this test finds with whatever that one already learned.

Usage:
    python stage7_run.py --dry_run
    python stage7_run.py --bank_dir ../stage1/direction_bank
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage1"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage2"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage3"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage4"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage5"))

from config import CFG
from build_direction_bank import load_real_model
from jlens import build_jspace_directions
from decorrelation import build_bipolar_direction, direction_following_index
from lora_training import setup_lora_model, train_stage4
from mechanism_check import jacobian_alignment_pre_post
from elicitation import elicit_reports
from scoring import exact_match_score

from covert_cognition import (
    ELICITATION_PROMPT, LABEL_INJECTION, LABEL_SAFE, TARGET_TOKENS,
    build_cell_prompt, scenario_texts,
)
from mechanism import ablation_test


def build_direction(model, tokenizer, layer: int, contexts, device: str) -> torch.Tensor:
    jspace = build_jspace_directions(model, tokenizer, [layer], list(TARGET_TOKENS), contexts, device)
    return build_bipolar_direction(jspace[(layer, LABEL_INJECTION)], jspace[(layer, LABEL_SAFE)])


def train_covert_cognition_adapter(model, tokenizer, layer, direction, alpha, device, n_epochs, lr, target_modules):
    examples = []
    for i in range(len(scenario_texts(LABEL_INJECTION, "train"))):
        examples.append({"layer": layer, "direction": direction, "alpha": alpha,
                          "prompt": build_cell_prompt(LABEL_INJECTION, split="train", index=i),
                          "target_label": LABEL_INJECTION})
    for i in range(len(scenario_texts(LABEL_SAFE, "train"))):
        examples.append({"layer": layer, "direction": direction, "alpha": -alpha,
                          "prompt": build_cell_prompt(LABEL_SAFE, split="train", index=i),
                          "target_label": LABEL_SAFE})

    peft_model = setup_lora_model(model, target_modules, r=8, lora_alpha=16)
    history = train_stage4(peft_model, tokenizer, examples, device, n_epochs=n_epochs, lr=lr)
    return peft_model, history


def run_decorrelation(peft_model, tokenizer, layer, direction, alpha, device, n_trials, max_new_tokens):
    # both cells use HELD-OUT phrasings -- never seen during training at all, so this tests
    # generalization to unseen scenario text, not just recall of the exact training sentences
    prompt_cell1 = build_cell_prompt(LABEL_SAFE, split="heldout")       # direction=injection, scenario=benign
    reports_dirInjection_inputSafe = elicit_reports(peft_model, tokenizer, layer, direction, alpha,
                                                      prompt_cell1, device, n_trials=n_trials, max_new_tokens=max_new_tokens)
    prompt_cell2 = build_cell_prompt(LABEL_INJECTION, split="heldout")  # direction=safe, scenario=injection
    reports_dirSafe_inputInjection = elicit_reports(peft_model, tokenizer, layer, direction, -alpha,
                                                      prompt_cell2, device, n_trials=n_trials, max_new_tokens=max_new_tokens)

    dfi = direction_following_index(reports_dirInjection_inputSafe, reports_dirSafe_inputInjection,
                                     LABEL_INJECTION, LABEL_SAFE)
    return {
        "reports_dirInjection_inputSafe": reports_dirInjection_inputSafe,
        "reports_dirSafe_inputInjection": reports_dirSafe_inputInjection,
        **dfi,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--bank_dir", default="../stage1/direction_bank")
    parser.add_argument("--layer", type=int, default=19)  # mid-point of CFG.sae_layers ([3,7,11,15,19,23,27]) for
                                                             # llama-3.1-8b-instruct's 32 layers
    parser.add_argument("--alpha", type=float, default=None,
                         help="Defaults to CFG.stage2_alphas[--layer] -- the calibrated per-layer value "
                              "from the alpha_sweep.py calibration pass (see stage1/config.py). Since "
                              "this stage only ever runs at ONE layer per invocation (--layer, default "
                              "19), a plain per-layer dict lookup is enough here -- no multi-layer loop "
                              "like Stage 4/5, so no need to resolve alpha per-example.")
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3,
                         help="Lowered from 5e-3 -- that value caused severe training instability in Stage 4 "
                              "on this same model before being fixed. Applying the same fix here preemptively.")
    parser.add_argument("--target_modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"],
                         help="LoRA target module names. Qwen2/Llama/Gemma-2 family (default) vs "
                              "GPT-2 family: ['c_attn'].")
    parser.add_argument("--n_contexts", type=int, default=30)
    parser.add_argument("--n_trials", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--out_dir", default="stage7_results")
    args = parser.parse_args()
    if args.alpha is None:
        if args.dry_run:
            # dry_run's mock bank doesn't use the real sae_layers, so the calibrated dict
            # (keyed by 3/7/19/23/27) may not cover --layer's default -- use a fixed
            # placeholder alpha for the dry run regardless (see stage3/stage4's dry_run
            # paths for the same reasoning).
            args.alpha = 4.0
        else:
            try:
                args.alpha = CFG.stage2_alphas[args.layer]
            except KeyError:
                raise KeyError(
                    f"No calibrated stage2_alphas entry for layer {args.layer!r}. Run "
                    f"alpha_sweep.py for this layer and add it to CFG.stage2_alphas, or pass "
                    f"--alpha explicitly to override."
                )

    device = CFG.device if torch.cuda.is_available() else "cpu"

    if args.dry_run:
        from transformers import GPT2LMHeadModel, GPT2Config
        from tests.mock_model import MockGenerativeTokenizer

        gpt2_cfg = GPT2Config(n_layer=5, n_embd=32, n_head=2, vocab_size=200)
        model = GPT2LMHeadModel(gpt2_cfg).to(device)
        tokenizer = MockGenerativeTokenizer(vocab_size=200)
        contexts = [f"Sample background context number {i}." for i in range(8)]
        layer, n_epochs, n_trials, max_new_tokens = 2, 5, 2, 5
        out_dir = "stage7_results_dryrun"
        # dry_run always tests against a hardcoded GPT-2 architecture (no download needed) --
        # GPT-2 uses a fused c_attn Conv1D, not separate q_proj/k_proj/v_proj/o_proj Linears
        args.target_modules = ["c_attn"]
    else:
        model, tokenizer = load_real_model(CFG.model_name, device, CFG.dtype)
        from data import load_context_corpus
        contexts = load_context_corpus(args.n_contexts)
        layer, n_epochs, n_trials, max_new_tokens = args.layer, args.n_epochs, args.n_trials, args.max_new_tokens
        out_dir = args.out_dir

    print(f"Building injection-detection direction at layer {layer} from {len(contexts)} contexts...")
    direction = build_direction(model, tokenizer, layer, contexts, device)

    n_train_examples = len(scenario_texts(LABEL_INJECTION, "train")) + len(scenario_texts(LABEL_SAFE, "train"))
    print(f"Training fresh adapter on {n_train_examples} correlated-condition examples "
          f"({n_epochs} epochs = {n_epochs * n_train_examples} total steps)...")
    peft_model, history = train_covert_cognition_adapter(
        model, tokenizer, layer, direction, args.alpha, device, n_epochs, args.lr, args.target_modules
    )

    print("Post-training sanity check (correlated condition, TRAIN phrasings, should now be high):")
    for label in (LABEL_INJECTION, LABEL_SAFE):
        sign = 1.0 if label == LABEL_INJECTION else -1.0
        reports = elicit_reports(peft_model, tokenizer, layer, direction, sign * args.alpha,
                                  build_cell_prompt(label, split="train"), device,
                                  n_trials=n_trials, max_new_tokens=max_new_tokens)
        scores = [s for s in (exact_match_score(r, label) for r in reports) if s is not None]
        rate = sum(scores) / len(scores) if scores else None
        print(f"  target='{label}': match_rate={rate}  e.g. {reports[0]!r}")

    print("\nRunning decorrelation test on HELD-OUT phrasings (the actual question this stage asks)...")
    decorrelation_result = run_decorrelation(peft_model, tokenizer, layer, direction, args.alpha,
                                              device, n_trials, max_new_tokens)
    print(f"  DFI overall: {decorrelation_result['dfi_overall']}")
    print(f"  P(report=injection | direction=injection): {decorrelation_result['p_dirX_matches_X']}")
    print(f"  P(report=safe | direction=injection, scenario=benign) [input following]: {decorrelation_result['p_dirX_matches_Y_input']}")

    print("\nRunning mechanism check (H4) -- on TRAIN phrasings first...")
    jac = jacobian_alignment_pre_post(peft_model, tokenizer, layer, direction,
                                       build_cell_prompt(LABEL_INJECTION, split="train"), LABEL_INJECTION, device)
    ab_injection_train = ablation_test(peft_model, tokenizer, layer, direction, args.alpha, LABEL_INJECTION,
                                        device, n_trials=n_trials, max_new_tokens=max_new_tokens, split="train")
    ab_safe_train = ablation_test(peft_model, tokenizer, layer, direction, args.alpha, LABEL_SAFE,
                                   device, n_trials=n_trials, max_new_tokens=max_new_tokens, split="train")

    print("Running mechanism check (H4) -- on HELD-OUT phrasings (the regime that actually matters, "
          "same as the decorrelation test above)...")
    ab_injection_heldout = ablation_test(peft_model, tokenizer, layer, direction, args.alpha, LABEL_INJECTION,
                                          device, n_trials=n_trials, max_new_tokens=max_new_tokens, split="heldout")
    ab_safe_heldout = ablation_test(peft_model, tokenizer, layer, direction, args.alpha, LABEL_SAFE,
                                     device, n_trials=n_trials, max_new_tokens=max_new_tokens, split="heldout")

    print(f"  jacobian alignment: before={jac['before_training']:.4f} after={jac['after_training']:.4f} delta={jac['delta']:.4f}")
    for tag, ab_inj, ab_safe in (("TRAIN", ab_injection_train, ab_safe_train),
                                  ("HELDOUT", ab_injection_heldout, ab_safe_heldout)):
        print(f"  [{tag}] ablation (injection): direction_necessary={ab_inj['direction_necessary']} "
              f"input_necessary={ab_inj['input_necessary']} hook_presence_alone_sufficient={ab_inj['hook_presence_alone_sufficient']}")
        print(f"  [{tag}] ablation (safe): direction_necessary={ab_safe['direction_necessary']} "
              f"input_necessary={ab_safe['input_necessary']} hook_presence_alone_sufficient={ab_safe['hook_presence_alone_sufficient']}")

    if ab_injection_heldout["direction_necessary"] != ab_injection_train["direction_necessary"] or \
       ab_safe_heldout["direction_necessary"] != ab_safe_train["direction_necessary"]:
        print("\n  NOTE: direction_necessary DIFFERS between train and held-out phrasings -- consistent with "
              "train-phrasing memorization masking the direction's real causal role. Trust the HELD-OUT "
              "numbers, not the train ones, when judging whether the decorrelation result is genuine.")

    verdict = ("GENERALIZES (trustworthy monitor)"
               if decorrelation_result["dfi_overall"] and decorrelation_result["dfi_overall"] > 0
               else "FAILS (danger of naive introspection training)")
    print(f"\nVERDICT: {verdict}")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "stage7_results.json"), "w") as f:
        json.dump({
            "layer": layer, "alpha": args.alpha,
            "training_history": history,
            "decorrelation": decorrelation_result,
            "jacobian_alignment": jac,
            "ablation_injection_train": ab_injection_train,
            "ablation_safe_train": ab_safe_train,
            "ablation_injection_heldout": ab_injection_heldout,
            "ablation_safe_heldout": ab_safe_heldout,
            "verdict": verdict,
        }, f, indent=2)
    print(f"Saved stage7_results.json to {out_dir}/")


if __name__ == "__main__":
    main()