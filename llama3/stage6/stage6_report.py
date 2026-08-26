"""
Stage 6 entry point: loads Stage 3's decorrelated results (pre-
training baseline), Stage 4's training_eval.json (in-sample
correlated accuracy), and Stage 5's results (post-training
generalization + mechanism check), computes all five metrics, and
saves both a JSON and a markdown table ready to drop into the paper.

Deliberately does NOT generate a preregistration -- see the project
README for why: preregistration only does its job before you've seen
results, and by the time this stage runs you've already fully
analyzed Stages 3-5. Writing "predictions" now would be post-hoc
rationalization labeled as something it isn't. If/when a genuine
confirmatory follow-up run happens (more directions, a different or
larger model), that's the point to write a real preregistration --
happy to draft one at that point.

Usage:
    python stage6_report.py \
        --stage3 ../stage3/stage3_results/stage3_results.json \
        --stage4_eval ../stage4/stage4_adapter/training_eval.json \
        --stage5 ../stage5/stage5_results/stage5_results.json
"""

import argparse
import json
import os

from metrics import compute_all_metrics


def format_pct(x):
    return "n/a" if x is None else f"{x:.1%}"


def format_num(x, decimals=4):
    return "n/a" if x is None else f"{x:.{decimals}f}"


def to_markdown(m: dict) -> str:
    pre = m["pre_training"]
    post = m["post_training"]
    gap = m["generalization_gap"]
    jac = m["jacobian_alignment_change"]
    ab = m["ablation"]

    lines = []
    lines.append("## Stage 6 metrics summary\n")

    lines.append("### 1-2. Accuracy vs. direction / vs. input, and direction-following index\n")
    lines.append("| | vs. direction | vs. input | DFI (mean) | DFI (range) | n |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(f"| Pre-training (Stage 3, base model) | {format_pct(pre['accuracy_vs_direction_and_input']['accuracy_vs_direction'])} "
                  f"| {format_pct(pre['accuracy_vs_direction_and_input']['accuracy_vs_input'])} "
                  f"| {format_num(pre['direction_following_index']['dfi_mean'], 2)} "
                  f"| [{format_num(pre['direction_following_index']['dfi_min'], 2)}, {format_num(pre['direction_following_index']['dfi_max'], 2)}] "
                  f"| {pre['accuracy_vs_direction_and_input']['n_entries']} |")
    lines.append(f"| Post-training (Stage 5, trained adapter) | {format_pct(post['accuracy_vs_direction_and_input']['accuracy_vs_direction'])} "
                  f"| {format_pct(post['accuracy_vs_direction_and_input']['accuracy_vs_input'])} "
                  f"| {format_num(post['direction_following_index']['dfi_mean'], 2)} "
                  f"| [{format_num(post['direction_following_index']['dfi_min'], 2)}, {format_num(post['direction_following_index']['dfi_max'], 2)}] "
                  f"| {post['accuracy_vs_direction_and_input']['n_entries']} |")

    lines.append("\n### 3. Correlated-train -> decorrelated-test generalization gap\n")
    lines.append(f"- Correlated (in-sample, training-like) accuracy: **{format_pct(gap['correlated_accuracy'])}**")
    lines.append(f"- Decorrelated direction-tracking accuracy: **{format_pct(gap['decorrelated_direction_accuracy'])}**")
    lines.append(f"- Gap: **{format_pct(gap['gap'])}** (large gap = shortcut-learning signature: "
                  f"strong in-sample performance not actually explained by the direction)")

    lines.append("\n### 4. Pre/post-training Jacobian alignment change\n")
    lines.append(f"- Before training: {format_num(jac['before_mean'])}")
    lines.append(f"- After training: {format_num(jac['after_mean'])}")
    lines.append(f"- Mean delta: {format_num(jac['delta_mean'])}")
    lines.append(f"- Increased in {jac['n_increased']}/{jac['n_total']} tested combinations")

    lines.append("\n### 5. Report accuracy under direction-ablation vs. input-ablation\n")
    lines.append("| condition | match rate |")
    lines.append("|---|---|")
    lines.append(f"| baseline (no ablation) | {format_pct(ab['baseline_mean'])} |")
    lines.append(f"| direction ablated | {format_pct(ab['ablate_direction_mean'])} |")
    lines.append(f"| input ablated | {format_pct(ab['ablate_input_mean'])} |")
    lines.append(f"\n- Direction necessary in {format_pct(ab['direction_necessary_fraction'])} of tested combinations")
    lines.append(f"- Input necessary in {format_pct(ab['input_necessary_fraction'])} of tested combinations")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3", default="../stage3/stage3_results/stage3_results.json")
    parser.add_argument("--stage4_eval", default="../stage4/stage4_adapter/training_eval.json")
    parser.add_argument("--stage5", default="../stage5/stage5_results/stage5_results.json")
    parser.add_argument("--stage3_condition", choices=["raw", "orthogonalized"], default="raw",
                         help="Which Stage 3 confound-(b) condition to use as the pre-training baseline.")
    parser.add_argument("--out_dir", default="stage6_report")
    args = parser.parse_args()

    stage3 = json.load(open(args.stage3))
    stage4_eval = json.load(open(args.stage4_eval))
    stage5 = json.load(open(args.stage5))

    metrics = compute_all_metrics(stage3, stage4_eval, stage5, stage3_condition=args.stage3_condition)
    markdown = to_markdown(metrics)

    print(markdown)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "stage6_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.out_dir, "stage6_summary.md"), "w") as f:
        f.write(markdown)
    print(f"\n\nSaved stage6_metrics.json and stage6_summary.md to {args.out_dir}/")


if __name__ == "__main__":
    main()
