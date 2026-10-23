"""
Scoring for Stage 2: automated exact-match scoring, plus utilities to
export elicited reports for blinded human judging and merge the
returned judgments back in.

Exact-match is a coarse, high-precision/low-recall automated signal
(catches only literal mentions of the target word) -- it's a cheap
check, not a replacement for human judging, which is why both are
called for in the plan. Both are tracked at trial-level granularity
(direction_key, trial_index), not aggregated per-direction, so a
human judge's per-report call doesn't silently overwrite a sibling
trial's score for the same direction.
"""

import csv
import json
import random
import re
from typing import List, Dict, Optional, Tuple


def exact_match_score(report: str, target_concept: Optional[str]) -> Optional[int]:
    """
    1 if target_concept appears as a whole word in report (case-
    insensitive), 0 if not, None if there's no well-defined target
    concept for this direction (e.g. a generic SAE feature with no
    single associated word) -- those rely on human judging instead.
    """
    if not target_concept:
        return None
    pattern = r"\b" + re.escape(target_concept.strip()) + r"\b"
    return 1 if re.search(pattern, report, flags=re.IGNORECASE) else 0


def infer_target_concept(direction_key: str) -> Optional[str]:
    """
    J-space directions are named "jspace::{layer}::{token}" -- the
    token IS the target concept, so exact-match is well-defined for
    them. Other classes don't have one known target word by
    construction, so this returns None for them.
    """
    parts = direction_key.split("::")
    if parts[0] == "jspace" and len(parts) >= 3:
        return parts[2]
    return None


def export_for_blind_judging(records: List[Dict], out_path: str, seed: int = 0) -> str:
    """
    records: [{"direction_key":..., "trial": int, "report": str}, ...]
    Writes a shuffled, ID-only CSV for a human judge to fill in a
    `judge_score` column (1 = report describes the intervened
    concept, 0 = it doesn't) with no direction_key visible. Returns
    the path to a separate answer-key JSON (not shown to the judge)
    used to merge scores back to (direction, trial) pairs afterward.
    """
    shuffled = records[:]
    random.Random(seed).shuffle(shuffled)

    answer_key = {}
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["blind_id", "report", "judge_score"])
        for i, r in enumerate(shuffled):
            blind_id = f"item_{i:04d}"
            writer.writerow([blind_id, r["report"], ""])
            answer_key[blind_id] = {"direction_key": r["direction_key"], "trial": r["trial"]}

    answer_key_path = out_path.replace(".csv", "_answer_key.json")
    with open(answer_key_path, "w") as f:
        json.dump(answer_key, f, indent=2)
    return answer_key_path


def merge_judge_scores(judged_csv_path: str, answer_key_path: str) -> Dict[Tuple[str, int], int]:
    """
    Reads a completed blind-judging CSV + its answer key. Returns
    {(direction_key, trial): judge_score}, skipping rows the judge
    left blank.
    """
    answer_key = json.load(open(answer_key_path))
    scores = {}
    with open(judged_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["judge_score"] not in ("", None):
                entry = answer_key[row["blind_id"]]
                scores[(entry["direction_key"], entry["trial"])] = int(row["judge_score"])
    return scores
