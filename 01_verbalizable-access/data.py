"""
Data loading for Stage 1.

- FineWeb sample: generic natural-language contexts, used both to
  compute J-lens directions (averaging over many contexts is the
  whole point of the technique) and to provide a background
  distribution for causal-efficacy scoring.
- CounterFact: subject-relation-object triples with paired
  counterfactual subjects (e.g. Paris/Rome - capital of). Used to fit
  probe directions for known attributes here in Stage 1, and reused
  in Stage 3 to build the decorrelation conflict design (direction
  X/Y vs. input X/Y).

`datasets` is imported lazily inside each function so this module —
and anything that imports it — stays importable in the dry-run test
without requiring the package or network access.
"""

from typing import List, Dict, Optional


def load_fineweb_contexts(n: int, seed: int = 0) -> List[str]:
    """Stream `n` short natural-language passages from FineWeb."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    out = []
    for row in ds:
        text = row["text"].strip()
        if len(text) > 50:
            out.append(text[:512])  # truncate to a manageable context length
        if len(out) >= n:
            break
    return out


def load_counterfact(n: Optional[int] = None) -> List[Dict]:
    """
    Load CounterFact (Meng et al., 2022) subject-relation-object triples.

    NOTE: field names below match the commonly-used
    "NeelNanda/counterfact-tracing" HF release; adjust the key lookups
    if you use a different CounterFact release with different schema.
    """
    from datasets import load_dataset

    ds = load_dataset("NeelNanda/counterfact-tracing", split="train")
    records = []
    for row in ds:
        records.append({
            "subject": row.get("subject"),
            "relation": row.get("relation_prefix", row.get("template", "")),
            "object_true": row.get("target_true", row.get("object_true", "")),
            "object_new": row.get("target_new", row.get("object_new", "")),
        })
        if n and len(records) >= n:
            break
    return records
