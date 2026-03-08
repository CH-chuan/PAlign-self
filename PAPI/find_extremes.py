#!/usr/bin/env python3
"""Find subjects with the highest and lowest scores for each Big Five trait.

Reports extremes across three item subsets:
  - Train (120 items from mpi_300_split.json)
  - Test  (180 items)
  - All   (300 items)
"""

import json
from pathlib import Path

import pandas as pd

TRAITS = ["A", "C", "E", "N", "O"]
TRAIT_NAMES = {
    "A": "Agreeableness",
    "C": "Conscientiousness",
    "E": "Extraversion",
    "N": "Neuroticism",
    "O": "Openness",
}


def compute_trait_scores(subjects, trait_items, item_subset=None):
    """Compute per-subject mean trait scores.

    Args:
        subjects: list of subject dicts with i1..i300 responses
        trait_items: {trait: [(item_num, is_forward), ...]}
        item_subset: optional set of item numbers to restrict to

    Returns:
        {trait: [(subject_index, case_id, mean_score), ...]}
    """
    scores = {t: [] for t in TRAITS}
    for idx, subj in enumerate(subjects):
        for trait in TRAITS:
            vals = []
            for item_num, is_forward in trait_items[trait]:
                if item_subset is not None and item_num not in item_subset:
                    continue
                raw = subj[f"i{item_num}"]
                vals.append(raw if is_forward else 6 - raw)
            if vals:
                scores[trait].append((idx, subj["case"], sum(vals) / len(vals)))
    return scores


def print_extremes(scores, label, show_dist=True):
    """Print high/low extremes and optional distribution summary."""
    print(f"=== {label} ===")
    print(f"{'Trait':<20} {'Extreme':>7}  {'SubjIdx':>7}  {'CaseID':>8}  {'Score':>6}")
    print("-" * 55)

    for trait in TRAITS:
        by_score = sorted(scores[trait], key=lambda x: x[2])
        lo_idx, lo_case, lo_score = by_score[0]
        hi_idx, hi_case, hi_score = by_score[-1]
        print(f"{TRAIT_NAMES[trait]:<20} {'LOW':>7}  {lo_idx:>7}  {lo_case:>8}  {lo_score:>6.3f}")
        print(f"{'':<20} {'HIGH':>7}  {hi_idx:>7}  {hi_case:>8}  {hi_score:>6.3f}")

    if show_dist:
        print()
        print(f"{'Trait':<20} {'Mean':>6} {'Std':>6} {'Min':>6} {'Max':>6}")
        print("-" * 48)
        for trait in TRAITS:
            vals = [s[2] for s in scores[trait]]
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            print(f"{TRAIT_NAMES[trait]:<20} {mean:>6.3f} {std:>6.3f} {min(vals):>6.3f} {max(vals):>6.3f}")
    print()


def main():
    data_dir = Path(__file__).parent

    with open(data_dir / "Test-set.json", encoding="utf-8") as f:
        subjects = json.load(f)

    item_key = pd.read_excel(data_dir / "IPIP-NEO-ItemKey.xls")

    with open(data_dir / "mpi_300_split.json", encoding="utf-8") as f:
        split = json.load(f)
    train_set = set(split["train_index"])
    test_set = set(split["test_index"])

    # Build per-trait item lists: {trait: [(item_num, is_forward), ...]}
    trait_items: dict[str, list[tuple[int, bool]]] = {t: [] for t in TRAITS}
    for _, row in item_key.iterrows():
        trait = row["Key"][0]
        direction = row["Sign"][0]  # '+' or '-'
        item_num = int(row["Full#"])
        trait_items[trait].append((item_num, direction == "+"))

    # Count items per subset per trait
    for label, subset in [("Train", train_set), ("Test", test_set), ("All", None)]:
        counts = {}
        for t in TRAITS:
            if subset is None:
                counts[t] = len(trait_items[t])
            else:
                counts[t] = sum(1 for num, _ in trait_items[t] if num in subset)
        print(f"{label} items per trait: {counts}")
    print()

    # Compute and print for each subset
    for label, subset in [
        ("Train (120 items)", train_set),
        ("Test (180 items)", test_set),
        ("All (300 items)", None),
    ]:
        scores = compute_trait_scores(subjects, trait_items, item_subset=subset)
        print_extremes(scores, label)


if __name__ == "__main__":
    main()
