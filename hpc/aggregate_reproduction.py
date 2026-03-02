#!/usr/bin/env python3
"""Aggregate reproduction results into a Table 1 comparison.

Reads JSON result files from all 8 methods and prints a formatted table
comparing per-trait MAE scores against the paper's reported values.

Usage:
    python hpc/aggregate_reproduction.py [--output-dir ./reproduction]
"""
import argparse
import json
import os
import sys

TRAITS = ['A', 'C', 'E', 'N', 'O']

# Paper Table 1 targets (MAE, lower is better)
PAPER_TARGETS = {
    'Few-Shot+PAS': {'A': 0.94, 'C': 0.91, 'E': 0.86, 'N': 0.98, 'O': 0.72},
}

# Method name -> (result file pattern, display name)
# Pattern uses {model} placeholder for the short model name
METHODS = [
    ('PAS',               '{output_dir}/PAS_{model}_OOD.json',                          'PAS (pure)'),
    ('few-shot-PAS',      '{output_dir}/few-shot-PAS_{model}_OOD.json',                 'Few-Shot+PAS'),
    ('DPO',               '{output_dir}/benchmarks/dpo/DPO_{model}_OOD.json',           'DPO'),
    ('PPO',               '{output_dir}/benchmarks/ppo/PPO_{model}_OOD.json',           'PPO'),
    ('Prompt-MORL',       '{output_dir}/benchmarks/prompt_morl/Prompt-MORL_{model}_OOD.json', 'Prompt-MORL'),
    ('Soups',             '{output_dir}/benchmarks/soups/Soups_{model}_OOD.json',       'Pers. Soups'),
    ('few-shot',          '{output_dir}/few-shot_{model}_OOD.json',                     'Few-Shot'),
    ('personality_prompt', '{output_dir}/personality_prompt_{model}_OOD.json',           'P² Prompt'),
]


def load_scores(filepath):
    """Load trait MAE scores from a result JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    scores = data.get('score', {})
    return {t: scores.get(f'mean_{t}_abs', float('nan')) for t in TRAITS}


def main():
    parser = argparse.ArgumentParser(description='Aggregate reproduction results into Table 1')
    parser.add_argument('--output-dir', default='./reproduction',
                        help='Base output directory (default: ./reproduction)')
    parser.add_argument('--model', default='Meta-Llama-3-8B-Instruct',
                        help='Short model name for filename matching')
    args = parser.parse_args()

    output_dir = args.output_dir
    model = args.model

    # Header
    header = f"{'Method':<20s} |"
    for t in TRAITS:
        header += f"  {t:>5s}"
    header += "  |    Sum"
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    found_any = False
    for method_key, pattern, display_name in METHODS:
        filepath = pattern.format(output_dir=output_dir, model=model)
        if not os.path.exists(filepath):
            continue
        found_any = True
        try:
            scores = load_scores(filepath)
            row = f"{display_name:<20s} |"
            total = 0.0
            for t in TRAITS:
                val = scores[t]
                row += f"  {val:5.2f}"
                total += val
            row += f"  | {total:6.2f}"
            print(row)
        except Exception as e:
            print(f"{display_name:<20s} |  ERROR: {e}")

    # Paper targets
    print("-" * len(header))
    for name, targets in PAPER_TARGETS.items():
        row = f"{'Paper (' + name + ')':<20s} |"
        total = 0.0
        for t in TRAITS:
            val = targets[t]
            row += f"  {val:5.2f}"
            total += val
        row += f"  | {total:6.2f}"
        print(row)
    print("=" * len(header))

    if not found_any:
        print(f"\nNo result files found in {output_dir}/")
        print(f"Expected model name in filenames: {model}")
        sys.exit(1)


if __name__ == '__main__':
    main()
