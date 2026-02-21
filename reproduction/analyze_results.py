"""
Analyze PAS reproduction results and compare against Table 1 from
"Personality Alignment of Large Language Models" (ICLR 2025).
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Paper's Table 1 values: PAS on Llama-3-8B-Instruct (Aligned Score, lower=better, range 0-4)
PAPER_VALUES = {
    'A': 0.94,
    'C': 0.91,
    'E': 0.86,
    'N': 0.98,
    'O': 0.72,
}

TRAIT_NAMES = {
    'A': 'Agreeableness',
    'C': 'Conscientiousness',
    'E': 'Extraversion',
    'N': 'Neuroticism',
    'O': 'Openness',
}

TOLERANCE = 0.15


def load_results(results_path):
    with open(results_path, 'r') as f:
        return json.load(f)


def compare_with_paper(log):
    """Compare per-trait Aligned Scores against paper's Table 1."""
    print("=" * 60)
    print("Comparison with Paper Table 1 (PAS / Llama-3-8B-Instruct)")
    print("=" * 60)
    print(f"{'Trait':<20} {'Paper':>8} {'Ours':>8} {'Delta':>8} {'Status':>10}")
    print("-" * 60)

    our_values = {}
    for trait in ['A', 'C', 'E', 'N', 'O']:
        key = f'mean_{trait}_abs'
        our_val = log['score'][key]
        paper_val = PAPER_VALUES[trait]
        delta = our_val - paper_val
        status = 'MATCH' if abs(delta) <= TOLERANCE else 'DIFFER'
        our_values[trait] = our_val
        print(f"{TRAIT_NAMES[trait]:<20} {paper_val:>8.3f} {our_val:>8.3f} {delta:>+8.3f} {status:>10}")

    print("-" * 60)
    paper_sum = sum(PAPER_VALUES.values())
    our_sum = sum(our_values.values())
    print(f"{'Sum':<20} {paper_sum:>8.3f} {our_sum:>8.3f} {our_sum - paper_sum:>+8.3f}")
    print()
    return our_values


def compute_correlations(log, data_dir='PAPI'):
    """Compute Pearson correlation between model predictions and ground truth per trait."""
    test_set_path = os.path.join(data_dir, 'Test-set.json')
    item_key_path = os.path.join(data_dir, 'IPIP-NEO-ItemKey.xls')
    split_path = os.path.join(data_dir, 'mpi_300_split.json')

    if not all(os.path.exists(p) for p in [test_set_path, item_key_path, split_path]):
        print("Warning: PAPI data files not found, skipping correlation analysis.")
        return None

    with open(test_set_path) as f:
        test_data = json.load(f)
    with open(split_path) as f:
        split = json.load(f)
    item_key = pd.read_excel(item_key_path)
    test_index = split['test_index']

    # Build ground truth trait scores per subject
    gt_scores = {t: [] for t in ['A', 'C', 'E', 'N', 'O']}
    for subject in test_data:
        trait_vals = {t: [] for t in ['A', 'C', 'E', 'N', 'O']}
        for t_i in test_index:
            row = item_key[item_key['Full#'] == t_i].iloc[0]
            trait = row.iloc[3][0]  # First char of label
            key_dir = 1 if row.iloc[2][0] == '+' else -1
            val = subject[f'i{t_i}']
            if key_dir == 1:
                trait_vals[trait].append(val)
            else:
                trait_vals[trait].append(6 - val)
        for t in trait_vals:
            gt_scores[t].append(np.mean(trait_vals[t]))

    # Model predictions per subject
    pred_scores = {t: [] for t in ['A', 'C', 'E', 'N', 'O']}
    for trait_idx, trait in enumerate(['A', 'C', 'E', 'N', 'O']):
        pred_scores[trait] = log['mean'][trait]

    print("=" * 60)
    print("Pearson Correlations (model predictions vs ground truth)")
    print("=" * 60)
    print(f"{'Trait':<20} {'r':>8} {'p-value':>12} {'N':>6}")
    print("-" * 60)

    correlations = {}
    for trait in ['A', 'C', 'E', 'N', 'O']:
        gt = np.array(gt_scores[trait])
        pred = np.array(pred_scores[trait])
        n = min(len(gt), len(pred))
        gt, pred = gt[:n], pred[:n]
        r, p = pearsonr(gt, pred)
        correlations[trait] = (r, p, n)
        print(f"{TRAIT_NAMES[trait]:<20} {r:>8.4f} {p:>12.2e} {n:>6}")

    print()
    return correlations, gt_scores, pred_scores


def make_scatter_plots(correlations, gt_scores, pred_scores, output_path='reproduction/scatter_comparison.png'):
    """Generate scatter plots for each trait."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle('PAS Reproduction: Model Predictions vs Ground Truth', fontsize=14)

    for ax, trait in zip(axes, ['A', 'C', 'E', 'N', 'O']):
        gt = np.array(gt_scores[trait])
        pred = np.array(pred_scores[trait])
        n = min(len(gt), len(pred))
        gt, pred = gt[:n], pred[:n]
        r, p, _ = correlations[trait]

        ax.scatter(gt, pred, alpha=0.3, s=10)
        # Fit line
        if len(gt) > 1:
            z = np.polyfit(gt, pred, 1)
            x_line = np.linspace(gt.min(), gt.max(), 100)
            ax.plot(x_line, np.polyval(z, x_line), 'r-', linewidth=1)
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Model Prediction')
        ax.set_title(f'{TRAIT_NAMES[trait]}\nr={r:.3f}, p={p:.2e}')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Scatter plots saved to {output_path}")


def main():
    results_dir = 'reproduction'
    results_file = os.path.join(results_dir, 'PAS_Meta-Llama-3-8B-Instruct_OOD.json')

    if not os.path.exists(results_file):
        print(f"Error: Results file not found at {results_file}")
        print("Run the main experiment first.")
        sys.exit(1)

    log = load_results(results_file)

    # Compare with paper
    our_values = compare_with_paper(log)

    # Compute correlations
    result = compute_correlations(log)
    if result:
        correlations, gt_scores, pred_scores = result
        make_scatter_plots(correlations, gt_scores, pred_scores)


if __name__ == '__main__':
    main()
