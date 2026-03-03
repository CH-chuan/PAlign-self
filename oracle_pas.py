"""
Oracle PAS: Pre-determined Alpha + Full-Data Probes

Leverages prior PAS runs to skip the expensive 6-alpha sweep by using a
pre-determined alpha, and improves probe quality by training on all 300 items
(not just the 120 train items).

Two modes:
  --analyze_only  (no GPU): Load pickles, print alpha consistency report
  Full run        (GPU):    Load model, train probes on all 300 items, evaluate
"""

import argparse
import json
import os
import pickle
import numpy as np
import torch
from collections import Counter
from copy import deepcopy
from datetime import datetime
from tqdm import tqdm

from main import (
    SCORES, SCORES_BACK, TEMPLATE, NEUTRAL_SYSTEM_PROMPT,
    prompt_to_tokens, getItems, generateAnswer, from_index_to_data,
    print_and_save_results, setup_raw_logger, lmean,
)
from baseline_utils import process_answers, calc_mean_and_var
from PAlign.pas import get_model


# ---------------------------------------------------------------------------
# 1. Load alphas from prior runs
# ---------------------------------------------------------------------------

def load_alphas(result_dirs, num_subjects=300):
    """Scan subject pickles from prior runs and extract alpha + MAE sum.

    Returns:
        dict: subject_idx -> [{'alpha', 'mae_sum', 'run_dir'}, ...]
    """
    alphas_data = {}
    for rd in result_dirs:
        subj_dir = os.path.join(rd, 'subject_results')
        if not os.path.isdir(subj_dir):
            print(f"Warning: {subj_dir} not found, skipping")
            continue
        for idx in range(num_subjects):
            pkl_path = os.path.join(subj_dir, f'subject_{idx:04d}.pkl')
            if not os.path.exists(pkl_path):
                continue
            with open(pkl_path, 'rb') as f:
                rs = pickle.load(f)
            alpha = rs.get('alpha', None)
            if alpha is None:
                continue
            mae_sum = sum(v for _, v in rs['mean_ver_abs']['mean'])
            alphas_data.setdefault(idx, []).append({
                'alpha': alpha,
                'mae_sum': mae_sum,
                'run_dir': rd,
            })
    return alphas_data


# ---------------------------------------------------------------------------
# 2. Analyze alpha consistency across runs
# ---------------------------------------------------------------------------

def analyze_alpha_consistency(alphas_data):
    """Print alpha consistency report across prior runs."""
    n_subjects = len(alphas_data)
    if n_subjects == 0:
        print("No subjects found in prior runs.")
        return

    # Count how many runs per subject
    run_counts = Counter(len(v) for v in alphas_data.values())
    print(f"\n=== Alpha Consistency Report ({n_subjects} subjects) ===")
    print(f"Runs per subject: {dict(sorted(run_counts.items()))}")

    # Only analyze subjects with 2+ runs
    multi = {k: v for k, v in alphas_data.items() if len(v) >= 2}
    if not multi:
        print("No subjects with multiple runs to compare.")
        return

    n_multi = len(multi)
    exact_match = 0
    two_thirds_agree = 0
    all_disagree = 0

    for idx, runs in multi.items():
        alphas = [r['alpha'] for r in runs]
        counts = Counter(alphas)
        most_common_count = counts.most_common(1)[0][1]
        if most_common_count == len(alphas):
            exact_match += 1
        if most_common_count >= 2:
            two_thirds_agree += 1
        if most_common_count == 1 and len(alphas) >= 3:
            all_disagree += 1

    print(f"\nAmong {n_multi} subjects with multiple runs:")
    print(f"  Exact match (all agree): {exact_match} ({100*exact_match/n_multi:.1f}%)")
    print(f"  2/3+ agree:              {two_thirds_agree} ({100*two_thirds_agree/n_multi:.1f}%)")
    print(f"  All disagree (3 runs):   {all_disagree} ({100*all_disagree/n_multi:.1f}%)")

    # Per-alpha distribution
    print("\nPer-alpha distribution (across all runs):")
    all_alphas = [r['alpha'] for runs in multi.values() for r in runs]
    for alpha, cnt in sorted(Counter(all_alphas).items()):
        print(f"  alpha={alpha}: {cnt} ({100*cnt/len(all_alphas):.1f}%)")

    # Strategy comparison
    print("\nStrategy comparison (per-subject alpha):")
    for strategy in ('majority', 'best_mae'):
        chosen = determine_oracle_alpha(alphas_data, strategy)
        dist = Counter(chosen.values())
        mae_info = []
        for idx, runs in alphas_data.items():
            best_run = min(runs, key=lambda r: r['mae_sum'])
            mae_info.append(best_run['mae_sum'])
        print(f"  {strategy}: alpha distribution = {dict(sorted(dist.items()))}, "
              f"best-run MAE avg = {np.mean(mae_info):.3f}")


# ---------------------------------------------------------------------------
# 3. Determine oracle alpha per subject
# ---------------------------------------------------------------------------

def determine_oracle_alpha(alphas_data, strategy):
    """Compute per-subject alpha from prior runs.

    Args:
        alphas_data: from load_alphas()
        strategy: 'majority' or 'best_mae'

    Returns:
        dict: subject_idx -> alpha (int)
    """
    oracle = {}
    for idx, runs in alphas_data.items():
        if strategy == 'majority':
            alphas = [r['alpha'] for r in runs]
            counts = Counter(alphas)
            max_count = counts.most_common(1)[0][1]
            # Ties → lowest alpha
            candidates = [a for a, c in counts.items() if c == max_count]
            oracle[idx] = min(candidates)
        elif strategy == 'best_mae':
            best = min(runs, key=lambda r: r['mae_sum'])
            oracle[idx] = best['alpha']
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    return oracle


# ---------------------------------------------------------------------------
# 4. Core Oracle PAS loop
# ---------------------------------------------------------------------------

def process_oracle_pas(data, model, tokenizer, model_file, oracle_alphas,
                       output_dir='./reproduction/oracle_pas',
                       batch_size=16, eval_set='both'):
    """Run PAS with full-data probes and pre-determined alpha.

    Key differences from process_pas():
      - Probes trained on all 300 items (train + test) instead of 120
      - Single alpha per subject (no sweep), 1 inference pass instead of 6
      - Evaluates on both all-300 and OOD-180 when eval_set='both'
    """
    raw_logger = setup_raw_logger(output_dir)

    os.makedirs(os.path.join(output_dir, 'subject_results'), exist_ok=True)
    progress_path = os.path.join(output_dir, 'oracle_pas_progress.jsonl')

    # Load existing results for resume
    results = [None] * len(data)
    done_indices = set()
    for idx in range(len(data)):
        pkl_path = os.path.join(output_dir, 'subject_results', f'subject_{idx:04d}.pkl')
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                results[idx] = pickle.load(f)
            done_indices.add(idx)
    if done_indices:
        print(f"Resuming: {len(done_indices)} subjects already completed, skipping them.")

    # -----------------------------------------------------------------------
    # Prepare activation data: ALL 300 items (train + test), grouped by trait
    # -----------------------------------------------------------------------
    personal_data = []
    for personal in ['A', 'C', 'E', 'N', 'O']:
        for item in data[0]['train'] + data[0]['test']:
            if item['label_ocean'] == personal:
                personal_data.append({
                    'question': TEMPLATE.format(item['text']),
                    'answer_matching_behavior': 'A',
                    'answer_not_matching_behavior': 'E',
                })

    print(f"Preprocessing activations for {len(personal_data)} items "
          f"(all 300, vs 120 in original PAS)...")
    all_head_wise_activations = model.preprocess_activate_dataset(personal_data)

    # -----------------------------------------------------------------------
    # Per-subject loop
    # -----------------------------------------------------------------------
    for index, sample in enumerate(tqdm(data)):
        if index in done_indices:
            continue

        model.reset_all()

        alpha = oracle_alphas.get(index)
        if alpha is None:
            print(f"Warning: no oracle alpha for subject {index}, skipping")
            continue

        system_prompt_text = NEUTRAL_SYSTEM_PROMPT

        # Build labels from ALL 300 items — same trait × item order as personal_data
        labels = []
        head_wise_activations = []
        personal_number = 0
        for personal in ['A', 'C', 'E', 'N', 'O']:
            for item in sample['train'] + sample['test']:
                if item['label_ocean'] == personal:
                    if item['value'] not in [0, 3]:
                        if item['value'] > 3:
                            labels.extend([1, 0])
                        else:
                            labels.extend([0, 1])
                        head_wise_activations.extend([
                            deepcopy(all_head_wise_activations[personal_number]),
                            deepcopy(all_head_wise_activations[personal_number + 1]),
                        ])
                    personal_number += 2

        # Train probes and get intervention vectors
        activate = model.get_activations(
            deepcopy(head_wise_activations), labels, num_to_intervene=24
        )

        # Apply fixed alpha
        model.reset_all()
        model.set_activate(activate, alpha)

        case_id = sample['test'][0]['case']
        raw_logger.info(f"=== subject={index} case={case_id} oracle_alpha={alpha} ===")

        # Generate answers
        if eval_set == 'both':
            # Generate for all 300 items in one pass
            all_items = data[0]['train'] + data[0]['test']
            all_answers = generateAnswer(
                tokenizer, model, all_items, TEMPLATE,
                system_prompt=system_prompt_text, model_file=model_file,
                raw_logger=raw_logger, batch_size=batch_size,
            )
            n_train = len(data[0]['train'])

            # OOD-180: answers for test items only
            ood_answers = all_answers[n_train:]
            ood_result = process_answers(ood_answers, sample)

            # ALL-300: answers for all items
            all_sample = {'test': data[0]['train'] + sample['test']}
            # Remap train items to have correct subject values
            all_sample_items = []
            for item in sample['train']:
                all_sample_items.append(item)
            for item in sample['test']:
                all_sample_items.append(item)
            all_sample = {'test': all_sample_items}
            all_result = process_answers(all_answers, all_sample)

            rs = {
                'ood': ood_result,
                'all': all_result,
                'oracle_alpha': alpha,
                # Keep ood as the primary result for print_and_save_results
                **ood_result,
            }
        else:
            # OOD only
            answers = generateAnswer(
                tokenizer, model, data[0]['test'], TEMPLATE,
                system_prompt=system_prompt_text, model_file=model_file,
                raw_logger=raw_logger, batch_size=batch_size,
            )
            ood_result = process_answers(answers, sample)
            rs = {**ood_result, 'oracle_alpha': alpha}

        results[index] = rs

        # Save per-subject pickle
        pkl_path = os.path.join(output_dir, 'subject_results', f'subject_{index:04d}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(rs, f)

        # Progress log
        ood_mae = {k: v for k, v in rs['mean_ver_abs']['mean']}
        score_sum = sum(ood_mae.values())
        progress_entry = {
            'index': index,
            'case': case_id,
            'oracle_alpha': alpha,
            'score_sum': score_sum,
            'mean_abs': ood_mae,
            'timestamp': datetime.now().isoformat(),
        }
        with open(progress_path, 'a') as f:
            f.write(json.dumps(progress_entry) + '\n')

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Subject {index}/{len(data)} done | "
              f"case={case_id} | oracle_alpha={alpha} | "
              f"score_sum={score_sum:.3f}")

    results = [r for r in results if r is not None]
    return results


# ---------------------------------------------------------------------------
# Main / CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Oracle PAS: Pre-determined Alpha + Full-Data Probes"
    )
    parser.add_argument(
        '--result_dirs', nargs='+', required=True,
        help='Directories containing prior PAS run results (with subject_results/ subdirs)',
    )
    parser.add_argument(
        '--analyze_only', action='store_true',
        help='Only print alpha consistency report (no GPU needed)',
    )
    parser.add_argument(
        '--alpha_strategy', choices=['majority', 'best_mae'], default='majority',
        help='How to pick per-subject alpha from prior runs',
    )
    parser.add_argument(
        '--model_file', default='meta-llama/Meta-Llama-3-8B-Instruct',
        help='HuggingFace model name',
    )
    parser.add_argument('--num_subjects', type=int, default=0, help='0 = all 300')
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--output_dir', default='./reproduction/oracle_pas')
    parser.add_argument(
        '--eval_set', choices=['ood', 'both'], default='both',
        help='Evaluate on OOD-180 only, or both OOD-180 and ALL-300',
    )
    args = parser.parse_args()

    # Load alphas from prior runs
    alphas_data = load_alphas(args.result_dirs)
    print(f"Loaded alpha data for {len(alphas_data)} subjects "
          f"from {len(args.result_dirs)} run(s)")

    # Always show analysis
    analyze_alpha_consistency(alphas_data)

    if args.analyze_only:
        return

    # Determine oracle alphas
    oracle_alphas = determine_oracle_alpha(alphas_data, args.alpha_strategy)
    print(f"\nOracle alphas ({args.alpha_strategy}): "
          f"{dict(sorted(Counter(oracle_alphas.values()).items()))}")

    # Load data
    dataset, text_file, train_index, test_index = getItems('PAPI')
    data = from_index_to_data(train_index, test_index, text_file, dataset, 'OOD')

    if args.num_subjects > 0:
        data = data[:args.num_subjects]
        print(f"Using {args.num_subjects} subjects (out of {len(dataset)} total)")

    # Load model
    model, tokenizer = get_model(args.model_file)
    if 'llama-3' in args.model_file.lower():
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

    # Run
    results = process_oracle_pas(
        data, model, tokenizer, args.model_file, oracle_alphas,
        output_dir=args.output_dir, batch_size=args.batch_size,
        eval_set=args.eval_set,
    )

    # Save OOD results
    strategy_label = f"Oracle-PAS-{args.alpha_strategy}"
    print_and_save_results(results, strategy_label, args.model_file, 'OOD',
                           output_dir=args.output_dir)

    # Save ALL-300 results if we have them
    if args.eval_set == 'both':
        all_results = []
        for r in results:
            if 'all' in r:
                all_results.append(r['all'])
        if all_results:
            print_and_save_results(all_results, strategy_label, args.model_file, 'ALL',
                                   output_dir=args.output_dir)


if __name__ == '__main__':
    main()
