"""
Extreme Trait PAS Steering

Creates synthetic "extreme" personality profiles — maximally high or low on a
single Big Five trait — and runs the PAS pipeline on them. Tests PAS's ability
to steer toward a pure trait extreme.

For a target trait (e.g., high-A):
  - A-related items → extreme value (forward-keyed: 5, reverse-keyed: 1)
  - All other items → value 3 (neutral, filtered out of probe training)

This means probes train exclusively on the target trait's items, all maximally
polarized. Non-target traits serve as a specificity control (should stay near 3).

Usage:
  python extreme_pas.py --model_file meta-llama/Meta-Llama-3-8B-Instruct \
    --profiles high_A low_A --batch_size 3
"""

import argparse
import json
import os
import numpy as np
import torch
from copy import deepcopy
from datetime import datetime
from tqdm import tqdm

from main import (
    TEMPLATE, NEUTRAL_SYSTEM_PROMPT,
    getItems, generateAnswer, from_index_to_data,
    setup_raw_logger, lmean, build_few_shot_prompt,
)
from baseline_utils import (
    process_answers, save_subject_meta,
)
from PAlign.pas import get_model


import csv


TRAITS = ['A', 'C', 'E', 'N', 'O']
ALL_PROFILES = [f'{d}_{t}' for d in ('high', 'low') for t in TRAITS]


def _save_answers(result, profile, output_dir, alpha=None):
    """Save answers CSV with string-based profile name."""
    results_dir = os.path.join(output_dir, 'subject_results')
    os.makedirs(results_dir, exist_ok=True)
    rows = result.get('rows', [])
    if not rows:
        return
    suffix = f'_alpha{alpha}' if alpha is not None else ''
    path = os.path.join(results_dir, f'extreme_{profile}_answers{suffix}.csv')
    fieldnames = ['question_idx', 'trait', 'key', 'ground_truth',
                  'raw_answer', 'parsed', 'score', 'error']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_probes(all_head_accs, top_heads, profile, output_dir):
    """Save probes CSV with string-based profile name."""
    results_dir = os.path.join(output_dir, 'subject_results')
    os.makedirs(results_dir, exist_ok=True)
    num_layers, num_heads = all_head_accs.shape[:2]
    selected_set = set((l, h) for l, h in top_heads)
    path = os.path.join(results_dir, f'extreme_{profile}_probes.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['layer', 'head', 'train_acc', 'val_acc', 'selected'])
        for layer in range(num_layers):
            for head in range(num_heads):
                writer.writerow([
                    layer, head,
                    round(float(all_head_accs[layer, head, 1]), 4),
                    round(float(all_head_accs[layer, head, 0]), 4),
                    1 if (layer, head) in selected_set else 0,
                ])


def build_extreme_subject(template_sample, target_trait, direction):
    """Build a synthetic extreme subject from a template sample.

    Args:
        template_sample: dict with 'train' and 'test' item lists (from data[0])
        target_trait: one of 'A', 'C', 'E', 'N', 'O'
        direction: 'high' or 'low'

    Returns:
        dict with 'train' and 'test' lists, values set to extremes for the
        target trait and 3 (neutral) for all others.
    """
    case = f'extreme_{direction}_{target_trait}'
    subject = {'train': [], 'test': []}

    for split in ('train', 'test'):
        for item in template_sample[split]:
            new_item = deepcopy(item)
            if item['label_ocean'] == target_trait:
                if direction == 'high':
                    new_item['value'] = 5 if item['key'] == 1 else 1
                else:
                    new_item['value'] = 1 if item['key'] == 1 else 5
            else:
                new_item['value'] = 3
            new_item['case'] = case
            subject[split].append(new_item)

    return subject


def process_extreme_pas(data_template, model, tokenizer, model_file, profiles,
                        output_dir='./reproduction/extreme_pas',
                        use_few_shot=True, batch_size=16):
    """Run PAS on synthetic extreme profiles.

    Args:
        data_template: data[0] — the first subject's items (used as template)
        model: PASLM model
        tokenizer: tokenizer
        model_file: HF model name
        profiles: list of profile strings like 'high_A', 'low_E'
        output_dir: output directory
        use_few_shot: whether to use few-shot prompt
        batch_size: inference batch size

    Returns:
        dict: profile_name -> result summary
    """
    raw_logger = setup_raw_logger(output_dir)
    os.makedirs(os.path.join(output_dir, 'subject_results'), exist_ok=True)
    progress_path = os.path.join(output_dir, 'extreme_pas_progress.jsonl')

    # Check which profiles are already done
    done_profiles = set()
    for profile in profiles:
        meta_path = os.path.join(output_dir, 'subject_results',
                                 f'extreme_{profile}_meta.json')
        if os.path.exists(meta_path):
            done_profiles.add(profile)
    if done_profiles:
        print(f"Resuming: {len(done_profiles)} profiles already completed, skipping them.")

    # Build activations once (depends only on question text, not subject values)
    personal_data = []
    for personal in TRAITS:
        for item in data_template['train']:
            if item['label_ocean'] == personal:
                personal_data.append({
                    'question': TEMPLATE.format(item['text']),
                    'answer_matching_behavior': 'A',
                    'answer_not_matching_behavior': 'E',
                })

    print(f"Preprocessing activations for {len(personal_data)} items...")
    all_head_wise_activations = model.preprocess_activate_dataset(personal_data)

    summary = {}

    for profile in tqdm(profiles, desc='Profiles'):
        if profile in done_profiles:
            # Load existing result for summary
            meta_path = os.path.join(output_dir, 'subject_results',
                                     f'extreme_{profile}_meta.json')
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            summary[profile] = {
                'best_alpha': meta.get('alpha'),
                'mae': meta.get('per_trait_mae'),
                'mean_score': meta.get('per_trait_mean'),
            }
            continue

        direction, target_trait = profile.split('_', 1)
        subject = build_extreme_subject(data_template, target_trait, direction)

        model.reset_all()

        # Build system prompt
        if use_few_shot:
            system_prompt_text = build_few_shot_prompt(subject['train'])
        else:
            system_prompt_text = NEUTRAL_SYSTEM_PROMPT

        # Build labels from train items (only target trait items pass value!=3 filter)
        labels = []
        head_wise_activations = []
        personal_number = 0
        for personal in TRAITS:
            for item in subject['train']:
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
        activate, top_heads, all_head_accs = model.get_activations(
            deepcopy(head_wise_activations), labels, num_to_intervene=24
        )

        # Alpha sweep
        alpha_values = [0, 1, 2, 4, 6, 8]
        result_cache = []

        for alpha in alpha_values:
            model.reset_all()
            model.set_activate(activate, alpha)

            raw_logger.info(f"=== profile={profile} alpha={alpha} ===")

            answers = generateAnswer(
                tokenizer, model, data_template['test'], TEMPLATE,
                system_prompt=system_prompt_text, model_file=model_file,
                raw_logger=raw_logger, batch_size=batch_size,
            )

            result = process_answers(answers, subject)
            result_cache.append(result)
            _save_answers(result, profile, output_dir, alpha=alpha)

        # Pick best alpha by MAE sum
        scores = []
        for p in result_cache:
            score = sum(v for _, v in p['mean_ver_abs']['mean'])
            if str(score) == 'nan':
                score = 1e6
            scores.append(score)
        best_idx = int(np.array(scores).argmin())
        rs = result_cache[best_idx]
        rs['alpha'] = alpha_values[best_idx]

        # Save meta and probes
        meta_path = os.path.join(output_dir, 'subject_results',
                                 f'extreme_{profile}_meta.json')
        method_name = 'Extreme-few-shot-PAS' if use_few_shot else 'Extreme-PAS'
        save_subject_meta(
            meta_path, result=rs, subject_index=profile,
            model_file=model_file, method=method_name,
            alpha=alpha_values[best_idx], alpha_mode='sweep',
            num_to_intervene=24,
            modified_heads=top_heads,
            modified_layers=sorted(set(l for l, h in top_heads)),
            profile=profile, direction=direction, target_trait=target_trait,
        )
        _save_probes(all_head_accs, top_heads, profile, output_dir)

        # Build summary entry
        mae_dict = {k: round(v, 4) for k, v in rs['mean_ver_abs']['mean']}
        mean_dict = {k: round(v, 4) for k, v in rs['mean_ver']['mean']}
        summary[profile] = {
            'best_alpha': alpha_values[best_idx],
            'mae': mae_dict,
            'mean_score': mean_dict,
        }

        # Progress log
        progress_entry = {
            'profile': profile,
            'alpha': alpha_values[best_idx],
            'score_sum': scores[best_idx],
            'mae': mae_dict,
            'mean_score': mean_dict,
            'timestamp': datetime.now().isoformat(),
        }
        with open(progress_path, 'a') as f:
            f.write(json.dumps(progress_entry) + '\n')

        print(f"[{datetime.now().strftime('%H:%M:%S')}] {profile} done | "
              f"best_alpha={alpha_values[best_idx]} | "
              f"target_mae={mae_dict[target_trait]:.3f} | "
              f"target_mean={mean_dict[target_trait]:.3f} | "
              f"score_sum={scores[best_idx]:.3f}")

    return summary


def save_summary(summary, model_file, output_dir):
    """Save the aggregated summary JSON."""
    model_short = model_file.split('/')[-1]
    path = os.path.join(output_dir, f'Extreme-PAS_{model_short}_summary.json')
    os.makedirs(output_dir, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extreme Trait PAS Steering"
    )
    parser.add_argument(
        '--model_file', default='meta-llama/Meta-Llama-3-8B-Instruct',
        help='HuggingFace model name',
    )
    parser.add_argument(
        '--profiles', nargs='+', default=['all'],
        help='Profiles to test: high_A, low_A, ..., or "all" for all 10',
    )
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--output_dir', default='./reproduction/extreme_pas')
    parser.add_argument(
        '--no_few_shot', action='store_true',
        help='Use neutral system prompt instead of few-shot',
    )
    args = parser.parse_args()

    # Resolve profiles
    if args.profiles == ['all'] or 'all' in args.profiles:
        profiles = ALL_PROFILES
    else:
        for p in args.profiles:
            if p not in ALL_PROFILES:
                parser.error(f"Invalid profile '{p}'. Must be one of: {ALL_PROFILES}")
        profiles = args.profiles

    print(f"Profiles to run: {profiles}")

    # Load data (need template items for question text + structure)
    dataset, text_file, train_index, test_index = getItems('PAPI')
    data = from_index_to_data(train_index, test_index, text_file, dataset, 'OOD')
    data_template = data[0]  # first subject as template (question text is same for all)

    # Load model
    model, tokenizer = get_model(args.model_file)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

    # Run
    summary = process_extreme_pas(
        data_template, model, tokenizer, args.model_file, profiles,
        output_dir=args.output_dir,
        use_few_shot=not args.no_few_shot,
        batch_size=args.batch_size,
    )

    # Save summary
    save_summary(summary, args.model_file, args.output_dir)

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Profile':<12} {'Alpha':>5} {'Target MAE':>10} {'Target Mean':>11} "
          f"{'Non-tgt MAE':>11}")
    print("-" * 70)
    for profile in profiles:
        if profile not in summary:
            continue
        s = summary[profile]
        _, target_trait = profile.split('_', 1)
        target_mae = s['mae'][target_trait]
        target_mean = s['mean_score'][target_trait]
        non_target_maes = [s['mae'][t] for t in TRAITS if t != target_trait]
        avg_non_target = lmean(non_target_maes) if non_target_maes else 0
        print(f"{profile:<12} {s['best_alpha']:>5} {target_mae:>10.3f} "
              f"{target_mean:>11.3f} {avg_non_target:>11.3f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
