"""
Synthetic single-trait PAS steering experiment.

Tests whether PAS can steer a single Big Five personality trait to an extreme
(high or low) while keeping all other traits neutral. Constructs a synthetic
subject whose questionnaire answers are extreme on one chosen trait and neutral
(value=3, skipped by PAS probe training) on all others.

Keying is respected: for "high C", positive-keyed C items are set to 5 (strongly
agree) and negative-keyed C items to 1 (strongly disagree), so the probes learn
to steer toward high Conscientiousness specifically.

Requires:
    - conda env: palign_repro  (with torch, transformers, einops, baukit, etc.)
    - GPU with >= 24 GB VRAM (model uses ~20.5 GB)
    - PAPI/ data directory in the project root

Arguments:
    --trait        Required. Target Big Five trait: A, C, E, N, or O.
    --direction    Required. Steering direction: high or low.
    --model_file   HuggingFace model name.
                   Default: meta-llama/Meta-Llama-3-8B-Instruct
    --alphas       Comma-separated alpha (steering strength) values to test.
                   Default: 0,1,2,4,6,8,12,16
    --output_dir   Directory for JSON result files.
                   Default: ./synthetic_HL
    --max_new_tokens  Max new tokens for generation. Default: 50

Examples:
    # Steer Conscientiousness high (basic run, ~7 min):
    python run_synthetic.py --trait C --direction high

    # Steer Neuroticism low:
    python run_synthetic.py --trait N --direction low

    # Steer Extraversion high with custom alpha sweep:
    python run_synthetic.py --trait E --direction high --alphas "0,2,4,8,12,16"

    # Using conda env explicitly:
    conda run -n palign_repro python run_synthetic.py --trait C --direction high

Output:
    Console: diagnostic item counts, per-alpha score table, UNK rates, best-alpha
    summary. JSON file saved to <output_dir>/synthetic_<trait>_<direction>_<timestamp>.json

Interpreting results:
    The score table shows per-trait "aligned scores" (lower = better alignment with
    the synthetic target). For a successful run:
    - The target trait should have a low score at higher alphas
    - Other traits should remain at baseline levels (near alpha=0 values)
    - The best alpha is chosen by minimum sum across all traits
"""

import json
import logging
import os
import argparse
import torch
import numpy as np
from datetime import datetime
from copy import deepcopy

from PAlign.pas import get_model
from main import getItems, from_index_to_data, TEMPLATE, SCORES_BACK, SYSTEM_PROMPT, prompt_to_tokens
from baseline_utils import process_answers

TRAITS = ['A', 'C', 'E', 'N', 'O']


def setup_raw_logger(output_dir):
    """Set up a dedicated file logger for raw model generations."""
    os.makedirs(output_dir, exist_ok=True)
    raw_logger = logging.getLogger('synthetic_raw_generations')
    raw_logger.setLevel(logging.INFO)
    if not raw_logger.handlers:
        fh = logging.FileHandler(os.path.join(output_dir, 'raw_generations.log'), mode='a')
        fh.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%dT%H:%M:%S'))
        raw_logger.addHandler(fh)
    return raw_logger


def generateAnswer(tokenizer, model, dataset, template, scores=None, system_prompt=SYSTEM_PROMPT,
                   model_file=None, max_new_tokens=50):
    """Generate answers using the model, with configurable max_new_tokens."""
    batch_size = 3
    questions = [item["text"].lower() for item in dataset]
    answers = []

    for batch in range(0, len(questions), batch_size):
        with torch.no_grad():
            outputs = model.generate(
                [prompt_to_tokens(tokenizer, system_prompt, template.format(prompt), 'Option', model_file)
                 for prompt in questions[batch:batch + batch_size]],
                max_new_tokens=max_new_tokens,
            )
            output_text = tokenizer.batch_decode(outputs)
            if 'llama-3' in model_file.lower():
                answer = [text.split("<|end_header_id|>")[-1] for text in output_text]
            else:
                answer = [text.split("[/INST]")[-1] for text in output_text]
            answers.extend(answer)

    return answers


def build_synthetic_subject(data_0, target_trait, direction):
    """Deep-copy data[0] and set values for synthetic single-trait experiment.

    Target trait items: 5 for +keyed, 1 for -keyed (high), or reversed (low).
    All other trait items: 3 (neutral, skipped in probe training).
    """
    synthetic = deepcopy(data_0)

    for split in ['train', 'test']:
        for item in synthetic[split]:
            if item['label_ocean'] == target_trait:
                if direction == 'high':
                    item['value'] = 5 if item['key'] == 1 else 1
                else:  # low
                    item['value'] = 1 if item['key'] == 1 else 5
            else:
                item['value'] = 3  # neutral

    return synthetic


def run_single_subject_pas(data_0, synthetic, model, tokenizer, model_file, alpha_values,
                           output_dir, target_trait, direction, max_new_tokens=50):
    """Run PAS on a single synthetic subject. Mirrors process_pas single-subject logic."""
    raw_logger = setup_raw_logger(output_dir)

    # Build personal_data from data_0['train'] (shared activations)
    personal_data = []
    for personal in TRAITS:
        for item in data_0['train']:
            if item['label_ocean'] == personal:
                personal_data.append({
                    'question': TEMPLATE.format(item['text']),
                    'answer_matching_behavior': 'A',
                    'answer_not_matching_behavior': 'E'
                })

    print("Precomputing activations...")
    all_head_wise_activations = model.preprocess_activate_dataset(personal_data)

    # Build labels from synthetic['train'] (only target trait items survive value!=3 filter)
    labels = []
    head_wise_activations = []
    personal_number = 0
    included = 0
    skipped = 0
    for personal in TRAITS:
        for item in synthetic['train']:
            if item['label_ocean'] == personal:
                if item['value'] not in [0, 3]:
                    if item['value'] > 3:
                        labels.extend([1, 0])
                    else:
                        labels.extend([0, 1])
                    head_wise_activations.extend([
                        deepcopy(all_head_wise_activations[personal_number]),
                        deepcopy(all_head_wise_activations[personal_number + 1])
                    ])
                    included += 1
                else:
                    skipped += 1
                personal_number += 2

    print(f"Diagnostic: {included} items included, {skipped} items skipped")
    print(f"Activation-label pairs: {len(labels)}")

    # Train probes and get interventions
    print("Training probes...")
    activate = model.get_activations(deepcopy(head_wise_activations), labels, num_to_intervene=24)

    # Build system prompt from synthetic['train']
    system_prompt_text = (
        'Here are some of your behaviors and your level of recognition towards them;'
        + ';'.join([f"{it['text']}:{SCORES_BACK[it['value']]}" for it in synthetic['train']])
    )

    # Test each alpha
    result_cache = []
    for alpha in alpha_values:
        model.reset_all()
        model.set_activate(activate, alpha)

        print(f"  Generating answers for alpha={alpha}...")
        answers = generateAnswer(tokenizer, model, data_0['test'], TEMPLATE,
                                 system_prompt=system_prompt_text, model_file=model_file,
                                 max_new_tokens=max_new_tokens)

        raw_logger.info(f"=== trait={target_trait} direction={direction} alpha={alpha} ===")
        for q_idx, ans in enumerate(answers):
            raw_logger.info(f"q={q_idx} | {ans.strip()}")

        result = process_answers(answers, synthetic)
        result_cache.append(result)

    model.reset_all()
    return result_cache


def print_report(result_cache, alpha_values, target_trait, direction):
    """Print formatted per-alpha, per-trait score table."""
    print("\n" + "=" * 70)
    print(f"Synthetic PAS Results: trait={target_trait}, direction={direction}")
    print("=" * 70)

    print(f"{'Alpha':>6} | {'A':>6} {'C':>6} {'E':>6} {'N':>6} {'O':>6} | {'Sum':>6}")
    print("-" * 70)

    scores_by_alpha = []
    for i, alpha in enumerate(alpha_values):
        mean_abs = dict(result_cache[i]['mean_ver_abs']['mean'])
        trait_scores = {t: mean_abs.get(t, float('nan')) for t in TRAITS}
        score_sum = sum(v for v in trait_scores.values() if not np.isnan(v))
        scores_by_alpha.append(score_sum)

        print(f"{alpha:>6} | {trait_scores['A']:>6.2f} {trait_scores['C']:>6.2f} "
              f"{trait_scores['E']:>6.2f} {trait_scores['N']:>6.2f} {trait_scores['O']:>6.2f} | "
              f"{score_sum:>6.2f}")

    # UNK counts
    print("-" * 70)
    print("UNK counts per alpha:")
    for i, alpha in enumerate(alpha_values):
        unk = result_cache[i]['count'].get('UNK', 0)
        total = sum(result_cache[i]['count'].values())
        pct = 100 * unk / total if total > 0 else 0
        print(f"  alpha={alpha}: {unk}/{total} ({pct:.1f}%)")

    # Best alpha (by total aligned score sum)
    best_idx = int(np.array(scores_by_alpha).argmin())
    best_alpha = alpha_values[best_idx]
    best_mean_abs = dict(result_cache[best_idx]['mean_ver_abs']['mean'])

    print(f"\nBest alpha: {best_alpha} (sum={scores_by_alpha[best_idx]:.2f})")
    print(f"Target trait {target_trait} aligned score: "
          f"{best_mean_abs.get(target_trait, float('nan')):.2f} (lower=better)")
    other_scores = [best_mean_abs.get(t, float('nan')) for t in TRAITS if t != target_trait]
    print(f"Other traits mean aligned score: {np.nanmean(other_scores):.2f}")

    return best_idx


def main():
    parser = argparse.ArgumentParser(description="Synthetic single-trait PAS steering experiment")
    parser.add_argument("--trait", required=True, choices=TRAITS,
                        help="Target Big Five trait (A/C/E/N/O)")
    parser.add_argument("--direction", required=True, choices=['high', 'low'],
                        help="Steering direction (high or low)")
    parser.add_argument("--model_file", default='meta-llama/Meta-Llama-3-8B-Instruct',
                        help="HuggingFace model name")
    parser.add_argument("--alphas", default='0,1,2,4,6,8,12,16',
                        help="Comma-separated alpha values")
    parser.add_argument("--output_dir", default='./synthetic_HL',
                        help="Output directory")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="Max new tokens for generation (default: 50)")
    args = parser.parse_args()

    alpha_values = [int(x) for x in args.alphas.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model_file}")
    model, tokenizer = get_model(args.model_file)
    if 'llama-3' in args.model_file.lower():
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

    # Load data (process 1 real subject for shared activations and question text)
    print("Loading PAPI data...")
    dataset, text_file, train_index, test_index = getItems('PAPI')
    data = from_index_to_data(train_index, test_index, text_file, dataset[:1], 'OOD')
    data_0 = data[0]

    # Build synthetic subject
    synthetic = build_synthetic_subject(data_0, args.trait, args.direction)

    train_target = sum(1 for it in synthetic['train'] if it['label_ocean'] == args.trait)
    train_other = sum(1 for it in synthetic['train'] if it['label_ocean'] != args.trait)
    print(f"Synthetic subject: {train_target} {args.trait} train items (active), "
          f"{train_other} other items (neutral)")

    # Run PAS
    result_cache = run_single_subject_pas(data_0, synthetic, model, tokenizer,
                                          args.model_file, alpha_values,
                                          args.output_dir, args.trait, args.direction,
                                          args.max_new_tokens)

    # Report
    best_idx = print_report(result_cache, alpha_values, args.trait, args.direction)

    # Save JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(args.output_dir,
                            f'synthetic_{args.trait}_{args.direction}_{timestamp}.json')
    output = {
        'trait': args.trait,
        'direction': args.direction,
        'model_file': args.model_file,
        'alphas': alpha_values,
        'best_alpha': alpha_values[best_idx],
        'results': []
    }
    for i, alpha in enumerate(alpha_values):
        r = result_cache[i]
        output['results'].append({
            'alpha': alpha,
            'mean_ver_abs': r['mean_ver_abs'],
            'mean_ver': r['mean_ver'],
            'count': r['count']
        })

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
