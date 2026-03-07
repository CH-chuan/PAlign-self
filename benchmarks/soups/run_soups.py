"""CLI entry point for Personalized-Soups: train 10 extremes → merge per subject → eval."""
import argparse
import os
import sys
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from benchmarks.common.data import load_all_data
from benchmarks.common.model_utils import load_base_model_and_tokenizer
from benchmarks.common.evaluation import evaluate_subject, aggregate_and_save_results, setup_raw_logger
from benchmarks.common.resume import (
    save_subject_result, load_completed_results, append_progress,
)
from benchmarks.soups.trainer import train_all_extreme_models
from benchmarks.soups.merger import merge_soup_for_subject


def run_soups(model_name, num_subjects=0, output_dir=None, data_dir='PAPI'):
    """Run the full Personalized-Soups benchmark pipeline."""
    if output_dir is None:
        output_dir = os.path.join('reproduction', 'benchmarks', 'soups')

    adapters_dir = os.path.join(output_dir, 'adapters')

    print(f"=== Personalized-Soups Benchmark ===")
    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")

    # Load data
    data = load_all_data(data_dir=data_dir, num_subjects=num_subjects)
    total = len(data)
    print(f"Subjects: {total}")

    # Phase 1: Train extreme models
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

    print("\n=== Phase 1: Training 10 extreme models ===")
    adapter_paths = train_all_extreme_models(
        model_name, tokenizer, data, adapters_dir,
    )

    # Set up raw generation logger
    raw_logger = setup_raw_logger(output_dir)

    # Phase 2: Per-subject merge + eval
    print("\n=== Phase 2: Per-subject merge and evaluation ===")
    results, done_indices = load_completed_results(total, output_dir)
    if done_indices:
        print(f"Resuming eval: {len(done_indices)} subjects already completed.")

    for idx in tqdm(range(total), desc="Soups eval"):
        if idx in done_indices:
            continue

        subject_data = data[idx]
        case_id = subject_data['test'][0]['case']

        # Fresh base model each subject (PeftModel.from_pretrained wraps in-place)
        base_model, _ = load_base_model_and_tokenizer(model_name)

        # Merge adapters with subject-specific weights
        peft_model = merge_soup_for_subject(
            base_model, subject_data, adapter_paths,
        )

        # Evaluate
        raw_logger.info(f"=== subject={idx} case={case_id} ===")
        result = evaluate_subject(
            peft_model, tokenizer, subject_data, model_name,
            raw_logger=raw_logger,
        )
        results[idx] = result

        # Save checkpoint
        save_subject_result(result, idx, output_dir, method='Soups')

        # Log progress
        score_sum = sum(k[1] for k in result['mean_ver_abs']['mean'])
        append_progress({
            'index': idx,
            'case': case_id,
            'score_sum': score_sum,
            'mean_abs': {k: v for k, v in result['mean_ver_abs']['mean']},
        }, output_dir, filename='soups_progress.jsonl')

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Subject {idx}/{total} | "
              f"case={case_id} | score_sum={score_sum:.3f}")

        # Free merged model and base model
        del peft_model, base_model
        torch.cuda.empty_cache()

    # Aggregate results
    aggregate_and_save_results(results, 'Soups', model_name, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Personalized-Soups Benchmark")
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--num_subjects", type=int, default=0, help="0=all 300")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--data_dir", default="PAPI")
    args = parser.parse_args()

    run_soups(
        model_name=args.model_name,
        num_subjects=args.num_subjects,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
