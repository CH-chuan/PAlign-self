"""CLI entry point for PPO benchmark: per-subject train → eval → discard."""
import argparse
import os
import sys
from datetime import datetime

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from benchmarks.common.data import load_all_data
from benchmarks.common.evaluation import evaluate_subject, aggregate_and_save_results, setup_raw_logger
from benchmarks.common.model_utils import delete_adapter
from benchmarks.common.resume import (
    save_subject_result, load_completed_results, append_progress,
)
from benchmarks.ppo.trainer import train_ppo_for_subject, load_ppo_model_for_eval
from transformers import AutoTokenizer


def run_ppo(model_name, num_subjects=0, output_dir=None, data_dir='PAPI'):
    """Run the full PPO benchmark pipeline."""
    if output_dir is None:
        output_dir = os.path.join('reproduction', 'benchmarks', 'ppo')

    print(f"=== PPO Benchmark ===")
    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")

    # Load data
    data = load_all_data(data_dir=data_dir, num_subjects=num_subjects)
    total = len(data)
    print(f"Subjects: {total}")

    # Check for completed subjects
    results, done_indices = load_completed_results(total, output_dir)
    if done_indices:
        print(f"Resuming: {len(done_indices)} subjects already completed.")

    # Set up raw generation logger
    raw_logger = setup_raw_logger(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

    for idx in tqdm(range(total), desc="PPO subjects"):
        if idx in done_indices:
            continue

        subject_data = data[idx]
        case_id = subject_data['test'][0]['case']

        # Train PPO adapter for this subject
        tmp_adapter_dir = os.path.join(output_dir, 'tmp_adapter')
        ppo_model = train_ppo_for_subject(
            model_name, tokenizer, subject_data, tmp_adapter_dir,
        )

        if ppo_model is None:
            print(f"Subject {idx}: training failed, skipping.")
            continue

        # Free PPO model and load clean model for eval
        del ppo_model
        torch.cuda.empty_cache()

        eval_model = load_ppo_model_for_eval(model_name, tmp_adapter_dir, tokenizer)

        # Evaluate
        raw_logger.info(f"=== subject={idx} case={case_id} ===")
        result = evaluate_subject(
            eval_model, tokenizer, subject_data, model_name,
            raw_logger=raw_logger,
        )
        results[idx] = result

        # Save checkpoint
        save_subject_result(result, idx, output_dir, method='PPO')

        # Log progress
        score_sum = sum(k[1] for k in result['mean_ver_abs']['mean'])
        append_progress({
            'index': idx,
            'case': case_id,
            'score_sum': score_sum,
            'mean_abs': {k: v for k, v in result['mean_ver_abs']['mean']},
        }, output_dir, filename='ppo_progress.jsonl')

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Subject {idx}/{total} | "
              f"case={case_id} | score_sum={score_sum:.3f}")

        # Discard adapter
        del eval_model
        torch.cuda.empty_cache()
        delete_adapter(tmp_adapter_dir)

    # Aggregate results
    aggregate_and_save_results(results, 'PPO', model_name, output_dir)


def main():
    parser = argparse.ArgumentParser(description="PPO Benchmark")
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--num_subjects", type=int, default=0, help="0=all 300")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--data_dir", default="PAPI")
    args = parser.parse_args()

    run_ppo(
        model_name=args.model_name,
        num_subjects=args.num_subjects,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
