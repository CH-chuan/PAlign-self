"""CLI entry point for DPO benchmark: per-subject train → eval → discard."""
import argparse
import os
import sys
import tempfile
from datetime import datetime

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from benchmarks.common.config import QLoRAConfig, DPOHyperparams
from benchmarks.common.data import load_all_data
from benchmarks.common.model_utils import load_base_model_and_tokenizer, delete_adapter
from benchmarks.common.evaluation import evaluate_subject, aggregate_and_save_results, setup_raw_logger
from benchmarks.common.resume import (
    save_subject_result, load_completed_results, append_progress,
)
from benchmarks.dpo.trainer import train_dpo_for_subject


def run_dpo(model_name, num_subjects=0, output_dir=None, data_dir='PAPI'):
    """Run the full DPO benchmark pipeline."""
    if output_dir is None:
        output_dir = os.path.join('reproduction', 'benchmarks', 'dpo')

    print(f"=== DPO Benchmark ===")
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

    # Load tokenizer once (base model is reloaded per subject to avoid stale LoRA wrappers)
    _, tokenizer = load_base_model_and_tokenizer(model_name)

    for idx in tqdm(range(total), desc="DPO subjects"):
        if idx in done_indices:
            continue

        subject_data = data[idx]
        case_id = subject_data['test'][0]['case']

        # Fresh base model each subject (apply_qlora wraps in-place)
        base_model, _ = load_base_model_and_tokenizer(model_name)

        # Train DPO adapter for this subject
        tmp_adapter_dir = os.path.join(output_dir, 'tmp_adapter')
        peft_model = train_dpo_for_subject(
            base_model, tokenizer, subject_data, tmp_adapter_dir,
        )

        if peft_model is None:
            print(f"Subject {idx}: no valid training pairs, skipping.")
            del base_model
            torch.cuda.empty_cache()
            continue

        # Evaluate
        raw_logger.info(f"=== subject={idx} case={case_id} ===")
        result = evaluate_subject(
            peft_model, tokenizer, subject_data, model_name,
            raw_logger=raw_logger,
        )
        results[idx] = result

        # Save checkpoint
        save_subject_result(result, idx, output_dir, method='DPO')

        # Log progress
        score_sum = sum(k[1] for k in result['mean_ver_abs']['mean'])
        append_progress({
            'index': idx,
            'case': case_id,
            'score_sum': score_sum,
            'mean_abs': {k: v for k, v in result['mean_ver_abs']['mean']},
        }, output_dir, filename='dpo_progress.jsonl')

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Subject {idx}/{total} | "
              f"case={case_id} | score_sum={score_sum:.3f}")

        # Discard adapter and base model: free memory
        del peft_model, base_model
        torch.cuda.empty_cache()
        delete_adapter(tmp_adapter_dir)

    # Aggregate results
    aggregate_and_save_results(results, 'DPO', model_name, output_dir)


def main():
    parser = argparse.ArgumentParser(description="DPO Benchmark")
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--num_subjects", type=int, default=0, help="0=all 300")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--data_dir", default="PAPI")
    args = parser.parse_args()

    run_dpo(
        model_name=args.model_name,
        num_subjects=args.num_subjects,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
