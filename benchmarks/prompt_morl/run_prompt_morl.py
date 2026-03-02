"""CLI entry point for Prompt-MORL benchmark: train once → eval all subjects."""
import argparse
import os
import sys
from datetime import datetime

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from benchmarks.common.data import load_all_data, get_subject_trait_means, get_personality_prefix
from benchmarks.common.model_utils import load_base_model_and_tokenizer, load_adapter
from benchmarks.common.evaluation import evaluate_subject, aggregate_and_save_results
from benchmarks.common.resume import (
    save_subject_result, load_completed_results, append_progress,
)
from benchmarks.prompt_morl.trainer import train_prompt_morl


def run_prompt_morl(model_name, num_subjects=0, output_dir=None, data_dir='PAPI'):
    """Run the full Prompt-MORL benchmark pipeline."""
    if output_dir is None:
        output_dir = os.path.join('reproduction', 'benchmarks', 'prompt_morl')

    adapter_dir = os.path.join(output_dir, 'adapter')

    print(f"=== Prompt-MORL Benchmark ===")
    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")

    # Load data
    data = load_all_data(data_dir=data_dir, num_subjects=num_subjects)
    total = len(data)
    print(f"Subjects: {total}")

    # Load base model
    base_model, tokenizer = load_base_model_and_tokenizer(model_name)

    # Train phase: check if adapter already exists
    if os.path.exists(os.path.join(adapter_dir, 'adapter_config.json')):
        print("Adapter already trained, loading...")
        peft_model = load_adapter(base_model, adapter_dir)
    else:
        print("Training Prompt-MORL adapter...")
        peft_model = train_prompt_morl(
            base_model, tokenizer, data, adapter_dir,
        )

    # Eval phase: per-subject evaluation with personality prefix
    results, done_indices = load_completed_results(total, output_dir)
    if done_indices:
        print(f"Resuming eval: {len(done_indices)} subjects already completed.")

    for idx in tqdm(range(total), desc="Prompt-MORL eval"):
        if idx in done_indices:
            continue

        subject_data = data[idx]
        case_id = subject_data['test'][0]['case']

        # Build personality-conditioned system prompt
        trait_means = get_subject_trait_means(subject_data)
        personality_prefix = get_personality_prefix(trait_means)
        system_prompt = personality_prefix + " You are a helpful, honest and concise assistant."

        # Evaluate with this subject's personality prefix
        result = evaluate_subject(
            peft_model, tokenizer, subject_data, model_name,
            system_prompt=system_prompt,
        )
        results[idx] = result

        # Save checkpoint
        save_subject_result(result, idx, output_dir)

        # Log progress
        score_sum = sum(k[1] for k in result['mean_ver_abs']['mean'])
        append_progress({
            'index': idx,
            'case': case_id,
            'score_sum': score_sum,
            'mean_abs': {k: v for k, v in result['mean_ver_abs']['mean']},
        }, output_dir, filename='prompt_morl_progress.jsonl')

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Subject {idx}/{total} | "
              f"case={case_id} | score_sum={score_sum:.3f}")

    # Aggregate results
    aggregate_and_save_results(results, 'Prompt-MORL', model_name, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Prompt-MORL Benchmark")
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--num_subjects", type=int, default=0, help="0=all 300")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--data_dir", default="PAPI")
    args = parser.parse_args()

    run_prompt_morl(
        model_name=args.model_name,
        num_subjects=args.num_subjects,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
