"""Unified entry point for running all benchmark methods."""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

METHODS = {
    'dpo': ('benchmarks.dpo.run_dpo', 'run_dpo'),
    'ppo': ('benchmarks.ppo.run_ppo', 'run_ppo'),
    'prompt_morl': ('benchmarks.prompt_morl.run_prompt_morl', 'run_prompt_morl'),
    'soups': ('benchmarks.soups.run_soups', 'run_soups'),
}


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmarking baselines (DPO, PPO, Prompt-MORL, Soups)",
    )
    parser.add_argument(
        "--methods", nargs='+',
        choices=list(METHODS.keys()) + ['all'],
        default=['all'],
        help="Methods to run (default: all)",
    )
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--num_subjects", type=int, default=0, help="0=all 300")
    parser.add_argument("--output_base", default=os.path.join('reproduction', 'benchmarks'),
                        help="Base output directory")
    parser.add_argument("--data_dir", default="PAPI")
    args = parser.parse_args()

    methods_to_run = list(METHODS.keys()) if 'all' in args.methods else args.methods

    for method in methods_to_run:
        module_path, func_name = METHODS[method]
        print(f"\n{'='*60}")
        print(f"Running: {method}")
        print(f"{'='*60}\n")

        module = __import__(module_path, fromlist=[func_name])
        run_func = getattr(module, func_name)

        output_dir = os.path.join(args.output_base, method)
        run_func(
            model_name=args.model_name,
            num_subjects=args.num_subjects,
            output_dir=output_dir,
            data_dir=args.data_dir,
        )

    print(f"\n{'='*60}")
    print("All methods complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
