"""
Evaluate a baked PAS-steered model served via vLLM using the OpenAI SDK.

Usage:
  python eval_served.py \
    --model_dir ./baked_model \
    --api_base http://localhost:8000/v1 \
    --subject_index 42 \
    --output eval_result.json
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

from openai import OpenAI
from tqdm import tqdm

from main import getItems, from_index_to_data, TEMPLATE, build_few_shot_prompt
from baseline_utils import process_answers


def query_model(client, model_name, system_prompt, user_prompt, prefill="Option"):
    """Send a single completion request via OpenAI SDK with assistant prefill."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    extra = {}
    if prefill:
        messages.append({"role": "assistant", "content": prefill})
        extra["extra_body"] = {
            "add_generation_prompt": False,
            "continue_final_message": True,
        }

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=10,
        temperature=0,
        **extra,
    )
    text = response.choices[0].message.content or ""
    if prefill:
        text = prefill + text
    return text


def main():
    parser = argparse.ArgumentParser(description="Evaluate a served PAS-steered model")
    parser.add_argument("--model_dir", default=None,
                        help="Path to baked model dir (reads persona_meta.json for subject_index)")
    parser.add_argument("--api_base", default="http://localhost:8000/v1",
                        help="vLLM server base URL")
    parser.add_argument("--model_name", default=None,
                        help="Model name for the API (default: auto-detect from vLLM)")
    parser.add_argument("--subject_index", type=int, default=None,
                        help="Subject index (0-299). Overrides persona_meta.json.")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--raw_log", default=None, help="Path to raw generation log file")
    args = parser.parse_args()

    # Resolve subject_index
    subject_index = args.subject_index
    if subject_index is None and args.model_dir:
        meta_path = os.path.join(args.model_dir, "persona_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            subject_index = meta["subject_index"]
            print(f"Read subject_index={subject_index} from {meta_path}")
        else:
            print(f"Error: No persona_meta.json in {args.model_dir} and --subject_index not provided")
            sys.exit(1)
    if subject_index is None:
        print("Error: Provide --subject_index or --model_dir with persona_meta.json")
        sys.exit(1)

    # Create OpenAI client pointing at vLLM
    client = OpenAI(base_url=args.api_base, api_key="unused")

    # Auto-detect model name from vLLM
    model_name = args.model_name
    if model_name is None:
        try:
            models = client.models.list()
            model_name = models.data[0].id
            print(f"Auto-detected model: {model_name}")
        except Exception as e:
            print(f"Error: Could not auto-detect model name: {e}")
            print("Provide --model_name explicitly.")
            sys.exit(1)

    # Load PAPI data
    dataset, text_file, train_index, test_index = getItems("PAPI")
    data = from_index_to_data(train_index, test_index, text_file, dataset, "OOD")

    if subject_index >= len(data):
        print(f"Error: subject_index {subject_index} out of range (max {len(data) - 1})")
        sys.exit(1)

    sample = data[subject_index]
    system_prompt = build_few_shot_prompt(sample["train"])

    # Set up raw generation logger
    raw_logger = None
    if args.raw_log:
        os.makedirs(os.path.dirname(args.raw_log) or ".", exist_ok=True)
        raw_logger = logging.getLogger("raw_gen")
        raw_logger.setLevel(logging.INFO)
        raw_logger.propagate = False
        fh = logging.FileHandler(args.raw_log, mode="w")
        fh.setFormatter(logging.Formatter("%(message)s"))
        raw_logger.addHandler(fh)

    # Query the model for each test item
    test_items = sample["test"]
    answers = []
    print(f"Evaluating subject {subject_index} ({len(test_items)} test items)...")
    for qi, item in enumerate(tqdm(test_items)):
        question = TEMPLATE.format(item["text"].lower())
        ans = query_model(client, model_name, system_prompt, question)
        answers.append(ans)
        if raw_logger:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
            raw_logger.info("[%s] q=%d | %s", ts, qi, ans.replace("\n", " "))

    # Score
    result = process_answers(answers, sample)

    # Print per-trait MAE
    print("\nPer-trait MAE:")
    mae_dict = {}
    for trait, value in result["mean_ver_abs"]["mean"]:
        mae_dict[trait] = round(value, 4)
        print(f"  {trait}: {value:.4f}")
    mae_sum = sum(mae_dict.values())
    print(f"  SUM: {mae_sum:.4f}")

    print(f"\nAnswer distribution: {result['count']}")

    # Save if requested
    if args.output:
        out = {
            "subject_index": subject_index,
            "case_id": result["case"],
            "per_trait_mae": mae_dict,
            "mae_sum": round(mae_sum, 4),
            "count": result["count"],
            "mean_ver": result["mean_ver"],
            "mean_ver_abs": result["mean_ver_abs"],
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
