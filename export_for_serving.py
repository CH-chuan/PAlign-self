"""
Export PAS-steered models for vLLM serving.

Two subcommands:
  export-biases  Run PAS for a subject, then save bias deltas to a compact file.
  bake           Apply saved bias deltas to a base model and write a full HF checkpoint.

Usage:
  # Full PAS alpha sweep (default — picks best alpha automatically)
  python export_for_serving.py export-biases \
    --model_file meta-llama/Meta-Llama-3-8B-Instruct \
    --subject_index 42 --output persona_biases.pt

  # Fixed alpha (skip sweep)
  python export_for_serving.py export-biases \
    --model_file meta-llama/Meta-Llama-3-8B-Instruct \
    --subject_index 42 --alpha 4 --output persona_biases.pt

  python export_for_serving.py bake \
    --model_file meta-llama/Meta-Llama-3-8B-Instruct \
    --biases persona_biases.pt --output_dir ./baked_model

  vllm serve ./baked_model --dtype float16
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from copy import deepcopy
from tqdm import tqdm

from PAlign.pas import get_model
from main import getItems, from_index_to_data, TEMPLATE, NEUTRAL_SYSTEM_PROMPT, generateAnswer, build_few_shot_prompt
from baseline_utils import process_answers


def run_export_biases(args):
    """Run PAS pipeline for one subject, export the resulting biases."""
    # Load data
    dataset, text_file, train_index, test_index = getItems('PAPI')
    data = from_index_to_data(train_index, test_index, text_file, dataset, 'OOD')

    if args.subject_index >= len(data):
        print(f"Error: subject_index {args.subject_index} out of range (max {len(data) - 1})")
        sys.exit(1)

    sample = data[args.subject_index]

    # Load model
    model, tokenizer = get_model(args.model_file)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

    # Build activation dataset from training items (same as process_pas)
    personal_data = []
    for personal in ['A', 'C', 'E', 'N', 'O']:
        for item in data[0]['train']:
            if item['label_ocean'] == personal:
                personal_data.append({
                    'question': TEMPLATE.format(item['text']),
                    'answer_matching_behavior': 'A',
                    'answer_not_matching_behavior': 'E'
                })

    print(f"Extracting activations from {len(personal_data)} items...")
    all_head_wise_activations = model.preprocess_activate_dataset(personal_data)

    # Prepare labels and activations for this subject
    labels = []
    head_wise_activations = []
    personal_number = 0
    for personal in ['A', 'C', 'E', 'N', 'O']:
        for item in sample['train']:
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
                personal_number += 2

    # Train probes and get intervention vectors
    print("Training probes and computing interventions...")
    interventions, top_heads, all_head_accs = model.get_activations(
        deepcopy(head_wise_activations), labels, num_to_intervene=24
    )

    if args.alpha is not None:
        # Fixed alpha mode — skip sweep
        model.reset_all()
        model.set_activate(interventions, args.alpha)
        best_alpha = args.alpha
        alpha_mode = 'fixed'
        per_trait_mae = {}
        print(f"Using fixed alpha={best_alpha}")
    else:
        # Full PAS alpha sweep (paper's method)
        alpha_values = [0, 1, 2, 4, 6, 8]
        result_cache = []
        print(f"Running alpha sweep {alpha_values} on {len(sample['test'])} test items...")

        for alpha in alpha_values:
            model.reset_all()
            model.set_activate(interventions, alpha)

            answers = generateAnswer(
                tokenizer, model, data[0]['test'], TEMPLATE,
                system_prompt=build_few_shot_prompt(sample['train']), model_file=args.model_file,
                batch_size=args.batch_size,
            )
            result = process_answers(answers, sample)
            result_cache.append(result)

            score = sum(k[1] for k in result['mean_ver_abs']['mean'])
            print(f"  alpha={alpha}: MAE sum={score:.3f}")

        # Pick best alpha (lowest MAE sum)
        scores = []
        for p in result_cache:
            score = sum(k[1] for k in p['mean_ver_abs']['mean'])
            if str(score) == 'nan':
                score = 1e6
            scores.append(score)
        best_idx = int(np.array(scores).argmin())
        best_alpha = alpha_values[best_idx]
        alpha_mode = 'sweep'
        per_trait_mae = dict(result_cache[best_idx]['mean_ver_abs']['mean'])

        print(f"Best alpha={best_alpha} (MAE sum={scores[best_idx]:.3f})")
        for trait, mae in sorted(per_trait_mae.items()):
            print(f"  {trait}: {mae:.3f}")

        # Re-apply best alpha
        model.reset_all()
        model.set_activate(interventions, best_alpha)

    # Export biases
    bias_deltas = model.export_biases(args.output)

    # Save metadata alongside
    meta_path = args.output.rsplit('.', 1)[0] + '_meta.json'
    meta = {
        'model_file': args.model_file,
        'subject_index': args.subject_index,
        'case_id': sample['train'][0]['case'],
        'alpha': best_alpha,
        'alpha_mode': alpha_mode,
        'num_to_intervene': 24,
        'modified_heads': [[int(x) for x in h] for h in top_heads],
        'num_modified_layers': len(bias_deltas),
        'modified_layers': sorted(bias_deltas.keys()),
    }
    if per_trait_mae:
        meta['per_trait_mae'] = {k: round(v, 4) for k, v in per_trait_mae.items()}
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")


def run_bake(args):
    """Load base model, apply saved bias deltas, save as full HF checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading bias deltas from {args.biases}...")
    bias_deltas = torch.load(args.biases, map_location='cpu', weights_only=True)
    print(f"  {len(bias_deltas)} layers to modify: {sorted(bias_deltas.keys())}")

    print(f"Loading base model {args.model_file}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_file, dtype=torch.float16, low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_file)

    # Enable attention_bias in config so that from_pretrained() will create
    # o_proj with bias=True and correctly load our bias tensors on reload.
    # Without this, config says attention_bias=false → bias-free Linear layers
    # → saved bias weights become "unexpected keys" and are silently dropped.
    model.config.attention_bias = True

    # Add zero biases to all attention projection layers. attention_bias=True
    # makes from_pretrained() expect biases on q/k/v/o_proj. Without saving
    # them, q/k/v biases get randomly initialized on reload, corrupting output.
    # Zero bias is mathematically identical to no bias.
    for layer in model.model.layers:
        attn = layer.self_attn
        for proj in [attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj]:
            if proj.bias is None:
                proj.bias = torch.nn.Parameter(
                    torch.zeros(proj.out_features, dtype=proj.weight.dtype)
                )

    for layer_idx, bias_value in bias_deltas.items():
        o_proj = model.model.layers[layer_idx].self_attn.o_proj
        o_proj.bias = torch.nn.Parameter(bias_value.to(torch.float16))
        print(f"  Layer {layer_idx}: PAS bias applied")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving baked model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Copy bias metadata if available
    meta_path = args.biases.rsplit('.', 1)[0] + '_meta.json'
    if os.path.exists(meta_path):
        import shutil
        shutil.copy2(meta_path, os.path.join(args.output_dir, 'persona_meta.json'))

    print(f"Done. Serve with: vllm serve {args.output_dir} --dtype float16")


def main():
    parser = argparse.ArgumentParser(
        description="Export PAS-steered models for vLLM serving"
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # export-biases subcommand
    p_export = subparsers.add_parser(
        'export-biases',
        help='Run PAS for one subject and export bias deltas'
    )
    p_export.add_argument('--model_file', required=True, help='HuggingFace model name or path')
    p_export.add_argument('--subject_index', type=int, required=True, help='Subject index (0-299)')
    p_export.add_argument('--alpha', type=float, default=None,
                          help='Fixed steering strength (e.g. 4). If omitted, runs full 6-alpha sweep.')
    p_export.add_argument('--batch_size', type=int, default=16,
                          help='Inference batch size for alpha sweep (default 16 for A100, use 3 for RTX 4090)')
    p_export.add_argument('--output', default='persona_biases.pt', help='Output .pt file path')

    # bake subcommand
    p_bake = subparsers.add_parser(
        'bake',
        help='Apply bias deltas to base model and save full checkpoint'
    )
    p_bake.add_argument('--model_file', required=True, help='HuggingFace model name or path')
    p_bake.add_argument('--biases', required=True, help='Path to bias deltas .pt file')
    p_bake.add_argument('--output_dir', required=True, help='Output directory for baked model')

    args = parser.parse_args()

    if args.command == 'export-biases':
        run_export_biases(args)
    elif args.command == 'bake':
        run_bake(args)


if __name__ == '__main__':
    main()
