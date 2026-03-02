"""Shared evaluation pipeline for all benchmark methods."""
import json
import os
import re

import numpy as np
import torch

from benchmarks.common.config import SCORES, TEMPLATE, SYSTEM_PROMPT


def prompt_to_tokens_hf(tokenizer, system_prompt, instruction, model_output, model_name):
    """Convert prompt to token IDs, handling different model templates.

    Works with standard HF/PEFT models (not the custom PASLM class).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},
    ]
    if model_output:
        messages.append({"role": "assistant", "content": model_output})

    token_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=not model_output)

    # Trim the trailing EOS if model_output is provided (for forced prefix)
    if model_output:
        token_ids = token_ids[:-1]

    return token_ids


def generate_answers(model, tokenizer, test_items, model_name,
                     system_prompt=SYSTEM_PROMPT, batch_size=3):
    """Generate answers for test items using a standard HF/PEFT model.

    Follows the same pattern as main.py:generateAnswer() but works with
    any HuggingFace model (including PEFT-wrapped models).

    Returns:
        list of raw answer strings
    """
    questions = [item["text"].lower() for item in test_items]
    answers = []

    model.eval()
    for batch_start in range(0, len(questions), batch_size):
        batch_questions = questions[batch_start:batch_start + batch_size]
        input_ids_list = [
            prompt_to_tokens_hf(
                tokenizer, system_prompt,
                TEMPLATE.format(q), 'Option', model_name
            )
            for q in batch_questions
        ]

        # Pad to same length
        max_len = max(len(ids) for ids in input_ids_list)
        pad_id = tokenizer.pad_token_id
        padded = []
        attention_masks = []
        for ids in input_ids_list:
            pad_len = max_len - len(ids)
            padded.append([pad_id] * pad_len + ids)  # left-pad
            attention_masks.append([0] * pad_len + [1] * len(ids))

        input_ids = torch.tensor(padded, device=model.device)
        attention_mask = torch.tensor(attention_masks, device=model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=15,
                do_sample=False,
            )

        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=False)

        for text in output_text:
            # Extract the last assistant response
            if '<|end_header_id|>' in text:
                answer = text.split("<|end_header_id|>")[-1]
            elif '<|im_start|>assistant' in text:
                answer = text.split("<|im_start|>assistant")[-1]
            elif '[/INST]' in text:
                answer = text.split("[/INST]")[-1]
            else:
                answer = text
            answers.append(answer)

    return answers


def evaluate_subject(model, tokenizer, subject_data, model_name,
                     system_prompt=SYSTEM_PROMPT, batch_size=3):
    """Evaluate a single subject: generate answers and score.

    Uses process_answers from baseline_utils.

    Returns:
        result dict from process_answers()
    """
    from baseline_utils import process_answers
    answers = generate_answers(
        model, tokenizer, subject_data['test'], model_name,
        system_prompt=system_prompt, batch_size=batch_size,
    )
    return process_answers(answers, subject_data)


def aggregate_and_save_results(results, method_name, model_name, output_dir,
                               dataset_set='OOD'):
    """Aggregate per-subject results and save JSON matching PAS output format.

    The output JSON has the same structure as reproduction/PAS_*_OOD.json.
    """
    def lmean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    # Filter out None results
    results = [r for r in results if r is not None]

    mean_data = [r['mean_ver']['mean'] for r in results]
    mean_A, mean_C, mean_E, mean_N, mean_O = (
        [row[j][1] for row in mean_data] for j in range(5)
    )

    std_data = [r['mean_ver']['std'] for r in results]
    std_A, std_C, std_E, std_N, std_O = (
        [row[j][1] for row in std_data] for j in range(5)
    )

    mean_abs_data = [r['mean_ver_abs']['mean'] for r in results]
    mean_A_abs, mean_C_abs, mean_E_abs, mean_N_abs, mean_O_abs = (
        [row[j][1] for row in mean_abs_data] for j in range(5)
    )

    std_abs_data = [r['mean_ver_abs']['std'] for r in results]
    std_A_abs, std_C_abs, std_E_abs, std_N_abs, std_O_abs = (
        [row[j][1] for row in std_abs_data] for j in range(5)
    )

    log = {
        'score': {
            'mean_A': lmean(mean_A), 'mean_C': lmean(mean_C),
            'mean_E': lmean(mean_E), 'mean_N': lmean(mean_N),
            'mean_O': lmean(mean_O),
            'std_A': lmean(std_A), 'std_C': lmean(std_C),
            'std_E': lmean(std_E), 'std_N': lmean(std_N),
            'std_O': lmean(std_O),
            'mean_A_abs': lmean(mean_A_abs), 'mean_C_abs': lmean(mean_C_abs),
            'mean_E_abs': lmean(mean_E_abs), 'mean_N_abs': lmean(mean_N_abs),
            'mean_O_abs': lmean(mean_O_abs),
            'std_A_abs': lmean(std_A_abs), 'std_C_abs': lmean(std_C_abs),
            'std_E_abs': lmean(std_E_abs), 'std_N_abs': lmean(std_N_abs),
            'std_O_abs': lmean(std_O_abs),
        },
        'mean': {
            'A': mean_A, 'C': mean_C, 'E': mean_E, 'N': mean_N, 'O': mean_O,
        },
        'std': {
            'A': std_A, 'C': std_C, 'E': std_E, 'N': std_N, 'O': std_O,
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    short_model = model_name.split('/')[-1]
    filename = os.path.join(output_dir, f'{method_name}_{short_model}_{dataset_set}.json')
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {filename}")
    from pprint import pprint
    pprint(log['score'])

    return log
