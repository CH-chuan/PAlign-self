import csv
import json
import os
import pickle
import re
import numpy as np
from datetime import datetime
from tqdm import tqdm

# Constants (you might want to import these from a common config file)
SCORES = {
    "A": 5, "B": 4, "C": 3, "D": 2, "E": 1
}

SCORES_BACK = {
    5: 'Very Accurate',
    4: 'Moderately Accurate',
    3: 'Neither Accurate Nor Inaccurate',
    2: 'Moderately Inaccurate',
    1: 'Very Inaccurate',
    0: 'Unknown'
}


def calc_mean_and_var(result):
    """
    Calculate mean and variance of results.
    """
    mean = {}
    std = {}
    for key, item in result.items():
        mean[key] = np.mean(np.array(item))
        std[key] = np.std(np.array(item))

    return {
        "mean": list(sorted(mean.items(), key=lambda item: item[0])),
        "std": list(sorted(std.items(), key=lambda item: item[0])),
    }



def process_answers(answers,sample):
    """
    Process answers and calculate results.
    """

    global_result = {'A': [], 'C': [], 'E': [], 'N': [], 'O': []}
    global_cnt = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "UNK": 0}
    global_result_abs = {'A': [], 'C': [], 'E': [], 'N': [], 'O': []}
    rows = []
    answer_number = 0
    for item in sample['test']:
        label = item["label_ocean"]
        key = item["key"]
        raw_answer = answers[answer_number]
        parsed_result = re.search(r"[abcdeABCDE][^a-zA-Z]", raw_answer[:12], flags=0)
        row = {
            'question_idx': answer_number,
            'trait': label,
            'key': key,
            'ground_truth': item['value'],
            'raw_answer': raw_answer[:50].replace('\n', ' ').strip(),
        }
        if parsed_result:
            parsed_result = parsed_result.group()[0].upper()
            # this step is to calculate the alignment between the model's answer and the item's value
            error = abs(SCORES[parsed_result] - item['value'])
            global_cnt[parsed_result] += 1
            global_result_abs[label].append(error)

            # now we re-arrange score to be 1-5, so that we can use it to calculate the mean of the model's answer
            score = SCORES[parsed_result]
            if key == 1:
                global_result[label].append(score)
            else:
                global_result[label].append(6 - score)
            row['parsed'] = parsed_result
            row['score'] = score if key == 1 else 6 - score
            row['error'] = error
        else:
            global_cnt["UNK"] += 1
            row['parsed'] = 'UNK'
            row['score'] = ''
            row['error'] = ''
        rows.append(row)
        answer_number += 1

    mean_var = calc_mean_and_var(global_result)
    mean_var_abs = calc_mean_and_var(global_result_abs)
    result_file = {
        'case': sample['test'][0]['case'],
        'result': global_result,
        'count': global_cnt,
        'mean_ver': mean_var,
        'mean_ver_abs': mean_var_abs,
        'rows': rows,
    }

    return result_file


def _deep_convert(obj):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _deep_convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_deep_convert(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_subject_meta(path, *, result=None, subject_index=None, **fields):
    """Write a subject meta JSON file.

    Args:
        path: Output file path (e.g. .../subject_0000_meta.json)
        result: Optional result dict from process_answers(). If provided, extracts
                case_id, count, per_trait_mae/std, per_trait_mean/std.
        subject_index: Subject index (added to meta if provided).
        **fields: Additional fields (model_file, method, alpha, alpha_mode,
                  modified_heads, modified_layers, per_trait_mae, etc.).
                  Override result-derived fields if both provided.
    """
    meta = {}
    if subject_index is not None:
        meta['subject_index'] = subject_index

    if result:
        meta['case_id'] = result['case']
        meta['count'] = result['count']
        meta['per_trait_mae'] = {k: round(v, 4) for k, v in result['mean_ver_abs']['mean']}
        meta['per_trait_mae_std'] = {k: round(v, 4) for k, v in result['mean_ver_abs']['std']}
        meta['per_trait_mean'] = {k: round(v, 4) for k, v in result['mean_ver']['mean']}
        meta['per_trait_std'] = {k: round(v, 4) for k, v in result['mean_ver']['std']}

    # Explicit fields override result-derived ones
    meta.update({k: v for k, v in fields.items() if v is not None})

    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = _deep_convert(meta)
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2)


def save_subject_answers(result, index, output_dir, alpha=None):
    """Save subject_XXXX_answers.csv from result['rows']."""
    results_dir = os.path.join(output_dir, 'subject_results')
    os.makedirs(results_dir, exist_ok=True)
    rows = result.get('rows', [])
    if not rows:
        return
    if alpha is not None:
        path = os.path.join(results_dir, f'subject_{index:04d}_answers_alpha{alpha}.csv')
    else:
        path = os.path.join(results_dir, f'subject_{index:04d}_answers.csv')
    fieldnames = ['question_idx', 'trait', 'key', 'ground_truth', 'raw_answer', 'parsed', 'score', 'error']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_subject_probes(all_head_accs, top_heads, index, output_dir):
    """Save subject_XXXX_probes.csv — all heads' probe train/val accuracy."""
    results_dir = os.path.join(output_dir, 'subject_results')
    os.makedirs(results_dir, exist_ok=True)
    num_layers, num_heads = all_head_accs.shape[:2]
    selected_set = set((l, h) for l, h in top_heads)
    path = os.path.join(results_dir, f'subject_{index:04d}_probes.csv')
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


def load_completed_indices(num_subjects, output_dir):
    """Load completed subject results from meta JSON files.

    Returns:
        results: list of length num_subjects (None for incomplete)
        done_indices: set of completed subject indices
    """
    results = [None] * num_subjects
    done_indices = set()
    results_dir = os.path.join(output_dir, 'subject_results')
    if not os.path.isdir(results_dir):
        return results, done_indices
    for idx in range(num_subjects):
        meta_path = os.path.join(results_dir, f'subject_{idx:04d}_meta.json')
        if not os.path.exists(meta_path):
            # Backward compat: check for legacy pickle
            pkl_path = os.path.join(results_dir, f'subject_{idx:04d}.pkl')
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    results[idx] = pickle.load(f)
                done_indices.add(idx)
            continue
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        # Reconstruct the result dict format expected by aggregation
        results[idx] = _meta_to_result(meta)
        done_indices.add(idx)
    return results, done_indices


def _meta_to_result(meta):
    """Reconstruct the result dict from a meta JSON for aggregation compatibility."""
    trait_order = ['A', 'C', 'E', 'N', 'O']
    mean_ver = {
        'mean': [(t, meta['per_trait_mean'][t]) for t in trait_order],
        'std': [(t, meta['per_trait_std'][t]) for t in trait_order],
    }
    mae_std = meta.get('per_trait_mae_std', {t: 0.0 for t in trait_order})
    mean_ver_abs = {
        'mean': [(t, meta['per_trait_mae'][t]) for t in trait_order],
        'std': [(t, mae_std[t]) for t in trait_order],
    }
    result = {
        'case': meta['case_id'],
        'count': meta['count'],
        'mean_ver': mean_ver,
        'mean_ver_abs': mean_ver_abs,
    }
    if 'alpha' in meta:
        result['alpha'] = meta['alpha']
    return result


def process_few_shot(data, model, tokenizer, model_file, batch_size=16,
                     output_dir=None, raw_logger=None):
    """
    Process data using few-shot learning method.
    """
    from main import generateAnswer, TEMPLATE

    # Resume setup
    if output_dir:
        results, done_indices = load_completed_indices(len(data), output_dir)
        progress_path = os.path.join(output_dir, 'few-shot_progress.jsonl')
        if done_indices:
            print(f"Resuming: {len(done_indices)} subjects already completed, skipping them.")
    else:
        results = [None] * len(data)
        done_indices = set()

    for index, i in enumerate(tqdm(data)):
        if index in done_indices:
            continue

        system_prompt_text = 'Here are some of your behaviors and your level of recognition towards them' + \
                             ';'.join([f"{it['text']}:{SCORES_BACK[it['value']]}" for it in i['train']])
        answers = generateAnswer(tokenizer, model, i['test'], TEMPLATE, scores=SCORES,
                                  system_prompt=system_prompt_text, model_file=model_file,
                                  raw_logger=raw_logger, batch_size=batch_size)
        rs = process_answers(answers, i)
        results[index] = rs

        # Save meta + answers + progress
        if output_dir:
            meta_path = os.path.join(output_dir, 'subject_results', f'subject_{index:04d}_meta.json')
            save_subject_meta(meta_path, result=rs, subject_index=index, method='few-shot')
            save_subject_answers(rs, index, output_dir)
            progress_entry = {
                'index': index,
                'case': rs['case'],
                'mean_abs': {k: v for k, v in rs['mean_ver_abs']['mean']},
                'timestamp': datetime.now().isoformat()
            }
            with open(progress_path, 'a') as f:
                f.write(json.dumps(progress_entry) + '\n')

    return [r for r in results if r is not None]


def process_personality_prompt(data, model, tokenizer, model_file, batch_size=16,
                               output_dir=None, raw_logger=None):
    """
    Process data using personality prompts method.
    """
    from main import generateAnswer, TEMPLATE
    system_prompt = json.load(open('PAPI/personality_prompt.json'))

    # Resume setup
    if output_dir:
        results, done_indices = load_completed_indices(len(data), output_dir)
        progress_path = os.path.join(output_dir, 'personality_prompt_progress.jsonl')
        if done_indices:
            print(f"Resuming: {len(done_indices)} subjects already completed, skipping them.")
    else:
        results = [None] * len(data)
        done_indices = set()

    for index, i in enumerate(tqdm(data)):
        if index in done_indices:
            continue

        system_prompt_text = system_prompt[index]['output'][0]
        answers = generateAnswer(tokenizer, model, i['test'], TEMPLATE, system_prompt=system_prompt_text,
                                 model_file=model_file, raw_logger=raw_logger, batch_size=batch_size)
        rs = process_answers(answers, i)
        results[index] = rs

        # Save meta + answers + progress
        if output_dir:
            meta_path = os.path.join(output_dir, 'subject_results', f'subject_{index:04d}_meta.json')
            save_subject_meta(meta_path, result=rs, subject_index=index, method='personality_prompt')
            save_subject_answers(rs, index, output_dir)
            progress_entry = {
                'index': index,
                'case': rs['case'],
                'mean_abs': {k: v for k, v in rs['mean_ver_abs']['mean']},
                'timestamp': datetime.now().isoformat()
            }
            with open(progress_path, 'a') as f:
                f.write(json.dumps(progress_entry) + '\n')

    return [r for r in results if r is not None]
