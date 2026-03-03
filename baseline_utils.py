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
    answer_number = 0
    for item in sample['test']:
        label = item["label_ocean"]
        key = item["key"]
        parsed_result = re.search(r"[abcdeABCDE][^a-zA-Z]", answers[answer_number][:12], flags=0)
        if parsed_result:
            parsed_result = parsed_result.group()[0].upper()
            # this step is to calculate the alignment between the model's answer and the item's value
            score = abs(SCORES[parsed_result] - item['value'])
            global_cnt[parsed_result] += 1
            global_result_abs[label].append(score)

            # now we re-arrange score to be 1-5, so that we can use it to calculate the mean of the model's answer
            score = SCORES[parsed_result]
            if key == 1:
                global_result[label].append(score)
            else:
                global_result[label].append(6 - score)
        else:
            global_cnt["UNK"] += 1
        answer_number += 1

    mean_var = calc_mean_and_var(global_result)
    mean_var_abs = calc_mean_and_var(global_result_abs)
    result_file = {
        'case': sample['test'][0]['case'],
        'result': global_result,
        'count': global_cnt,
        'mean_ver': mean_var, 
        'mean_ver_abs': mean_var_abs # 
    }

    return result_file


def process_few_shot(data, model, tokenizer, model_file, batch_size=16,
                     output_dir=None, raw_logger=None):
    """
    Process data using few-shot learning method.
    """
    from main import generateAnswer, TEMPLATE

    # Resume setup
    results = [None] * len(data)
    done_indices = set()
    if output_dir:
        results_dir = os.path.join(output_dir, 'subject_results')
        os.makedirs(results_dir, exist_ok=True)
        progress_path = os.path.join(output_dir, 'few-shot_progress.jsonl')
        for idx in range(len(data)):
            pkl_path = os.path.join(results_dir, f'subject_{idx:04d}.pkl')
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    results[idx] = pickle.load(f)
                done_indices.add(idx)
        if done_indices:
            print(f"Resuming: {len(done_indices)} subjects already completed, skipping them.")

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

        # Save pickle + progress
        if output_dir:
            pkl_path = os.path.join(results_dir, f'subject_{index:04d}.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump(rs, f)
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
    results = [None] * len(data)
    done_indices = set()
    if output_dir:
        results_dir = os.path.join(output_dir, 'subject_results')
        os.makedirs(results_dir, exist_ok=True)
        progress_path = os.path.join(output_dir, 'personality_prompt_progress.jsonl')
        for idx in range(len(data)):
            pkl_path = os.path.join(results_dir, f'subject_{idx:04d}.pkl')
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    results[idx] = pickle.load(f)
                done_indices.add(idx)
        if done_indices:
            print(f"Resuming: {len(done_indices)} subjects already completed, skipping them.")

    for index, i in enumerate(tqdm(data)):
        if index in done_indices:
            continue

        system_prompt_text = system_prompt[index]['output'][0]
        answers = generateAnswer(tokenizer, model, i['test'], TEMPLATE, system_prompt=system_prompt_text,
                                 model_file=model_file, raw_logger=raw_logger, batch_size=batch_size)
        rs = process_answers(answers, i)
        results[index] = rs

        # Save pickle + progress
        if output_dir:
            pkl_path = os.path.join(results_dir, f'subject_{index:04d}.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump(rs, f)
            progress_entry = {
                'index': index,
                'case': rs['case'],
                'mean_abs': {k: v for k, v in rs['mean_ver_abs']['mean']},
                'timestamp': datetime.now().isoformat()
            }
            with open(progress_path, 'a') as f:
                f.write(json.dumps(progress_entry) + '\n')

    return [r for r in results if r is not None]
