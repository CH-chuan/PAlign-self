"""Training data builders for all benchmark methods."""
import sys
import os

from datasets import Dataset

# Add project root to path so we can import main.py functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from main import getItems, from_index_to_data
from benchmarks.common.config import (
    SCORES, SCORES_BACK, TRAITS, TRAIT_NAMES, TEMPLATE, SYSTEM_PROMPT,
)


def load_all_data(data_dir='PAPI', dataset_set='OOD', num_subjects=0):
    """Load and preprocess all subject data.

    Returns:
        data: list of dicts with 'train' and 'test' keys
    """
    dataset, text_file, train_index, test_index = getItems(data_dir)
    data = from_index_to_data(train_index, test_index, text_file, dataset, dataset_set)
    if num_subjects > 0:
        data = data[:num_subjects]
    return data


def _score_to_option_text(score):
    """Convert a numeric score (1-5) to the option letter and text."""
    letter = {5: 'A', 4: 'B', 3: 'C', 2: 'D', 1: 'E'}[score]
    return f"({letter}). {SCORES_BACK[score]}"


def _opposite_score(score):
    """Return the opposite extreme score: 5->1, 4->2, 2->4, 1->5."""
    return 6 - score


def build_dpo_pairs(subject_data, system_prompt=SYSTEM_PROMPT):
    """Build DPO preference pairs from a subject's training items.

    Excludes items with value=3 (neutral). For each item:
    - chosen = option text matching the subject's score
    - rejected = option text at the opposite extreme

    Returns:
        datasets.Dataset with columns: prompt, chosen, rejected
    """
    prompts = []
    chosen_list = []
    rejected_list = []

    for item in subject_data['train']:
        value = item['value']
        if value == 3:
            continue

        prompt_text = TEMPLATE.format(item['text'].lower())
        # Format as chat-style prompt
        prompt = f"System: {system_prompt}\nUser: {prompt_text}\nAssistant:"

        chosen = " " + _score_to_option_text(value)
        rejected = " " + _score_to_option_text(_opposite_score(value))

        prompts.append(prompt)
        chosen_list.append(chosen)
        rejected_list.append(rejected)

    return Dataset.from_dict({
        'prompt': prompts,
        'chosen': chosen_list,
        'rejected': rejected_list,
    })


def build_ppo_queries(subject_data, system_prompt=SYSTEM_PROMPT):
    """Build PPO query dataset from a subject's training items.

    Returns:
        datasets.Dataset with columns: query, correct_score
    """
    queries = []
    correct_scores = []

    for item in subject_data['train']:
        prompt_text = TEMPLATE.format(item['text'].lower())
        query = f"System: {system_prompt}\nUser: {prompt_text}\nAssistant:"
        queries.append(query)
        correct_scores.append(item['value'])

    return Dataset.from_dict({
        'query': queries,
        'correct_score': correct_scores,
    })


def _personality_prefix(trait_scores):
    """Build personality-conditioning prefix from trait scores dict.

    Args:
        trait_scores: dict like {'A': 3.2, 'C': 4.1, ...} (mean trait scores)
    """
    parts = []
    for trait in TRAITS:
        name = TRAIT_NAMES[trait]
        score = trait_scores[trait]
        parts.append(f"{name} level {score:.1f}")
    return "You are an AI with " + ", ".join(parts) + "."


def _compute_subject_trait_means(subject_data):
    """Compute mean trait scores from a subject's training items."""
    trait_sums = {t: [] for t in TRAITS}
    for item in subject_data['train']:
        label = item['label_ocean']
        value = item['value']
        key = item['key']
        # Normalize reverse-keyed items
        if key == -1:
            value = 6 - value
        trait_sums[label].append(value)
    return {t: sum(v) / len(v) if v else 3.0 for t, v in trait_sums.items()}


def build_prompt_morl_dataset(all_data, system_prompt=SYSTEM_PROMPT):
    """Build Prompt-MORL SFT dataset: all subjects' items with personality prefix.

    Returns:
        datasets.Dataset with columns: text (formatted for SFT)
    """
    texts = []

    for subject_data in all_data:
        trait_means = _compute_subject_trait_means(subject_data)
        prefix = _personality_prefix(trait_means)

        for item in subject_data['train']:
            value = item['value']
            if value == 0:
                continue
            prompt_text = TEMPLATE.format(item['text'].lower())
            answer = _score_to_option_text(value)

            text = (
                f"System: {prefix} {system_prompt}\n"
                f"User: {prompt_text}\n"
                f"Assistant: {answer}"
            )
            texts.append(text)

    return Dataset.from_dict({'text': texts})


def build_soups_extreme_dataset(all_data, trait, direction, system_prompt=SYSTEM_PROMPT):
    """Build training dataset for one extreme Soups model.

    Args:
        all_data: list of all subject data dicts
        trait: one of 'A', 'C', 'E', 'N', 'O'
        direction: 'high' (target score 5) or 'low' (target score 1)

    Returns:
        datasets.Dataset with columns: query, correct_score
    """
    target_score = 5 if direction == 'high' else 1
    queries = []
    correct_scores = []

    for subject_data in all_data:
        for item in subject_data['train']:
            if item['label_ocean'] != trait:
                continue
            prompt_text = TEMPLATE.format(item['text'].lower())
            key = item['key']
            # For reverse-keyed items, flip the target
            if key == -1:
                effective_target = 6 - target_score
            else:
                effective_target = target_score
            query = f"System: {system_prompt}\nUser: {prompt_text}\nAssistant:"
            queries.append(query)
            correct_scores.append(effective_target)

    return Dataset.from_dict({
        'query': queries,
        'correct_score': correct_scores,
    })


def get_subject_trait_means(subject_data):
    """Public wrapper for computing subject trait means."""
    return _compute_subject_trait_means(subject_data)


def get_personality_prefix(trait_scores):
    """Public wrapper for building personality prefix."""
    return _personality_prefix(trait_scores)
