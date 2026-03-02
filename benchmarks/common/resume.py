"""Checkpoint and progress tracking utilities, matching PAS resume pattern."""
import json
import os
import pickle
from datetime import datetime


def save_subject_result(result, index, output_dir):
    """Save a single subject's result as a pickle file."""
    results_dir = os.path.join(output_dir, 'subject_results')
    os.makedirs(results_dir, exist_ok=True)
    pkl_path = os.path.join(results_dir, f'subject_{index:04d}.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(result, f)


def load_completed_results(num_subjects, output_dir):
    """Load all completed subject results from pickle files.

    Returns:
        results: list of length num_subjects (None for incomplete)
        done_indices: set of completed subject indices
    """
    results = [None] * num_subjects
    done_indices = set()
    results_dir = os.path.join(output_dir, 'subject_results')
    if not os.path.exists(results_dir):
        return results, done_indices
    for idx in range(num_subjects):
        pkl_path = os.path.join(results_dir, f'subject_{idx:04d}.pkl')
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                results[idx] = pickle.load(f)
            done_indices.add(idx)
    return results, done_indices


def append_progress(entry_dict, output_dir, filename='progress.jsonl'):
    """Append a progress entry to the JSONL log."""
    os.makedirs(output_dir, exist_ok=True)
    entry_dict['timestamp'] = datetime.now().isoformat()
    progress_path = os.path.join(output_dir, filename)
    with open(progress_path, 'a') as f:
        f.write(json.dumps(entry_dict) + '\n')
