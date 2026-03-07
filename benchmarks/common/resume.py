"""Checkpoint and progress tracking utilities, matching PAS resume pattern."""
import json
import os
import pickle
from datetime import datetime

from baseline_utils import (
    save_subject_meta as _save_meta,
    save_subject_answers as _save_answers,
    load_completed_indices as _load_completed,
)


def save_subject_result(result, index, output_dir, method=None):
    """Save a single subject's result as meta JSON + answers CSV."""
    extra = {}
    if method:
        extra['method'] = method
    path = os.path.join(output_dir, 'subject_results', f'subject_{index:04d}_meta.json')
    _save_meta(path, result=result, subject_index=index, **extra)
    _save_answers(result, index, output_dir)


def load_completed_results(num_subjects, output_dir):
    """Load all completed subject results.

    Returns:
        results: list of length num_subjects (None for incomplete)
        done_indices: set of completed subject indices
    """
    return _load_completed(num_subjects, output_dir)


def append_progress(entry_dict, output_dir, filename='progress.jsonl'):
    """Append a progress entry to the JSONL log."""
    os.makedirs(output_dir, exist_ok=True)
    entry_dict['timestamp'] = datetime.now().isoformat()
    progress_path = os.path.join(output_dir, filename)
    with open(progress_path, 'a') as f:
        f.write(json.dumps(entry_dict) + '\n')
