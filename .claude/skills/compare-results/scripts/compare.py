#!/usr/bin/env python3
"""Compare reproduction MAE scores against paper Table 1 targets.

Scans one or more result directories for *_OOD.json files, extracts per-trait
MAE scores, and writes a formatted markdown comparison table.

Usage:
    python .claude/skills/compare-results/scripts/compare.py <dir1> [dir2 ...]
    python .claude/skills/compare-results/scripts/compare.py reproduction_0302 --output results.md
"""
import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path

TRAITS = ['A', 'C', 'E', 'N', 'O']

# Filename prefix -> display method name
PREFIX_TO_METHOD = {
    'PAS':                'PAS',
    'DPO':                'DPO',
    'PPO':                'PPO',
    'Prompt-MORL':        'Prompt-MORL',
    'Soups':              'Personalized-Soups',
    'few-shot-PAS':       'Few-Shot+PAS',
    'few_shot_pas':       'Few-Shot+PAS',
    'few-shot':           'Few-Shot',
    'personality_prompt':  'P\u00b2',
}

# Preferred display order for methods
METHOD_ORDER = [
    'PPO', 'DPO', 'Prompt-MORL', 'Personalized-Soups',
    'Few-Shot', 'P\u00b2', 'Few-Shot+PAS', 'PAS',
]


def parse_paper_targets(paper_path):
    """Parse paper target table from a markdown file.

    Returns dict: {method_name: {trait: float, ..., 'Score': float, 'Mode': str}}
    """
    targets = {}
    with open(paper_path, 'r') as f:
        lines = f.readlines()

    # Find header line to get column indices
    header_line = None
    for line in lines:
        if 'Method' in line and '|' in line:
            header_line = line
            break
    if not header_line:
        return targets

    headers = [h.strip() for h in header_line.split('|')]
    # Remove empty strings from leading/trailing pipes
    headers = [h for h in headers if h]

    # Map header names to trait keys
    col_map = {}
    for i, h in enumerate(headers):
        h_clean = re.sub(r'\s*↓\s*', '', h)
        if 'Agreeableness' in h_clean:
            col_map[i] = 'A'
        elif 'Conscientiousness' in h_clean:
            col_map[i] = 'C'
        elif 'Extraversion' in h_clean:
            col_map[i] = 'E'
        elif 'Neuroticism' in h_clean:
            col_map[i] = 'N'
        elif 'Openness' in h_clean:
            col_map[i] = 'O'
        elif h_clean == 'Score':
            col_map[i] = 'Score'
        elif 'Alignment Mode' in h_clean or h_clean == 'Mode':
            col_map[i] = 'Mode'

    for line in lines:
        if '---' in line or 'Method' in line:
            continue
        parts = [p.strip() for p in line.split('|')]
        parts = [p for p in parts if p]  # drop empty from pipes
        if len(parts) < 3:
            continue

        # Extract method name, stripping markdown bold/italic
        method_raw = parts[0]
        method_name = re.sub(r'[*_]', '', method_raw).strip()
        # Skip model header rows (italic model names like "Llama-3-8B-Instruct")
        if not any(c.isdigit() for c in ''.join(parts[1:])):
            continue
        # Normalize known method names
        if 'PAS' in method_name and 'Ours' in method_name:
            method_name = 'PAS'
        elif method_name == 'P²' or method_name == 'P2':
            method_name = 'P\u00b2'

        entry = {}
        for i, val_str in enumerate(parts):
            if i in col_map:
                key = col_map[i]
                # Strip bold markers from values
                val_clean = re.sub(r'[*]', '', val_str).strip()
                try:
                    entry[key] = float(val_clean)
                except ValueError:
                    if key == 'Mode':
                        entry[key] = val_clean

        if any(t in entry for t in TRAITS):
            targets[method_name] = entry

    return targets


def aggregate_pickles(pickle_dir):
    """Build scores and subject count from per-subject pickle files.

    Each pickle has mean_ver_abs['mean'] = [(trait, value), ...].
    Returns (trait_scores, num_subjects) matching load_scores output,
    or None if no pickles found.
    """
    results_dir = Path(pickle_dir) / 'subject_results'
    if not results_dir.is_dir():
        return None
    pkl_files = sorted(results_dir.glob('subject_*.pkl'))
    if not pkl_files:
        return None

    per_trait = {t: [] for t in TRAITS}
    for pkl_path in pkl_files:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        mae_dict = dict(data['mean_ver_abs']['mean'])
        for t in TRAITS:
            if t in mae_dict:
                per_trait[t].append(float(mae_dict[t]))

    if not any(per_trait.values()):
        return None

    trait_scores = {}
    for t in TRAITS:
        vals = per_trait[t]
        trait_scores[t] = sum(vals) / len(vals) if vals else float('nan')

    return trait_scores, len(pkl_files)


def discover_json_files(data_dir):
    """Find all *_OOD.json files in a data directory.

    Searches:
      1. Direct files in data_dir
      2. Immediate subdirs (pas/, dpo/, etc.)
      3. benchmarks/*/ subdirs
    """
    found = []
    data_path = Path(data_dir)
    if not data_path.is_dir():
        print(f"Warning: {data_dir} is not a directory, skipping", file=sys.stderr)
        return found

    # 1. Direct files
    for f in data_path.glob('*_OOD.json'):
        found.append(f)

    # 2. Immediate subdirs
    for subdir in data_path.iterdir():
        if subdir.is_dir() and subdir.name != 'benchmarks':
            for f in subdir.glob('*_OOD.json'):
                found.append(f)

    # 3. benchmarks/*/
    benchmarks_dir = data_path / 'benchmarks'
    if benchmarks_dir.is_dir():
        for subdir in benchmarks_dir.iterdir():
            if subdir.is_dir():
                for f in subdir.glob('*_OOD.json'):
                    found.append(f)

    return found


def classify_file(filepath):
    """Map a JSON filename to its method name.

    Returns (method_name, model_name) or (None, None) if unrecognized.
    """
    stem = filepath.stem  # e.g. "PAS_Meta-Llama-3-8B-Instruct_OOD"
    if not stem.endswith('_OOD'):
        return None, None
    # Strip _OOD suffix
    name_part = stem[:-4]  # e.g. "PAS_Meta-Llama-3-8B-Instruct"

    # Try each prefix, longest first to avoid "few-shot" matching "few-shot-PAS"
    sorted_prefixes = sorted(PREFIX_TO_METHOD.keys(), key=len, reverse=True)
    for prefix in sorted_prefixes:
        if name_part.startswith(prefix + '_'):
            model = name_part[len(prefix) + 1:]
            return PREFIX_TO_METHOD[prefix], model
    return None, None


def load_scores(filepath):
    """Load trait MAE scores and subject count from a result JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    scores = data.get('score', {})
    trait_scores = {}
    for t in TRAITS:
        trait_scores[t] = scores.get(f'mean_{t}_abs', float('nan'))

    # Count subjects from per-subject arrays
    mean_data = data.get('mean', {})
    num_subjects = 0
    for t in TRAITS:
        arr = mean_data.get(t, [])
        if isinstance(arr, list) and len(arr) > num_subjects:
            num_subjects = len(arr)

    return trait_scores, num_subjects


def dir_suffix(dir_label):
    """Extract a short suffix from a directory label.

    Strips common prefixes like 'reproduction_', 'reproduce_', 'repro_'.
    Returns empty string if the dir name is exactly a common prefix with no suffix.
    """
    for prefix in ('reproduction_', 'reproduce_', 'repro_'):
        if dir_label.startswith(prefix):
            return dir_label[len(prefix):]
    # If the dir name is just "reproduction" etc., no suffix
    if dir_label in ('reproduction', 'reproduce', 'repro'):
        return ''
    return dir_label


def format_table(paper_targets, repro_results, output_path):
    """Write a markdown comparison table.

    Args:
        paper_targets: {method: {trait: val, 'Score': val, 'Mode': str}}
        repro_results: [(method, model, dir_label, {trait: val}, num_subjects), ...]
        output_path: path to write the .md file
    """
    lines = []
    lines.append('# MAE Comparison: Paper vs Reproduction\n')

    # Header
    lines.append('| Method | A \u2193 | C \u2193 | E \u2193 | N \u2193 | O \u2193 | Score \u2193 | #Subj |')
    lines.append('|--------|----:|----:|----:|----:|----:|-------:|------:|')

    # Collect best reproduction score per trait for bolding
    best_repro = {}
    for t in TRAITS + ['Score']:
        best_repro[t] = float('inf')
    for method, model, dir_label, scores, n_subj in repro_results:
        for t in TRAITS:
            v = scores.get(t, float('nan'))
            if v == v and v < best_repro[t]:  # not NaN
                best_repro[t] = v
        total = sum(scores.get(t, 0) for t in TRAITS)
        if total < best_repro['Score']:
            best_repro['Score'] = total

    # Collect paper scores for comparison
    paper_best = {}
    for t in TRAITS + ['Score']:
        paper_best[t] = float('inf')
    for method, entry in paper_targets.items():
        for t in TRAITS:
            v = entry.get(t, float('inf'))
            if v < paper_best[t]:
                paper_best[t] = v
        s = entry.get('Score', float('inf'))
        if s < paper_best['Score']:
            paper_best['Score'] = s

    # Paper rows (ordered)
    for method in METHOD_ORDER:
        if method not in paper_targets:
            continue
        entry = paper_targets[method]
        score = entry.get('Score', sum(entry.get(t, 0) for t in TRAITS))
        vals = []
        for t in TRAITS:
            v = entry.get(t, float('nan'))
            vals.append(f'{v:.2f}')
        vals.append(f'{score:.2f}')
        lines.append(f'| {method} (paper) | {" | ".join(vals)} | - |')

    # Separator
    lines.append('| | | | | | | | |')

    # Reproduction rows (ordered by method, then by dir)
    ordered_repro = sorted(repro_results,
                           key=lambda x: (METHOD_ORDER.index(x[0])
                                          if x[0] in METHOD_ORDER else 999, x[2]))
    for method, model, dir_label, scores, n_subj in ordered_repro:
        total = sum(scores.get(t, 0) for t in TRAITS)
        vals = []
        for t in TRAITS:
            v = scores.get(t, float('nan'))
            s = f'{v:.2f}'
            # Bold if this is the best repro score AND beats the paper best
            if v == best_repro[t] and v < paper_best[t] and len(repro_results) > 1:
                s = f'**{s}**'
            vals.append(s)
        # Score column
        score_s = f'{total:.2f}'
        if total == best_repro['Score'] and total < paper_best['Score'] and len(repro_results) > 1:
            score_s = f'**{score_s}**'
        vals.append(score_s)
        # Build display name: method_suffix or just method
        suffix = dir_suffix(dir_label)
        display = f'{method}_{suffix}' if suffix else method
        subj_str = str(n_subj) if n_subj >= 300 else f'{n_subj}*'
        lines.append(f'| {display} | {" | ".join(vals)} | {subj_str} |')

    lines.append('')

    content = '\n'.join(lines)
    with open(output_path, 'w') as f:
        f.write(content)
    return content


def main():
    parser = argparse.ArgumentParser(
        description='Compare reproduction MAE scores against paper Table 1 targets')
    parser.add_argument('data_dirs', nargs='+',
                        help='One or more result directories to scan')
    parser.add_argument('--paper', default='papers/benchmarking_target.md',
                        help='Path to paper target markdown table (default: papers/benchmarking_target.md)')
    parser.add_argument('--output', default='comparison_table.md',
                        help='Output markdown file (default: comparison_table.md)')
    args = parser.parse_args()

    # Parse paper targets
    paper_targets = {}
    if os.path.exists(args.paper):
        paper_targets = parse_paper_targets(args.paper)
        print(f"Loaded {len(paper_targets)} paper targets from {args.paper}", file=sys.stderr)
    else:
        print(f"Warning: paper targets file not found: {args.paper}", file=sys.stderr)

    # Discover and load reproduction results
    repro_results = []
    for data_dir in args.data_dirs:
        dir_label = os.path.basename(os.path.normpath(data_dir))
        data_path = Path(data_dir)

        # 1. Load from *_OOD.json files
        json_files = discover_json_files(data_dir)
        json_dirs_used = set()  # track which subdirs had JSON results
        for fpath in json_files:
            method, model = classify_file(fpath)
            if method is None:
                print(f"Warning: unrecognized file {fpath}, skipping", file=sys.stderr)
                continue
            json_dirs_used.add(fpath.parent.name)
            try:
                scores, n_subj = load_scores(fpath)
                repro_results.append((method, model, dir_label, scores, n_subj))
                total = sum(scores.get(t, 0) for t in TRAITS)
                print(f"  {method:<22s} ({dir_label}): "
                      f"A={scores['A']:.2f} C={scores['C']:.2f} E={scores['E']:.2f} "
                      f"N={scores['N']:.2f} O={scores['O']:.2f}  Sum={total:.2f}  "
                      f"[{n_subj} subjects]", file=sys.stderr)
            except Exception as e:
                print(f"Error loading {fpath}: {e}", file=sys.stderr)

        # 2. Check subdirs for pickle-only results (no OOD.json)
        if data_path.is_dir():
            for subdir in data_path.iterdir():
                if not subdir.is_dir() or subdir.name in json_dirs_used:
                    continue
                # Check if this subdir name maps to a known method
                method_name = PREFIX_TO_METHOD.get(subdir.name)
                if method_name is None:
                    continue
                # Skip if it already has an OOD.json
                if list(subdir.glob('*_OOD.json')):
                    continue
                result = aggregate_pickles(subdir)
                if result is None:
                    continue
                scores, n_subj = result
                repro_results.append((method_name, 'pickle', dir_label, scores, n_subj))
                total = sum(scores.get(t, 0) for t in TRAITS)
                print(f"  {method_name:<22s} ({dir_label}): "
                      f"A={scores['A']:.2f} C={scores['C']:.2f} E={scores['E']:.2f} "
                      f"N={scores['N']:.2f} O={scores['O']:.2f}  Sum={total:.2f}  "
                      f"[{n_subj} subjects, from pickles]", file=sys.stderr)

        if not json_files and not any(r[2] == dir_label for r in repro_results):
            print(f"Warning: no results found in {data_dir}", file=sys.stderr)

    if not repro_results:
        print("Error: no results loaded from any directory", file=sys.stderr)
        sys.exit(1)

    # Generate table
    content = format_table(paper_targets, repro_results, args.output)
    print(f"\nWrote comparison table to {args.output}", file=sys.stderr)
    print(content)


if __name__ == '__main__':
    main()
