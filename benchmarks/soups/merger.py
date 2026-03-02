"""LoRA adapter weighted merge for Personalized-Soups."""
from benchmarks.common.config import TRAITS
from benchmarks.common.model_utils import merge_adapters_weighted, load_merged_adapter
from benchmarks.common.data import get_subject_trait_means


def compute_soup_weights(subject_data):
    """Compute per-adapter merge weights from a subject's trait scores.

    Normalizes each trait mean from [1, 5] → [0, 1] range.
    For 'high' adapters, weight = normalized_score.
    For 'low' adapters, weight = 1 - normalized_score.

    Returns:
        dict mapping (trait, direction) to weight, normalized to sum=1
    """
    trait_means = get_subject_trait_means(subject_data)

    raw_weights = {}
    for trait in TRAITS:
        score = trait_means[trait]
        # Normalize [1, 5] → [0, 1]
        normalized = (score - 1.0) / 4.0
        normalized = max(0.0, min(1.0, normalized))
        raw_weights[(trait, 'high')] = normalized
        raw_weights[(trait, 'low')] = 1.0 - normalized

    # Normalize weights to sum to 1
    total = sum(raw_weights.values())
    if total > 0:
        weights = {k: v / total for k, v in raw_weights.items()}
    else:
        # Uniform fallback
        weights = {k: 1.0 / len(raw_weights) for k in raw_weights}

    return weights


def merge_soup_for_subject(base_model, subject_data, adapter_paths):
    """Create a merged LoRA model for a specific subject.

    Args:
        base_model: quantized base model (no adapters)
        subject_data: dict with 'train' and 'test' keys
        adapter_paths: dict mapping (trait, direction) → adapter dir path

    Returns:
        peft_model with merged weights loaded
    """
    weights_dict = compute_soup_weights(subject_data)

    # Build ordered lists
    paths = []
    weights = []
    for key in sorted(adapter_paths.keys()):
        paths.append(adapter_paths[key])
        weights.append(weights_dict[key])

    # Merge adapter state dicts
    merged_state_dict = merge_adapters_weighted(paths, weights)

    # Load merged weights onto base model
    # Use the first adapter's config as reference
    first_adapter_path = paths[0]
    peft_model = load_merged_adapter(base_model, merged_state_dict, first_adapter_path)

    return peft_model
