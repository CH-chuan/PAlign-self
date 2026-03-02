"""Reward function for PPO-based personality alignment."""
import re

from benchmarks.common.config import SCORES


def calculate_reward(generated_text, correct_score):
    """Calculate reward for a generated answer.

    Parses the A-E option from generated text, converts to score,
    and returns negative absolute error vs the correct score.

    Args:
        generated_text: raw generated text from the model
        correct_score: target score (1-5) for this item

    Returns:
        float: reward value. -abs(predicted - correct) for valid answers,
               -6.0 for unparseable answers.
    """
    parsed = re.search(r"[abcdeABCDE][^a-zA-Z]", generated_text[:12], flags=0)
    if parsed:
        letter = parsed.group()[0].upper()
        predicted_score = SCORES[letter]
        return -abs(predicted_score - correct_score)
    return -6.0
