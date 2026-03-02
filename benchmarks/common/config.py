"""Shared configuration for all benchmark methods."""
from dataclasses import dataclass, field
from typing import List


# Constants shared with main.py
SCORES = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}

SCORES_BACK = {
    5: 'Very Accurate',
    4: 'Moderately Accurate',
    3: 'Neither Accurate Nor Inaccurate',
    2: 'Moderately Inaccurate',
    1: 'Very Inaccurate',
    0: 'Unknown',
}

TRAITS = ['A', 'C', 'E', 'N', 'O']

TRAIT_NAMES = {
    'A': 'Agreeableness',
    'C': 'Conscientiousness',
    'E': 'Extraversion',
    'N': 'Neuroticism',
    'O': 'Openness',
}

SYSTEM_PROMPT = "You are a helpful, honest and concise assistant."

TEMPLATE = """Given a statement of you: "You {}."
Please choose from the following options to identify how accurately this statement describes you.
Options:
(A). Very Accurate
(B). Moderately Accurate
(C). Neither Accurate Nor Inaccurate
(D). Moderately Inaccurate
(E). Very Inaccurate
"""


@dataclass
class QLoRAConfig:
    """QLoRA configuration shared across all training methods."""
    lora_r: int = 16
    lora_alpha: int = 100
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class DPOHyperparams:
    """DPO training hyperparameters from the paper."""
    learning_rate: float = 5e-4
    warmup_steps: int = 100
    weight_decay: float = 0.05
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    max_steps: int = 250
    beta: float = 0.1
    max_length: int = 512
    max_prompt_length: int = 384


@dataclass
class PPOHyperparams:
    """PPO training hyperparameters from the paper."""
    learning_rate: float = 1.41e-5
    per_device_train_batch_size: int = 32
    max_steps: int = 250
    max_new_tokens: int = 15
    kl_penalty: str = "kl"
    init_kl_coef: float = 0.2


@dataclass
class PromptMORLHyperparams:
    """Prompt-MORL training hyperparameters from the paper."""
    learning_rate: float = 5e-4
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    max_steps: int = 250
    max_seq_length: int = 512


@dataclass
class SoupsHyperparams:
    """Personalized-Soups hyperparameters from the paper."""
    # Uses PPO hyperparameters for training extreme models
    ppo: PPOHyperparams = field(default_factory=PPOHyperparams)
    num_extreme_models: int = 10  # high/low × 5 traits
