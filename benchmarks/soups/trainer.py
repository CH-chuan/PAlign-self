"""Soups trainer: 10 extreme PPO models (high/low × 5 traits)."""
import os

import torch
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, TaskType

from benchmarks.common.config import QLoRAConfig, SoupsHyperparams, TRAITS
from benchmarks.common.data import build_soups_extreme_dataset
from benchmarks.common.model_utils import save_adapter
from benchmarks.ppo.reward import calculate_reward


def train_one_extreme_model(model_name, tokenizer, all_data, trait, direction,
                            adapter_dir, qlora_cfg=None, hparams=None):
    """Train one extreme PPO model for a trait/direction.

    Args:
        model_name: HF model name
        tokenizer: tokenizer
        all_data: list of all subject data dicts
        trait: one of 'A', 'C', 'E', 'N', 'O'
        direction: 'high' or 'low'
        adapter_dir: where to save the adapter
        qlora_cfg: QLoRA configuration
        hparams: SoupsHyperparams (uses .ppo for PPO config)

    Returns:
        adapter_dir path
    """
    if qlora_cfg is None:
        qlora_cfg = QLoRAConfig()
    if hparams is None:
        hparams = SoupsHyperparams()
    ppo_hp = hparams.ppo

    # Build extreme dataset
    dataset = build_soups_extreme_dataset(all_data, trait, direction)
    print(f"Training {direction}_{trait}: {len(dataset)} examples")

    lora_config = LoraConfig(
        r=qlora_cfg.lora_r,
        lora_alpha=qlora_cfg.lora_alpha,
        lora_dropout=qlora_cfg.lora_dropout,
        target_modules=qlora_cfg.target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    ppo_config = PPOConfig(
        learning_rate=ppo_hp.learning_rate,
        batch_size=ppo_hp.per_device_train_batch_size,
        mini_batch_size=min(4, ppo_hp.per_device_train_batch_size),
        ppo_epochs=4,
        kl_penalty=ppo_hp.kl_penalty,
        init_kl_coef=ppo_hp.init_kl_coef,
        log_with=None,
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        peft_config=lora_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
    )

    queries = dataset['query']
    correct_scores = dataset['correct_score']
    num_queries = len(queries)

    steps_done = 0
    while steps_done < ppo_hp.max_steps:
        batch_size = min(ppo_hp.per_device_train_batch_size, num_queries)
        indices = torch.randint(0, num_queries, (batch_size,)).tolist()

        query_tensors = [
            tokenizer.encode(queries[i], return_tensors="pt").squeeze(0)
            for i in indices
        ]

        try:
            response_tensors = ppo_trainer.generate(
                query_tensors,
                max_new_tokens=ppo_hp.max_new_tokens,
                do_sample=True,
                top_k=0,
                top_p=1.0,
            )
        except (RuntimeError, torch.cuda.CudaError) as e:
            print(f"PPO training diverged at step {steps_done}: {e}")
            break

        responses_text = [
            tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors
        ]
        rewards = [
            torch.tensor(calculate_reward(resp, correct_scores[idx]))
            for resp, idx in zip(responses_text, indices)
        ]

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        steps_done += 1

        if 'objective/kl' in stats and stats['objective/kl'] < -100:
            print(f"PPO KL diverged at step {steps_done} (kl={stats['objective/kl']:.1f}), stopping early.")
            break

    # Save adapter
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)

    del model, ppo_trainer
    torch.cuda.empty_cache()

    return adapter_dir


def train_all_extreme_models(model_name, tokenizer, all_data, adapters_base_dir,
                             qlora_cfg=None, hparams=None):
    """Train all 10 extreme models (high/low × 5 traits).

    Skips models whose adapters already exist on disk.

    Returns:
        dict mapping (trait, direction) to adapter_dir path
    """
    adapter_paths = {}
    for trait in TRAITS:
        for direction in ['high', 'low']:
            adapter_dir = os.path.join(adapters_base_dir, f'{direction}_{trait}')
            adapter_paths[(trait, direction)] = adapter_dir

            # Skip if already trained
            if os.path.exists(os.path.join(adapter_dir, 'adapter_config.json')):
                print(f"Adapter {direction}_{trait} already exists, skipping.")
                continue

            print(f"\n--- Training {direction}_{trait} ---")
            train_one_extreme_model(
                model_name, tokenizer, all_data, trait, direction,
                adapter_dir, qlora_cfg, hparams,
            )

    return adapter_paths
