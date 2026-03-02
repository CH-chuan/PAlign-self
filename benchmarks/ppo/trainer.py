"""PPO trainer for per-subject personality alignment."""
import torch
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, TaskType

from benchmarks.common.config import QLoRAConfig, PPOHyperparams
from benchmarks.common.data import build_ppo_queries
from benchmarks.ppo.reward import calculate_reward


def train_ppo_for_subject(model_name, tokenizer, subject_data, output_dir,
                          qlora_cfg=None, hparams=None):
    """Train a PPO adapter for a single subject.

    Note: PPO with trl requires AutoModelForCausalLMWithValueHead,
    so we load the model fresh inside this function.

    Args:
        model_name: HF model name (will be loaded with quantization)
        tokenizer: tokenizer
        subject_data: dict with 'train' and 'test' keys
        output_dir: where to save the temporary adapter
        qlora_cfg: QLoRA configuration
        hparams: PPO hyperparameters

    Returns:
        peft_model: model with trained LoRA adapter (for evaluation)
        or None on failure
    """
    if qlora_cfg is None:
        qlora_cfg = QLoRAConfig()
    if hparams is None:
        hparams = PPOHyperparams()

    # Build query dataset
    query_dataset = build_ppo_queries(subject_data)

    lora_config = LoraConfig(
        r=qlora_cfg.lora_r,
        lora_alpha=qlora_cfg.lora_alpha,
        lora_dropout=qlora_cfg.lora_dropout,
        target_modules=qlora_cfg.target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    ppo_config = PPOConfig(
        learning_rate=hparams.learning_rate,
        batch_size=hparams.per_device_train_batch_size,
        mini_batch_size=min(4, hparams.per_device_train_batch_size),
        ppo_epochs=4,
        kl_penalty=hparams.kl_penalty,
        init_kl_coef=hparams.init_kl_coef,
        log_with=None,
    )

    # Load model with value head for PPO
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

    # Training loop
    queries = query_dataset['query']
    correct_scores = query_dataset['correct_score']
    num_queries = len(queries)

    steps_done = 0
    while steps_done < hparams.max_steps:
        # Sample a batch
        batch_size = min(hparams.per_device_train_batch_size, num_queries)
        indices = torch.randint(0, num_queries, (batch_size,)).tolist()

        # Tokenize queries
        query_tensors = [
            tokenizer.encode(queries[i], return_tensors="pt").squeeze(0)
            for i in indices
        ]

        # Generate responses
        response_tensors = ppo_trainer.generate(
            query_tensors,
            max_new_tokens=hparams.max_new_tokens,
            do_sample=True,
            top_k=0,
            top_p=1.0,
        )

        # Compute rewards
        responses_text = [
            tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors
        ]
        rewards = [
            torch.tensor(calculate_reward(resp, correct_scores[idx]))
            for resp, idx in zip(responses_text, indices)
        ]

        # PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        steps_done += 1

    # Save the adapter
    model.save_pretrained(output_dir)

    return model


def load_ppo_model_for_eval(model_name, adapter_path, tokenizer):
    """Load a PPO-trained model for evaluation (without value head)."""
    from benchmarks.common.model_utils import load_base_model_and_tokenizer, load_adapter
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    peft_model = load_adapter(base_model, adapter_path)
    return peft_model
