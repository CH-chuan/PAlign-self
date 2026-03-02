"""Prompt-MORL trainer: single shared model with personality-conditioned prompts."""
import torch
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig

from benchmarks.common.config import QLoRAConfig, PromptMORLHyperparams
from benchmarks.common.model_utils import apply_qlora, save_adapter
from benchmarks.common.data import build_prompt_morl_dataset


def train_prompt_morl(model, tokenizer, all_data, adapter_dir,
                      qlora_cfg=None, hparams=None):
    """Train a single Prompt-MORL adapter on all subjects' data.

    The training data includes personality-conditioned system prompts,
    so the model learns to respond differently based on personality levels.

    Args:
        model: base quantized model
        tokenizer: tokenizer
        all_data: list of all subject data dicts
        adapter_dir: where to save the trained adapter
        qlora_cfg: QLoRA configuration
        hparams: Prompt-MORL hyperparameters

    Returns:
        peft_model: model with trained LoRA adapter
    """
    if qlora_cfg is None:
        qlora_cfg = QLoRAConfig()
    if hparams is None:
        hparams = PromptMORLHyperparams()

    # Build the pooled SFT dataset
    train_dataset = build_prompt_morl_dataset(all_data)
    print(f"Prompt-MORL training set: {len(train_dataset)} examples")

    # Apply LoRA
    peft_model = apply_qlora(model, qlora_cfg)

    training_args = SFTConfig(
        output_dir=adapter_dir,
        learning_rate=hparams.learning_rate,
        per_device_train_batch_size=hparams.per_device_train_batch_size,
        gradient_accumulation_steps=hparams.gradient_accumulation_steps,
        max_steps=hparams.max_steps,
        max_seq_length=hparams.max_seq_length,
        logging_steps=50,
        save_strategy="no",
        bf16=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    # Save the adapter
    save_adapter(peft_model, adapter_dir)

    return peft_model
