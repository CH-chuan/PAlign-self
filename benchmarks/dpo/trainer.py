"""DPO trainer for per-subject personality alignment."""
import torch
from transformers import TrainingArguments
from trl import DPOTrainer, DPOConfig

from benchmarks.common.config import QLoRAConfig, DPOHyperparams
from benchmarks.common.model_utils import apply_qlora
from benchmarks.common.data import build_dpo_pairs


def train_dpo_for_subject(model, tokenizer, subject_data, output_dir,
                          qlora_cfg=None, hparams=None):
    """Train a DPO adapter for a single subject.

    Args:
        model: base quantized model (will be wrapped with LoRA)
        tokenizer: tokenizer
        subject_data: dict with 'train' and 'test' keys
        output_dir: where to save the temporary adapter
        qlora_cfg: QLoRA configuration
        hparams: DPO hyperparameters

    Returns:
        peft_model: model with trained LoRA adapter
    """
    if qlora_cfg is None:
        qlora_cfg = QLoRAConfig()
    if hparams is None:
        hparams = DPOHyperparams()

    # Build preference pairs
    train_dataset = build_dpo_pairs(subject_data)
    if len(train_dataset) == 0:
        return None

    # Apply LoRA
    peft_model = apply_qlora(model, qlora_cfg)

    training_args = DPOConfig(
        output_dir=output_dir,
        learning_rate=hparams.learning_rate,
        warmup_steps=hparams.warmup_steps,
        weight_decay=hparams.weight_decay,
        per_device_train_batch_size=hparams.per_device_train_batch_size,
        gradient_accumulation_steps=hparams.gradient_accumulation_steps,
        max_steps=hparams.max_steps,
        beta=hparams.beta,
        max_length=hparams.max_length,
        max_prompt_length=hparams.max_prompt_length,
        logging_steps=50,
        save_strategy="no",
        remove_unused_columns=False,
        bf16=True,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    return peft_model
