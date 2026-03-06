"""Model loading with QLoRA, adapter management, and weighted LoRA merge."""
import os
import shutil
from collections import OrderedDict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
    prepare_model_for_kbit_training,
)

from benchmarks.common.config import QLoRAConfig


def load_base_model_and_tokenizer(model_name, qlora_cfg=None):
    """Load a model with 4-bit quantization and prepare for QLoRA.

    Returns:
        model: quantized HF model (not yet wrapped with LoRA)
        tokenizer: corresponding tokenizer
    """
    if qlora_cfg is None:
        qlora_cfg = QLoRAConfig()

    compute_dtype = getattr(torch, qlora_cfg.bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qlora_cfg.load_in_4bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=qlora_cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=qlora_cfg.bnb_4bit_use_double_quant,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=compute_dtype,
    )
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

    return model, tokenizer


def apply_qlora(model, qlora_cfg=None, task_type=TaskType.CAUSAL_LM):
    """Wrap model with LoRA adapters.

    Returns:
        peft_model: model with LoRA adapters applied
    """
    if qlora_cfg is None:
        qlora_cfg = QLoRAConfig()

    lora_config = LoraConfig(
        r=qlora_cfg.lora_r,
        lora_alpha=qlora_cfg.lora_alpha,
        lora_dropout=qlora_cfg.lora_dropout,
        target_modules=qlora_cfg.target_modules,
        task_type=task_type,
        bias="none",
    )
    return get_peft_model(model, lora_config)


def save_adapter(model, path):
    """Save only the LoRA adapter weights."""
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)


def load_adapter(base_model, adapter_path):
    """Load a saved LoRA adapter onto a base model."""
    return PeftModel.from_pretrained(base_model, adapter_path)


def delete_adapter(path):
    """Remove a saved adapter directory."""
    if os.path.exists(path):
        shutil.rmtree(path)


def merge_adapters_weighted(adapter_paths, weights):
    """Merge multiple LoRA adapters with specified weights.

    Loads adapter state_dicts and computes weighted average of
    lora_A and lora_B matrices.

    Args:
        adapter_paths: list of adapter directory paths
        weights: list of floats, same length as adapter_paths

    Returns:
        merged_state_dict: OrderedDict of merged LoRA parameters
    """
    assert len(adapter_paths) == len(weights)
    assert abs(sum(weights) - 1.0) < 1e-6, f"Weights must sum to 1, got {sum(weights)}"

    merged = OrderedDict()
    for path, w in zip(adapter_paths, weights):
        adapter_file = os.path.join(path, 'adapter_model.safetensors')
        if not os.path.exists(adapter_file):
            adapter_file = os.path.join(path, 'adapter_model.bin')

        if adapter_file.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(adapter_file)
        else:
            state_dict = torch.load(adapter_file, map_location='cpu', weights_only=True)

        for key, value in state_dict.items():
            if key in merged:
                merged[key] = merged[key] + w * value.float()
            else:
                merged[key] = w * value.float()

    # Convert back to original dtype
    for key in merged:
        merged[key] = merged[key].to(torch.bfloat16)

    return merged


def load_merged_adapter(base_model, merged_state_dict, adapter_config_path):
    """Load a merged adapter state dict onto a base model.

    Args:
        base_model: the quantized base model
        merged_state_dict: output of merge_adapters_weighted()
        adapter_config_path: path to any one of the source adapters (for config)

    Returns:
        peft_model with merged weights loaded
    """
    peft_model = PeftModel.from_pretrained(base_model, adapter_config_path)
    peft_model.load_state_dict(merged_state_dict, strict=False)
    return peft_model
