# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reproduction and extension of **"Personality Alignment of Large Language Models" (ICLR 2025)**. The core method is **PAS (Persona Activation Steering)**: at inference time, train per-head logistic regression probes on contrastive prompt pairs, then apply activation interventions (bias shifts) to steer LLM responses toward a target Big Five personality profile.

## Common Commands

```bash
# Environment
conda activate palign_repro

# Full reproduction (300 subjects, ~35h on RTX 4090)
python main.py --modes PAS --model_file meta-llama/Meta-Llama-3-8B-Instruct --num_subjects 0

# Quick validation (5 subjects, ~35min)
python main.py --modes PAS --model_file meta-llama/Meta-Llama-3-8B-Instruct --num_subjects 5

# Synthetic single-trait experiment (~7min)
python run_synthetic.py --trait C --direction high --alphas "0,4"

# Analyze reproduction results vs paper Table 1
python reproduction/analyze_results.py

# HPC: submit all jobs for a model
./hpc/submit_palign.sh -m qwen3coder-30b --all
```

There is no test suite or linter configured.

## Important: Transformers Version Sensitivity

This project uses custom modeling files (`modeling_llama.py`, `modeling_mistral.py`) that patch internal HuggingFace `transformers` classes. These files are tightly coupled to specific `transformers` internals (attention layer structure, forward signature, KV cache API, etc.) which change across versions. When debugging model-loading errors, generation failures, or hook-related issues, always check the installed `transformers` version and read the raw source of both our custom modeling files and the upstream `transformers` code to verify compatibility. Do not rely on assumptions about the API — search the actual code.

## Architecture

### PAS Pipeline (core data flow)

1. **Data**: 300 IPIP-NEO subjects in `PAPI/Test-set.json`, each with 120 training + 180 test questions split via `PAPI/mpi_300_split.json`
2. **Activation extraction**: `PASLM.preprocess_activate_dataset()` runs contrastive prompt pairs through the model and captures per-head outputs via `_ModuleOutputCapture` hooks on `model.layers.{i}.self_attn.head_out` (an `nn.Identity()` added by custom modeling files)
3. **Probe training**: `PASLM.get_activations()` trains a logistic regression probe per attention head, selects top-K heads by validation accuracy, returns intervention directions
4. **Alpha search**: `process_pas()` in `main.py` sweeps alpha values `[0,1,2,4,6,8]`, picks the one minimizing MAE on training questions
5. **Intervention**: `PASLM.set_activate()` modifies layer biases; `PASLM.reset_all()` restores originals from cache
6. **Evaluation**: `generateAnswer()` batch-generates on 180 test questions, `process_answers()` in `baseline_utils.py` parses A-E choices into scores

### Key Files

| File | Role |
|------|------|
| `PAlign/pas.py` | `get_model()` returns a `PASLM` instance; contains activation capture, probe training, intervention logic |
| `PAlign/modeling_llama.py` | Custom `LlamaForCausalLM` with `head_out = nn.Identity()` hooks on attention outputs |
| `PAlign/modeling_mistral.py` | Same hooks for `MistralForCausalLM` |
| `main.py` | Full reproduction entry point: tokenization, generation, per-subject PAS pipeline, resume via pickle |
| `run_synthetic.py` | Synthetic single-trait steering experiment (one extreme trait, others neutral) |
| `baseline_utils.py` | Answer parsing (`process_answers`) and score aggregation |

### Multi-Model Chat Template Support

`prompt_to_tokens()` (in both `main.py` and `run_synthetic.py`) and `PAlign/pas.py` have branching logic:
- **Llama-3**: Manual `<|begin_of_text|>...<|eot_id|>` formatting
- **Qwen / GPT-OSS**: `tokenizer.apply_chat_template()`
- Answer extraction splits on `<|im_start|>assistant\n` for Qwen/GPT-OSS vs `<|start_header_id|>assistant` for Llama

Adding a new model architecture requires a custom `modeling_*.py` with the `head_out` identity hook.

### Resume Support

`main.py` saves per-subject results to `reproduction/subject_results/subject_XXXX.pkl` and skips already-completed subjects on restart. Progress is logged to `reproduction/pas_progress.jsonl`.

## Hardware

- GPU: 24+ GB VRAM required (Llama-3-8B uses ~20.5 GB)
- batch_size=3 in `generateAnswer()` is tuned for 24 GB; increase for larger GPUs
- HPC jobs target A100-80GB via SLURM configs in `exp_configs/models/`
