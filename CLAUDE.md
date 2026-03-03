# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Reproduction of "Personality Alignment of Large Language Models" (ICLR 2025). Implements the PAS (Persona Activation Steering) method to align LLMs with individual personality traits using the Big Five (OCEAN) model. Primary model: Meta-Llama-3-8B-Instruct, with support for other architectures (Mistral, Qwen, etc.).

## Commands

```bash
# Install
pip install .
```

### PAS (Persona Activation Steering)

Activation-steering method — the main contribution of the paper. Trains logistic regression probes to identify personality-relevant attention heads, then steers them at inference via `o_proj.bias` modification. Runs at fp16, ~20.5 GB VRAM.

```bash
# Full run (all 300 subjects, ~35 hours)
python main.py --modes PAS --model_file meta-llama/Meta-Llama-3-8B-Instruct --num_subjects 0

# Quick test (5 subjects)
python main.py --modes PAS --model_file meta-llama/Meta-Llama-3-8B-Instruct --num_subjects 5
```

Interrupted runs resume automatically from pickle files in `reproduction/subject_results/`.
Output: `reproduction/PAS_Meta-Llama-3-8B-Instruct_OOD.json`

### Benchmarking Baselines

Six comparison methods from the paper's Table 1. Two black-box baselines run via `main.py`; four white-box training-based methods run via `benchmarks/`.

#### 1. DPO (Direct Preference Optimization)

Per-subject method. For each of 300 subjects, trains a QLoRA adapter using DPO on preference pairs built from the subject's 120 train items (chosen = correct option, rejected = opposite extreme; neutral items excluded). The adapter is evaluated on 180 test items, then discarded.

```bash
# Full run (~20-25 hours)
python -m benchmarks.dpo.run_dpo --model_name meta-llama/Meta-Llama-3-8B-Instruct --num_subjects 0

# Quick test (1 subject)
python -m benchmarks.dpo.run_dpo --model_name meta-llama/Meta-Llama-3-8B-Instruct --num_subjects 1
```

Output: `reproduction/benchmarks/dpo/DPO_Meta-Llama-3-8B-Instruct_OOD.json`

#### 2. PPO (Proximal Policy Optimization)

Per-subject method. For each subject, trains a QLoRA adapter using PPO with a custom reward function: `-abs(predicted_score - correct_score)` (or -6 for unparseable answers). No separate reward model.

```bash
# Full run (~25-30 hours)
python -m benchmarks.ppo.run_ppo --model_name meta-llama/Meta-Llama-3-8B-Instruct --num_subjects 0

# Quick test (1 subject)
python -m benchmarks.ppo.run_ppo --model_name meta-llama/Meta-Llama-3-8B-Instruct --num_subjects 1
```

Output: `reproduction/benchmarks/ppo/PPO_Meta-Llama-3-8B-Instruct_OOD.json`

#### 3. Few-Shot Prompting

Black-box baseline. Injects all 120 training items (question text + ground-truth rating) into the system prompt as behavioral examples, then evaluates on 180 test items. No training required.

```bash
# Full run
python main.py --modes few-shot --model_file meta-llama/Meta-Llama-3-8B-Instruct --num_subjects 0

# Quick test (5 subjects)
python main.py --modes few-shot --model_file meta-llama/Meta-Llama-3-8B-Instruct --num_subjects 5
```

Output: `reproduction/few-shot_Meta-Llama-3-8B-Instruct_OOD.json`

#### 4. Prompt-MORL (Prompt-based Multi-Objective RL)

Shared-model method. Trains a single QLoRA adapter via SFT on all subjects' training items pooled together (~27K examples). Each training example has a personality-conditioned system prompt ("You are an AI with Agreeableness level X, ..."). At evaluation, each subject's trait scores are embedded in the system prompt.

```bash
# Full run (~2 hours: ~20 min train + ~2 min/subject eval)
python -m benchmarks.prompt_morl.run_prompt_morl --model_name meta-llama/Meta-Llama-3-8B-Instruct --num_subjects 0

# Quick test (5 subjects)
python -m benchmarks.prompt_morl.run_prompt_morl --model_name meta-llama/Meta-Llama-3-8B-Instruct --num_subjects 5
```

Output: `reproduction/benchmarks/prompt_morl/Prompt-MORL_Meta-Llama-3-8B-Instruct_OOD.json`

#### 5. Personalized-Soups

Two-phase method. First trains 10 extreme QLoRA adapters via PPO: `{high, low} × {A, C, E, N, O}`, each targeting maximum/minimum scores for its trait. Then for each subject, computes merge weights from trait scores (normalized to [0,1]), produces a weighted average of the 10 LoRA adapters, and evaluates.

```bash
# Full run (~12 hours: ~100 min train + ~10 hours eval)
python -m benchmarks.soups.run_soups --model_name meta-llama/Meta-Llama-3-8B-Instruct --num_subjects 0

# Quick test (5 subjects)
python -m benchmarks.soups.run_soups --model_name meta-llama/Meta-Llama-3-8B-Instruct --num_subjects 5
```

Output: `reproduction/benchmarks/soups/Soups_Meta-Llama-3-8B-Instruct_OOD.json`

#### 6. Personality Prompt (P²)

Black-box baseline. Uses a pre-generated personality description (from `PAPI/personality_prompt.json`) as the system prompt for each subject, then evaluates on 180 test items. No training required.

```bash
# Full run
python main.py --modes personality_prompt --model_file meta-llama/Meta-Llama-3-8B-Instruct --num_subjects 0

# Quick test (5 subjects)
python main.py --modes personality_prompt --model_file meta-llama/Meta-Llama-3-8B-Instruct --num_subjects 5
```

Output: `reproduction/personality_prompt_Meta-Llama-3-8B-Instruct_OOD.json`

#### Running Multiple Training-Based Benchmarks Together

```bash
# Run all 4 white-box methods
python -m benchmarks.run_all --methods all --model_name meta-llama/Meta-Llama-3-8B-Instruct --num_subjects 0

# Run a subset
python -m benchmarks.run_all --methods dpo prompt_morl --num_subjects 5
```

### Common Options

| Flag | `main.py` | `benchmarks` | Description |
|------|-----------|-------------|-------------|
| Model | `--model_file` | `--model_name` | HuggingFace model name or path |
| Subjects | `--num_subjects` | `--num_subjects` | Number of subjects (0 = all 300) |
| Output | `--output_dir` | `--output_dir` | Output directory |
| Data | — | `--data_dir` | PAPI data directory (default: `PAPI`) |

All training-based benchmarks use 4-bit QLoRA (~5-6 GB VRAM) and support resume via pickle checkpoints. Interrupted runs automatically skip completed subjects on restart.

## Architecture

### Core Pipeline (`main.py`)

1. **Load data**: `getItems()` reads PAPI questionnaires from `PAPI/` (IPIP-NEO-120 train set + 180-item test set)
2. **Preprocess activations**: For each Big Five trait, extract head-wise activations from `o_proj` layers using baukit's `TraceDict`
3. **Per-subject loop** (`process_pas`): For each of 300 subjects:
   - Train logistic regression probes on (layer, head) activations → select top 24 heads
   - Compute intervention vectors (direction + std) per selected head
   - Test 6 alpha values [0, 1, 2, 4, 6, 8] by modifying `o_proj.bias`
   - Pick alpha minimizing MAE against ground-truth personality scores
4. **Aggregate results**: Save per-trait MAE to `reproduction/PAS_*.json`

### Key Modules

- **`PAlign/pas.py`** — `PASLM` class: model loading, activation extraction (`preprocess_activate_dataset`), probe training (`get_activations`), intervention application (`set_activate`). Uses native PyTorch forward pre-hooks on `o_proj` layers.
- **`baseline_utils.py`** — Answer parsing (A-E → 1-5 scores), MAE calculation, few-shot and personality-prompt baselines.
- **`PAlign/modeling_llama.py`, `modeling_mistral.py`** — Custom model implementations with activation hook support (legacy; new code uses standard HuggingFace models with hooks).

### Intervention Mechanism

PAS modifies `o_proj.bias` in selected attention layers:
```
displacement[head] = alpha × std × direction_vector
new_bias = F.linear(displacement, o_proj.weight)
```
This steers model behavior toward the target personality profile without fine-tuning weights.

### Multi-Model Support

`prompt_to_tokens()` and `generateAnswer()` handle different chat templates:
- Llama-3: `apply_chat_template()` with `<|eot_id|>` splitting
- Qwen/GPT-OSS: `apply_chat_template()` with `<|im_start|>assistant\n` splitting
- Mistral: `[INST]` template

## Key Dependencies

Core: `torch`, `transformers` (>=4.50), `baukit`, `einops`, `scikit-learn`, `numpy`, `pandas`
Benchmarks: `peft` (>=0.7), `trl` (>=0.11, <0.20), `bitsandbytes` (>=0.43), `accelerate` (>=0.27), `datasets`

## Data

- `PAPI/Test-set.json` — 300 clustered subjects with personality scores
- `PAPI/mpi_300_split.json` — Train/test question indices (120/180 split)
- `PAPI/IPIP-NEO-ItemKey.xls` — Question text lookup

## Known Constraints

- Batch size hardcoded to 3 (fits 24GB VRAM on RTX 4090, model uses ~20.5GB)
- Each subject takes ~6-7 minutes (6 alphas × 180 questions in batches of 3)
- Paper Table 1 targets: A=0.94, C=0.91, E=0.86, N=0.98, O=0.72 (MAE, lower=better)
