# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Reproduction of "Personality Alignment of Large Language Models" (ICLR 2025). Implements the PAS (Persona Activation Steering) method to align LLMs with individual personality traits using the Big Five (OCEAN) model. Primary model: Meta-Llama-3-8B-Instruct, with support for other architectures (Mistral, Qwen, etc.).

## Commands

```bash
# Core (PAS pipeline only)
pip install .

# With benchmark dependencies (DPO, PPO, Prompt-MORL, Soups)
pip install .[benchmarks]

# With serving dependencies (vLLM, eval_served.py)
pip install .[serving]

# Everything
pip install .[benchmarks,serving]
```

### PAS (Persona Activation Steering)

Activation-steering method — the main contribution of the paper. Trains logistic regression probes to identify personality-relevant attention heads, then steers them at inference via `o_proj.bias` modification. Runs at fp16, ~20.5 GB VRAM.

Two variants:
- **few-shot-PAS** (`--modes few-shot-PAS`, **default**): Activation steering + few-shot prompt containing all 120 training items (the paper's original method)
- **PAS** (`--modes PAS`): Pure activation steering with a neutral system prompt

```bash
# Full run — default mode (few-shot-PAS, all 300 subjects, ~35 hours on RTX 4090)
python main.py --model_file meta-llama/Meta-Llama-3-8B-Instruct --num_subjects 0

# Full run — pure PAS (no few-shot prompt)
python main.py --modes PAS --model_file meta-llama/Meta-Llama-3-8B-Instruct --num_subjects 0

# Quick test (5 subjects)
python main.py --model_file meta-llama/Meta-Llama-3-8B-Instruct --num_subjects 5
```

Interrupted runs resume automatically from pickle files in `reproduction/subject_results/`.
Output: `reproduction/{PAS,few-shot-PAS}_Meta-Llama-3-8B-Instruct_OOD.json`

### Oracle PAS (Pre-determined Alpha + Full-Data Probes)

Variant of PAS that leverages prior runs to skip the 6-alpha sweep and improve probe quality. Uses a pre-determined alpha from prior runs and trains probes on all 300 items (train + test) instead of just 120 train items. Uses few-shot prompts during evaluation (same as default few-shot-PAS). Reduces runtime from ~35 hours to ~8 hours for 300 subjects.

```bash
# Analysis only (no GPU) — print alpha consistency report
python oracle_pas.py --analyze_only \
  --result_dirs reproduction/run1 reproduction/run2 reproduction/run3

# Full run with majority-vote alpha
python oracle_pas.py \
  --result_dirs reproduction/run1 reproduction/run2 reproduction/run3 \
  --alpha_strategy majority \
  --model_file meta-llama/Meta-Llama-3-8B-Instruct \
  --num_subjects 0 --batch_size 3 --eval_set both

# Full run with best-MAE alpha
python oracle_pas.py \
  --result_dirs reproduction/run1 reproduction/run2 reproduction/run3 \
  --alpha_strategy best_mae --num_subjects 0

# Quick test (5 subjects)
python oracle_pas.py \
  --result_dirs reproduction/run1 reproduction/run2 reproduction/run3 \
  --alpha_strategy majority --num_subjects 5
```

Alpha strategies: `majority` (most common alpha across runs, ties → lowest) or `best_mae` (alpha from the run with the smallest MAE sum).

Output:
- `reproduction/oracle_pas/subject_results/subject_XXXX.pkl` — per-subject results
- `reproduction/oracle_pas/Oracle-PAS-{strategy}_{model}_OOD.json` — OOD-180 metrics
- `reproduction/oracle_pas/Oracle-PAS-{strategy}_{model}_ALL.json` — ALL-300 metrics
- `reproduction/oracle_pas/oracle_pas_progress.jsonl` — progress log

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

### NO_CHANGE Baseline

Runs inference on a single subject without any steering or prompting — measures unmodified model behavior.

```bash
python main.py --modes NO_CHANGE --model_file meta-llama/Meta-Llama-3-8B-Instruct
```

### Common Options

| Flag | `main.py` | `benchmarks` | Description |
|------|-----------|-------------|-------------|
| Model | `--model_file` | `--model_name` | HuggingFace model name or path |
| Subjects | `--num_subjects` | `--num_subjects` | Number of subjects (0 = all 300) |
| Batch size | `--batch_size` | — | Inference batch size (default 16 for A100-80GB, use 3 for RTX 4090 24GB) |
| Output | `--output_dir` | `--output_dir` | Output directory |
| Data | — | `--data_dir` | PAPI data directory (default: `PAPI`) |

All training-based benchmarks use 4-bit QLoRA (~5-6 GB VRAM) and support resume via pickle checkpoints. Interrupted runs automatically skip completed subjects on restart.

### Serving with vLLM

Export a PAS-steered model as a standard HuggingFace checkpoint, then serve it with vLLM (or any HF-compatible runtime). The alpha sweep uses the few-shot prompt during evaluation (same as default few-shot-PAS).

```bash
# Step 1: Export bias deltas with full PAS alpha sweep (default, ~7 min on RTX 4090)
python export_for_serving.py export-biases \
  --model_file meta-llama/Meta-Llama-3-8B-Instruct \
  --subject_index 42 --output persona_biases.pt

# Or with a fixed alpha (skip sweep)
python export_for_serving.py export-biases \
  --model_file meta-llama/Meta-Llama-3-8B-Instruct \
  --subject_index 42 --alpha 4 --output persona_biases.pt

# Step 2: Bake bias deltas into a full model checkpoint
python export_for_serving.py bake \
  --model_file meta-llama/Meta-Llama-3-8B-Instruct \
  --biases persona_biases.pt \
  --output_dir ./baked_model

# Step 3: Serve with vLLM (no special config needed)
vllm serve ./baked_model --dtype float16
```

When `--alpha` is omitted, the export runs the full 6-alpha sweep `[0, 1, 2, 4, 6, 8]` on the subject's test set and picks the alpha with lowest MAE — matching the paper's method. Use `--batch_size 3` for RTX 4090 24GB.

The bias file is portable — you can store many persona files (~500KB each) and bake on demand. The `PASLM.export_biases()` method can also be called programmatically after `set_activate()`.

#### Evaluating a Served Model

Once a baked model is served via vLLM, evaluate it against ground-truth personality scores using the OpenAI-compatible API:

```bash
python eval_served.py \
  --model_dir ./baked_model \
  --api_base http://localhost:8000/v1 \
  --output eval_result.json \
  --raw_log raw_gen.log
```

`--model_dir` reads `persona_meta.json` for the subject index (or use `--subject_index` directly). Model name is auto-detected from the vLLM server.

#### Automated Serving Pipeline

`run_serving_pipeline.sh` automates the full loop (export → bake → serve → eval → cleanup) for multiple subjects:

```bash
# All 300 subjects
bash run_serving_pipeline.sh --num_subjects 0

# 5 subjects starting from index 10
bash run_serving_pipeline.sh --num_subjects 5 --start_index 10

# Custom model / port / output
bash run_serving_pipeline.sh --model_file meta-llama/Meta-Llama-3-8B-Instruct \
  --port 8001 --output_dir ./served_eval
```

Output structure under `--output_dir` (default `./served_eval`):
- `biases/s{i}.pt` — per-subject bias files
- `results/eval_s{i}.json` — per-subject evaluation results
- `raw_logs/s{i}.log` — raw generation logs

Supports resume: skips subjects whose `eval_s{i}.json` already exists. Handles vLLM lifecycle (start/stop/GPU cleanup) automatically.

## Architecture

### Core Pipeline (`main.py`)

1. **Load data**: `getItems()` reads PAPI questionnaires from `PAPI/` (IPIP-NEO-120 train set + 180-item test set)
2. **Preprocess activations**: For each Big Five trait, extract head-wise activations from `o_proj` layers using native PyTorch forward pre-hooks
3. **Per-subject loop** (`process_pas`): For each of 300 subjects:
   - Train logistic regression probes on (layer, head) activations → select top 24 heads
   - Compute intervention vectors (direction + std) per selected head
   - Test 6 alpha values [0, 1, 2, 4, 6, 8] by modifying `o_proj.bias`
   - Pick alpha minimizing MAE against ground-truth personality scores
4. **Aggregate results**: Save per-trait MAE to `reproduction/PAS_*.json`

### Key Modules

- **`PAlign/pas.py`** — `PASLM` class: model loading, activation extraction (`preprocess_activate_dataset`), probe training (`get_activations`), intervention application (`set_activate`). Uses native PyTorch forward pre-hooks on `o_proj` layers.
- **`baseline_utils.py`** — Answer parsing (A-E → 1-5 scores), MAE calculation, few-shot and personality-prompt baselines.

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

Core: `torch`, `transformers` (>=4.50), `einops`, `scikit-learn`, `numpy`, `pandas`
Benchmarks (`.[benchmarks]`): `peft` (>=0.7), `trl` (>=0.11, <0.12), `bitsandbytes` (>=0.43), `accelerate` (>=0.27), `datasets`
Serving (`.[serving]`): `vllm`, `openai`

## Data

- `PAPI/Test-set.json` — 300 clustered subjects with personality scores
- `PAPI/mpi_300_split.json` — Train/test question indices (120/180 split)
- `PAPI/IPIP-NEO-ItemKey.xls` — Question text lookup

### HPC Reproduction Pipeline

Full Table 1 reproduction on SLURM (A100-80GB). Submits all 8 methods (PAS, few-shot-PAS, few-shot, personality_prompt, DPO, PPO, Prompt-MORL, Soups) as separate jobs.

```bash
# One-time setup
bash hpc/setup_env.sh && chmod +x hpc/submit_reproduction.sh

# Submit all methods
./hpc/submit_reproduction.sh --all

# Submit specific methods
./hpc/submit_reproduction.sh --method pas few_shot_pas dpo

# Preview without submitting
./hpc/submit_reproduction.sh --all --dry-run

# Aggregate results after completion
python hpc/aggregate_reproduction.py
```

See `hpc/README.md` for batch size tuning, method list, and file structure.

## Known Constraints

- Default batch size: 16 (A100-80GB); use `--batch_size 3` for RTX 4090 24GB (~20.5GB VRAM)
- Each subject takes ~6-7 minutes on RTX 4090 (6 alphas × 180 questions in batches of 3)
- Paper Table 1 targets: A=0.94, C=0.91, E=0.86, N=0.98, O=0.72 (MAE, lower=better)
