# HPC Reproduction Pipeline

Run the full Table 1 reproduction (PAS + 6 baselines) on SLURM with A100-80GB GPUs.

## Quick Start

```bash
# 1. One-time setup (on HPC login node)
bash hpc/setup_env.sh
chmod +x hpc/submit_reproduction.sh

# 2. Submit all 8 methods
./hpc/submit_reproduction.sh --all

# 3. Monitor
squeue -u $USER
tail -f logs/reproduction/*.out

# 4. Aggregate results
python hpc/aggregate_reproduction.py
```

## Methods

| Method | Mode Flag | Description | Est. Time |
|--------|-----------|-------------|-----------|
| `pas` | `--modes PAS` | Pure activation steering (no few-shot prompt) | ~24h |
| `few_shot_pas` | `--modes few-shot-PAS` | Steering + few-shot prompt (paper's method) | ~24h |
| `dpo` | benchmarks.dpo | Per-subject DPO QLoRA | ~18h |
| `ppo` | benchmarks.ppo | Per-subject PPO QLoRA | ~20h |
| `prompt_morl` | benchmarks.prompt_morl | Shared SFT with personality prompt | ~3h |
| `soups` | benchmarks.soups | 10 extreme PPO + merge | ~10h |
| `few_shot` | `--modes few-shot` | Few-shot prompting only | ~2h |
| `personality_prompt` | `--modes personality_prompt` | P² personality description | ~2h |

## Usage

```bash
# Submit specific methods
./hpc/submit_reproduction.sh --method pas few_shot_pas few_shot personality_prompt 
./hpc/submit_reproduction.sh --method dpo ppo prompt_morl soups

# Override defaults
./hpc/submit_reproduction.sh --all --model meta-llama/Meta-Llama-3-8B-Instruct --num-subjects 10

# Custom output directory (default: reproduction/)
./hpc/submit_reproduction.sh --all --output-dir reproduction_v2

# Preview without submitting
./hpc/submit_reproduction.sh --all --dry-run

# List methods and resources
./hpc/submit_reproduction.sh --list-methods
```

## Batch Sizes

Tuned for A100-80GB:

| Component | RTX 4090 (24GB) | A100 (80GB) |
|-----------|----------------|-------------|
| Inference (main.py) | 3 | 16 |
| Inference (benchmarks) | 3 | 16 |
| DPO training | 4 (×4 accum) | 16 (×1 accum) |
| PPO training | 16 | 32 |
| Prompt-MORL training | 4 (×4 accum) | 16 (×1 accum) |

Override inference batch size: `--batch-size 8`

## PAS Variants

The original paper's PAS method combines activation steering **and** a few-shot system prompt containing all 120 training items. We separate these:

- **PAS** (`--modes PAS`): Pure activation steering with a neutral system prompt
- **few-shot** (`--modes few-shot`): Few-shot prompt only, no steering
- **few-shot-PAS** (`--modes few-shot-PAS`): Both (paper's original method)

## File Structure

```
hpc/
├── setup_env.sh              # One-time conda setup
├── reproduction.slurm        # SLURM job template
├── submit_reproduction.sh    # CLI submit wrapper
├── aggregate_reproduction.py # Results aggregator
└── README.md

exp_configs/methods/          # Per-method SLURM resources
├── pas.conf
├── few_shot_pas.conf
├── dpo.conf
├── ppo.conf
├── prompt_morl.conf
├── soups.conf
├── few_shot.conf
└── personality_prompt.conf
```

## Output

Results are saved to `reproduction/` with the standard naming:
```
reproduction/
├── PAS_Meta-Llama-3-8B-Instruct_OOD.json
├── few-shot-PAS_Meta-Llama-3-8B-Instruct_OOD.json
├── few-shot_Meta-Llama-3-8B-Instruct_OOD.json
├── personality_prompt_Meta-Llama-3-8B-Instruct_OOD.json
└── benchmarks/
    ├── dpo/DPO_Meta-Llama-3-8B-Instruct_OOD.json
    ├── ppo/PPO_Meta-Llama-3-8B-Instruct_OOD.json
    ├── prompt_morl/Prompt-MORL_Meta-Llama-3-8B-Instruct_OOD.json
    └── soups/Soups_Meta-Llama-3-8B-Instruct_OOD.json
```

## Resume

All methods support resume via pickle checkpoints. If a job times out or fails, resubmit the same method — it will skip completed subjects automatically.
