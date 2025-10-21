# Intervention Parameter Storage

This guide explains how PAS intervention parameters are saved, structured, and used. Understanding this is essential for working with PAS interventions and preparing them for deployment.

## Overview

PAS (Personality Activate Search) computes **sample-specific interventions** during training. Each intervention represents personality-aligned directions that can be applied to attention heads to steer the model's behavior.

**Why save interventions?**
- ✅ Reuse computed interventions without retraining
- ✅ Deploy interventions to production (via vLLM)
- ✅ Analyze which attention heads are personality-relevant
- ✅ Create multiple personality-aligned model variants
- ✅ Archive training results for reproducibility

---

## What Gets Saved

### File Types

PAS creates two types of files:

1. **Individual Files**: `./interventions/PAS_{model_name}_sample{N}.pkl`
   - One file per training sample
   - Contains intervention for a specific personality profile
   - Standalone and self-contained

2. **Aggregated File**: `./interventions/PAS_{model_name}_all.pkl`
   - Single file with all interventions
   - Includes model metadata
   - Useful for batch processing

### Directory Structure

After running PAS training:

```
PAlign-self/
├── interventions/
│   ├── PAS_qwen3-0.6b_sample0.pkl      # Sample 0's intervention
│   ├── PAS_qwen3-0.6b_sample1.pkl      # Sample 1's intervention
│   ├── PAS_qwen3-0.6b_sample2.pkl      # Sample 2's intervention
│   ├── ...
│   └── PAS_qwen3-0.6b_all.pkl          # All interventions aggregated
├── log/
│   └── PAS_qwen3-0.6b_OOD.json         # Evaluation results
```

---

## Data Structure

### Individual Intervention File

Each `.pkl` file contains a dictionary with this structure:

```python
{
    'sample_id': str,                    # Unique identifier (e.g., "case_001")
    'sample_index': int,                 # Index in dataset (0, 1, 2, ...)
    
    'activate': dict,                    # *** CORE: Intervention parameters ***
    # Format: {
    #     'model.layers.5.self_attn.head_out': [
    #         (head_idx, direction_vector, std),  # Head 0
    #         (head_idx, direction_vector, std),  # Head 3
    #         ...
    #     ],
    #     'model.layers.12.self_attn.head_out': [...],
    #     ...
    # }
    
    'best_alpha': float,                 # Selected alpha value (0, 1, 2, 4, 6, 8)
    'alpha_scores': list,                # MAE for each alpha tested [score_0, score_1, ...]
    'all_alpha_values': list,            # Alpha values tested [0, 1, 2, 4, 6, 8]
    
    'labels': list,                      # Training labels (0s and 1s for binary classification)
    'personality_scores': {              # Target personality (OCEAN model)
        'A': int,  # Agreeableness (1-5)
        'C': int,  # Conscientiousness (1-5)
        'E': int,  # Extraversion (1-5)
        'N': int,  # Neuroticism (1-5)
        'O': int,  # Openness (1-5)
    },
    
    'system_prompt': str,                # Context prompt used during training
}
```

### Aggregated File

The `_all.pkl` file contains:

```python
{
    'interventions': list,               # List of all individual intervention dicts above
    'model_file': str,                   # Path to base model
    'num_samples': int,                  # Total number of samples processed
    
    'model_config': {                    # Model architecture info
        'num_layers': int,               # e.g., 28 for Qwen3-0.6B
        'num_heads': int,                # e.g., 16
        'hidden_size': int,              # e.g., 1024
    }
}
```

---

## How Interventions Are Computed

Understanding the computation helps interpret the saved data:

### Step 1: Extract Activations

For each personality trait (A, C, E, N, O), PAS:
1. Feeds personality questionnaire items through the model
2. Captures attention head outputs (before `o_proj`)
3. Stores activations for positive/negative examples

### Step 2: Train Probes

For each attention head:
1. Train a logistic regression classifier
2. Identify heads that predict personality traits
3. Select top-N heads (default: 24)

### Step 3: Compute Directions

For selected heads:
1. Calculate "center of mass" between positive and negative examples
2. Normalize direction vectors
3. Store: `(head_index, direction_vector, std_deviation)`

### Step 4: Test Alpha Values

Test different intervention strengths:
```python
alpha_values = [0, 1, 2, 4, 6, 8]  # 0 = no intervention

for alpha in alpha_values:
    # Apply: activation' = activation + alpha * std * direction
    # Evaluate on validation set
    # Record MAE (Mean Absolute Error)

best_alpha = argmin(MAE_scores)
```

---

## Code Implementation

### Where Interventions Are Saved

In `main.py`, the `process_pas()` function handles saving:

**Key locations:**

```python
# Line 14: Import pickle
import pickle

# Lines 209-210: Create directories
os.makedirs('./interventions', exist_ok=True)
os.makedirs('./log', exist_ok=True)

# Lines 255-269: Create intervention data structure
intervention_data = {
    'sample_id': sample['test'][0]['case'],
    'sample_index': index,
    'activate': activate,           # Computed by model.get_activations()
    'labels': labels,
    'personality_scores': {
        'A': ..., 'C': ..., 'E': ..., 'N': ..., 'O': ...
    },
    'system_prompt': system_prompt_text,
    'best_alpha': best_alpha,
    'alpha_scores': alpha_scores,
    'all_alpha_values': [0, 1, 2, 4, 6, 8],
}

# Lines 299-312: Save individual file
intervention_file = f'./interventions/PAS_{model_name}_sample{index}.pkl'
with open(intervention_file, 'wb') as f:
    pickle.dump(intervention_data, f)

print(f"✅ Saved intervention for sample {index}: alpha={best_alpha}, MAE={min(alpha_scores)}")

# Lines 317-335: Save aggregated file (after processing all samples)
aggregated_file = f'./interventions/PAS_{model_name}_all.pkl'
with open(aggregated_file, 'wb') as f:
    pickle.dump({
        'interventions': all_interventions,
        'model_file': model_file,
        'num_samples': len(data),
        'model_config': {...}
    }, f)
```

---

## Loading and Using Interventions

### Load Individual Intervention

```python
import pickle

# Load one intervention
with open('./interventions/PAS_qwen3-0.6b_sample0.pkl', 'rb') as f:
    intervention = pickle.load(f)

# Access components
print(f"Sample ID: {intervention['sample_id']}")
print(f"Best alpha: {intervention['best_alpha']}")
print(f"Personality: {intervention['personality_scores']}")

# Access intervention parameters
activate = intervention['activate']
for layer_name, head_interventions in activate.items():
    print(f"\nLayer: {layer_name}")
    for head_idx, direction, std in head_interventions:
        print(f"  Head {head_idx}: std={std:.4f}, direction shape={direction.shape}")
```

### Load Aggregated File

```python
import pickle

# Load all interventions
with open('./interventions/PAS_qwen3-0.6b_all.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"Model: {data['model_file']}")
print(f"Num samples: {data['num_samples']}")
print(f"Model config: {data['model_config']}")

# Access individual interventions
for intervention in data['interventions']:
    print(f"Sample {intervention['sample_index']}: alpha={intervention['best_alpha']}")
```

### Apply Intervention to Model

```python
from PAlign.pas import get_model

# Load model
model, tokenizer = get_model("path/to/model")

# Load intervention
with open('./interventions/PAS_qwen3-0.6b_sample0.pkl', 'rb') as f:
    intervention = pickle.load(f)

# Apply intervention
model.set_activate(intervention['activate'], intervention['best_alpha'])

# Now generate with personality alignment
prompt = "What is your opinion on teamwork?"
tokens = model.prompt_to_tokens(prompt)
output = model.model.generate(**tokens, max_new_tokens=100)
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
```

---

## Inspecting Interventions

### Analyze Intervention Distribution

```python
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load aggregated file
with open('./interventions/PAS_qwen3-0.6b_all.pkl', 'rb') as f:
    data = pickle.load(f)

# Analyze alpha distribution
alphas = [i['best_alpha'] for i in data['interventions']]
print(f"Alpha distribution: {np.bincount(alphas)}")

# Analyze which layers are most intervened
layer_counts = {}
for intervention in data['interventions']:
    for layer_name in intervention['activate'].keys():
        layer_idx = int(layer_name.split('.')[2])
        layer_counts[layer_idx] = layer_counts.get(layer_idx, 0) + 1

print(f"Most intervened layers: {sorted(layer_counts.items(), key=lambda x: x[1], reverse=True)[:5]}")
```

### Visualize Personality Profiles

```python
import pickle
import pandas as pd

# Load interventions
with open('./interventions/PAS_qwen3-0.6b_all.pkl', 'rb') as f:
    data = pickle.load(f)

# Create DataFrame
profiles = []
for intervention in data['interventions']:
    profile = intervention['personality_scores'].copy()
    profile['sample_id'] = intervention['sample_id']
    profile['alpha'] = intervention['best_alpha']
    profiles.append(profile)

df = pd.DataFrame(profiles)
print(df.describe())

# Personality trait correlations
print("\nTrait correlations:")
print(df[['A', 'C', 'E', 'N', 'O']].corr())
```

---

## File Size and Performance

### Expected File Sizes

For typical models:

| Model Size | Per Sample | Aggregated (300 samples) |
|------------|-----------|--------------------------|
| 0.5B - 1B  | ~50-100 KB | ~15-30 MB |
| 7B - 13B   | ~200-500 KB | ~60-150 MB |
| 70B        | ~1-2 MB | ~300-600 MB |

**Why these sizes?**
- Most space: `direction_vector` (float arrays, shape ~64-128 per head)
- Number of heads intervened: Usually 24 total across selected layers
- String fields: Negligible size

### Loading Speed

```python
import time
import pickle

# Individual file (fast)
start = time.time()
with open('./interventions/PAS_qwen3-0.6b_sample0.pkl', 'rb') as f:
    data = pickle.load(f)
print(f"Loaded in {time.time() - start:.3f}s")  # ~0.001-0.01s

# Aggregated file (slower, but still fast)
start = time.time()
with open('./interventions/PAS_qwen3-0.6b_all.pkl', 'rb') as f:
    data = pickle.load(f)
print(f"Loaded in {time.time() - start:.3f}s")  # ~0.1-1s for 300 samples
```

---

## Sample-Specific vs Trait-Specific

### Current Implementation: Sample-Specific

Each intervention corresponds to a **complete personality profile** (one survey respondent):

```python
personality_scores = {
    'A': 4,  # High agreeableness
    'C': 3,  # Medium conscientiousness
    'E': 5,  # High extraversion
    'N': 2,  # Low neuroticism
    'O': 4,  # High openness
}
```

This means:
- 300 samples → 300 different personality combinations
- Each intervention represents a holistic personality profile
- Useful for creating diverse agents with complex personalities

### Future: Trait-Specific (Planned)

Would involve interventions for individual traits at extremes:

```python
# High Openness only
personality_scores = {'A': 3, 'C': 3, 'E': 3, 'N': 3, 'O': 5}

# Low Neuroticism only
personality_scores = {'A': 3, 'C': 3, 'E': 3, 'N': 1, 'O': 3}
```

**To implement trait-specific interventions:**
1. Replace training dataset with synthetic perfect scores
2. Run PAS with isolated trait variations
3. Save interventions as `PAS_{model}_trait_{trait}_{level}.pkl`
4. Combine multiple trait interventions at deployment

---

## Best Practices

### 1. Organize Interventions

```bash
interventions/
├── qwen3-0.6b/
│   ├── sample_specific/
│   │   ├── PAS_qwen3-0.6b_sample0.pkl
│   │   └── ...
│   └── trait_specific/  # Future
│       ├── PAS_qwen3-0.6b_trait_O_high.pkl
│       └── ...
└── llama3-8b/
    └── sample_specific/
        └── ...
```

### 2. Version Control

Include metadata in filenames or companion files:

```python
# Save with version info
intervention_data['version'] = '1.0'
intervention_data['timestamp'] = datetime.now().isoformat()
intervention_data['dataset'] = 'IPIP-NEO-300'
```

### 3. Backup Important Interventions

```bash
# Backup aggregated files
cp interventions/PAS_qwen3-0.6b_all.pkl backups/PAS_qwen3-0.6b_all_2025-10-21.pkl
```

### 4. Document Personality Profiles

Create a CSV index:

```python
import pickle
import pandas as pd

# Load interventions
with open('./interventions/PAS_qwen3-0.6b_all.pkl', 'rb') as f:
    data = pickle.load(f)

# Create index
index = []
for i in data['interventions']:
    index.append({
        'sample_index': i['sample_index'],
        'sample_id': i['sample_id'],
        'alpha': i['best_alpha'],
        **i['personality_scores']
    })

df = pd.DataFrame(index)
df.to_csv('./interventions/PAS_qwen3-0.6b_index.csv', index=False)
```

---

## Troubleshooting

### Issue: Files not being created

**Check:**
1. Directory exists: `mkdir -p ./interventions`
2. Write permissions: `ls -la ./interventions`
3. Disk space: `df -h`

### Issue: File size unexpectedly large

**Causes:**
- Too many samples
- Large models (more heads, larger hidden dims)

**Solutions:**
- Save only best N interventions
- Compress files: `import gzip; gzip.open(...)`

### Issue: Loading errors

**Causes:**
- Python version mismatch
- Missing dependencies

**Solutions:**
```python
# Use protocol for compatibility
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

# Or use JSON for simple data (slower, but portable)
import json
json.dump(data_as_dict, f)
```

---

## Next Steps

Now that interventions are saved:

1. **Deploy with vLLM**: See `VLLM_DEPLOYMENT.md`
2. **Analyze interventions**: Use inspection code above
3. **Create custom personalities**: Select specific samples
4. **Archive for research**: Back up important interventions

---

## Further Reading

- **Adding New Models**: See `ADDING_NEW_MODELS.md`
- **vLLM Deployment**: See `VLLM_DEPLOYMENT.md`
- **PAS Paper**: https://openreview.net/forum?id=0DZEs8NpUH

---

**Questions?** The intervention storage system is designed to be flexible and extensible. Understanding this structure is key to effectively deploying PAS in production!

