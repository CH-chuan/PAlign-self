# PAS-Qwen3 vLLM Integration Plan

## Overview
This document outlines the strategy for integrating PAS (Personality Activate Search) intervention parameters with vLLM for production serving in an agentic workflow.

## 🎯 Goals

1. **Save PAS intervention parameters** for reuse
2. **Package Qwen3 model with PAS interventions** as a deployable unit
3. **Serve via vLLM** with personality-aligned responses
4. **Enable dynamic personality switching** (optional advanced feature)

---

## 📋 Architecture Options

### Option 1: Baked-In Interventions (Recommended for Single Personality)
**Approach**: Apply PAS interventions and save the modified model weights directly.

**Pros**:
- ✅ Simplest integration with vLLM
- ✅ No runtime overhead
- ✅ Works with standard vLLM serving
- ✅ Easy to deploy and maintain

**Cons**:
- ❌ Fixed to one personality profile
- ❌ Need to save separate model for each personality
- ❌ Cannot switch personalities at runtime

### Option 2: Custom vLLM Model with Dynamic Interventions (Advanced)
**Approach**: Register custom Qwen3 model in vLLM that applies PAS interventions dynamically.

**Pros**:
- ✅ Switch personalities at runtime
- ✅ Single model serves multiple personalities
- ✅ More flexible for agentic workflows

**Cons**:
- ❌ More complex implementation
- ❌ Requires vLLM model plugin
- ❌ May have slight runtime overhead
- ❌ Need to maintain custom vLLM integration

---

## 🔧 Implementation Plan

### Phase 1: Save Intervention Parameters ✅

#### Step 1.1: Modify `main.py` to Save PAS Parameters

Add intervention parameter saving to `process_pas()`:

```python
def process_pas(data, model, tokenizer, model_file):
    """Process data using PAS and save intervention parameters."""
    
    # Create directories
    os.makedirs('./interventions', exist_ok=True)
    os.makedirs('./log', exist_ok=True)
    
    # Prepare personal data for activation
    personal_data = []
    for personal in ['A', 'C', 'E', 'N', 'O']:
        for item in data[0]['train']:
            if item['label_ocean'] == personal:
                personal_data.append({
                    'question': TEMPLATE.format(item['text']),
                    'answer_matching_behavior': 'A',
                    'answer_not_matching_behavior': 'E'
                })
    
    # Preprocess activation dataset
    all_head_wise_activations = model.preprocess_activate_dataset(personal_data)
    
    results = []
    all_interventions = []  # Store all interventions
    
    for index, sample in enumerate(tqdm(data)):
        model.reset_all()
        
        # Generate system prompt
        system_prompt_text = 'Here are some of your behaviors...' + \
                             ';'.join([f"{it['text']}:{SCORES_BACK[it['value']]}" 
                                      for it in sample['train']])
        
        # Prepare labels and activations
        labels = []
        head_wise_activations = []
        personal_number = 0
        for personal in ['A', 'C', 'E', 'N', 'O']:
            for item in sample['train']:
                if item['label_ocean'] == personal:
                    if item['value'] not in [0, 3]:
                        if item['value'] > 3:
                            labels.extend([1, 0])
                        else:
                            labels.extend([0, 1])
                        head_wise_activations.extend([
                            deepcopy(all_head_wise_activations[personal_number]),
                            deepcopy(all_head_wise_activations[personal_number + 1])
                        ])
                    personal_number += 2
        
        # Get activations for intervention
        activate = model.get_activations(deepcopy(head_wise_activations), 
                                        labels, num_to_intervene=24)
        
        # Test different activation levels
        result_cache = []
        alpha_scores = []
        for num in [0, 1, 2, 4, 6, 8]:
            model.reset_all()
            model.set_activate(activate, num)
            answers = generateAnswer(tokenizer, model, data[0]['test'], TEMPLATE,
                                   system_prompt=system_prompt_text, 
                                   model_file=model_file)
            result = process_answers(answers, sample)
            result_cache.append(result)
            
            # Calculate score for this alpha
            score = sum([k[1] for k in result['mean_ver_abs']['mean']])
            if str(score) == 'nan':
                score = 1e6
            alpha_scores.append(score)
        
        # Select best alpha
        best_alpha_idx = np.array(alpha_scores).argmin()
        best_alpha = [0, 1, 2, 4, 6, 8][best_alpha_idx]
        rs = result_cache[best_alpha_idx]
        rs['alpha'] = best_alpha
        results.append(rs)
        
        # Save intervention parameters for this sample
        intervention_data = {
            'sample_id': sample['test'][0]['case'],
            'sample_index': index,
            'activate': activate,  # Dictionary with intervention vectors
            'best_alpha': best_alpha,
            'labels': labels,
            'personality_scores': {
                'A': sample['train'][0]['value'] if sample['train'] else None,
                'C': sample['train'][1]['value'] if len(sample['train']) > 1 else None,
                'E': sample['train'][2]['value'] if len(sample['train']) > 2 else None,
                'N': sample['train'][3]['value'] if len(sample['train']) > 3 else None,
                'O': sample['train'][4]['value'] if len(sample['train']) > 4 else None,
            },
            'system_prompt': system_prompt_text,
            'alpha_scores': alpha_scores,
        }
        all_interventions.append(intervention_data)
        
        # Save individual intervention
        intervention_file = f'./interventions/PAS_{model_file.split("/")[-1]}_sample{index}.pkl'
        with open(intervention_file, 'wb') as f:
            pickle.dump(intervention_data, f)
        
        print(f"Saved intervention for sample {index} with alpha={best_alpha}")
    
    # Save aggregated interventions
    aggregated_file = f'./interventions/PAS_{model_file.split("/")[-1]}_all.pkl'
    with open(aggregated_file, 'wb') as f:
        pickle.dump({
            'interventions': all_interventions,
            'model_file': model_file,
            'num_samples': len(data),
            'model_config': {
                'num_layers': model.model.model.config.num_hidden_layers,
                'num_heads': model.model.model.config.num_attention_heads,
                'hidden_size': model.model.model.config.hidden_size,
            }
        }, f)
    
    print(f"✅ Saved all interventions to {aggregated_file}")
    
    return results
```

#### Step 1.2: Add Import
Add `pickle` import at top of `main.py`:
```python
import pickle
import os
```

---

### Phase 2: Create Model Baking Script (Option 1)

#### Step 2.1: Create `bake_pas_model.py`

This script applies PAS interventions permanently to model weights:

```python
#!/usr/bin/env python3
"""
Bake PAS interventions into Qwen3 model weights for vLLM serving.

This creates a new model with PAS interventions permanently applied,
suitable for serving via vLLM without custom modifications.
"""

import torch
import pickle
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig
from PAlign.pas import get_model
import torch.nn.functional as F
from einops import rearrange
import numpy as np


def apply_pas_to_model(model, intervention_data, alpha=None):
    """
    Apply PAS interventions to model weights permanently.
    
    Args:
        model: The PASLM model instance
        intervention_data: Dictionary with intervention parameters
        alpha: Override alpha value (uses best_alpha if None)
    """
    activate = intervention_data['activate']
    best_alpha = alpha if alpha is not None else intervention_data['best_alpha']
    
    num_heads = model.model.model.config.num_attention_heads
    head_dim = model.model.model.config.hidden_size // num_heads
    
    print(f"Applying PAS interventions with alpha={best_alpha}")
    
    for head_out_name, list_int_vec in activate.items():
        layer_no = int(head_out_name.split('.')[2])
        
        # Create displacement vector
        displacement = np.zeros((num_heads, head_dim))
        for head_no, head_vec, std in list_int_vec:
            displacement[head_no] = best_alpha * std * head_vec
        
        device = model.model.model.layers[layer_no].self_attn.o_proj.weight.device
        displacement = torch.tensor(
            rearrange(displacement, 'h d -> (h d)'), 
            device=device,
            dtype=model.model.model.layers[layer_no].self_attn.o_proj.weight.dtype
        )
        
        # Apply via linear transformation to get bias
        bias_addition = F.linear(
            displacement, 
            model.model.model.layers[layer_no].self_attn.o_proj.weight
        )
        
        # Get current bias (or create zeros if doesn't exist)
        current_bias = model.model.model.layers[layer_no].self_attn.o_proj.bias
        if current_bias is None:
            current_bias = torch.zeros(
                model.model.model.config.hidden_size,
                device=device,
                dtype=model.model.model.layers[layer_no].self_attn.o_proj.weight.dtype
            )
        
        # Add intervention to bias
        new_bias = current_bias + bias_addition
        model.model.model.layers[layer_no].self_attn.o_proj.bias = torch.nn.parameter.Parameter(new_bias)
        
        print(f"  ✓ Applied intervention to layer {layer_no}")
    
    return model


def save_baked_model(model, tokenizer, output_dir, intervention_metadata):
    """
    Save the model with baked PAS interventions.
    
    Args:
        model: Model with PAS interventions applied
        tokenizer: Tokenizer
        output_dir: Output directory path
        intervention_metadata: Metadata about the interventions
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving baked model to {output_dir}")
    
    # Save the model using HuggingFace format
    model.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save intervention metadata
    metadata_file = output_path / "pas_metadata.pkl"
    with open(metadata_file, 'wb') as f:
        pickle.dump(intervention_metadata, f)
    
    # Save a README
    readme_content = f"""# PAS-Enhanced Qwen3 Model

This model has Personality Activate Search (PAS) interventions baked into the weights.

## Intervention Details
- **Alpha**: {intervention_metadata['best_alpha']}
- **Sample ID**: {intervention_metadata['sample_id']}
- **Num Intervened Heads**: {intervention_metadata.get('num_heads_intervened', 24)}

## Personality Profile
{intervention_metadata.get('personality_scores', {})}

## Usage with vLLM

```python
from vllm import LLM, SamplingParams

# Load the model
llm = LLM(model="{output_dir}")

# Generate
prompts = ["Your prompt here"]
sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

## Original Model
Base model: {intervention_metadata['model_file']}
"""
    
    with open(output_path / "README.md", 'w') as f:
        f.write(readme_content)
    
    print(f"✅ Model saved successfully to {output_dir}")
    print(f"   - Model weights: {output_dir}")
    print(f"   - Metadata: {metadata_file}")
    print(f"   - README: {output_path / 'README.md'}")


def main():
    parser = argparse.ArgumentParser(description="Bake PAS interventions into model weights")
    parser.add_argument("--model_file", required=True, help="Path to base model")
    parser.add_argument("--intervention_file", required=True, 
                       help="Path to intervention .pkl file")
    parser.add_argument("--output_dir", required=True, 
                       help="Output directory for baked model")
    parser.add_argument("--alpha", type=float, default=None,
                       help="Override alpha value (uses best_alpha if not specified)")
    parser.add_argument("--sample_index", type=int, default=0,
                       help="Which sample's intervention to use")
    
    args = parser.parse_args()
    
    print("="*60)
    print("PAS Model Baking Tool")
    print("="*60)
    
    # Load base model
    print(f"\n1. Loading base model: {args.model_file}")
    model, tokenizer = get_model(args.model_file)
    print("   ✓ Model loaded")
    
    # Load intervention parameters
    print(f"\n2. Loading intervention parameters: {args.intervention_file}")
    with open(args.intervention_file, 'rb') as f:
        if 'all.pkl' in args.intervention_file:
            # Aggregated file
            data = pickle.load(f)
            intervention_data = data['interventions'][args.sample_index]
            print(f"   Using sample {args.sample_index} from aggregated file")
        else:
            # Individual file
            intervention_data = pickle.load(f)
    
    print(f"   ✓ Loaded intervention (alpha={intervention_data['best_alpha']})")
    
    # Apply interventions
    print(f"\n3. Applying PAS interventions...")
    model = apply_pas_to_model(model, intervention_data, alpha=args.alpha)
    print("   ✓ Interventions applied")
    
    # Save baked model
    print(f"\n4. Saving baked model...")
    intervention_metadata = {
        'model_file': args.model_file,
        'intervention_file': args.intervention_file,
        'sample_id': intervention_data.get('sample_id', 'unknown'),
        'sample_index': args.sample_index,
        'best_alpha': args.alpha if args.alpha else intervention_data['best_alpha'],
        'personality_scores': intervention_data.get('personality_scores', {}),
        'system_prompt': intervention_data.get('system_prompt', ''),
    }
    
    save_baked_model(model, tokenizer, args.output_dir, intervention_metadata)
    
    print("\n" + "="*60)
    print("✅ Done! Model ready for vLLM serving")
    print("="*60)
    print(f"\nTo serve with vLLM:")
    print(f"  vllm serve {args.output_dir}")
    print(f"\nOr in Python:")
    print(f"  from vllm import LLM")
    print(f"  llm = LLM(model='{args.output_dir}')")


if __name__ == "__main__":
    main()
```

---

### Phase 3: vLLM Integration (Option 1 - Simple)

#### Step 3.1: Test with vLLM

After baking the model:

```bash
# Install vLLM
pip install vllm

# Serve the baked model
vllm serve ./baked_models/pas_qwen3_personality_A

# Or use in Python
python -c "
from vllm import LLM, SamplingParams

llm = LLM(model='./baked_models/pas_qwen3_personality_A')
prompts = ['Tell me about yourself']
outputs = llm.generate(prompts, SamplingParams(temperature=0.7))
print(outputs[0].outputs[0].text)
"
```

---

### Phase 4: Custom vLLM Model (Option 2 - Advanced)

For dynamic personality switching, create a vLLM plugin:

#### Step 4.1: Create `vllm_pas_plugin/` Directory Structure

```
vllm_pas_plugin/
├── __init__.py
├── pas_qwen3.py        # Custom Qwen3 with PAS
├── pas_manager.py      # Manages interventions
└── setup.py            # Plugin setup
```

#### Step 4.2: Implement Custom vLLM Model

**File**: `vllm_pas_plugin/pas_qwen3.py`

```python
"""Custom Qwen3 model with PAS support for vLLM."""

from typing import Optional, Dict, Any
import torch
import torch.nn.functional as F
from einops import rearrange
import numpy as np

from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


class PASQwen3ForCausalLM(Qwen3ForCausalLM):
    """Qwen3 with Personality Activate Search (PAS) interventions."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pas_interventions: Optional[Dict[str, Any]] = None
        self.pas_alpha: float = 0.0
        self._original_biases = {}
        
        # Cache original biases
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer.self_attn, 'o_proj'):
                self._original_biases[i] = (
                    layer.self_attn.o_proj.bias.clone() 
                    if layer.self_attn.o_proj.bias is not None 
                    else None
                )
    
    def load_pas_interventions(self, intervention_data: Dict[str, Any], alpha: float = None):
        """Load PAS intervention parameters."""
        self.pas_interventions = intervention_data['activate']
        self.pas_alpha = alpha if alpha is not None else intervention_data['best_alpha']
        self._apply_pas_interventions()
    
    def _apply_pas_interventions(self):
        """Apply PAS interventions to model."""
        if self.pas_interventions is None:
            return
        
        num_heads = self.config.num_attention_heads
        head_dim = self.config.hidden_size // num_heads
        
        for head_out_name, list_int_vec in self.pas_interventions.items():
            layer_no = int(head_out_name.split('.')[2])
            
            # Create displacement
            displacement = np.zeros((num_heads, head_dim))
            for head_no, head_vec, std in list_int_vec:
                displacement[head_no] = self.pas_alpha * std * head_vec
            
            device = self.model.layers[layer_no].self_attn.o_proj.weight.device
            displacement = torch.tensor(
                rearrange(displacement, 'h d -> (h d)'),
                device=device,
                dtype=self.model.layers[layer_no].self_attn.o_proj.weight.dtype
            )
            
            # Apply bias
            bias_addition = F.linear(
                displacement,
                self.model.layers[layer_no].self_attn.o_proj.weight
            )
            
            current_bias = self._original_biases.get(layer_no)
            if current_bias is None:
                current_bias = torch.zeros(
                    self.config.hidden_size,
                    device=device,
                    dtype=self.model.layers[layer_no].self_attn.o_proj.weight.dtype
                )
            
            new_bias = current_bias + bias_addition
            self.model.layers[layer_no].self_attn.o_proj.bias = torch.nn.parameter.Parameter(new_bias)
    
    def reset_pas_interventions(self):
        """Reset to original model without interventions."""
        for i, orig_bias in self._original_biases.items():
            if orig_bias is not None:
                self.model.layers[i].self_attn.o_proj.bias = torch.nn.parameter.Parameter(orig_bias.clone())
```

#### Step 4.3: Register Plugin

**File**: `vllm_pas_plugin/__init__.py`

```python
"""vLLM Plugin for PAS-enhanced models."""

def register():
    """Register PAS models with vLLM."""
    from vllm import ModelRegistry
    
    # Use lazy import to avoid CUDA initialization issues
    ModelRegistry.register_model(
        "PASQwen3ForCausalLM",
        "vllm_pas_plugin.pas_qwen3:PASQwen3ForCausalLM"
    )
```

**File**: `vllm_pas_plugin/setup.py`

```python
from setuptools import setup, find_packages

setup(
    name="vllm-pas-plugin",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "vllm>=0.3.0",
        "einops",
        "numpy",
    ],
    entry_points={
        "vllm.plugins": [
            "pas = vllm_pas_plugin:register",
        ],
    },
)
```

---

## 📊 Workflow Summary

### For Single Personality (Recommended)

```bash
# 1. Run PAS training and save interventions
python main.py --modes PAS --model_file /home/chuan/projects/models/qwen3-0.6b

# 2. Bake interventions into model
python bake_pas_model.py \
  --model_file /home/chuan/projects/models/qwen3-0.6b \
  --intervention_file ./interventions/PAS_qwen3-0.6b_sample0.pkl \
  --output_dir ./baked_models/pas_qwen3_personality_A

# 3. Serve with vLLM
vllm serve ./baked_models/pas_qwen3_personality_A
```

### For Multiple Personalities (Advanced)

```bash
# 1. Run PAS training
python main.py --modes PAS --model_file /home/chuan/projects/models/qwen3-0.6b

# 2. Install vLLM plugin
cd vllm_pas_plugin && pip install -e . && cd ..

# 3. Serve with custom endpoint that switches personalities
python serve_pas_vllm.py \
  --model_file /home/chuan/projects/models/qwen3-0.6b \
  --interventions_dir ./interventions/
```

---

## 🚀 Next Steps

1. **Immediate**: Implement Step 1.1 and 1.2 to save intervention parameters
2. **Phase 2**: Create and test the baking script
3. **Phase 3**: Test vLLM serving with baked model
4. **Optional**: Implement Phase 4 for dynamic personality switching

---

## 📁 Expected File Structure After Implementation

```
PAlign-self/
├── interventions/                    # NEW: Intervention parameters
│   ├── PAS_qwen3-0.6b_sample0.pkl
│   ├── PAS_qwen3-0.6b_sample1.pkl
│   └── PAS_qwen3-0.6b_all.pkl
├── baked_models/                     # NEW: Models with baked interventions
│   └── pas_qwen3_personality_A/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer_config.json
│       ├── pas_metadata.pkl
│       └── README.md
├── vllm_pas_plugin/                  # NEW: vLLM plugin (optional)
│   ├── __init__.py
│   ├── pas_qwen3.py
│   └── setup.py
├── bake_pas_model.py                 # NEW: Baking script
├── serve_pas_vllm.py                 # NEW: Serving script (optional)
├── main.py                           # MODIFIED: Save interventions
└── ... (existing files)
```

---

**References**:
- [vLLM Model Registration](https://docs.vllm.ai/en/stable/contributing/model/registration.html)
- PyTorch Model Serialization: https://pytorch.org/tutorials/beginner/saving_loading_models.html

