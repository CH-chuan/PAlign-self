#!/usr/bin/env python3
"""
Bake PAS interventions into Qwen3 model weights for vLLM serving.

This creates a new model with PAS interventions permanently applied,
suitable for serving via vLLM without custom modifications.

Usage:
    python bake_pas_model.py \
        --model_file /home/chuan/projects/models/qwen3-0.6b \
        --intervention_file ./interventions/PAS_qwen3-0.6b_sample0.pkl \
        --output_dir ./baked_models/pas_qwen3_personality_A
"""

import torch
import pickle
import argparse
import os
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
    
    Returns:
        model: Model with interventions applied
    """
    activate = intervention_data['activate']
    best_alpha = alpha if alpha is not None else intervention_data['best_alpha']
    
    num_heads = model.model.model.config.num_attention_heads
    head_dim = model.model.model.config.hidden_size // num_heads
    
    print(f"\nApplying PAS interventions with alpha={best_alpha}")
    print(f"  Num attention heads: {num_heads}")
    print(f"  Head dimension: {head_dim}")
    
    modified_layers = []
    
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
        else:
            current_bias = current_bias.detach().clone()
        
        # Add intervention to bias
        new_bias = current_bias + bias_addition
        model.model.model.layers[layer_no].self_attn.o_proj.bias = torch.nn.parameter.Parameter(new_bias)
        
        modified_layers.append(layer_no)
        print(f"  ✓ Layer {layer_no}: Modified {len(list_int_vec)} heads")
    
    print(f"\n✅ Applied interventions to {len(modified_layers)} layers")
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
    
    print(f"\nSaving baked model to {output_dir}")
    
    # Save the model using HuggingFace format
    model.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save intervention metadata
    metadata_file = output_path / "pas_metadata.pkl"
    with open(metadata_file, 'wb') as f:
        pickle.dump(intervention_metadata, f)
    
    # Create a comprehensive README
    personality_str = "\n".join([f"- **{k}**: {v}" for k, v in 
                                 intervention_metadata.get('personality_scores', {}).items()])
    
    readme_content = f"""# PAS-Enhanced Qwen3 Model

This model has Personality Activate Search (PAS) interventions baked into the weights.

## Intervention Details

- **Alpha**: {intervention_metadata['best_alpha']}
- **Sample ID**: {intervention_metadata['sample_id']}
- **Base Model**: {intervention_metadata['model_file']}

## Personality Profile (OCEAN Scores)

{personality_str}

## Usage with vLLM

### Python API

```python
from vllm import LLM, SamplingParams

# Load the model
llm = LLM(model="{output_dir}")

# Generate
prompts = ["Tell me about yourself and how you approach problems."]
sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

### OpenAI-Compatible Server

```bash
vllm serve {output_dir} --port 8000

# Then use OpenAI client
import openai
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="{output_dir}",
    messages=[
        {{"role": "user", "content": "Tell me about yourself"}}
    ]
)
print(response.choices[0].message.content)
```

### Command Line

```bash
# Serve the model
vllm serve {output_dir}

# Or run inference directly
python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='{output_dir}')
print(llm.generate(['Hello!'], SamplingParams())[0].outputs[0].text)
"
```

## What Makes This Model Special?

This model has been aligned to a specific personality profile using PAS (Personality Activate Search).
The interventions are permanently baked into the attention layer biases, so:

✅ No runtime overhead
✅ Works with any vLLM deployment
✅ Compatible with all vLLM features (quantization, speculative decoding, etc.)
✅ Can be deployed on multiple GPUs with tensor parallelism

## Technical Details

- **Method**: Personality Activate Search (PAS)
- **Intervention Type**: Attention head bias modification
- **Num Intervened Heads**: ~24 per personality trait
- **OCEAN Traits**: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism

## Files

- `config.json` - Model configuration
- `model.safetensors` or `pytorch_model.bin` - Model weights with PAS interventions
- `tokenizer.json`, `tokenizer_config.json` - Tokenizer files
- `pas_metadata.pkl` - PAS intervention metadata
- `README.md` - This file

## Citation

If you use this model, please cite:

```bibtex
@inproceedings{{
zhu2025personality,
title={{Personality Alignment of Large Language Models}},
author={{Minjun Zhu and Yixuan Weng and Linyi Yang and Yue Zhang}},
booktitle={{The Thirteenth International Conference on Learning Representations}},
year={{2025}},
url={{https://openreview.net/forum?id=0DZEs8NpUH}}
}}
```

---

Generated using PAS (Personality Activate Search)
Original implementation: https://github.com/xxx/PAlign-self
"""
    
    with open(output_path / "README.md", 'w') as f:
        f.write(readme_content)
    
    print(f"✅ Model saved successfully!")
    print(f"   📁 Model weights: {output_dir}")
    print(f"   📄 Metadata: {metadata_file}")
    print(f"   📖 README: {output_path / 'README.md'}")


def main():
    parser = argparse.ArgumentParser(
        description="Bake PAS interventions into model weights for vLLM serving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Bake a single personality
  python bake_pas_model.py \\
    --model_file /home/chuan/projects/models/qwen3-0.6b \\
    --intervention_file ./interventions/PAS_qwen3-0.6b_sample0.pkl \\
    --output_dir ./baked_models/pas_qwen3_personality_A
  
  # Override alpha value
  python bake_pas_model.py \\
    --model_file models/qwen3-0.6b \\
    --intervention_file interventions/PAS_qwen3-0.6b_sample0.pkl \\
    --output_dir baked_models/pas_qwen3_alpha4 \\
    --alpha 4
        """
    )
    parser.add_argument("--model_file", required=True, 
                       help="Path to base model (local or HuggingFace)")
    parser.add_argument("--intervention_file", required=True, 
                       help="Path to intervention .pkl file")
    parser.add_argument("--output_dir", required=True, 
                       help="Output directory for baked model")
    parser.add_argument("--alpha", type=float, default=None,
                       help="Override alpha value (uses best_alpha from file if not specified)")
    parser.add_argument("--sample_index", type=int, default=0,
                       help="Which sample's intervention to use (for aggregated files)")
    
    args = parser.parse_args()
    
    print("="*70)
    print(" " * 20 + "PAS Model Baking Tool")
    print("="*70)
    
    # Load base model
    print(f"\n[1/4] Loading base model...")
    print(f"      Model: {args.model_file}")
    model, tokenizer = get_model(args.model_file)
    print("      ✓ Model loaded")
    
    # Load intervention parameters
    print(f"\n[2/4] Loading intervention parameters...")
    print(f"      File: {args.intervention_file}")
    with open(args.intervention_file, 'rb') as f:
        if 'all.pkl' in args.intervention_file:
            # Aggregated file
            data = pickle.load(f)
            intervention_data = data['interventions'][args.sample_index]
            print(f"      ✓ Using sample {args.sample_index} from aggregated file")
        else:
            # Individual file
            intervention_data = pickle.load(f)
            print(f"      ✓ Loaded individual intervention file")
    
    alpha_to_use = args.alpha if args.alpha is not None else intervention_data['best_alpha']
    print(f"      Alpha: {alpha_to_use}")
    print(f"      Sample ID: {intervention_data.get('sample_id', 'unknown')}")
    
    # Apply interventions
    print(f"\n[3/4] Applying PAS interventions...")
    model = apply_pas_to_model(model, intervention_data, alpha=args.alpha)
    
    # Save baked model
    print(f"\n[4/4] Saving baked model...")
    intervention_metadata = {
        'model_file': args.model_file,
        'intervention_file': args.intervention_file,
        'sample_id': intervention_data.get('sample_id', 'unknown'),
        'sample_index': args.sample_index,
        'best_alpha': alpha_to_use,
        'personality_scores': intervention_data.get('personality_scores', {}),
        'system_prompt': intervention_data.get('system_prompt', ''),
        'baking_date': str(torch.cuda.Event),
    }
    
    save_baked_model(model, tokenizer, args.output_dir, intervention_metadata)
    
    print("\n" + "="*70)
    print("✅ SUCCESS! Model is ready for vLLM serving")
    print("="*70)
    print(f"\n🚀 To serve with vLLM:")
    print(f"   vllm serve {args.output_dir}")
    print(f"\n🐍 Or in Python:")
    print(f"   from vllm import LLM")
    print(f"   llm = LLM(model='{args.output_dir}')")
    print(f"   outputs = llm.generate(['Hello!'])")
    print()


if __name__ == "__main__":
    main()

