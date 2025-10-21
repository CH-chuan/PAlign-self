# Adding New Model Architectures to PAS

This guide explains how to add support for new model architectures to the PAS (Personality Activate Search) framework. We use **Qwen3** as a concrete example throughout.

## Overview

PAS works by inserting "hooks" into the attention layers of transformer models. To add a new model architecture, you need to:

1. Create a custom modeling file with PAS hooks
2. Register the model in the PAS loader
3. Update prompt handling and response parsing
4. Test the integration

---

## Step 1: Create Custom Modeling File

### 1.1 Locate the Original Model Implementation

Find the model's implementation in the `transformers` library:

```python
from transformers.models.qwen3 import modeling_qwen3
```

Or check: `site-packages/transformers/models/qwen3/modeling_qwen3.py`

### 1.2 Copy the Attention Layer

Create a new file: `PAlign/modeling_<modelname>.py`

Copy the entire attention class from the original implementation. For Qwen3:

```python
# PAlign/modeling_qwen3.py
import torch
import torch.nn as nn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
    Qwen3Model,
    Qwen3ForCausalLM as BaseQwen3ForCausalLM,
    # ... other imports
)
```

### 1.3 Add PAS Hooks to Attention Layer

In the attention class `__init__` method, add three identity hooks:

```python
class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: Optional[int] = None):
        super().__init__()
        
        # ... existing initialization code ...
        
        # PAS Hooks - ADD THESE THREE LINES
        self.att_out = nn.Identity()    # Attention weights hook
        self.value_out = nn.Identity()  # Value states hook  
        self.head_out = nn.Identity()   # Main intervention hook (CRITICAL)
```

### 1.4 Modify the Forward Pass

In the attention layer's `forward` method, insert the `head_out` hook **before** the output projection:

```python
def forward(self, hidden_states, ...):
    # ... existing attention computation ...
    
    # Reshape back: [batch, seq, num_heads, head_dim] -> [batch, seq, hidden]
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    
    # *** INSERT THIS LINE - CRITICAL FOR PAS ***
    attn_output = self.head_out(attn_output)
    
    # Then apply output projection
    attn_output = self.o_proj(attn_output)
    
    return attn_output, attn_weights, past_key_value
```

**Why this matters**: PAS modifies attention outputs just before they're projected back to the hidden dimension. This is where personality-aligned directions are injected.

### 1.5 Keep All Other Components

Copy all supporting classes without modification:
- The main model class (e.g., `Qwen3Model`)
- The causal LM class (e.g., `Qwen3ForCausalLM`)
- Any decoder layers, embeddings, etc.

**Example structure**:
```python
# PAlign/modeling_qwen3.py

class Qwen3Attention(nn.Module):  # Modified with hooks
    # ... with PAS hooks added
    pass

class Qwen3DecoderLayer(nn.Module):  # Unchanged
    # ... uses Qwen3Attention above
    pass

class Qwen3Model(Qwen3PreTrainedModel):  # Unchanged
    # ... uses Qwen3DecoderLayer
    pass

class Qwen3ForCausalLM(Qwen3PreTrainedModel):  # Unchanged
    # ... uses Qwen3Model
    pass
```

---

## Step 2: Register Model in PAS Loader

Edit `PAlign/pas.py` to add your model architecture:

### 2.1 Add Architecture Detection

In the `PASLM.__init__` method, add a condition for your model:

```python
class PASLM(nn.Module):
    def __init__(self, model_file, batch_size=3):
        # ... existing code ...
        
        # Add this elif block for your model
        elif self.config.architectures[0] == 'Qwen3ForCausalLM':
            from PAlign.modeling_qwen3 import Qwen3ForCausalLM as ModelForCausalLM
            self.model = ModelForCausalLM.from_pretrained(
                model_file,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map='auto'
            )
```

### 2.2 Update Tokenization (if needed)

If your model uses a special chat template, update `prompt_to_tokens`:

```python
def prompt_to_tokens(self, instruction):
    if 'llama-3' in self.model_file.lower() or 'qwen3' in self.model_file.lower():
        # Use apply_chat_template for instruction-tuned models
        messages = [{"role": "user", "content": instruction}]
        tokens = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors='pt'
        )
    else:
        # Standard tokenization for base models
        tokens = self.tokenizer(instruction, return_tensors='pt')
    
    return tokens
```

---

## Step 3: Update Main Script

Edit `main.py` to handle your model's specific formats:

### 3.1 Update Response Parsing

Add a condition in `generateAnswer` to parse your model's output format:

```python
def generateAnswer(tokenizer, model, test_data, template, system_prompt=None, model_file=None):
    # ... existing code ...
    
    # Parse responses based on model type
    if 'llama-2' in model_file.lower():
        answer = [text.split("[/INST]")[-1] for text in output_text]
    elif 'llama-3' in model_file.lower():
        answer = [text.split("assistant\n\n")[-1].split("<|eot_id|>")[0] for text in output_text]
    elif 'qwen3' in model_file.lower():
        # Qwen3 uses <|im_start|> and <|im_end|> markers
        answer = [
            text.split("assistant\n")[-1].split("<|im_end|>")[0]
            if "assistant\n" in text
            else text.split("[/INST]")[-1]
            for text in output_text
        ]
    elif 'mistral' in model_file.lower():
        answer = [text.split("[/INST]")[-1] for text in output_text]
    else:
        answer = output_text
    
    return answer
```

### 3.2 Update Tokenizer Configuration

In the `main` function, configure the tokenizer for your model:

```python
def main():
    # ... existing code ...
    
    if 'llama-3' in model_file.lower() or 'qwen3' in model_file.lower():
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
```

---

## Step 4: Test the Integration

### 4.1 Create a Test Script

Create `test_<modelname>.py` to verify the integration:

```python
#!/usr/bin/env python3
"""Test script for Qwen3 integration with PAS."""

import torch
from PAlign.pas import get_model

def test_qwen3_integration():
    """Test Qwen3 model loading and PAS hooks."""
    
    print("Testing Qwen3 Integration with PAS")
    print("=" * 60)
    
    # 1. Load model
    print("\n[1/6] Loading Qwen3 model...")
    model_path = "Qwen/Qwen3-0.5B"  # or your local path
    model, tokenizer = get_model(model_path)
    print("✓ Model loaded successfully")
    
    # 2. Check model type
    print("\n[2/6] Checking model configuration...")
    assert 'Qwen3' in model.model.__class__.__name__, "Not a Qwen3 model!"
    print(f"✓ Model type: {model.model.__class__.__name__}")
    
    # 3. Verify hooks are installed
    print("\n[3/6] Verifying PAS hooks...")
    layer0 = model.model.model.layers[0]
    assert hasattr(layer0.self_attn, 'head_out'), "Missing head_out hook!"
    assert hasattr(layer0.self_attn, 'att_out'), "Missing att_out hook!"
    assert hasattr(layer0.self_attn, 'value_out'), "Missing value_out hook!"
    print("✓ All PAS hooks installed correctly")
    
    # 4. Test tokenization
    print("\n[4/6] Testing tokenization...")
    test_prompt = "What is your personality?"
    tokens = model.prompt_to_tokens(test_prompt)
    assert tokens is not None, "Tokenization failed!"
    print(f"✓ Tokenization works (tokens shape: {tokens['input_ids'].shape})")
    
    # 5. Test inference
    print("\n[5/6] Testing model inference...")
    with torch.no_grad():
        outputs = model.model.generate(
            tokens['input_ids'].to(model.model.device),
            max_new_tokens=20,
            do_sample=False
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"✓ Inference successful")
    print(f"  Sample output: {response[:100]}...")
    
    # 6. Verify PAS methods exist
    print("\n[6/6] Verifying PAS methods...")
    assert hasattr(model, 'preprocess_activate_dataset'), "Missing PAS method!"
    assert hasattr(model, 'get_activations'), "Missing PAS method!"
    assert hasattr(model, 'set_activate'), "Missing PAS method!"
    print("✓ All PAS methods available")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Qwen3 integration successful!")
    print("=" * 60)

if __name__ == "__main__":
    test_qwen3_integration()
```

### 4.2 Run Tests

```bash
# Quick test
python test_qwen3.py

# Full PAS run (small sample)
python main.py --modes PAS --model_file Qwen/Qwen3-0.5B
```

---

## Architecture-Specific Considerations

### Common Attention Patterns

Different models organize attention differently. Here's where to place hooks:

**Standard Multi-Head Attention** (Llama, Mistral):
```python
# After attention computation, before o_proj
attn_output = attn_output.transpose(1, 2).contiguous()
attn_output = attn_output.reshape(bsz, q_len, hidden_size)
attn_output = self.head_out(attn_output)  # ← Hook here
attn_output = self.o_proj(attn_output)
```

**Grouped-Query Attention** (Llama-3, some Qwen variants):
```python
# Same placement, just different K/V handling
attn_output = repeat_kv(attn_output, self.num_key_value_groups)
attn_output = attn_output.transpose(1, 2).contiguous()
attn_output = attn_output.reshape(bsz, q_len, hidden_size)
attn_output = self.head_out(attn_output)  # ← Hook here
attn_output = self.o_proj(attn_output)
```

**Multi-Query Attention** (Falcon, some variants):
```python
# Similar pattern applies
attn_output = attn_output.view(bsz, q_len, hidden_size)
attn_output = self.head_out(attn_output)  # ← Hook here
attn_output = self.o_proj(attn_output)
```

### Chat Template Formats

| Model | Chat Markers | Example |
|-------|--------------|---------|
| Llama-2 | `[INST]`, `[/INST]` | `[INST] Question [/INST] Answer` |
| Llama-3 | `<\|start_header_id\|>`, `<\|eot_id\|>` | Uses `apply_chat_template` |
| Qwen3 | `<\|im_start\|>`, `<\|im_end\|>` | Uses `apply_chat_template` |
| Mistral | `[INST]`, `[/INST]` | Same as Llama-2 |

**Recommendation**: Use `tokenizer.apply_chat_template()` for instruction-tuned models.

---

## Troubleshooting

### Issue: "PAS not implemented yet for XForCausalLM"

**Cause**: Model not registered in `pas.py`

**Solution**: Add the architecture check in Step 2.1

### Issue: Hooks not found in attention layer

**Cause**: Hooks not added or wrong attention class modified

**Solution**: 
1. Verify you're modifying the correct attention class
2. Check hooks are in `__init__`: `self.head_out = nn.Identity()`
3. Ensure the custom modeling file is being imported

### Issue: Response parsing returns empty strings

**Cause**: Output format doesn't match parsing logic

**Solution**:
1. Print raw `output_text` to see actual format
2. Adjust split logic in Step 3.1
3. Check tokenizer's `eos_token` and special tokens

### Issue: CUDA out of memory

**Cause**: Model too large or batch size too big

**Solution**:
```python
# In main.py, reduce batch size
batch_size = 3 if '70B' in model_file else 5  # Lower this
```

---

## Checklist for New Model Integration

Use this checklist when adding a new model:

- [ ] Created `PAlign/modeling_<modelname>.py`
- [ ] Added three PAS hooks to attention `__init__`
- [ ] Inserted `head_out` hook in forward pass (before `o_proj`)
- [ ] Added architecture check in `PAlign/pas.py`
- [ ] Updated prompt tokenization (if needed)
- [ ] Updated response parsing in `main.py`
- [ ] Configured tokenizer settings
- [ ] Created test script `test_<modelname>.py`
- [ ] Verified hooks are installed (test script passes)
- [ ] Ran small PAS experiment successfully
- [ ] Checked output logs are generated

---

## Example: Adding Gemma Support

Here's a quick example for adding Google's Gemma:

```python
# 1. PAlign/modeling_gemma.py
from transformers.models.gemma.modeling_gemma import *

class GemmaAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        # ... existing init ...
        
        # Add PAS hooks
        self.att_out = nn.Identity()
        self.value_out = nn.Identity()
        self.head_out = nn.Identity()
    
    def forward(self, hidden_states, ...):
        # ... attention computation ...
        
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.head_out(attn_output)  # ← Add this
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights, past_key_value

# 2. PAlign/pas.py
elif self.config.architectures[0] == 'GemmaForCausalLM':
    from PAlign.modeling_gemma import GemmaForCausalLM as ModelForCausalLM

# 3. main.py - response parsing
elif 'gemma' in model_file.lower():
    answer = [text.split("<end_of_turn>")[-1] for text in output_text]
```

---

## Supported Models

| Model | Status | File |
|-------|--------|------|
| Llama-2/3 | ✅ Supported | `modeling_llama.py` |
| Mistral | ✅ Supported | `modeling_mistral.py` |
| Qwen3 | ✅ Supported | `modeling_qwen3.py` |
| Gemma | 🟡 Example above | - |
| Phi-3 | 🟡 Add following this guide | - |

---

## Further Reading

- **PAS Paper**: https://openreview.net/forum?id=0DZEs8NpUH
- **Intervention Storage**: See `INTERVENTION_STORAGE.md`
- **vLLM Deployment**: See `VLLM_DEPLOYMENT.md`

---

**Questions?** The pattern is consistent across models. If you follow these steps, most transformer architectures should work smoothly with PAS!

