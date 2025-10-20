# Qwen3 Integration for PAS (Personality Activate Search)

## Overview

This document describes the Qwen3 model integration with the PAS (Personality Activate Search) framework for personality alignment of large language models.

## Changes Made

### 1. Custom Qwen3 Modeling File (`PAlign/modeling_qwen3.py`)

Created a modified version of the Qwen3 model with PAS hooks:

- **Added identity hooks** in `Qwen3Attention.__init__()`:
  ```python
  self.att_out = nn.Identity()
  self.value_out = nn.Identity()
  self.head_out = nn.Identity()
  ```

- **Modified forward pass** in `Qwen3Attention.forward()`:
  ```python
  attn_output = self.head_out(attn_output)  # Apply hook before o_proj
  attn_output = self.o_proj(attn_output)
  ```

These hooks allow the PAS system to intercept and modify attention outputs for personality alignment.

### 2. Updated PAS Loader (`PAlign/pas.py`)

Added Qwen3 support to the model loader:

```python
elif self.config.architectures[0] == 'Qwen3ForCausalLM':
    from PAlign.modeling_qwen3 import Qwen3ForCausalLM as ModelForCausalLM
```

Updated tokenization to support Qwen3's chat template format.

### 3. Updated Main Script (`main.py`)

**Fixed import issue:**
```python
from PAlign.pas import get_model  # Changed from PAlign.llama_pas
```

**Added Qwen3 prompt handling:**
- Updated `prompt_to_tokens()` to handle Qwen3 chat templates
- Modified response parsing in `generateAnswer()` to handle Qwen3 output format
- Updated tokenizer configuration for Qwen3 models

## Usage

### Basic Usage

```bash
python main.py --modes PAS --model_file Qwen/Qwen3-0.5B
```

### With Local Model

If you have a local Qwen3 model:

```bash
python main.py --modes PAS --model_file models/qwen3-0.6b
```

### Available Modes

- `PAS`: Personality Activate Search (recommended)
- `NO_CHANGE`: Baseline without intervention
- `few-shot`: Few-shot learning baseline
- `personality_prompt`: Personality prompt baseline

## Testing

Run the test script to verify the integration:

```bash
python test_qwen3.py
```

This will:
1. Load the Qwen3 model with PAS modifications
2. Verify all hooks are properly installed
3. Test tokenization and inference
4. Confirm all PAS methods are available

## Architecture Comparison

### Llama vs Qwen3 Key Differences

| Feature | Llama | Qwen3 |
|---------|-------|-------|
| Chat Template | `<\|end_header_id\|>` | `<\|im_start\|>`, `<\|im_end\|>` |
| Attention | Standard multi-head | Multi-head with Q/K normalization |
| Special Features | - | Sliding window attention |

### PAS Integration Points

Both models use the same PAS intervention strategy:

1. **Activation Extraction**: Extract head-wise activations from personality Q&A
2. **Probe Training**: Train logistic regression on attention heads
3. **Direction Calculation**: Compute personality-relevant directions
4. **Bias Injection**: Modify `o_proj` layer bias in attention mechanism

## Model Architecture Details

### Qwen3 Attention Layer

```
Input → Q/K/V Projections → Q/K Normalization → RoPE → 
Attention Computation → [head_out hook] → O Projection → Output
```

The `head_out` hook is placed after attention computation but before the output projection, allowing PAS to inject personality-aligned bias.

## Expected Results

When running PAS with Qwen3, you should see:

- Personality trait alignment metrics (OCEAN: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism)
- Mean absolute error reduction compared to baseline
- Alpha (intervention strength) selection between {0, 1, 2, 4, 6, 8}

Results are saved to: `./log/PAS_{model_name}_OOD.json`

## Troubleshooting

### Issue: Model not loading
- **Solution**: Ensure the model path is correct and the model is compatible with transformers library

### Issue: CUDA out of memory
- **Solution**: Reduce batch size in `main.py` line 100:
  ```python
  batch_size = 3 if '70B' in model_file else 10  # Reduce to smaller value
  ```

### Issue: Tokenizer errors
- **Solution**: Verify the tokenizer supports chat templates:
  ```python
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = 'left'
  ```

### Issue: Chat template format errors
- **Solution**: Check the model's specific chat template format and adjust the response parsing in `generateAnswer()` function

## Dependencies

Required packages (already in `setup.py`):
- `torch`
- `transformers >= 4.40.0` (for Qwen3 support)
- `scikit-learn`
- `numpy`
- `pandas`
- `einops`
- `baukit`

## Future Improvements

Potential enhancements for Qwen3 integration:

1. **Optimize for Qwen3-specific features**:
   - Leverage sliding window attention for longer contexts
   - Utilize Q/K normalization characteristics

2. **Support for Qwen3 variants**:
   - Qwen3-MoE (Mixture of Experts)
   - Qwen3-VL (Vision-Language)

3. **Performance optimization**:
   - Quantization support (4-bit, 8-bit)
   - Flash Attention 2 integration

## References

- [Qwen3 Technical Report](https://qwenlm.github.io/)
- [PAS Paper (ICLR 2025)](https://openreview.net/forum?id=0DZEs8NpUH)
- [PAPI Dataset](https://huggingface.co/datasets/WestlakeNLP/PAPI-300K)

## Contact

For issues specific to Qwen3 integration, please check:
1. The test script output (`test_qwen3.py`)
2. Linter errors in modified files
3. Model compatibility with transformers version

---

*Last Updated: October 19, 2025*


