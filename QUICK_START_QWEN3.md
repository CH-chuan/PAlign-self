# Quick Start Guide: Qwen3 with PAS

## 🚀 Quick Test (30 seconds)

```bash
cd /home/chuan/projects/PAlign-self
python test_qwen3.py
```

Expected output: "✅ All tests passed!"

## 📋 Usage Examples

### 1. Run PAS with HuggingFace Model
```bash
python main.py --modes PAS --model_file Qwen/Qwen3-0.5B
```

### 2. Run PAS with Local Model
```bash
python main.py --modes PAS --model_file models/qwen3-0.6b
```

### 3. Compare All Methods
```bash
# Baseline
python main.py --modes NO_CHANGE --model_file Qwen/Qwen3-0.5B

# Few-shot
python main.py --modes few-shot --model_file Qwen/Qwen3-0.5B

# PAS (best)
python main.py --modes PAS --model_file Qwen/Qwen3-0.5B
```

## 📁 Files Modified/Created

**Created**:
- ✅ `PAlign/modeling_qwen3.py` - Custom Qwen3 with PAS hooks
- ✅ `test_qwen3.py` - Test script
- ✅ `QWEN3_INTEGRATION.md` - Detailed docs
- ✅ `IMPLEMENTATION_SUMMARY.md` - Technical details

**Modified**:
- ✅ `PAlign/pas.py` - Added Qwen3 support
- ✅ `main.py` - Fixed import + added Qwen3 handling

## 🔧 What Was Changed

### 1. Custom Qwen3 Attention Layer
Added 3 identity hooks for PAS intervention:
```python
self.head_out = nn.Identity()   # Main hook for activation modification
self.att_out = nn.Identity()    # Attention weights hook
self.value_out = nn.Identity()  # Value states hook
```

### 2. Model Loader
Now recognizes and loads Qwen3ForCausalLM:
```python
elif self.config.architectures[0] == 'Qwen3ForCausalLM':
    from PAlign.modeling_qwen3 import Qwen3ForCausalLM as ModelForCausalLM
```

### 3. Chat Template Support
Handles Qwen3's `<|im_start|>` and `<|im_end|>` markers:
```python
if 'llama-3' in model_file.lower() or 'qwen3' in model_file.lower():
    # Use apply_chat_template for both
```

## ⚡ Quick Validation

```python
from PAlign.pas import get_model

# Load model
model, tokenizer = get_model("Qwen/Qwen3-0.5B")

# Check hooks exist
layer0 = model.model.model.layers[0]
assert hasattr(layer0.self_attn, 'head_out'), "Hook missing!"

print("✓ Qwen3 with PAS is ready!")
```

## 📊 Expected Output

After running PAS, check: `./log/PAS_Qwen3-0.5B_OOD.json`

Should contain:
- Personality scores (A, C, E, N, O)
- Mean absolute errors
- Standard deviations
- Per-sample results

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Module not found | Run `pip install -e .` in project root |
| CUDA OOM | Reduce batch_size in line 100 of main.py |
| Chat template error | Model must be instruction-tuned, not base |
| Import error | Fixed: use `from PAlign.pas import get_model` |

## 📚 Documentation

- **Full Guide**: `QWEN3_INTEGRATION.md`
- **Technical Details**: `IMPLEMENTATION_SUMMARY.md`
- **This File**: Quick reference only

## 🎯 Next Steps

1. Run test script to verify installation
2. Try with small Qwen3 model first (0.5B)
3. Run baseline comparison
4. Scale up to larger models as needed

## 💡 Pro Tips

- Use smaller batch size for large models
- First run NO_CHANGE mode to get baseline
- PAS typically gives best results with alpha ∈ {2, 4, 6}
- Check GPU memory before starting

---

**Quick Question?** Run: `python test_qwen3.py`  
**Need Help?** Check: `QWEN3_INTEGRATION.md`


