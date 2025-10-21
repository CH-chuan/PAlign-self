# vLLM Deployment for PAS-Intervened Models

This guide explains how to package PAS interventions into standard HuggingFace model format and serve them directly with vLLM.

## Goal

Transform this:
```
base_model/ + intervention.pkl → custom code to apply interventions
```

Into this:
```
baked_model/ → vllm serve baked_model/
```

**Result**: Standard HuggingFace model with interventions permanently baked into weights, ready for direct vLLM serving.

---

## Overview

### What Is "Baking"?

Baking means **permanently applying PAS interventions to model weights**. Instead of:
- Loading base model
- Loading intervention file
- Applying interventions at runtime (slower, requires custom code)

You get:
- Single model directory with interventions already applied
- Standard HuggingFace format (config.json, model weights, tokenizer)
- Direct serving with `vllm serve model-dir` (no custom code)

### Why Bake Models?

✅ **Production-ready**: Standard deployment, no custom code  
✅ **Performance**: No runtime overhead for applying interventions  
✅ **Portability**: Works with any vLLM-compatible infrastructure  
✅ **Multi-agent**: Create multiple personality variants easily  
✅ **Caching**: vLLM's KV cache and optimizations work seamlessly  

---

## Quick Start (3 Steps)

### Step 1: Run PAS Training

Generate intervention parameters:

```bash
cd /home/chuan/projects/PAlign-self

python main.py \
  --modes PAS \
  --model_file /home/chuan/projects/models/qwen3-0.6b
```

**Output**: Intervention files in `./interventions/`

### Step 2: Bake Model

Package interventions into model weights:

```bash
python bake_pas_model.py \
  --model_file /home/chuan/projects/models/qwen3-0.6b \
  --intervention_file ./interventions/PAS_qwen3-0.6b_sample0.pkl \
  --output_dir ./baked_models/agent_personality_0
```

**Output**: Complete HuggingFace model in `./baked_models/agent_personality_0/`

### Step 3: Serve with vLLM

Deploy directly:

```bash
pip install vllm

vllm serve ./baked_models/agent_personality_0 \
  --port 8000 \
  --host 0.0.0.0
```

**Done!** Your personality-aligned model is now serving.

---

## The Baking Process

### What `bake_pas_model.py` Does

The script performs these steps:

1. **Load base model**: Load the original, unmodified model
2. **Load interventions**: Read the `.pkl` file with intervention parameters
3. **Apply to biases**: Modify `o_proj.bias` in attention layers to bake interventions
4. **Save model**: Write complete HuggingFace model to disk
5. **Generate docs**: Create README with usage instructions

### Technical Details

**How interventions are baked:**

```python
# For each intervened attention head:
displacement = alpha * std * direction_vector

# Apply via output projection bias:
bias_addition = o_proj.weight @ displacement
new_bias = original_bias + bias_addition

# Save modified bias
layer.self_attn.o_proj.bias = new_bias
```

**Why this works:**
- PAS normally applies: `output' = o_proj(head_output + displacement)`
- Math: `output' = o_proj.weight @ (head_output + displacement) + o_proj.bias`
- Expand: `output' = (o_proj.weight @ head_output) + (o_proj.weight @ displacement) + o_proj.bias`
- Equivalent to: `output' = o_proj.weight @ head_output + new_bias`
- Where: `new_bias = o_proj.bias + (o_proj.weight @ displacement)`

This means we can pre-compute and add to the bias, making the intervention permanent!

---

## Using `bake_pas_model.py`

### Basic Usage

```bash
python bake_pas_model.py \
  --model_file <path_to_base_model> \
  --intervention_file <path_to_intervention_pkl> \
  --output_dir <output_directory>
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--model_file` | Yes | Path to base model (local or HuggingFace) |
| `--intervention_file` | Yes | Path to `.pkl` intervention file |
| `--output_dir` | Yes | Where to save baked model |
| `--alpha` | No | Override alpha value (uses best_alpha if not provided) |
| `--sample_index` | No | Which sample to use from `_all.pkl` file (default: 0) |

### Examples

**Example 1: Bake with individual intervention file**

```bash
python bake_pas_model.py \
  --model_file /home/chuan/projects/models/qwen3-0.6b \
  --intervention_file ./interventions/PAS_qwen3-0.6b_sample0.pkl \
  --output_dir ./baked_models/personality_analytical
```

**Example 2: Bake from aggregated file (specific sample)**

```bash
python bake_pas_model.py \
  --model_file /home/chuan/projects/models/qwen3-0.6b \
  --intervention_file ./interventions/PAS_qwen3-0.6b_all.pkl \
  --sample_index 5 \
  --output_dir ./baked_models/personality_sample5
```

**Example 3: Override alpha value**

```bash
python bake_pas_model.py \
  --model_file /home/chuan/projects/models/qwen3-0.6b \
  --intervention_file ./interventions/PAS_qwen3-0.6b_sample0.pkl \
  --alpha 6 \
  --output_dir ./baked_models/personality_alpha6
```

### Expected Output

```
======================================================================
                    PAS Model Baking Tool
======================================================================

[1/4] Loading base model...
      Model: /home/chuan/projects/models/qwen3-0.6b
      ✓ Model loaded

[2/4] Loading intervention parameters...
      File: ./interventions/PAS_qwen3-0.6b_sample0.pkl
      ✓ Loaded intervention
      Alpha: 4
      Sample ID: sample_0001

[3/4] Applying PAS interventions...

Applying PAS interventions with alpha=4
  Num attention heads: 16
  Head dimension: 64
  ✓ Applied intervention to layer 5
  ✓ Applied intervention to layer 8
  ✓ Applied intervention to layer 12
  ✓ Applied intervention to layer 18
  ✓ Applied intervention to layer 21
  ✓ Applied intervention to layer 25

✅ Applied interventions to 6 layers

[4/4] Saving baked model...

Saving baked model to ./baked_models/personality_analytical
✅ Model saved successfully!
   📁 Model weights: ./baked_models/personality_analytical
   📄 Metadata: ./baked_models/personality_analytical/pas_metadata.pkl
   📖 README: ./baked_models/personality_analytical/README.md

======================================================================
✅ SUCCESS! Model is ready for vLLM serving
======================================================================

🚀 To serve with vLLM:
   vllm serve ./baked_models/personality_analytical

🐍 Or in Python:
   from vllm import LLM
   llm = LLM(model='./baked_models/personality_analytical')
```

---

## Baked Model Structure

After baking, you get a complete HuggingFace model:

```
baked_models/personality_analytical/
├── config.json                 # Model configuration
├── model.safetensors          # Model weights (with baked interventions)
│   or pytorch_model.bin       # (alternative weight format)
├── tokenizer_config.json      # Tokenizer configuration
├── tokenizer.json            # Tokenizer vocabulary
├── special_tokens_map.json   # Special tokens
├── pas_metadata.pkl          # PAS intervention metadata
└── README.md                 # Usage instructions
```

**What's inside `pas_metadata.pkl`:**

```python
{
    'model_file': str,              # Original base model path
    'intervention_file': str,       # Intervention file used
    'sample_id': str,              # Sample identifier
    'sample_index': int,           # Sample number
    'best_alpha': float,           # Alpha value used
    'personality_scores': dict,    # OCEAN personality profile
    'system_prompt': str,          # Context prompt from training
}
```

---

## Serving with vLLM

### Method 1: Command-Line Server (OpenAI-Compatible)

```bash
# Basic serving
vllm serve ./baked_models/personality_analytical \
  --port 8000

# With custom settings
vllm serve ./baked_models/personality_analytical \
  --port 8000 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096
```

**Use with OpenAI client:**

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # vLLM doesn't require real key
)

response = client.chat.completions.create(
    model="./baked_models/personality_analytical",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's your approach to problem-solving?"}
    ],
    temperature=0.7,
    max_tokens=256
)

print(response.choices[0].message.content)
```

### Method 2: Python API (Direct)

```python
from vllm import LLM, SamplingParams

# Load model
llm = LLM(model="./baked_models/personality_analytical")

# Configure sampling
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256
)

# Generate
prompts = [
    "Tell me about your personality.",
    "How do you handle stressful situations?",
    "What are your strengths and weaknesses?"
]

outputs = llm.generate(prompts, sampling_params)

for prompt, output in zip(prompts, outputs):
    print(f"Q: {prompt}")
    print(f"A: {output.outputs[0].text}\n")
```

### Method 3: Multi-GPU Serving

For larger models:

```bash
# Tensor parallelism across 2 GPUs
vllm serve ./baked_models/personality_analytical \
  --tensor-parallel-size 2 \
  --port 8000

# Or in Python
llm = LLM(
    model="./baked_models/personality_analytical",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.9
)
```

---

## Multi-Agent Setup

Create multiple personality agents for different tasks:

### Creating Multiple Agents

```bash
# Create 5 different personality agents
for i in 0 1 2 3 4; do
  python bake_pas_model.py \
    --model_file /home/chuan/projects/models/qwen3-0.6b \
    --intervention_file ./interventions/PAS_qwen3-0.6b_sample${i}.pkl \
    --output_dir ./baked_models/agent_${i}
done
```

### Agent Router

```python
from vllm import LLM, SamplingParams

class PersonalityAgentRouter:
    """Route queries to different personality-aligned agents."""
    
    def __init__(self, agent_paths):
        """
        Args:
            agent_paths: Dict mapping agent names to model paths
        """
        self.agents = {
            name: LLM(model=path)
            for name, path in agent_paths.items()
        }
        self.sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=256
        )
    
    def query(self, agent_name, prompt):
        """Query a specific agent."""
        llm = self.agents[agent_name]
        outputs = llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text
    
    def query_all(self, prompt):
        """Get responses from all agents."""
        return {
            name: self.query(name, prompt)
            for name in self.agents.keys()
        }
    
    def best_agent_for_task(self, task_type):
        """Suggest best agent for task type (customize as needed)."""
        task_map = {
            'analytical': 'agent_0',
            'creative': 'agent_1',
            'empathetic': 'agent_2',
            'leadership': 'agent_3',
            'technical': 'agent_4',
        }
        return task_map.get(task_type, 'agent_0')

# Usage
router = PersonalityAgentRouter({
    'agent_0': './baked_models/agent_0',
    'agent_1': './baked_models/agent_1',
    'agent_2': './baked_models/agent_2',
    'agent_3': './baked_models/agent_3',
    'agent_4': './baked_models/agent_4',
})

# Route to specific agent
response = router.query('agent_1', "Brainstorm creative solutions for...")

# Get all perspectives
all_responses = router.query_all("How should we approach this problem?")
for agent, response in all_responses.items():
    print(f"{agent}: {response}\n")

# Automatic routing
best_agent = router.best_agent_for_task('creative')
response = router.query(best_agent, "Generate ideas for...")
```

### Serving Multiple Agents Simultaneously

```bash
# Serve different agents on different ports
vllm serve ./baked_models/agent_0 --port 8000 &
vllm serve ./baked_models/agent_1 --port 8001 &
vllm serve ./baked_models/agent_2 --port 8002 &

# Then use OpenAI client to connect to different ports
```

---

## Performance Optimization

### Quantization

Reduce memory usage with quantization:

```bash
# AWQ quantization (4-bit)
vllm serve ./baked_models/personality_analytical \
  --quantization awq \
  --port 8000

# In Python
llm = LLM(
    model="./baked_models/personality_analytical",
    quantization="awq"
)
```

**Note**: Quantize the base model before baking interventions for best results.

### GPU Memory Utilization

```bash
# Use 90% of GPU memory
vllm serve ./baked_models/personality_analytical \
  --gpu-memory-utilization 0.9

# Limit to specific GPU
CUDA_VISIBLE_DEVICES=0 vllm serve ./baked_models/personality_analytical
```

### Batch Processing

Process multiple prompts efficiently:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="./baked_models/personality_analytical")
sampling_params = SamplingParams(temperature=0.7)

# Batch of 100 prompts
prompts = [f"Question {i}: ..." for i in range(100)]
outputs = llm.generate(prompts, sampling_params)

# vLLM automatically batches and optimizes
```

### Continuous Batching

vLLM automatically uses continuous batching (no configuration needed):
- New requests join ongoing batches
- Maximizes GPU utilization
- Reduces latency for concurrent requests

---

## Verification and Testing

### Test Baked Model

```python
from vllm import LLM, SamplingParams

def test_baked_model(model_path):
    """Test a baked model."""
    print(f"Testing: {model_path}\n")
    
    # Load
    llm = LLM(model=model_path)
    
    # Test prompts
    prompts = [
        "Describe your personality in one sentence.",
        "How do you approach decision-making?",
        "What motivates you the most?",
    ]
    
    # Generate
    outputs = llm.generate(prompts, SamplingParams(temperature=0.7, max_tokens=100))
    
    # Display
    for prompt, output in zip(prompts, outputs):
        print(f"Q: {prompt}")
        print(f"A: {output.outputs[0].text}\n")
        print("-" * 70)
    
    print("\n✅ Test complete!")

# Run test
test_baked_model("./baked_models/personality_analytical")
```

### Compare Multiple Agents

```python
def compare_agents(agent_paths, prompt):
    """Compare responses from multiple agents."""
    from vllm import LLM, SamplingParams
    
    print(f"Prompt: {prompt}\n")
    print("=" * 70)
    
    for name, path in agent_paths.items():
        llm = LLM(model=path)
        output = llm.generate([prompt], SamplingParams(temperature=0.7))
        
        print(f"\n{name}:")
        print(output[0].outputs[0].text)
        print("-" * 70)

# Compare
compare_agents({
    'Agent 0': './baked_models/agent_0',
    'Agent 1': './baked_models/agent_1',
    'Agent 2': './baked_models/agent_2',
}, "What's your approach to teamwork?")
```

---

## Troubleshooting

### Issue: "No module named 'vllm'"

**Solution:**
```bash
pip install vllm
```

### Issue: "Cannot load model"

**Causes:**
- Model path incorrect
- Model files missing/corrupted

**Solution:**
```bash
# Verify baked model structure
ls -lh ./baked_models/personality_analytical/

# Should see: config.json, model files, tokenizer files
```

### Issue: "CUDA out of memory"

**Solutions:**

1. Reduce GPU memory utilization:
```bash
vllm serve ./baked_models/agent_0 --gpu-memory-utilization 0.7
```

2. Use quantization:
```bash
vllm serve ./baked_models/agent_0 --quantization awq
```

3. Reduce max model length:
```bash
vllm serve ./baked_models/agent_0 --max-model-len 2048
```

### Issue: Model responses don't reflect personality

**Possible causes:**
1. Wrong intervention file used during baking
2. Alpha value too low (no intervention effect)
3. Base model not instruction-tuned

**Debugging:**
```python
# Check metadata
import pickle
with open('./baked_models/agent_0/pas_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
    print(f"Alpha: {metadata['best_alpha']}")  # Should be > 0
    print(f"Personality: {metadata['personality_scores']}")
```

### Issue: Slow inference

**Solutions:**

1. Enable CUDA graphs (for fixed-size batches):
```python
llm = LLM(model="...", enforce_eager=False)  # Default uses CUDA graphs
```

2. Use Flash Attention:
```bash
pip install flash-attn
# vLLM automatically uses it if available
```

3. Increase batch size for throughput:
```python
outputs = llm.generate(large_batch_of_prompts, sampling_params)
```

---

## Best Practices

### 1. Model Naming

Use descriptive names:
```bash
# Good
./baked_models/qwen3_0.6b_creative_alpha4/
./baked_models/qwen3_0.6b_analytical_sample0/

# Less clear
./baked_models/model1/
./baked_models/test/
```

### 2. Documentation

Keep track of what each agent represents:
```python
# Create manifest.json
import json

manifest = {
    'agent_0': {
        'path': './baked_models/agent_0',
        'personality': {'O': 5, 'C': 4, 'E': 3, 'A': 4, 'N': 2},
        'best_for': ['creative tasks', 'brainstorming'],
        'alpha': 4,
    },
    # ... more agents
}

with open('./baked_models/manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
```

### 3. Testing Before Deployment

Always test baked models:
```bash
# Quick test
python -c "
from vllm import LLM
llm = LLM(model='./baked_models/agent_0')
output = llm.generate(['Hello!'])
print(output[0].outputs[0].text)
"
```

### 4. Version Control

Track which base model and intervention files were used:
```bash
# Include in path or metadata
./baked_models/qwen3-0.6b_v1.0_sample0/
```

---

## Production Deployment

### Docker Container

```dockerfile
FROM vllm/vllm-openai:latest

# Copy baked model
COPY ./baked_models/agent_0 /models/agent_0

# Expose port
EXPOSE 8000

# Run vLLM server
CMD ["vllm", "serve", "/models/agent_0", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t personality-agent .
docker run -p 8000:8000 --gpus all personality-agent
```

### Kubernetes Deployment

```yaml
apiVersion: v1
kind: Service
metadata:
  name: personality-agent-service
spec:
  selector:
    app: personality-agent
  ports:
    - port: 8000
      targetPort: 8000
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: personality-agent
spec:
  replicas: 2
  selector:
    matchLabels:
      app: personality-agent
  template:
    metadata:
      labels:
        app: personality-agent
    spec:
      containers:
      - name: vllm
        image: personality-agent:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
```

---

## Summary

**Workflow:**
1. ✅ Train PAS → Get intervention files
2. ✅ Bake interventions → Get HuggingFace model
3. ✅ Serve with vLLM → Standard deployment

**Key Benefits:**
- No custom code needed in production
- Standard vLLM infrastructure
- Multiple personality agents easily
- Production-ready performance

**Next Steps:**
- Create your first baked model
- Test with vLLM serving
- Deploy to your agentic workflow

---

## Further Reading

- **Intervention Storage**: See `INTERVENTION_STORAGE.md`
- **Adding New Models**: See `ADDING_NEW_MODELS.md`
- **vLLM Documentation**: https://docs.vllm.ai/
- **PAS Paper**: https://openreview.net/forum?id=0DZEs8NpUH

---

**Ready to deploy?** The baking process makes PAS production-ready with standard tools and infrastructure!

