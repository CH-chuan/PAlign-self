#!/usr/bin/env python3
"""
Test script for Qwen3 model integration with PAS (Personality Activate Search)
"""

import torch
from PAlign.pas import get_model
from transformers import AutoConfig

def test_qwen3_model_loading():
    """
    Test if Qwen3 model can be loaded with PAS modifications.
    """
    print("="*60)
    print("Testing Qwen3 Model Integration with PAS")
    print("="*60)
    
    # Test with a small Qwen3 model (you can change this to your local path)
    model_path = "Qwen/Qwen3-0.5B"  # Change to "models/qwen3-0.6b" if you have it locally
    
    try:
        print(f"\n1. Checking model configuration...")
        config = AutoConfig.from_pretrained(model_path)
        print(f"   Model architecture: {config.architectures[0]}")
        print(f"   Number of layers: {config.num_hidden_layers}")
        print(f"   Number of attention heads: {config.num_attention_heads}")
        print(f"   Hidden size: {config.hidden_size}")
        
        print(f"\n2. Loading model with PAS wrapper...")
        model, tokenizer = get_model(model_path)
        print(f"   ✓ Model loaded successfully!")
        
        print(f"\n3. Verifying PAS hooks...")
        # Check if the model has the required hooks
        first_layer = model.model.model.layers[0] if hasattr(model.model, 'model') else model.model.layers[0]
        if hasattr(first_layer.self_attn, 'head_out'):
            print(f"   ✓ head_out hook found in attention layer")
        else:
            print(f"   ✗ head_out hook NOT found - PAS modifications may not be applied")
            return False
        
        if hasattr(first_layer.self_attn, 'att_out'):
            print(f"   ✓ att_out hook found in attention layer")
        else:
            print(f"   ✗ att_out hook NOT found")
        
        print(f"\n4. Testing tokenizer...")
        test_prompt = "Hello, how are you?"
        tokens = tokenizer(test_prompt, return_tensors="pt")
        print(f"   ✓ Tokenizer working - Input shape: {tokens.input_ids.shape}")
        
        print(f"\n5. Testing model inference...")
        with torch.no_grad():
            outputs = model.model(tokens.input_ids.to(model.device))
        print(f"   ✓ Model inference working - Output shape: {outputs.logits.shape}")
        
        print(f"\n6. Testing PAS activation methods...")
        if hasattr(model, 'reset_all'):
            print(f"   ✓ reset_all() method found")
        if hasattr(model, 'preprocess_activate_dataset'):
            print(f"   ✓ preprocess_activate_dataset() method found")
        if hasattr(model, 'get_activations'):
            print(f"   ✓ get_activations() method found")
        if hasattr(model, 'set_activate'):
            print(f"   ✓ set_activate() method found")
        
        print("\n" + "="*60)
        print("✅ All tests passed! Qwen3 integration successful!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_qwen3_model_loading()
    exit(0 if success else 1)


