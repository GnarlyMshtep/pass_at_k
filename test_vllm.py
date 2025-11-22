#!/usr/bin/env python3
"""
Simple script to test vLLM installation and GPU functionality
"""




def test_hf_generate():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("\n" + "=" * 60)
    print("Testing standard HuggingFace generate")
    print("=" * 60)
    
    model_path = "/data/stalaei/passk/models/Qwen2.5-3B-Instruct"
    print(f"Loading model from: {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )
        print("✓ HF Model loaded successfully!")
        
        inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=20)
        print(f"Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
        print("✓ HF generate test passed!")
    except Exception as e:
        print(f"❌ HF generate test failed: {e}")

def test_vllm():
    from vllm import LLM, SamplingParams
    print("=" * 60)
    print("Testing vLLM Installation")
    print("=" * 60)

    
    print("\n" + "=" * 60)
    print("Testing vLLM with Qwen2.5-3B-Instruct")
    print("=" * 60)
    
    # Use local Qwen2.5-3B-Instruct model
    model_path = "/data/stalaei/passk/models/Qwen2.5-3B-Instruct"
    
    print(f"\nLoading model from: {model_path}")
    print("This may take a moment...")
    
    # Create LLM instance
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=2048,  # Reasonable context length
        gpu_memory_utilization=0.5  # Use 50% GPU memory
    )
    
    print("✓ Model loaded successfully!")
    
    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=50
    )
    
    # Test prompts (formatted for instruction model)
    prompts = [
        "What is the capital of France?",
        "Write a short poem about AI.",
        "Explain what vLLM is in one sentence.",
    ]
    
    print("\n" + "=" * 60)
    print("Generating text...")
    print("=" * 60)
    
    # Generate
    outputs = llm.generate(prompts, sampling_params)
    
    # Print results
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\n--- Prompt {i+1} ---")
        print(f"Input: {prompt}")
        print(f"Output: {generated_text}")
    
    print("\n" + "=" * 60)
    print("✓ vLLM test completed successfully!")
    print("=" * 60)

def test_torch():
    import torch
    print("=" * 60)
    print("Testing PyTorch")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    print("\n" + "=" * 60)
    a = torch.randn(10, 10).cuda()
    b = torch.randn(10, 10).cuda()
    c = a @ b
    print(c)
    print("=" * 60)

def tokenize_test():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/data/stalaei/passk/models/Qwen2.5-3B-Instruct", trust_remote_code=False)
    text = "Hello, how are you?"
    tokens = tokenizer.encode(text)
    print(tokens)

if __name__ == "__main__":
    tokenize_test()
    test_torch()
    test_hf_generate()
    test_vllm()


