#!/usr/bin/env python3
import argparse
import json
import sys

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)


def detect_chat_template(tokenizer, model_path):
    """Try to detect if this is an instruct/chat model"""
    # Check for common instruct indicators
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        return True
    
    # Check model name for instruct indicators
    model_name = model_path.lower()
    instruct_indicators = ['instruct', 'chat', 'assistant', 'alpaca', 'vicuna', 'llama-2-chat']
    return any(indicator in model_name for indicator in instruct_indicators)

def format_chat_message(tokenizer, messages):
    """Format messages using the tokenizer's chat template"""
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback: simple format
        formatted = ""
        for msg in messages:
            if msg["role"] == "user":
                formatted += f"Human: {msg['content']}\n"
            elif msg["role"] == "assistant":
                formatted += f"Assistant: {msg['content']}\n"
        formatted += "Assistant: "
        return formatted

def generate_response(model, tokenizer, prompt, max_new_tokens=512, temperature=0.7, do_sample=True, device="auto"):
    """Generate response from the model"""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Respect the requested device when moving inputs
    if device == "cuda" and torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id),
            repetition_penalty=1.1
        )
    
    # Decode only the new tokens
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()

def main():
    parser = argparse.ArgumentParser(description='Chat with or complete text using local HF models')
    parser.add_argument('--model-path', required=True, help='Path to HF model directory')
    parser.add_argument('--completion', default=False,  action='store_true', help='Single completion mode instead of chat')
    parser.add_argument('--max-tokens', type=int, default=2048, help='Max new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--quantize', action='store_true', default=False, help='Use 4-bit quantization (requires bitsandbytes)')
    parser.add_argument('--device', default='cuda', choices=['auto', 'cpu', 'cuda'], help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}...")
    
    # Set up quantization if requested
    quantization_config = None
    if args.quantize:
        if args.device != 'cuda' or not torch.cuda.is_available():
            print("Quantization requires CUDA; disabling quantization since device is not CUDA or CUDA unavailable.")
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model_kwargs = {"quantization_config": quantization_config} if quantization_config else {}
    if args.device in ['cuda', 'auto'] and torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16
        # Let accelerate infer device map when auto, or when quantization is enabled
        if args.device == 'auto' or quantization_config is not None:
            model_kwargs["device_map"] = "auto"
    
    # Filter out None values to avoid passing them
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **{k: v for k, v in model_kwargs.items() if v is not None})
    model.eval()
    
    # Detect if this is a chat model
    is_chat_model = detect_chat_template(tokenizer, args.model_path)
    print(f"Detected {'chat/instruct' if is_chat_model else 'base'} model")
    
    if args.completion:
        # Single completion mode
        print("\n=== COMPLETION MODE ===")
        prompt = input("Prompt: ")
        
        print(f"\n=== EXACT INPUT TO MODEL ===")
        print(repr(prompt))
        print(f"=== END INPUT ({len(tokenizer.encode(prompt))} tokens) ===\n")
        
        response = generate_response(model, tokenizer, prompt, args.max_tokens, args.temperature, device=args.device)
        print(f"Generated: {response}")
        
    else:
        # Interactive chat mode
        print(f"\n=== {'CHAT' if is_chat_model else 'COMPLETION'} MODE ===")
        print("Type 'quit' to exit, 'clear' to clear conversation\n")
        
        conversation = []
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'clear':
                    conversation = []
                    print("Conversation cleared.\n")
                    continue
                elif user_input == '':
                    continue
                
                if is_chat_model:
                    # Use proper chat formatting
                    conversation.append({"role": "user", "content": user_input})
                    full_prompt = format_chat_message(tokenizer, conversation)
                else:
                    # Simple back-and-forth for base models
                    conversation.append(f"Human: {user_input}")
                    full_prompt = "\n".join(conversation) + "\nAssistant: "
                
                print(f"\n=== EXACT INPUT TO MODEL ===")
                print(repr(full_prompt))
                print(f"=== END INPUT ({len(tokenizer.encode(full_prompt))} tokens) ===\n")
                
                response = generate_response(model, tokenizer, full_prompt, args.max_tokens, args.temperature, device=args.device)
                print(f"Assistant: {response}\n")
                
                if is_chat_model:
                    conversation.append({"role": "assistant", "content": response})
                else:
                    conversation.append(f"Assistant: {response}")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()