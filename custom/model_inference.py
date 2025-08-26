#!/usr/bin/env python3
"""
Simple model inference utility class.
Provides clean functions for loading and calling models using HuggingFace.
Supports both single and batch generation.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional, Union


class ModelInference:
    """
    Simple class for loading and running inference with HuggingFace models.
    Supports both single and batch generation.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        **kwargs
    ):
        """
        Initialize the model inference class.
        
        Args:
            model_path: Path to the model (local or HuggingFace hub)
            device: Device to load model on ("auto", "cuda", "cpu")
            dtype: Model precision ("bfloat16", "float16", "float32")
            trust_remote_code: Whether to trust remote code
            **kwargs: Additional arguments for model loading
        """
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.kwargs = kwargs
        
        self.model = None
        self.tokenizer = None
        self.default_system_message = self._select_default_system_message(model_path)
        
        # Load model and tokenizer
        self._load_model()
    
    def _select_default_system_message(self, model_path: str) -> str:
        """Return a sensible default system message based on the model name."""
        name = (model_path or "").lower()
        if "qwen" in name:
            return "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        return "You are a helpful assistant."

    def _load_model(self):
        """Load the model and tokenizer."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side="left",
            trust_remote_code=self.trust_remote_code
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        torch_dtype = getattr(torch, self.dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map=self.device,
            trust_remote_code=self.trust_remote_code,
            **self.kwargs
        )
        
        self.model.eval()
    
    def _generate_from_texts(
        self,
        texts: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        num_return_sequences: int,
        pad_token_id: Optional[int],
        eos_token_id: Optional[int],
        **kwargs
    ) -> List[str]:
        """Tokenize texts, generate continuations, and return decoded completions only."""
        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **kwargs
            )

        # Compute per-input real lengths using attention mask (exclude padding)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            input_lengths = [inputs["input_ids"][i].shape[-1] for i in range(inputs["input_ids"].shape[0])]
        else:
            input_lengths = [int(mask.sum().item()) for mask in attention_mask]

        total_outputs = outputs.shape[0]
        num_inputs = len(texts)
        decoded = []
        for row_idx in range(total_outputs):
            input_idx = row_idx // num_return_sequences
            cut = input_lengths[input_idx]
            new_tokens = outputs[row_idx][cut:]
            decoded.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        return decoded

    def generate(
        self,
        prompt: Union[str, List[str]],
        system_message: Optional[str] = None,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text from prompt(s) using chat template format. Supports both single and batch generation.
        
        Args:
            prompt: Input text prompt(s) - can be string or list of strings
            system_message: System message to include in the conversation
            max_length: Maximum length of generated text
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = random)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            num_return_sequences: Number of sequences to return per prompt
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text string(s) - string for single prompt, list for batch
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")
        
        # Handle single vs batch input
        is_batch = isinstance(prompt, list)
        prompts = prompt if is_batch else [prompt]
        sys_msg = system_message or self.default_system_message
        
        # Set default token IDs
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        
        # Convert prompts to chat-formatted texts
        texts = []
        for user_prompt in prompts:
            msgs = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_prompt},
            ]
            if hasattr(self.tokenizer, "apply_chat_template"):
                text = self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            else:
                text = f"system: {sys_msg}\nuser: {user_prompt}\nassistant: "
            texts.append(text)

        decoded = self._generate_from_texts(
            texts=texts,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        if num_return_sequences > 1:
            # Group per input
            grouped = [
                decoded[i : i + num_return_sequences]
                for i in range(0, len(decoded), num_return_sequences)
            ]
            return grouped if is_batch else grouped[0]
        return decoded if is_batch else decoded[0]
    
    def generate_chat(
        self,
        messages: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text using chat template. Supports both single and batch.
        
        Args:
            messages: Single conversation or list of conversations
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text string(s)
        """
        # Handle single vs batch input
        is_batch = isinstance(messages[0], list) if messages else False
        conversations = messages if is_batch else [messages]

        # Convert conversations to chat-formatted texts
        texts = []
        for conv in conversations:
            if hasattr(self.tokenizer, "apply_chat_template"):
                text = self.tokenizer.apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=True
                )
            else:
                text = "\n".join([f"{m['role']}: {m['content']}" for m in conv]) + "\nassistant: "
            texts.append(text)

        # Use shared generator
        decoded = self._generate_from_texts(
            texts=texts,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=50,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )
        return decoded if is_batch else decoded[0]
    
    def generate_batch(
        self,
        prompts: List[str],
        system_message: Optional[str] = None,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        batch_size: int = 8,
        **kwargs
    ) -> List[str]:
        """
        Generate text for a batch of prompts with automatic batching.
        
        Args:
            prompts: List of input prompts
            system_message: System message to include in the conversation (defaults based on model)
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            batch_size: Size of batches to process at once
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated text strings
        """
        results = []
        
        sys_msg = system_message or self.default_system_message

        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_results = self.generate(
                batch_prompts,
                system_message=sys_msg,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
            results.extend(batch_results)
        
        return results
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current GPU memory usage.
        
        Returns:
            Dictionary with memory usage information
        """
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        memory_info = {}
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            
            memory_info[f"gpu_{i}"] = {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "total_gb": total,
                "free_gb": total - reserved
            }
        
        return memory_info
    
    def clear_cache(self):
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
        self.clear_cache()


# Convenience functions for quick usage
def load_model(
    model_path: str,
    device: str = "auto",
    dtype: str = "bfloat16",
    **kwargs
) -> ModelInference:
    """
    Load a model and return ModelInference instance.
    
    Args:
        model_path: Path to the model
        device: Device to load model on
        dtype: Model precision
        **kwargs: Additional arguments
        
    Returns:
        ModelInference instance
    """
    return ModelInference(
        model_path=model_path,
        device=device,
        dtype=dtype,
        **kwargs
    )


def generate_text(
    model: ModelInference,
    prompt: Union[str, List[str]],
    system_message: str = "You are a helpful assistant.",
    max_length: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    **kwargs
) -> Union[str, List[str]]:
    """
    Generate text using a loaded model. Supports both single and batch.
    
    Args:
        model: ModelInference instance
        prompt: Input prompt(s) - string or list of strings
        system_message: System message to include in the conversation
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        **kwargs: Additional generation parameters
        
    Returns:
        Generated text - string for single prompt, list for batch
    """
    return model.generate(
        prompt=prompt,
        system_message=system_message,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        **kwargs
    )


def generate_batch(
    model: ModelInference,
    prompts: List[str],
    system_message: str = "You are a helpful assistant.",
    max_length: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    batch_size: int = 8,
    **kwargs
) -> List[str]:
    """
    Generate text for a batch of prompts with automatic batching.
    
    Args:
        model: ModelInference instance
        prompts: List of input prompts
        system_message: System message to include in the conversation
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        batch_size: Size of batches to process at once
        **kwargs: Additional generation parameters
        
    Returns:
        List of generated text strings
    """
    return model.generate_batch(
        prompts=prompts,
        system_message=system_message,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        batch_size=batch_size,
        **kwargs
    )
