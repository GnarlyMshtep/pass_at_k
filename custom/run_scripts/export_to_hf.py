import os
import shutil

import torch
from transformers import AutoConfig, AutoModelForCausalLM

# Try to import DTensor for unwrapping FSDP2 checkpoints
try:
	from torch.distributed._tensor import DTensor  # type: ignore
	_HAS_DTENSOR = True
except Exception:
	DTensor = None  # type: ignore
	_HAS_DTENSOR = False

ckpt_dir = "/mnt/xfs/home/aiilyas/rl-exploration/pass_at_k/checkpoints/sft/global_step_6"
hf_export_dir = os.path.join(ckpt_dir, "hf_export")

# Use the same base model/config you trained
base_model = "Qwen/Qwen2.5-1.5B-Instruct"
config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model, config=config, trust_remote_code=True)

# Load FSDP-saved full state (world_size_1 suggests full state in a single .pt)
state_path = os.path.join(ckpt_dir, "model_world_size_1_rank_0.pt")
state = torch.load(state_path, map_location="cpu")

# Unwrap DTensor -> local torch.Tensor if needed
if isinstance(state, dict):
	new_state = {}
	for k, v in state.items():
		if _HAS_DTENSOR and isinstance(v, DTensor):
			v = v.to_local()
		new_state[k] = v
	state = new_state

# Fallback: strip leading 'module.' if present in keys
def _maybe_strip_module_prefix(sd: dict):
	if not sd:
		return sd
	has_module = any(k.startswith("module.") for k in sd.keys())
	if not has_module:
		return sd
	return {k[len("module."):]: v for k, v in sd.items()}

state_try = state
try:
	missing, unexpected = model.load_state_dict(state_try, strict=False)
except RuntimeError:
	state_try = _maybe_strip_module_prefix(state)
	missing, unexpected = model.load_state_dict(state_try, strict=False)

print("missing:", missing, "unexpected:", unexpected)

model.save_pretrained(hf_export_dir, safe_serialization=True)
print("Saved HF model to:", hf_export_dir)

# Optionally copy tokenizer/config artifacts if present in the checkpoint folder
ckpt_hf_dir = os.path.join(ckpt_dir, "huggingface")
if os.path.isdir(ckpt_hf_dir):
	for fname in [
		"tokenizer.json",
		"tokenizer_config.json",
		"special_tokens_map.json",
		"vocab.json",
		"merges.txt",
		"added_tokens.json",
		"chat_template.jinja",
		"generation_config.json",
	]:
		src = os.path.join(ckpt_hf_dir, fname)
		if os.path.isfile(src):
			try:
				shutil.copy2(src, os.path.join(hf_export_dir, fname))
			except Exception:
				pass