# Paraphraser Runs Format

## Purpose
One-off scripts that transform rollout data. Each run logs to `logs/ParaphraserRuns/{MM}/{DD}/{desc}/`.

## Expected output format
- `config.json5` — what was run (source files, model, timestamp)
- `data.jsonl` — modified rollout entries (same schema as source, with modified `input` and `output`)
- `paraphrase_log.jsonl` — detailed per-entry log (hidden_ranges, raw LLM response, errors)

## Rollout entry schema (output)
Same as verl rollout JSONL entries:
- `input`: `"user\n{prompt}\nassistant\n"` — prompt with hidden-tag instruction suffix
- `output`: assistant response with `<hidden>`/`</hidden>` tags inserted
- All other fields preserved from original entry

## Notes for future paraphraser scripts
- Use `KimiK2_5` from `/shared/matan/code/APPS_inference_lim_hidden_scratchpad/std_setup_factored/LLMs/kimi_k2_5.py`
- Number lines of transcript, ask LLM for line ranges (avoids re-generating entire output)
- Support multiple `<hidden>`/`</hidden>` pairs per entry
- Save both the modified data AND the raw LLM responses for debugging
