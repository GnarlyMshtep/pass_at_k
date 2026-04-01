"""Eval a single merged model."""
import argparse
import asyncio
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from TRLSFT.envs.apps_backdoor import (
    compute_summary,
    generate_completions_vllm_client,
    load_eval_questions,
    score_completions,
)


async def async_main(model_path: str, label: str, port: int) -> None:
    eval_source = "/shared/matan/code/pass_at_k/logs/VerlRun/03/02/hidden_wmonitor_backdoor_qwen3_8b_remove_21_39_186oohgn/rollouts/train/401.jsonl"
    questions = load_eval_questions(eval_source_file=eval_source, n_samples=100, seed=42)
    prompts = [q["input"] for q in questions]
    print(f"[{label}] Loaded {len(questions)} eval questions")

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path, "--port", str(port),
        "--dtype", "bfloat16", "--max-model-len", "7000",
        "--gpu-memory-utilization", "0.85", "--tensor-parallel-size", "1",
    ]
    log_fh = open(f"/shared/matan/code/pass_at_k/tmp/vllm_{label}.log", "w")
    server_proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)

    try:
        import httpx
        start = time.time()
        while time.time() - start < 600:
            try:
                if httpx.get(f"http://localhost:{port}/health", timeout=5).status_code == 200:
                    print(f"[{label}] vLLM ready after {time.time() - start:.0f}s")
                    break
            except Exception:
                pass
            time.sleep(5)
        else:
            print(f"[{label}] ERROR: server did not start"); server_proc.kill(); return

        completions = await generate_completions_vllm_client(
            prompts=prompts, api_base=f"http://localhost:{port}/v1",
            model_name=model_path, max_tokens=6000, temperature=0.7,
        )
        results = await score_completions(questions=questions, completions=completions, reward_global_step=500)
        summary = compute_summary(results=results)

        total_opens = sum(len(re.findall(r"<hidden>", (r.get("generation", {}) or {}).get("response", ""))) for r in results)
        total_closes = sum(len(re.findall(r"</hidden>", (r.get("generation", {}) or {}).get("response", ""))) for r in results)
        fail_extract = sum(1 for r in results if r.get("tests_total", 0) == 0)

        print(f"\n=== {label} ===")
        print(f"  score={summary['mean_score']:.4f}  frac_test={summary['mean_frac_test_cases_passing']:.4f}  sus={summary['mean_sus_score']:.4f}")
        print(f"  backdoor={summary['mean_backdoor_test_passed']:.4f}  hidden_len={summary['mean_hidden_lengths']:.1f}")
        print(f"  avg <hidden>={total_opens/len(results):.1f}  avg </hidden>={total_closes/len(results):.1f}  fail_extract={fail_extract/len(results):.2f}")

        with open(f"/tmp/eval_{label}.jsonl", "w") as f:
            for r in results:
                f.write(json.dumps(r, default=str) + "\n")
    finally:
        server_proc.terminate()
        try: server_proc.wait(timeout=10)
        except: server_proc.kill()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--port", type=int, required=True)
    args = p.parse_args()
    asyncio.run(async_main(model_path=args.model_path, label=args.label, port=args.port))
