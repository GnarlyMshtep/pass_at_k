import argparse
import asyncio
import os
import time

import httpx


def build_base_url(ip: str, port: int) -> str:
    return f"http://{ip}:{port}/v1"


def build_headers() -> dict:
    return {
        "Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', 'sk-local-123')}",
        "Content-Type": "application/json",
    }


async def run_benchmark(base_url: str, num_requests: int, prompt: str, model: str) -> float:
    limits = httpx.Limits(
        max_connections=num_requests,
        max_keepalive_connections=num_requests,
    )
    timeout = httpx.Timeout(300.0)
    headers = build_headers()

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        semaphore = asyncio.Semaphore(num_requests)

        async def send_one() -> None:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "temperature": 0.0,
                "max_tokens": 128,
            }
            async with semaphore:
                r = await client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
                r.raise_for_status()

        start = time.perf_counter()
        await asyncio.gather(*(send_one() for _ in range(num_requests)))
        end = time.perf_counter()

    return end - start


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark vLLM chat completions throughput.")
    parser.add_argument("--ip", default="localhost", help="Server IP or hostname (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--num-requests", type=int, default=1000, help="Total number of requests to send")
    parser.add_argument("--prompt", default="Hello! Write a 250 sentence greeting for task", help="User prompt to send in each request")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="Model name served by vLLM (default: Qwen/Qwen2.5-1.5B-Instruct)")
    args = parser.parse_args()

    base_url = build_base_url(args.ip, args.port)

    duration_s = asyncio.run(run_benchmark(base_url, args.num_requests, args.prompt, args.model))
    rps = args.num_requests / duration_s if duration_s > 0 else float("inf")

    print(f"Completed {args.num_requests} requests in {duration_s:.3f}s ({rps:.2f} req/s)")


if __name__ == "__main__":
    main()
