

# Pick the GPUs and a friendly API model name
```
# pick your 4 GPUs and a model name clients will call
CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --served-model-name qwen1p5b \
  --host 0.0.0.0 --port 8000 \
  --api-key sk-local-123 \
  --data-parallel-size 4 --data-parallel-size-local 4 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --api-server-count 4 \
  --tokenizer-pool-size 8 \
  --max-num-batched-tokens 131072 \
  --max-num-seqs 1024
  ```

  or use the run/deploy_vllm.sh

  # Check your ip
  hostname -I | awk '{print $1}'

  # Query the model
