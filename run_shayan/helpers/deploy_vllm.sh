#   --tokenizer-pool-size 8 \

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu_ids) gpu_ids="$2"; shift ;;
        --model_path) model_path="$2"; shift ;;
        --model_id) model_id="$2"; shift ;;
        --port) port="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

IFS=, read -ra IDS <<< "$gpu_ids"
num_gpus=${#IDS[@]}

CUDA_VISIBLE_DEVICES=${gpu_ids} \
vllm serve ${model_path} \
  --served-model-name "${model_id}" \
  --host "0.0.0.0" --port "${port}" \
  --data-parallel-size "${num_gpus}" --data-parallel-size-local "${num_gpus}" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --api-server-count "$((4 * num_gpus))" \
  --max-num-batched-tokens 131072 \
  --max-num-seqs 1024

#   Example run: bash run/deploy_vllm.sh --gpu_ids 0,1,2,3 --model_path Qwen/Qwen2.5-1.5B-Instruct --model_id Qwen2.5-1.5B --port 8000