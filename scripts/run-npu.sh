#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python




# set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
export CUDA_VISIBLE_DEVICES=0,1,2,3
NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"



SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

RUN_ID=${RUN_ID:-"run_$(date +%Y%m%d_%H%M%S)"}
LOAD_SAVE_PATH="/home/zhanghaoze/myfork/slime/shared_data/${RUN_ID}/checkpoints"

CKPT_ARGS=(
   --hf-checkpoint /home/zhanghaoze/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct
   --load /home/zhanghaoze/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct
   --ref-load /home/zhanghaoze/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct
)

ROLLOUT_ARGS=(
   --prompt-data /home/zhanghaoze/.cache/huggingface/hub/datasets--zhuzilin--dapo-math-17k/snapshots/2e65612930298bde4c5d58fd97b3f23a483aaff9/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --balance-data
   --rm-type deepscaler
   --num-rollout 100
   --rollout-batch-size 2
   --n-samples-per-prompt 4
   --rollout-max-response-len 4096
   --rollout-temperature 0.8
   --global-batch-size 8
)

GRPO_ARGS=(
   --use-kl-loss
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime-dev-mcore-fsdp
   # --wandb-group qwen3-4B-fsdp-1130-ref
   # --wandb-key ${WANDB_API_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.75
   --sglang-decode-log-interval 1000
   --sglang-chunked-prefill-size 4096
   --sglang-device npu
   --sglang-attention-backend ascend
   # --sglang-enable-memory-saver False
   --sglang-disable-cuda-graph
   # --disable-cuda-graph
   # --sglang-attention-backend ascend
   
)

TRAIN_BACKEND_ARGS=(
   --train-backend fsdp
   --update-weight-buffer-size 536870912
   --gradient-checkpointing
   --attn-implementation eager
   --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
)

PERF_ARGS=(
   --use-dynamic-batch-size
   --max-tokens-per-gpu 2048
)

MISC_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node 4
   # --num-gpus-per-node 4
   --rollout-num-gpus 4
   # --colocate
   --use-fault-tolerance
   --dump-details /home/zhanghaoze/myfork/slime/shared_data/qwen3-4B-fsdp-1116-noref/dump_details
   --distributed-backend hccl
   # --fsdp-cpu-offload
)

# launch the master node of ray in container - 8 GPUs for training
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --resources '{"NPU":8}' --disable-usage-stats


RUNTIME_ENV_JSON="{
   \"env_vars\": {
     \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}\",
     \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\", 
     \"TORCHINDUCTOR_FORCE_DISABLE_CACHES\": \"1\",  
   }
}"

   # --runtime-env-json="${RUNTIME_ENV_JSON}" \
ray job submit --address="http://127.0.0.1:8265" \
   -- python3 train.py \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${TRAIN_BACKEND_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${MISC_ARGS[@]}



