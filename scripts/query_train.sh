source "$(conda info --base)/etc/profile.d/conda.sh" # 推荐这种更健壮的方式
# conda activate my_rag
# --runtime-env-json='{"working_dir": "/openrlhf"}' \
#    --flash_attn \
##### 1.  本地缓存根目录（用 /tmp/$USER 就够了） #####
LOCAL_SCRATCH="/tmp/${USER}"
mkdir -p "${LOCAL_SCRATCH}"/{torch_extensions,triton_cache,hf_cache,tmp}

##### 2.  让 Bash 先把变量替换成“绝对路径”后再塞进 JSON #####
RUNTIME_ENV_JSON=$(cat <<EOF
{
  "env_vars": {
    "TORCH_EXTENSIONS_DIR": "${LOCAL_SCRATCH}/torch_extensions",
    "TRITON_CACHE_DIR":     "${LOCAL_SCRATCH}/triton_cache",
    "XDG_CACHE_HOME":       "${LOCAL_SCRATCH}/hf_cache",
    "HF_HOME":              "${LOCAL_SCRATCH}/hf_cache/hf",
    "HF_DATASETS_CACHE":    "${LOCAL_SCRATCH}/hf_cache/datasets",
    "TMPDIR":               "${LOCAL_SCRATCH}/tmp"
  }
}
EOF
)



N_SAMPLES=16
EPISODE=1
WARMUP=0.0
TBS=128
RBS=1
KL=0.0001
LR=2e-6
MAX_LENGTH=16
TEMP=1.0



ray job submit --address="http://127.0.0.1:8267" \
   --runtime-env-json "$RUNTIME_ENV_JSON" \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 1 \
   --colocate_all_model \
   --deepspeed_enable_sleep \
   --vllm_enable_sleep \
   --vllm_gpu_memory_utilization 0.3 \
   --pretrain /scratch/Llama-3.1-8B-Instruct \
   --remote_rm_url http://138.25.54.180:5008/get_reward \
   --save_path openrlhf/only_query_data10000 \
   --train_batch_size 448 \
   --micro_train_batch_size 14 \
   --micro_rollout_batch_size 14 \
   --rollout_batch_size 28 \
   --n_samples_per_prompt 16 \
   --advantage_estimator group_norm \
   --max_samples 100000 \
   --max_epochs 1 \
   --num_episodes ${EPISODE} \
   --lr_warmup_ratio ${WARMUP} \
   --prompt_max_len 2560 \
   --generate_max_len 48 \
   --critic_learning_rate 9e-6 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate $LR \
   --init_kl_coef $KL \
   --prompt_data query_only_insufficient_data.jsonl \
   --input_key raw_prompt \
   --apply_chat_template \
   --flash_attn \
   --gradient_checkpointing \
   --save_steps 20 \
   --save_hf_ckpt \
   --max_ckpt_num 2 \
   --ckpt_path /scratch/query2 \
   --temperature $TEMP \
   --use_wandb  \
   --wandb_project \
   --wandb_run_name \
   --packing_samples \
   --adam_offload \
   --load_checkpoint \
   --enable_prefix_caching \