EPISODE=1
WARMUP=0.0
KL=0.0001
LR=2e-6
TEMP=1.0


deepspeed --module openrlhf.cli.train_sft \
  --pretrain /scratch/Llama-3.1-8B-Instruct \
  --dataset sft_final_data_2plus.jsonl \
  --input_key message \
  --apply_chat_template \
  --max_len 2560 \
  --train_batch_size 16 \
  --micro_train_batch_size 1 \
  --save_steps 100 \
  --logging_steps 1 \
  --eval_steps 50 \
  --zero_stage 3 \
  --max_epochs 1 \
  --bf16 \
  --learning_rate 1e-5 \
  --save_path /scratch/Debate-Augmented-RAG/sft_llama_full \
  --use_wandb  \
  --wandb_project \
  --wandb_run_name \
  --packing_samples \
  --max_ckpt_num 2 \
  --save_hf_ckpt \
  --ckpt_path /scratch/llama_judge_full \
