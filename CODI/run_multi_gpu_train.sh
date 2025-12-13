#!/bin/bash
# 多卡混合训练脚本示例
# 使用 torchrun 或 accelerate 进行分布式训练

SAVE_DIR="./hybrid-gpu-Llama-3.2-1B-Instruct"
NUM_GPUS=4  # 修改为你的GPU数量

# ============================================
# 方法1: 使用 torchrun (推荐)
# ============================================
torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 train.py \
    --output_dir "$SAVE_DIR" \
    --expt_name gsm8k_llama1b_hybrid_multi_gpu \
    --logging_dir "$SAVE_DIR/logs" \
    --logging_steps 10 \
    --model_name_or_path ./pretrained/Llama-3.2-1B-Instruct \
    --data_name icot \
    --seed 11 \
    --model_max_length 512 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --bf16 \
    --num_train_epochs 1 \
    --learning_rate 8e-4 \
    --max_grad_norm 2.0 \
    --use_lora True \
    --lora_r 128 --lora_alpha 32 --lora_init \
    --save_strategy "no" \
    --save_total_limit 1 \
    --save_safetensors False \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --do_train \
    --report_to tensorboard \
    --num_latent 6 \
    --logging_strategy "steps" \
    --use_prj True \
    --prj_dim 2048 \
    --prj_dropout 0.0 \
    --distill_loss_div_std True \
    --exp_mode False \
    --exp_data_num 200 \
    --remove_eos True \
    --distill_loss_factor 20 \
    --print_ref_model_stats False \
    --max_token_num 200 \
    --hybrid_training True \
    --cot_loss_factor 1.0 \
    --ddp_find_unused_parameters False

# ============================================
# 方法2: 使用 accelerate
# 首先运行: accelerate config 配置分布式环境
# ============================================
# accelerate launch --num_processes=$NUM_GPUS train.py \
#     --output_dir "$SAVE_DIR" \
#     ... (其他参数同上)

# ============================================
# 方法3: 使用 deepspeed (需要额外配置)
# ============================================
# deepspeed --num_gpus=$NUM_GPUS train.py \
#     --deepspeed ds_config.json \
#     --output_dir "$SAVE_DIR" \
#     ... (其他参数同上)

# ============================================
# 多卡注意事项：
# 1. per_device_train_batch_size 是每张卡的batch size
#    总batch size = per_device_train_batch_size * NUM_GPUS * gradient_accumulation_steps
# 2. 建议关闭 print_ref_model_stats，避免多卡输出混乱
# 3. 学习率可能需要根据总batch size调整
# ============================================
