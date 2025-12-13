#!/bin/bash
# 混合训练脚本示例
# 在原CODI基础上，同时训练隐式推理和显式CoT

export CUDA_VISIBLE_DEVICES="0"
SAVE_DIR="./ckpts"

python train.py \
    --output_dir "$SAVE_DIR" \
    --expt_name gsm8k_llama1b_adaptive \
    --logging_dir "$SAVE_DIR/logs" \
    --logging_steps 10 \
    --model_name_or_path ./pretrained/Llama-3.2-1B-Instruct \
    --data_name icot \
    --seed 11 \
    --model_max_length 512 \
    --per_device_train_batch_size 16 \
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
    --remove_eos True \
    --distill_loss_factor 1 \
    --print_ref_model_stats True \
    --max_token_num 200 \
    --ref_loss_factor 1.0 \
    --exp_mode True \
    --exp_data_num 1000 \
    --use_adaptive_loss True \
    --adaptive_loss_factor 20.0 \
    --adaptive_window_e_to_l 5 \
    --adaptive_window_l_to_e 0 \
    --adaptive_loss_type smooth_l1 \
    --restore_from ./pretrained/CODI-llama3.2-1b-Instruct/pytorch_model.bin


# To disable adaptive loss (baseline), simply set:
# --use_adaptive_loss False
# or
# --adaptive_loss_factor 0.0


# ============================================
# 方案1: CODI + 混合训练 (推荐)
# 同时训练隐式推理和显式CoT
# ============================================
python train.py \
    --output_dir "$SAVE_DIR" \
    --expt_name gsm8k_llama1b_hybrid \
    --logging_dir "$SAVE_DIR/logs" \
    --logging_steps 10 \
    --model_name_or_path ./pretrained/Llama-3.2-1B-Instruct \
    --data_name icot \
    --seed 11 \
    --model_max_length 512 \
    --per_device_train_batch_size 16 \
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
    --remove_eos True \
    --distill_loss_factor 20 \
    --print_ref_model_stats True \
    --max_token_num 200 \
    --ref_loss_factor 2.0 \
    --exp_mode True \
    --exp_data_num 100 \
    --restore_from ./pretrained/CODI-llama3.2-1b-Instruct/pytorch_model.bin

# --lora_init

# ============================================
# 方案2: SIM-CODI + 混合训练
# 在SIM-CODI基础上添加显式CoT训练
# ============================================
# python train_hybrid.py \
#     --output_dir "$SAVE_DIR" \
#     --expt_name gsm8k_llama1b_sim_hybrid \
#     --logging_dir "$SAVE_DIR/logs" \
#     --logging_steps 10 \
#     --model_name_or_path ./pretrained/SIM_COT-LLaMA3-CODI-1B \
#     --data_name icot \
#     --seed 11 \
#     --model_max_length 512 \
#     --per_device_train_batch_size 16 \
#     --gradient_accumulation_steps 4 \
#     --bf16 \
#     --num_train_epochs 1 \
#     --learning_rate 8e-4 \
#     --max_grad_norm 2.0 \
#     --use_lora True \
#     --lora_r 128 --lora_alpha 32 --lora_init \
#     --save_strategy "no" \
#     --save_total_limit 1 \
#     --save_safetensors False \
#     --weight_decay 0.1 \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --do_train \
#     --report_to tensorboard \
#     --num_latent 6 \
#     --logging_strategy "steps" \
#     --use_prj True \
#     --prj_dim 2048 \
#     --prj_dropout 0.0 \
#     --distill_loss_div_std True \
#     --exp_mode False \
#     --exp_data_num 200 \
#     --remove_eos True \
#     --distill_loss_factor 20 \
#     --print_ref_model_stats True \
#     --max_token_num 200 \
#     --use_decoder True \
#     --ref_loss_factor 2.0 \
    # --restore_from ./pretrained/CODI-llama3.2-1b-Instruct/pytorch_model.bin


# ============================================
# 方案3: 混合训练 + 部分纯CoT样本
# 一定比例的样本只用CoT loss训练，其他用混合loss
# 适合想要更强显式CoT能力的场景
# ============================================
python train.py \
    --output_dir "$SAVE_DIR" \
    --expt_name gsm8k_llama1b_hybrid_mixed_ratio \
    --logging_dir "$SAVE_DIR/logs" \
    --logging_steps 10 \
    --model_name_or_path ./pretrained/Llama-3.2-1B-Instruct \
    --data_name icot \
    --seed 11 \
    --model_max_length 512 \
    --per_device_train_batch_size 16 \
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
    --remove_eos True \
    --distill_loss_factor 20 \
    --print_ref_model_stats True \
    --max_token_num 200 \
    --ref_loss_factor 20 \
    --hybrid_cot_only_ratio 1 \
    --exp_mode True \
    --exp_data_num 100 \
    --restore_from ./pretrained/CODI-llama3.2-1b-Instruct/pytorch_model.bin


