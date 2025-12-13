#!/bin/bash
# ============================================================================
# 半显半隐推理训练 - 完整流程脚本
# ============================================================================

set -e

# 配置路径
MODEL_PATH="./CODI/pretrained/Llama-3.2-1B-Instruct"
DATA_DIR="./data/hybrid_training"
CHECKPOINT_DIR="./checkpoints/hybrid_codi"
export WANDB_DISABLED=true
# ============================================================================
# Phase 1: 数据生成 (SwiReasoning风格)
# ============================================================================

echo "========================================"
echo "Phase 1: Generating hybrid training data"
echo "Using SwiReasoning logic from step4"
echo "========================================"

# 如果数据目录不存在，运行Phase 1
if [ ! -f "${DATA_DIR}/hybrid_training_data.json" ]; then
    python phase1_data_generation.py \
        --model_path ${MODEL_PATH} \
        --data_name icot \
        --output_dir ${DATA_DIR} \
        --window_e_to_l 5 \
        --window_l_to_e 0 \
        --max_switch_count 5 \
        --max_latent_steps 6 \
        --entropy_guided_ratio 0.5 \
        --random_ratio 0.3 \
        --explicit_ratio 0.2 \
        --bf16 \
        --seed 42
    
    echo "Phase 1 complete! Data saved to ${DATA_DIR}"
else
    echo "Phase 1 data already exists, skipping..."
fi

# ============================================================================
# Phase 2: 训练
# ============================================================================

echo "========================================"
echo "Phase 2: Training hybrid CODI model"
echo "========================================"

python phase2_train_compatible.py \
    --model_name_or_path ${MODEL_PATH} \
    --hybrid_data_path ${DATA_DIR}/hybrid_training_data.json \
    --output_dir ${CHECKPOINT_DIR} \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --num_latent 6 \
    --prj_dim 2048 \
    --distill_loss_factor 1.0 \
    --ref_loss_factor 1.0 \
    --hybrid_cot_only_ratio 0.2 \
    --logging_steps 50 \
    --save_steps 500 \
    --bf16 \
    --seed 42

echo "Phase 2 complete! Model saved to ${CHECKPOINT_DIR}"

# ============================================================================
# Phase 3: 评估
# ============================================================================

echo "========================================"
echo "Phase 3: Evaluating on GSM8K"
echo "========================================"

python step4_adaptive_eval.py \
    --model_type codi \
    --base_model_path ${MODEL_PATH} \
    --ckpt_dir ${CHECKPOINT_DIR} \
    --prj_dim 2048 \
    --data_name gsm8k \
    --bf16 \
    --baseline_mode adaptive \
    --max_switch_count 5 \
    --window_e_to_l 5 \
    --window_l_to_e 0 \
    --max_samples 100

echo "========================================"
echo "All phases complete!"
echo "========================================"
