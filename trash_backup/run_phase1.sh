#!/bin/bash
# ============================================================================
# Phase 1: 半显半隐训练数据生成 - 运行脚本 (SwiReasoning风格)
# ============================================================================

# 设置路径
MODEL_PATH="./CODI/pretrained/Llama-3.2-1B-Instruct"  # 用于计算熵的模型
OUTPUT_DIR="./data/hybrid_training"
export CUDA_VISIBLE_DEVICES="4,5,6,7"
# ============================================================================
# 生成半显半隐训练数据 (使用SwiRController逻辑)
# ============================================================================
echo "=========================================="
echo "Generating hybrid training data..."
echo "Using SwiReasoning logic from step4"
echo "=========================================="

python phase1_data_generation.py \
    --model_path ${MODEL_PATH} \
    --data_name icot \
    --output_dir ${OUTPUT_DIR} \
    --window_e_to_l 5 \
    --window_l_to_e 0 \
    --max_switch_count 5 \
    --max_latent_steps 6 \
    --entropy_guided_ratio 0.5 \
    --random_ratio 0.3 \
    --explicit_ratio 0.2 \
    --bf16 \
    --seed 42
    # --max_samples 1000  # 测试时先用少量样本

echo "=========================================="
echo "Phase 1 Complete!"
echo "=========================================="
echo "Generated files:"
echo "  - ${OUTPUT_DIR}/hybrid_training_data.json     (所有样本)"
echo "  - ${OUTPUT_DIR}/hybrid_training_entropy_guided.json"
echo "  - ${OUTPUT_DIR}/hybrid_training_random.json"
echo "  - ${OUTPUT_DIR}/hybrid_training_full_explicit.json"
echo "  - ${OUTPUT_DIR}/generation_stats.json         (统计信息)"
echo ""
echo "每个样本包含:"
echo "  - token_modes: 每个token的模式 (0=latent, 1=normal)"
echo "  - entropies: 每个token的熵值"
echo "  - switch_events: 切换事件"
echo "=========================================="
