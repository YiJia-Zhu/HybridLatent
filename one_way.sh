#!/bin/bash
# 我基于CODI改进了方法，实现了半显半隐层推理方法，loss计算的对齐方法都参考官方实现，仅仅范式变为了半显半隐（核心./CODI/src/model_adaptive.py）。目前我one_way.sh训练出来的模型，理论上应该就是一般的CoT - SFT？应该和我训练数据的格式<<3+4=7>> <<16-7=9>>The answer is: 18这样一致，但是为什么是“## Step 1: Calculate the number of eggs ”但是这里为什么推理结果是这样貌似和没有训练的llama格式一致？
# 帮我看看CODI官方怎么搞的/mnt/8T/xgr/zhuyijia/SIM-CoT/CODI/src/model.py为什么他们可以训练成功？



# ========== 配置变量 ==========
EXPT_NAME="gsm8k_llama1b_adaptive"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$SCRIPT_DIR}"
SAVE_DIR="${SAVE_DIR:-$BASE_DIR/CODI/ckpts}"

# 训练参数（用于构建 CKPT_DIR 路径）
MODEL_NAME="Llama-3.2-1B-Instruct"
NUM_EPOCHS=10
LEARNING_RATE=0.0008
SEED=11

# 自动构建 CKPT_DIR
CKPT_DIR="${SAVE_DIR}/${EXPT_NAME}/${MODEL_NAME}/ep_${NUM_EPOCHS}/lr_${LEARNING_RATE}/seed_${SEED}"

# GPU 配置（单卡）
GPU=1

# ========== 日志配置 ==========
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${SAVE_DIR}/logs_${TIMESTAMP}"
mkdir -p "$LOG_DIR"
cp "$0" "$LOG_DIR/run_script.sh"


TRAIN_LOG="${LOG_DIR}/train_gpu${GPU}.log"
TEST_LATENT_LOG="${LOG_DIR}/test_latent_gpu${GPU}.log"
TEST_COT_LOG="${LOG_DIR}/test_cot_gpu${GPU}.log"
TEST_ADAPTIVE_LOG="${LOG_DIR}/test_adaptive_gpu${GPU}.log"

echo "Logs will be saved to: $LOG_DIR"

# ========== 信号处理：Ctrl+C 时终止 ==========
cleanup() {
    echo ""
    echo "=========================================="
    echo "Caught interrupt signal! Cleaning up..."
    echo "=========================================="
    pkill -P $$ 2>/dev/null
    echo "Process terminated."
    echo "Partial logs saved in: $LOG_DIR"
    exit 1
}

trap cleanup SIGINT SIGTERM SIGHUP

# ========== Training ==========
echo "Starting Training on GPU ${GPU}..."
echo "Training log: $TRAIN_LOG"
cd ${BASE_DIR}/CODI


# CUDA_VISIBLE_DEVICES="6,7" torchrun --nproc_per_node=2 --master_port 29501 train_adaptive.py \
CUDA_VISIBLE_DEVICES="${GPU}" python train_adaptive.py \
    --output_dir "$SAVE_DIR" \
    --expt_name "$EXPT_NAME" \
    --logging_dir "$SAVE_DIR/logs" \
    --logging_steps 10 \
    --model_name_or_path ./pretrained/${MODEL_NAME} \
    --data_name icot \
    --seed ${SEED} \
    --model_max_length 512 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --bf16 \
    --num_train_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --max_grad_norm 2.0 \
    --use_lora True \
    --lora_r 128 --lora_alpha 32 --lora_init \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 30 \
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
    --print_ref_model_stats True \
    --max_token_num 200 \
    --remove_eos True \
    --adaptive_training True\
    --window_e_to_l 5 \
    --window_l_to_e 0 \
    --baseline_mode random \
    --random_prob 0.5 \
    --use_decoder \
    --ce_loss_factor 1 --ref_loss_factor 1 --explain_loss_factor 0.1 --align_loss_factor 20 --distill_loss_factor 20 \
    2>&1 | tee "$TRAIN_LOG"

    # --restore_from ./pretrained/CODI-llama3.2-1b-Instruct/pytorch_model.bin \
    # --use_adaptive_loss False \
    # --adaptive_loss_factor 5.0 \
    # --adaptive_window_e_to_l 5 \
    # --adaptive_window_l_to_e 0 \
    # --adaptive_loss_type smooth_l1 \
    # 

# --hybrid_cot_only_ratio 1 \
# 检查训练是否成功
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "Training failed! Check log: $TRAIN_LOG"
    exit 1
fi
echo "Training completed!"

# ========== Resolve checkpoint for eval ==========
# `train_adaptive.py` saves weights under `checkpoint-*/pytorch_model.bin` during training.
# If the run finishes cleanly it may also save to `$CKPT_DIR`, but when interrupted only
# the step checkpoints exist.
CKPT_FOR_EVAL="$CKPT_DIR"
if [ ! -f "$CKPT_FOR_EVAL/pytorch_model.bin" ]; then
    LATEST_CKPT="$(ls -d "$CKPT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1)"
    if [ -n "$LATEST_CKPT" ] && [ -f "$LATEST_CKPT/pytorch_model.bin" ]; then
        CKPT_FOR_EVAL="$LATEST_CKPT"
    fi
fi
if [ ! -f "$CKPT_FOR_EVAL/pytorch_model.bin" ]; then
    echo "ERROR: No usable checkpoint found under: $CKPT_DIR" >&2
    echo "Hint: expected either '$CKPT_DIR/pytorch_model.bin' or '$CKPT_DIR/checkpoint-*/pytorch_model.bin'." >&2
    exit 1
fi
echo "Using checkpoint for evaluation: $CKPT_FOR_EVAL"

# ========== 串行测试 ==========
echo ""
echo "=========================================="
echo "Starting sequential testing on GPU ${GPU}..."
echo "=========================================="

cd ${BASE_DIR}

python step4_adaptive_step.py \
    --model_type codi \
    --base_model_path ./CODI/pretrained/${MODEL_NAME} \
    --ckpt_dir "$CKPT_FOR_EVAL" \
    --data_name gsm8k \
    --bf16 \
    --baseline_mode adaptive \
    --prj_dim 2048 \
    --max_switch_count 0 \
    --window_e_to_l 0 \
    --window_l_to_e 0 \
    --max_latent 3


    # --baseline_mode random \
    # --random_prob 1 \
