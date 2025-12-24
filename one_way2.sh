#!/bin/bash

# ========== 配置变量 ==========
EXPT_NAME="gsm8k_llama1b_param"
BASE_DIR="/storage/zyj_data/swilatent/SIM-CoT"
SAVE_DIR="/storage/zyj_data/swilatent/SIM-CoT/CODI/ckpts"

# 训练参数（用于构建 CKPT_DIR 路径）
MODEL_NAME="Llama-3.2-1B-Instruct"
NUM_EPOCHS=3
LEARNING_RATE=0.0008
SEED=11

# 自动构建 CKPT_DIR
CKPT_DIR="${SAVE_DIR}/${EXPT_NAME}/${MODEL_NAME}/ep_${NUM_EPOCHS}/lr_${LEARNING_RATE}/seed_${SEED}"

# CKPT_DIR="/storage/zyj_data/swilatent/SIM-CoT/CODI/pretrained/SIM_COT-LLaMA3-CODI-1B"

# GPU 列表（改为2张卡）
GPUS=(6 7)

# ========== 日志配置 ==========
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${SAVE_DIR}/logs_${TIMESTAMP}"
mkdir -p "$LOG_DIR"
cp "$0" "$LOG_DIR/run_script.sh"

TRAIN_LOG="${LOG_DIR}/train_gpu${GPUS[0]}.log"
TEST_LATENT_LOG="${LOG_DIR}/test_latent_gpu${GPUS[0]}.log"
TEST_COT_LOG="${LOG_DIR}/test_cot_gpu${GPUS[0]}.log"
TEST_ADAPTIVE_LOG="${LOG_DIR}/test_adaptive_gpu${GPUS[1]}.log"

echo "Logs will be saved to: $LOG_DIR"

# ========== 信号处理：Ctrl+C 时终止所有子进程 ==========
PIDS=()

cleanup() {
    echo ""
    echo "=========================================="
    echo "Caught interrupt signal! Cleaning up..."
    echo "=========================================="
    
    # 终止所有记录的子进程
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Killing process $pid..."
            kill -TERM "$pid" 2>/dev/null
            # 等待一小段时间后强制杀死
            sleep 1
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null
            fi
        fi
    done
    
    # 同时杀掉所有同组的子进程（更彻底）
    pkill -P $$ 2>/dev/null
    
    echo "All processes terminated."
    echo "Partial logs saved in: $LOG_DIR"
    exit 1
}

# 捕获 SIGINT (Ctrl+C), SIGTERM, SIGHUP
trap cleanup SIGINT SIGTERM SIGHUP

# ========== Training (使用1个GPU) ==========
echo "Starting Training on GPU ${GPUS[0]}..."
echo "Training log: $TRAIN_LOG"
cd ${BASE_DIR}/CODI

CUDA_VISIBLE_DEVICES="6,7" torchrun --nproc_per_node=2 --master_port 29510 train_group.py \
    --output_dir "$SAVE_DIR" \
    --expt_name "$EXPT_NAME" \
    --logging_dir "$SAVE_DIR/logs" \
    --logging_steps 10 \
    --model_name_or_path ./pretrained/${MODEL_NAME} \
    --data_name icot \
    --seed ${SEED} \
    --model_max_length 512 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --bf16 \
    --num_train_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --max_grad_norm 2.0 \
    --use_lora True \
    --lora_r 128 --lora_alpha 32 --lora_init \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 30 \
    --save_safetensors False \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --do_train \
    --report_to tensorboard \
    --num_latent 10 \
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
    --include_last_cot True\
    --use_decoder \
    --k_step_latent_num 1 \
    --ce_loss_factor 1 --ref_loss_factor 1 --explain_loss_factor 0.1 --align_loss_factor 20 --distill_loss_factor 20 \
    2>&1 | tee "$TRAIN_LOG"


   


    # --restore_from ./ckpts/sft_cot_llama1b/Llama-3.2-1B-Instruct/ep_3/lr_0.0008/seed_11/pytorch_model.bin \

    # --restore_from ./ckpts/sft_cot_llama1b/Llama-3.2-1B-Instruct/ep_3/lr_0.0008/seed_11/pytorch_model.bin \
    # --restore_from ./pretrained/CODI-llama3.2-1b-Instruct/pytorch_model.bin \

    # --hybrid_cot_only_ratio 1 \

# 检查训练是否成功
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "Training failed! Check log: $TRAIN_LOG"
    exit 1
fi
echo "Training completed!"

# # ========== Test Latent (顺序执行) ==========
# echo ""
# echo "=========================================="
# echo "Starting Test Latent on GPU ${GPUS[0]}..."
# echo "=========================================="
# echo "Test Latent log: $TEST_LATENT_LOG"

# CUDA_VISIBLE_DEVICES="${GPUS[0]}" python test.py \
#     --data_name "gsm8k" \
#     --output_dir "$SAVE_DIR" \
#     --model_name_or_path ./pretrained/${MODEL_NAME} \
#     --seed ${SEED} \
#     --model_max_length 512 \
#     --bf16 \
#     --lora_r 128 --lora_alpha 32 --lora_init \
#     --batch_size 128 \
#     --greedy True \
#     --num_latent 6 \
#     --use_prj True \
#     --prj_dim 2048 \
#     --prj_no_ln False \
#     --prj_dropout 0.0 \
#     --inf_latent_iterations 6 \
#     --inf_num_iterations 1 \
#     --remove_eos True \
#     --use_lora True \
#     --ckpt_dir "$CKPT_DIR" \
#     2>&1 | tee "$TEST_LATENT_LOG"

# # 检查 Test Latent 是否成功
# if [ ${PIPESTATUS[0]} -ne 0 ]; then
#     echo "Test Latent failed! Check log: $TEST_LATENT_LOG"
#     exit 1
# fi
# echo "Test Latent completed!"

# ========== 并行测试 (2个GPU并行运行 step4_adaptive_eval) ==========
echo ""
echo "=========================================="
echo "Starting parallel testing (step4_adaptive_eval)..."
echo "=========================================="

cd ${BASE_DIR}

# Test CoT (GPU 0)
echo "Test CoT on GPU ${GPUS[0]}... Log: $TEST_COT_LOG"
CUDA_VISIBLE_DEVICES="${GPUS[0]}" python step4_adaptive_eval.py \
    --model_type codi \
    --base_model_path ./CODI/pretrained/${MODEL_NAME} \
    --ckpt_dir "$CKPT_DIR" \
    --data_name gsm8k \
    --bf16 \
    --batch_size 8 \
    --baseline_mode random \
    --random_prob 1 \
    --prj_dim 2048 \
    --max_switch_count 5 \
    --window_e_to_l 256 \
    --window_l_to_e 0 \
    --max_samples 50 \
    2>&1 | tee "$TEST_COT_LOG" &
    # --max_samples 50 \

PID1=$!
PIDS+=($PID1)

# Test Adaptive (GPU 1)
echo "Test Adaptive on GPU ${GPUS[1]}... Log: $TEST_ADAPTIVE_LOG"
CUDA_VISIBLE_DEVICES="${GPUS[1]}" python step4_adaotive_k.py \
    --model_type codi \
    --base_model_path ./CODI/pretrained/${MODEL_NAME} \
    --ckpt_dir "$CKPT_DIR" \
    --data_name gsm8k \
    --bf16 \
    --baseline_mode adaptive \
    --prj_dim 2048 \
    --max_switch_count 5 \
    --window_e_to_l 0 \
    --window_l_to_e 0 \
    --max_latent_steps 6 \
    --k_step_latent_num 3 \
    --max_samples 50 \
    2>&1 | tee "$TEST_ADAPTIVE_LOG" &
    # --max_samples 50 \

PID2=$!
PIDS+=($PID2)

# 等待所有并行任务完成
echo ""
echo "Waiting for all tests to complete..."
echo "Running PIDs: ${PIDS[*]}"

wait $PID1
STATUS1=$?
echo "Test CoT completed! (exit code: $STATUS1)"

wait $PID2
STATUS2=$?
echo "Test Adaptive completed! (exit code: $STATUS2)"

echo ""
echo "=========================================="
echo "All experiments finished!"
echo "=========================================="
echo ""
echo "Log files saved in: $LOG_DIR"
echo "  - Training:      $TRAIN_LOG"
echo "  - Test Latent:   $TEST_LATENT_LOG"
echo "  - Test CoT:      $TEST_COT_LOG"
echo "  - Test Adaptive: $TEST_ADAPTIVE_LOG"