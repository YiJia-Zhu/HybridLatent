set -e

export CUDA_VISIBLE_DEVICES="1"
python step1_collect_entropy_data.py \
    --model_path ./CODI/pretrained/Llama-3.2-1B-Instruct \
    --data_name icot \
    --output_path data/Llama-3.2-1B-Instruct_entropy_data.pt \
    --max_samples 10000 \
    --compute_dynamics


python step2_train_entropy_predictor.py \
    --data_path data/Llama-3.2-1B-Instruct_entropy_data.pt \
    --output_path checkpoints/Llama-3.2-1B-Instruct/entropy_predictor.pt \
    --hidden_dim 2048 \
    --epochs 100

python step3_adaptive_inference.py \
    --model_type coconut \
    --base_model_path ./Coconut/pretrained/gpt2 \
    --model_path ./Coconut/ckpts/gsm-coconut/checkpoint_6 \
    --predictor_path checkpoints/entropy_predictor.pt \
    --input "Question: What is 2+3?" \
    --bf16


python step3_adaptive_inference.py \
    --model_type codi \
    --base_model_path ./CODI/pretrained/Llama-3.2-1B-Instruct \
    --ckpt_dir ./CODI/pretrained/CODI-llama3.2-1b-Instruct \
    --predictor_path checkpoints/entropy_predictor.pt \
    --input "Question: What is 2+3?" \
    --bf16

python step4_adaptive_eval.py \
    --model_type codi \
    --base_model_path ./CODI/pretrained/Llama-3.2-1B-Instruct \
    --ckpt_dir ./CODI/pretrained/CODI-llama3.2-1b-Instruct \
    --predictor_path checkpoints/entropy_predictor.pt \
    --data_name gsm8k \
    --bf16 \
    --batch_size 8 \
    --baseline_mode adaptive \
    --prj_dim 2048

python step4_adaptive_eval.py \
    --model_type codi \
    --base_model_path ./CODI/pretrained/Llama-3.2-1B-Instruct \
    --ckpt_dir ./CODI/codi_llama1b_full/gsm8k_llama1b_latent_baseline/Llama-3.2-1B-Instruct/ep_1/lr_0.0008/seed_11 \
    --predictor_path checkpoints/entropy_predictor.pt \
    --data_name gsm8k \
    --bf16 \
    --batch_size 8 \
    --baseline_mode adaptive \
    --prj_dim 2048

# 全隐层推理
python step4_adaptive_eval.py \
    --model_type codi \
    --base_model_path ./CODI/pretrained/Llama-3.2-1B-Instruct \
    --ckpt_dir ./CODI/pretrained/CODI-llama3.2-1b-Instruct \
    --predictor_path checkpoints/entropy_predictor.pt \
    --data_name gsm8k \
    --bf16 \
    --batch_size 8 \
    --baseline_mode random \
    --random_prob 0.3 \
    --prj_dim 2048 \
    --max_samples 100





# 传统CoT测试
python step0_baseline_eval.py --model_path ./CODI/pretrained/Llama-3.2-1B-Instruct --data_name gsm8k --max_new_tokens 8192