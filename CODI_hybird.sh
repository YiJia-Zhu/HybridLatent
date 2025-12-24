set -e

export CUDA_VISIBLE_DEVICES="7"
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




export CUDA_VISIBLE_DEVICES="6"
python step4_adaptive_step.py \
    --model_type codi \
    --base_model_path ./CODI/pretrained/Llama-3.2-1B-Instruct \
    --ckpt_dir /storage/zyj_data/swilatent/SIM-CoT/CODI/ckpts/gsm8k_llama1b_param/Llama-3.2-1B-Instruct/ep_3/lr_0.0008/seed_11/checkpoint-2000 \
    --data_name gsm8k \
    --bf16 \
    --baseline_mode adaptive \
    --prj_dim 2048 \
    --max_switch_count 5 \
    --window_e_to_l 0 \
    --window_l_to_e 0 \
    --max_latent_steps 1 \
    --max_samples 50

export CUDA_VISIBLE_DEVICES="6"
python step4_adaptive_step.py \
    --model_type codi \
    --base_model_path ./CODI/pretrained/Llama-3.2-1B-Instruct \
    --ckpt_dir /storage/zyj_data/swilatent/SIM-CoT/CODI/ckpts/gsm8k_llama1b_param/Llama-3.2-1B-Instruct/ep_3/lr_0.0008/seed_11/checkpoint-2000 \
    --data_name gsm8k \
    --bf16 \
    --baseline_mode adaptive \
    --random_prob 0.5 \
    --prj_dim 2048 \
    --max_switch_count 5 \
    --window_e_to_l 0 \
    --window_l_to_e 0 \
    --max_latent_steps 3 \
    --max_samples 50

export CUDA_VISIBLE_DEVICES="7"
python step4_adaptive_k.py \
    --model_type codi \
    --base_model_path ./CODI/pretrained/Llama-3.2-1B-Instruct \
    --ckpt_dir ./CODI/pretrained/CODI-llama3.2-1b-Instruct \
    --data_name gsm8k \
    --bf16 \
    --baseline_mode alternating \
    --random_prob 1 \
    --prj_dim 2048 \
    --max_switch_count 6 \
    --window_e_to_l 0 \
    --window_l_to_e 0 \
    --max_latent_steps 9 \
    --k_step_latent_num 3 \
    --max_samples 10

python step4_adaptive_k.py \
    --model_type codi \
    --base_model_path ./CODI/pretrained/Llama-3.2-1B-Instruct \
    --ckpt_dir ./CODI/pretrained/CODI-llama3.2-1b-Instruct \
    --data_name gsm8k \
    --bf16 \
    --baseline_mode random \
    --random_prob 0.5 \
    --prj_dim 2048 \
    --max_switch_count 6 \
    --window_e_to_l 0 \
    --window_l_to_e 0 \
    --max_latent_steps 9 \
    --k_step_latent_num 3 \
    --max_samples 10


export CUDA_VISIBLE_DEVICES="0"

python step4_adaptive_eval.py \
    --model_type codi \
    --base_model_path ./CODI/pretrained/Llama-3.2-1B-Instruct \
    --ckpt_dir /storage/zyj_data/swilatent/SIM-CoT/CODI/ckpts/gsm8k_llama1b_adaptive_random/Llama-3.2-1B-Instruct/ep_5/lr_0.0008/seed_11/checkpoint-5000 \
    --predictor_path checkpoints/entropy_predictor.pt \
    --data_name gsm8k \
    --bf16 \
    --batch_size 8 \
    --baseline_mode random \
    --random_prob 1 \
    --prj_dim 2048 \
    --max_switch_count 5 \
    --window_e_to_l 5 \
    --window_l_to_e 0 \
    --max_latent_steps 3

python step4_adaptive_eval.py \
    --model_type codi \
    --base_model_path ./CODI/pretrained/Llama-3.2-1B-Instruct \
    --ckpt_dir ./CODI/pretrained/Llama-3.2-1B-Instruct \
    --predictor_path checkpoints/entropy_predictor.pt \
    --data_name gsm8k \
    --bf16 \
    --batch_size 8 \
    --baseline_mode random \
    --random_prob 1 \
    --prj_dim 2048 \
    --max_switch_count 5 \
    --window_e_to_l 5 \
    --window_l_to_e 0 

python step4_adaptive_eval.py \
    --model_type codi \
    --base_model_path ./CODI/pretrained/Llama-3.2-1B-Instruct \
    --ckpt_dir ./CODI/pretrained/CODI-llama3.2-1b-Instruct \
    --predictor_path checkpoints/entropy_predictor.pt \
    --data_name gsm8k \
    --bf16 \
    --batch_size 8 \
    --baseline_mode random \
    --random_prob 1 \
    --prj_dim 2048 \
    --max_switch_count 5 \
    --window_e_to_l 5 \
    --window_l_to_e 0 

python step4_adaptive_eval.py \
    --model_type codi \
    --base_model_path ./CODI/pretrained/Llama-3.2-1B-Instruct \
    --ckpt_dir ./CODI/pretrained/CODI-llama3.2-1b-Instruct \
    --predictor_path checkpoints/entropy_predictor.pt \
    --data_name gsm8k \
    --bf16 \
    --batch_size 8 \
    --baseline_mode adaptive \
    --prj_dim 2048 \
    --max_switch_count 5 \
    --window_e_to_l 5 \
    --window_l_to_e 0 


python step4_adaptive_eval.py \
    --model_type codi \
    --base_model_path ./CODI/pretrained/Llama-3.2-8B-Instruct \
    --ckpt_dir ./CODI/pretrained/SIM_COT-LLaMA3-CODI-8B \
    --predictor_path checkpoints/entropy_predictor.pt \
    --data_name gsm8k \
    --bf16 \
    --batch_size 8 \
    --baseline_mode adaptive \
    --prj_dim 4096 \
    --max_switch_count 5 \
    --window_e_to_l 5 \
    --window_l_to_e 0 

python step4_adaptive_eval.py \
    --model_type codi \
    --base_model_path ./CODI/pretrained/Llama-3.2-8B-Instruct \
    --ckpt_dir ./CODI/pretrained/SIM_COT-LLaMA3-CODI-8B \
    --predictor_path checkpoints/entropy_predictor.pt \
    --data_name gsm8k \
    --bf16 \
    --batch_size 8 \
    --baseline_mode random \
    --random_prob 1 \
    --prj_dim 4096 \
    --max_switch_count 5 \
    --window_e_to_l 5 \
    --window_l_to_e 0 

cd CODI
bash ./src


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