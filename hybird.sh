set -e

export CUDA_VISIBLE_DEVICES="6,7"
python step1_collect_entropy_data.py \
    --model_path ./Coconut/pretrained/gpt2 \
    --data_name icot \
    --output_path data/entropy_data.pt \
    --max_samples 10000 \
    --compute_dynamics


python step2_train_entropy_predictor.py \
    --data_path data/entropy_data.pt \
    --output_path checkpoints/entropy_predictor.pt \
    --hidden_dim 768 \
    --epochs 10

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
    --batch_size 128 \
    --baseline_mode adaptive

python step4_adaptive_eval.py \
    --model_type codi \
    --base_model_path ./CODI/pretrained/Llama-3.2-1B-Instruct \
    --ckpt_dir ./CODI/pretrained/CODI-llama3.2-1b-Instruct \
    --predictor_path checkpoints/entropy_predictor.pt \
    --data_name gsm8k \
    --bf16 \
    --batch_size 128 \
    --baseline_mode random \
    --random_prob 0

python step4_adaptive_eval.py \
    --model_type codi \
    --base_model_path ./CODI/pretrained/Llama-3.2-1B-Instruct \
    --ckpt_dir ./CODI/pretrained/Llama-3.2-1B-Instruct \
    --predictor_path checkpoints/entropy_predictor.pt \
    --data_name gsm8k \
    --bf16 \
    --batch_size 128 \
    --baseline_mode all_normal
