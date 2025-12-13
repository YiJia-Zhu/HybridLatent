export CUDA_VISIBLE_DEVICES="6"


python step4_adaptive_eval.py \
    --model_type coconut \
    --base_model_path ./Coconut/pretrained/gpt2 \
    --checkpoint_path /storage/zyj_data/swilatent/HybridLatentReasoning/checkpoints/coconut_Reproduction/stage_1_training_ck/checkpoint_12 \
    --predictor_path checkpoints/entropy_predictor.pt \
    --data_name gsm8k \
    --baseline_mode adaptive \
    --max_switch_count 5 \
    --window_e_to_l 5 \
    --window_l_to_e 0


python step4_adaptive_eval.py \
    --model_type coconut \
    --base_model_path ./Coconut/pretrained/gpt2 \
    --checkpoint_path ./Coconut/ckpts/gsm-coconut-gpt2/checkpoint_6 \
    --predictor_path checkpoints/entropy_predictor.pt \
    --data_name gsm8k \
    --baseline_mode random \
    --random_prob 1 \
    --max_switch_count 5 \
    --window_e_to_l 5 \
    --window_l_to_e 0