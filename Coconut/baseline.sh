
export CUDA_VISIBLE_DEVICES="1,2,3,4"
torchrun --nnodes 1 --nproc_per_node 2 run.py args/gsm_cot.yaml 
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_cot.yaml 


# 第二个任务（使用不同端口，如 29501）
torchrun --nnodes 1 --nproc_per_node 2 --master_port 29501 run.py args/tmp.yaml


export CUDA_VISIBLE_DEVICES="5,6"
torchrun --nnodes 1 --nproc_per_node 2 --master_port 29501 run.py args/gsm_cot_eval.yaml 


export CUDA_VISIBLE_DEVICES="3,4"
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_cot_eval.yaml 
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut_eval.yaml 