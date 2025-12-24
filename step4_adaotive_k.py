"""
阶段3: 半显半隐式推理 (SwiReasoning风格控制) - 数据集评估版本

扩展功能:
- 支持多种数据集 (gsm8k, gsm-hard, multi-arith, svamp, commonsense)
- 计算测试精度
- 统计 token 分布 (Normal mode vs Latent mode)
- 记录两种模式的平均时间

使用方法:
    # CODI 数据集评估
    python step4_adaptive_eval.py \
        --model_type codi \
        --base_model_path ./CODI/pretrained/Llama-3.2-1B-Instruct \
        --ckpt_dir ./CODI/pretrained/CODI-llama3.2-1b-Instruct \
        --predictor_path checkpoints/entropy_predictor.pt \
        --data_name gsm8k \
        --bf16 \
        --batch_size 8
    
    # Coconut 数据集评估
    python step4_adaptive_eval.py \
        --model_type coconut \
        --base_model_path ./Coconut/pretrained/gpt2 \
        --checkpoint_path ./Coconut/ckpts/gsm-coconut-gpt2/checkpoint_6 \
        --predictor_path checkpoints/entropy_predictor.pt \
        --data_name gsm8k \
        --baseline_mode adaptive \
        --max_switch_count 5 \
        --window_e_to_l 5 \
        --window_l_to_e 0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import json
import re
import math
import time
import logging
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from datasets import load_dataset, concatenate_datasets
from safetensors.torch import load_file
import glob

# 设置路径
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'Coconut'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'CODI'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'CODI', 'src'))

# 导入依赖
from safetensors.torch import load_file
from peft import LoraConfig, TaskType

# 导入 Coconut
from coconut import Coconut

# 导入 CODI
from model_adaptive import CODI, ModelArguments, TrainingArguments

from Swicontrol import EntropyPredictor, compute_entropy_swir, TimeStats, SwiRModeState, SwiRController, AdaptiveController



# ============================================================================
# 数据集加载 (参考 CODI/test.py)
# ============================================================================

def load_eval_dataset(data_name: str) -> Tuple[List[str], List, str, str]:
    """
    加载评估数据集
    返回: (questions, answers, question_key, answer_key)
    """
    logging.info(f"Loading dataset: {data_name}")
    
    if data_name == "gsm8k":
        dataset = load_dataset("gsm8k", "main")
        test_set = dataset['test']
        question_name = "question"
        answer_name = "answer"
    elif data_name == "gsm-hard":
        dataset = load_dataset("juyoung-trl/gsm-hard")
        test_set = dataset['train']
        question_name = "instruction"
        answer_name = "response"
    elif data_name == "multi-arith":
        dataset = load_dataset("ChilleD/MultiArith")
        test_set = dataset['test']
        question_name = "question"
        answer_name = "final_ans"
    elif data_name == "svamp":
        dataset = load_dataset("ChilleD/SVAMP")
        test_set = concatenate_datasets([dataset["train"], dataset["test"]])
        question_name = "question_concat"
        answer_name = "Answer"
    elif data_name == "commonsense":
        dataset = load_dataset("zen-E/CommonsenseQA-GPT4omini")
        test_set = dataset['validation']
        question_name = "question"
        answer_name = "answer"
    else:
        raise ValueError(f"Unknown dataset: {data_name}")
    
    questions = [f"{example[question_name].strip().replace('  ', ' ')}" for example in test_set]
    
    answers = []
    for example in test_set:
        ans_raw = example[answer_name]
        
        if isinstance(ans_raw, bool):
            answers.append(ans_raw)
            continue
        if ans_raw in ["True", "False"]:
            answers.append(ans_raw == "True")
            continue
        if ans_raw in "ABCDE":
            answers.append(ans_raw)
            continue
        
        if "####" in str(ans_raw):
            ans = str(ans_raw).split('####')[-1]
        else:
            ans = str(ans_raw)
        ans = ans.replace(',', '')
        try:
            ans = float(ans)
        except ValueError:
            ans = float("inf")
        answers.append(ans)
    
    logging.info(f"Loaded {len(questions)} examples from {data_name}")
    return questions, answers, question_name, answer_name


def extract_answer_number(sentence: str, data_name: str = "gsm8k") -> float:
    """从模型输出中提取答案"""
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    
    if not pred:
        if "commonsense" in data_name:
            pred_text = sentence.split("The answer is:")[-1].strip()
            if pred_text and pred_text[0] in "ABCDE":
                return pred_text[0]
        elif "strategy" in data_name or "prontoqa" in data_name.lower():
            if "True" in sentence:
                return True
            elif "False" in sentence:
                return False
        return float('inf')
    
    return float(pred[-1])


def compute_accuracy(gold: List, pred: List) -> float:
    """计算精度"""
    acc = 0.0
    for p, g in zip(pred, gold):
        if isinstance(p, list):
            if g in p:
                acc += 1
        else:
            if p == g:
                acc += 1
    return acc / len(gold) if gold else 0.0


# ============================================================================
# CODI 模型加载
# ============================================================================

def load_codi_model(
    base_model_path: str,
    ckpt_dir: str,
    device: torch.device,
    num_latent: int = 6,
    use_bf16: bool = True,
    lora_r: int = 128,
    lora_alpha: int = 32,
    prj_dim: int = 768,
):
    """加载 CODI 模型"""
    print(f"Loading CODI model...")
    print(f"  Base model: {base_model_path}")
    print(f"  Checkpoint dir: {ckpt_dir}")
    
    task_type = TaskType.CAUSAL_LM
    if any(name in base_model_path.lower() for name in ["llama", "mistral", "falcon", "qwen"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    elif any(name in base_model_path.lower() for name in ["phi"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
    elif any(name in base_model_path.lower() for name in ["gpt2"]):
        target_modules = ["c_attn", "c_proj", "c_fc"]
    else:
        raise ValueError(f"Unsupported model: {base_model_path}")
    
    lora_config = LoraConfig(
        task_type=task_type,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=target_modules,
        init_lora_weights=True,
    )
    
    model_args = ModelArguments(
        model_name_or_path=base_model_path,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_init=True,
        train=False,
    )
    
    training_args = TrainingArguments(
        output_dir="./tmp",
        num_latent=num_latent,
        use_lora=True,
        bf16=use_bf16,
        use_prj=True,
        prj_dim=prj_dim,
    )
    
    model = CODI(model_args, training_args, lora_config)
    
    if ckpt_dir and os.path.exists(ckpt_dir):
        shard_files = sorted(glob.glob(os.path.join(ckpt_dir, "model-*.safetensors")))
        
        if shard_files:
            print(f"Loading {len(shard_files)} sharded safetensors files...")
            state_dict = {}
            for shard_file in shard_files:
                state_dict.update(load_file(shard_file))
        elif os.path.exists(os.path.join(ckpt_dir, "model.safetensors")):
            state_dict = load_file(os.path.join(ckpt_dir, "model.safetensors"))
        else:
            state_dict = torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"), map_location='cpu')
        
        model.load_state_dict(state_dict, strict=False)
        model.codi.tie_weights()
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model_path,
        padding_side="left",
        use_fast=False,
    )
    
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
    
    model = model.to(device)
    if use_bf16:
        model = model.to(torch.bfloat16)
    model.eval()
    
    return model, tokenizer


# ============================================================================
# Coconut 模型加载 (修复版)
# ============================================================================

def load_coconut_model(
    base_model_path: str,
    checkpoint_path: str,
    device: torch.device,
    use_bf16: bool = False,
):
    """
    加载 Coconut 模型 (修复版)
    
    参考官方 Coconut 的 run.py 配置:
    - mode: coconut_baseline
    - 单文件 checkpoint 加载
    
    关键：需要根据 checkpoint 类型决定加载顺序
    - 如果是基础模型权重（无 base_causallm 前缀）：先加载，再 resize
    - 如果是 Coconut 完整权重（有 base_causallm 前缀）：先 resize，再加载
    """
    print(f"Loading Coconut model...")
    print(f"  Base model: {base_model_path}")
    print(f"  Checkpoint: {checkpoint_path}")
    
    # 1. 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    
    # 2. 加载 tokenizer 并添加特殊 token
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    
    # 3. 预加载 checkpoint 以检查类型
    saved_weights = None
    has_coconut_keys = False
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"  Loading checkpoint from: {checkpoint_path}")
        
        # 判断是文件还是目录
        if os.path.isfile(checkpoint_path):
            saved_weights = torch.load(checkpoint_path, map_location="cpu")
        else:
            possible_files = [
                os.path.join(checkpoint_path, "pytorch_model.bin"),
                os.path.join(checkpoint_path, "model.safetensors"),
            ]
            for f in possible_files:
                if os.path.exists(f):
                    if f.endswith(".safetensors"):
                        saved_weights = load_file(f)
                    else:
                        saved_weights = torch.load(f, map_location="cpu")
                    break
            
            if saved_weights is None:
                raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")
        
        # 检查是否是 Coconut 模型的 checkpoint
        has_coconut_keys = any(k.startswith("base_causallm") for k in saved_weights.keys())
        print(f"  Checkpoint has Coconut keys: {has_coconut_keys}")
        
        # 打印一些 keys 用于调试
        sample_keys = list(saved_weights.keys())[:5]
        print(f"  Sample keys: {sample_keys}")
    
    # 4. 根据 checkpoint 类型决定加载顺序
    if saved_weights is not None and not has_coconut_keys:
        # 基础模型权重：先加载到基础模型，再 resize
        print(f"  Loading base model weights first, then resize...")
        missing, unexpected = base_model.load_state_dict(saved_weights, strict=False)
        print(f"  Loaded base model checkpoint. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        
        # 然后 resize token embeddings
        base_model.resize_token_embeddings(len(tokenizer))
        embeddings = base_model.get_input_embeddings()
        target_id = tokenizer.convert_tokens_to_ids("<<")
        
        # 初始化新 token 的 embeddings
        for token_id in [latent_id, start_id, end_id]:
            embeddings.weight.data[token_id] = embeddings.weight.data[target_id].clone()
            if hasattr(base_model, 'lm_head'):
                base_model.lm_head.weight.data[token_id] = base_model.lm_head.weight.data[target_id].clone()
        
        # 创建 Coconut 模型
        model = Coconut(base_model, latent_id, start_id, end_id, tokenizer.eos_token_id)
        
    else:
        # Coconut 完整权重或无 checkpoint：先 resize，再加载
        print(f"  Resize first, then load Coconut weights...")
        
        # 先 resize token embeddings
        base_model.resize_token_embeddings(len(tokenizer))
        embeddings = base_model.get_input_embeddings()
        target_id = tokenizer.convert_tokens_to_ids("<<")
        
        # 初始化新 token 的 embeddings
        for token_id in [latent_id, start_id, end_id]:
            embeddings.weight.data[token_id] = embeddings.weight.data[target_id].clone()
            if hasattr(base_model, 'lm_head'):
                base_model.lm_head.weight.data[token_id] = base_model.lm_head.weight.data[target_id].clone()
        
        # 创建 Coconut 模型
        model = Coconut(base_model, latent_id, start_id, end_id, tokenizer.eos_token_id)
        
        # 加载 Coconut 完整权重
        if saved_weights is not None:
            missing, unexpected = model.load_state_dict(saved_weights, strict=False)
            print(f"  Loaded Coconut checkpoint. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            if missing:
                print(f"  Missing keys sample: {missing[:5]}")
            if unexpected:
                print(f"  Unexpected keys sample: {unexpected[:5]}")
    
    # 5. 移动到设备并设置精度
    model = model.to(device)
    if use_bf16:
        model = model.to(torch.bfloat16)
    model.eval()
    
    print(f"  Model loaded successfully!")
    print(f"  - latent_id: {latent_id}")
    print(f"  - start_id: {start_id}")
    print(f"  - end_id: {end_id}")
    print(f"  - eos_token_id: {tokenizer.eos_token_id}")
    
    return model, tokenizer, latent_id, start_id, end_id


# ============================================================================
# CODI 自适应生成 (带时间统计)
# ============================================================================


def codi_adaptive_generate_with_timing(
    model: CODI,
    tokenizer,
    controller: AdaptiveController,
    input_text: str,
    max_new_tokens: int = 256,
    device: torch.device = None,
    verbose: bool = False,
    use_prj: bool = True,
    greedy: bool = True,
    answer_triggers: List[str] = None,
    max_latent_steps: int = 6,
    random_prob: float = 0.5,
    k_step_latent_num: int = 1,
    # ===== 新增参数 =====
    step_start_ids: Tuple[int, ...] = (2501, 1134),  # << 的 token IDs
    step_end_id: int = 2511,  # >> 的 token ID
) -> Tuple[Dict, TimeStats]:
    """
    修正版：保证每个 Latent 区域 <bot>LLL<eot> 结构完整
    新增：只在 step 边界（>> 之后）才允许从 normal 切换到 latent
    """
    
    if answer_triggers is None:
        answer_triggers = ["The answer is", "#### ", "the answer is", "Answer:"]
    
    if device is None:
        device = next(model.parameters()).device
    
    time_stats = TimeStats()
    batch = tokenizer(input_text, return_tensors="pt", padding=True)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    batch_size = input_ids.shape[0]
    
    controller.init_state(batch_size, device)
    
    generated_tokens = []
    token_modes = []
    token_entropies = []
    switch_events = []
    
    generated_text_buffer = [""] * batch_size
    consecutive_latent_count = 0
    current_latent_group_count = 0  # ← 新增：当前latent group中的计数

    
    # ========== 关键状态：追踪是否在 Latent 区域内 ==========
    in_latent_region = False
    
    # ========== 新增：追踪是否在 step 内部 (<<...>> 之间) ==========
    in_step = False
    
    def get_embd(token_ids):
        return model.get_embd(model.codi, model.model_name)(token_ids)
    
    # 预计算 bot 和 eot embeddings
    bot_emb = get_embd(torch.tensor([model.bot_id], device=device)).unsqueeze(0).expand(batch_size, -1, -1)
    eot_emb = get_embd(torch.tensor([model.eot_id], device=device)).unsqueeze(0).expand(batch_size, -1, -1)
    
    # ========== 辅助函数：插入 <bot> ==========
    def insert_bot(past_kv):
        nonlocal in_latent_region
        with torch.no_grad():
            out = model.codi(
                inputs_embeds=bot_emb,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_kv,
            )
        in_latent_region = True
        if verbose:
            print(f"  [INSERT] <bot> token")
        return out.past_key_values, out.hidden_states[-1][:, -1, :]
    
    # ========== 辅助函数：插入 <eot> ==========
    def insert_eot(past_kv):
        nonlocal in_latent_region
        with torch.no_grad():
            out = model.codi(
                inputs_embeds=eot_emb,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_kv,
            )
        in_latent_region = False
        if verbose:
            print(f"  [INSERT] <eot> token")
        return out.past_key_values, out.hidden_states[-1][:, -1, :], out.logits[:, -1, :]
    
    # ========== 1. Encode 问题 ==========
    past_key_values = None
    with torch.no_grad():
        outputs = model.codi(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
        )
        past_key_values = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]
    
    logits = outputs.logits[:, -1, :model.codi.config.vocab_size - 1]
    
    # ========== 2. 初始模式判断 ==========
    # 初始时允许切换（还没开始生成 step）
    at_step_boundary = torch.ones(batch_size, dtype=torch.bool, device=device)
    mode, to_normal, to_soft, cur_entropy = controller.step(last_hidden, 0, logits, at_step_boundary=at_step_boundary)
    current_mode = "normal" if mode[0].item() == 1 else "soft"
    
    if verbose:
        print(f"[Init] Initial mode: {current_mode}, entropy: {cur_entropy[0].item():.4f}")
    
    # 如果初始就是 soft 模式，先插入 <bot>
    if current_mode == "soft":
        past_key_values, last_hidden = insert_bot(past_key_values)
    
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    pred_tokens = [[] for _ in range(batch_size)]
    
    # ========== 3. 生成循环 ==========
    for step in range(max_new_tokens):
        start_time = time.perf_counter()
        
        # ========== 3.1 检查是否需要强制退出 Latent 区域 ==========
        force_exit_latent = False
        if current_mode == "soft" and consecutive_latent_count >= max_latent_steps:
            # ← 修改：达到max_latent_steps直接强制退出，不检查k的倍数
            force_exit_latent = True
            if verbose:
                print(f"[Step {step}] Max latent steps ({max_latent_steps}) reached, forcing exit")

        
        # 如果强制退出，先插入 <eot>
        if force_exit_latent and in_latent_region:
            past_key_values, last_hidden, logits = insert_eot(past_key_values)
            current_mode = "normal"
            controller.state.mode[0] = 1
            controller.state.mode_stay_steps[0] = 0
            controller.state.ref_entropy[0] = cur_entropy[0]
            consecutive_latent_count = 0
            current_latent_group_count = 0
            switch_events.append((step, "soft->normal (forced)", cur_entropy[0].item()))
        
        # ========== 3.2 根据当前模式执行推理 ==========
        with torch.no_grad():
            if current_mode == "soft":
                # Latent 模式：使用 hidden state 作为输入
                if use_prj and hasattr(model, 'prj'):
                    input_embd = model.prj(last_hidden.unsqueeze(1))
                else:
                    input_embd = last_hidden.unsqueeze(1)
            else:
                # Normal 模式：使用 token embedding
                if greedy:
                    next_token_ids = torch.argmax(logits, dim=-1)
                else:
                    probs = F.softmax(logits / 0.1, dim=-1)
                    next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
                input_embd = get_embd(next_token_ids).unsqueeze(1)
            
            out = model.codi(
                inputs_embeds=input_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            past_key_values = out.past_key_values
            last_hidden = out.hidden_states[-1][:, -1, :]
        
        step_time = time.perf_counter() - start_time
        logits = out.logits[:, -1, :model.codi.config.vocab_size - 1]
        
        # ========== 3.3 记录时间统计 ==========
        if current_mode == "normal":
            time_stats.add_normal(step_time)
        else:
            time_stats.add_soft(step_time)
        
        # ========== 3.4 处理 Normal 模式的 token 生成 ==========
        if current_mode == "normal":
            consecutive_latent_count = 0
            current_latent_group_count = 0  # ← 重置group计数

            
            # ===== 新增：检测 step 边界 =====
            token_id = next_token_ids[0].item()
            if token_id in step_start_ids:
                in_step = True
                if verbose:
                    print(f"[Step {step}] Detected step start '<<', in_step=True")
            elif token_id == step_end_id:
                in_step = False
                if verbose:
                    print(f"[Step {step}] Detected step end '>>', in_step=False")
            # ================================
            
            for b in range(batch_size):
                if not finished[b]:
                    token_text = tokenizer.decode([next_token_ids[b].item()])
                    generated_text_buffer[b] += token_text
                    
                    # 检查答案触发
                    if not controller.state.answer_locked[b]:
                        for trigger in answer_triggers:
                            if trigger in generated_text_buffer[b]:
                                controller.state.answer_locked[b] = True
                                controller.state.mode[b] = 1
                                if verbose:
                                    print(f"[Step {step}] Answer trigger '{trigger}' detected")
                                break
                    
                    pred_tokens[b].append(next_token_ids[b].item())
                    if next_token_ids[b] == tokenizer.eos_token_id:
                        finished[b] = True
            
            generated_tokens.append({
                "token_id": next_token_ids[0].item(),
                "token_text": tokenizer.decode([next_token_ids[0].item()]),
                "mode": current_mode,
                "entropy": cur_entropy[0].item(),
                "answer_locked": controller.state.answer_locked[0].item(),
                "in_step": in_step,  # 新增：记录是否在 step 内
            })
        else:
            # Latent 模式
            consecutive_latent_count += 1
            current_latent_group_count += 1  # ← 增加group计数

            generated_tokens.append({
                "token_id": -1,
                "token_text": "<latent>",
                "mode": current_mode,
                "entropy": cur_entropy[0].item(),
                "answer_locked": controller.state.answer_locked[0].item(),
                "in_step": in_step,  # 新增
            })
        
        token_modes.append(current_mode)
        token_entropies.append(cur_entropy[0].item())
        
        # ========== 3.5 检查是否结束 ==========
        if finished.all():
            # 如果在 Latent 区域内结束，插入 <eot>
            if in_latent_region:
                past_key_values, last_hidden, logits = insert_eot(past_key_values)
                switch_events.append((step, "soft->normal (end)", cur_entropy[0].item()))
            break
        
        # ========== 3.6 更新模式（基于当前输出） ==========
        end_token_mask = None
        if current_mode == "normal":
            end_token_mask = (next_token_ids == tokenizer.eos_token_id)
        
        old_mode = current_mode
        
        # ===== 新增：计算 at_step_boundary =====
        # 只有在 step 外（in_step=False）才允许从 normal 切换到 latent
        at_step_boundary = torch.tensor([not in_step], dtype=torch.bool, device=device)
        if batch_size > 1:
            at_step_boundary = at_step_boundary.expand(batch_size)
        # =======================================
        
        mode, to_normal, to_soft, cur_entropy = controller.step(
            last_hidden, step + 1, logits, end_token_mask, at_step_boundary
        )
        
        # ========== 3.7 处理模式切换 ==========
        can_exit_latent = (current_latent_group_count % k_step_latent_num == 0) or (consecutive_latent_count >= max_latent_steps)

        # soft -> normal: 需要插入 <eot> 且必须满足退出条件
        if to_normal[0].item() and in_latent_region:
            if can_exit_latent:
                past_key_values, last_hidden, logits = insert_eot(past_key_values)
                current_latent_group_count = 0 
                consecutive_latent_count = 0  # ← 添加这一行！
                switch_events.append((step, "soft->normal", cur_entropy[0].item()))
                if verbose:
                    print(f"[Step {step}] soft->normal, entropy: {cur_entropy[0].item():.4f}")
            else:
                # ← 未达到k的倍数且未达到max_latent_steps，强制保持soft模式
                mode[0] = 0
                controller.state.mode[0] = 0
                if verbose:
                    print(f"[Step {step}] soft->normal blocked: latent_count={current_latent_group_count}, need multiple of {k_step_latent_num} or reach max_latent_steps")


        
        # normal -> soft: 需要插入 <bot>
        if to_soft[0].item() and not in_latent_region:
            past_key_values, last_hidden = insert_bot(past_key_values)
            current_latent_group_count = 0  # ← 开始新的latent group
            switch_events.append((step, "normal->soft", cur_entropy[0].item()))
            if verbose:
                print(f"[Step {step}] normal->soft, entropy: {cur_entropy[0].item():.4f}, at_step_boundary: {at_step_boundary[0].item()}")
        
        # ========== 3.8 决定下一步模式 ==========
        if controller.state.answer_locked[0]:
            next_mode = "normal"
        else:
            if current_mode == "soft" and current_latent_group_count % k_step_latent_num != 0:
                next_mode = "soft"
            else:
                next_mode = "normal" if mode[0].item() == 1 else "soft"

        
        current_mode = next_mode
    
    # ========== 4. 循环结束后，确保 Latent 区域闭合 ==========
    if in_latent_region:
        past_key_values, last_hidden, logits = insert_eot(past_key_values)
        switch_events.append((max_new_tokens, "soft->normal (final)", cur_entropy[0].item()))
        if verbose:
            print(f"[Final] Closing latent region with <eot>")
    
    # ========== 5. 生成结果 ==========
    output_text = tokenizer.decode(pred_tokens[0], skip_special_tokens=True)
    
    mode_counts = {"normal": 0, "soft": 0}
    for m in token_modes:
        mode_counts[m] += 1
    
    return {
        "input": input_text,
        "output": output_text,
        "generated_tokens": generated_tokens,
        "mode_distribution": mode_counts,
        "switch_events": switch_events,
        "avg_entropy": sum(token_entropies) / len(token_entropies) if token_entropies else 0,
        "total_steps": len(generated_tokens),
        "total_switches": len(switch_events),
    }, time_stats

# ============================================================================
# Coconut 自适应生成 (修复版 - 带时间统计)
# ============================================================================

def coconut_adaptive_generate_with_timing(
    model: Coconut,
    tokenizer,
    controller: AdaptiveController,
    input_text: str,
    max_new_tokens: int = 256,
    max_latent_steps: int = 6,
    device: torch.device = None,
    verbose: bool = False,
    answer_triggers: List[str] = None,
) -> Tuple[Dict, TimeStats]:
    """
    Coconut 自适应生成 (修复版)
    
    参考官方 Coconut 的 generate 方法实现
    """
    if answer_triggers is None:
        answer_triggers = ["####", "The answer is", "the answer is"]
    
    if device is None:
        device = next(model.parameters()).device
    
    time_stats = TimeStats()
    
    # 1. Tokenize 输入
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    batch_size = input_ids.shape[0]
    
    assert batch_size == 1, "Coconut currently only supports batch_size == 1"
    
    # 2. 初始化控制器状态
    controller.init_state(batch_size, device)
    
    # 记录变量
    generated_tokens = []
    token_modes = []
    token_entropies = []
    switch_events = []
    generated_text_buffer = ""
    consecutive_latent_count = 0
    
    # 3. 初始前向传播 (处理输入)
    labels = input_ids.clone()
    position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=device).reshape(1, -1)
    
    with torch.no_grad():
        outputs = model.forward(
            input_ids,
            torch.ones_like(input_ids, device=device),
            labels,
            position_ids,
        )
    
    inputs_embeds = outputs.inputs_embeds
    
    # 4. 获取初始 hidden states 和 logits
    with torch.no_grad():
        base_outputs = model.base_causallm(inputs_embeds=inputs_embeds, output_hidden_states=True)
    
    hidden_states = base_outputs.hidden_states[-1][:, -1, :]  # [batch_size, hidden_dim]
    logits = base_outputs.logits[:, -1, :]  # [batch_size, vocab_size]
    
    # 5. 获取第一个 token
    next_token = torch.argmax(logits, dim=-1)
    next_token_id = next_token[0].item()
    
    # 6. 初始模式判断
    mode, to_normal, to_soft, cur_entropy = controller.step(hidden_states, 0, logits)
    current_mode = "normal" if mode[0].item() == 1 else "soft"
    
    # 记录第一个 token
    token_text = tokenizer.decode([next_token_id])
    generated_tokens.append({
        "token_id": next_token_id,
        "token_text": token_text,
        "mode": current_mode,
        "entropy": cur_entropy[0].item(),
    })
    token_modes.append(current_mode)
    token_entropies.append(cur_entropy[0].item())
    generated_text_buffer += token_text
    
    # 7. 更新 inputs_embeds
    new_token_embed = model.embedding(next_token).unsqueeze(1)  # [1, 1, hidden_dim]
    new_inputs_embeds = torch.cat([inputs_embeds, new_token_embed], dim=1)
    
    # 8. 生成循环
    for step in range(1, max_new_tokens):
        start_time = time.perf_counter()
        
        # 检查是否达到连续 latent 上限
        if current_mode == "soft" and consecutive_latent_count >= max_latent_steps:
            current_mode = "normal"
            controller.state.mode[0] = 1
            controller.state.mode_stay_steps[0] = 0  # ← 新增
            controller.state.ref_entropy[0] = cur_entropy[0] 
            consecutive_latent_count = 0
            if verbose:
                print(f"[Step {step}] Max latent steps ({max_latent_steps}) reached, forcing normal mode")
        
        # 前向传播
        with torch.no_grad():
            base_outputs = model.base_causallm(inputs_embeds=new_inputs_embeds, output_hidden_states=True)
        
        step_time = time.perf_counter() - start_time
        
        hidden_states = base_outputs.hidden_states[-1][:, -1, :]
        logits = base_outputs.logits[:, -1, :]
        
        # 获取下一个 token
        next_token = torch.argmax(logits, dim=-1)
        next_token_id = next_token[0].item()
        
        # 检查是否结束
        end_token_mask = (next_token == tokenizer.eos_token_id)
        
        # 更新模式
        mode, to_normal, to_soft, cur_entropy = controller.step(
            hidden_states, step, logits, end_token_mask
        )
        
        # 记录时间
        if current_mode == "normal":
            time_stats.add_normal(step_time)
        else:
            time_stats.add_soft(step_time)
        
        # 记录切换事件
        if to_normal[0].item():
            switch_events.append((step, "soft->normal", cur_entropy[0].item()))
        if to_soft[0].item():
            switch_events.append((step, "normal->soft", cur_entropy[0].item()))
        
        token_modes.append(current_mode)
        token_entropies.append(cur_entropy[0].item())
        
        # 根据当前模式决定下一步操作
        is_soft = (mode[0].item() == 0) and (not controller.state.locked_normal[0].item())
        
        if current_mode == "soft" and is_soft and consecutive_latent_count < max_latent_steps:
            # Latent 模式: 使用 hidden states 作为下一个输入
            consecutive_latent_count += 1
            new_token_embed = hidden_states.unsqueeze(1)  # [1, 1, hidden_dim]
            
            generated_tokens.append({
                "token_id": -1,
                "token_text": "<latent>",
                "mode": current_mode,
                "entropy": cur_entropy[0].item(),
            })
        else:
            # Normal 模式: 使用 token embedding
            consecutive_latent_count = 0
            
            token_text = tokenizer.decode([next_token_id])
            generated_text_buffer += token_text
            
            # 检查答案触发
            if not controller.state.answer_locked[0]:
                for trigger in answer_triggers:
                    if trigger in generated_text_buffer:
                        controller.state.answer_locked[0] = True
                        controller.state.mode[0] = 1
                        mode[0] = 1
                        if verbose:
                            print(f"[Step {step}] Answer trigger '{trigger}' detected, locking normal mode")
                        break
            
            generated_tokens.append({
                "token_id": next_token_id,
                "token_text": token_text,
                "mode": current_mode,
                "entropy": cur_entropy[0].item(),
            })
            
            new_token_embed = model.embedding(next_token).unsqueeze(1)
            
            # 检查是否结束
            if next_token_id == tokenizer.eos_token_id:
                break
        
        # 更新 inputs_embeds
        new_inputs_embeds = torch.cat([new_inputs_embeds, new_token_embed], dim=1)
        
        # 决定下一步模式
        if controller.state.answer_locked[0]:
            current_mode = "normal"
        elif consecutive_latent_count >= max_latent_steps:
            current_mode = "normal"
        else:
            current_mode = "normal" if mode[0].item() == 1 else "soft"
    
    # 9. 提取输出文本 (只包含实际 token)
    output_tokens = [t["token_id"] for t in generated_tokens if t["token_id"] != -1]
    output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
    
    # 10. 统计模式分布
    mode_counts = {"normal": 0, "soft": 0}
    for m in token_modes:
        mode_counts[m] += 1
    
    result = {
        "input": input_text,
        "output": output_text,
        "generated_tokens": generated_tokens,
        "mode_distribution": mode_counts,
        "switch_events": switch_events,
        "avg_entropy": sum(token_entropies) / len(token_entropies) if token_entropies else 0,
        "total_steps": len(generated_tokens),
        "total_switches": len(switch_events),
        "latent_count": sum(1 for t in generated_tokens if t["token_id"] == -1),
    }
    
    return result, time_stats


# ============================================================================
# 数据集评估
# ============================================================================

def evaluate_dataset(
    model,
    tokenizer,
    controller: AdaptiveController,
    questions: List[str],
    answers: List,
    data_name: str,
    generate_fn,
    batch_size: int = 1,
    max_samples: Optional[int] = None,
    verbose: bool = False,
    output_file: Optional[str] = None,
) -> Dict:
    """评估数据集"""
    
    if max_samples is not None:
        questions = questions[:max_samples]
        answers = answers[:max_samples]
    
    total_samples = len(questions)
    print(f"\n{'='*60}")
    print(f"Evaluating {data_name}: {total_samples} samples")
    print(f"{'='*60}")
    
    predictions = []
    results = []
    global_time_stats = TimeStats()
    global_mode_counts = {"normal": 0, "soft": 0}
    total_switches = 0
    total_entropy = 0.0
    
    for i, (question, gold_answer) in enumerate(zip(questions, answers)):
        controller.state = None
        
        result, time_stats = generate_fn(question)
        
        pred_answer = extract_answer_number(result["output"], data_name)
        predictions.append(pred_answer)
        
        global_time_stats.merge(time_stats)
        global_mode_counts["normal"] += result["mode_distribution"]["normal"]
        global_mode_counts["soft"] += result["mode_distribution"]["soft"]
        total_switches += result["total_switches"]
        total_entropy += result["avg_entropy"]
        
        result["gold_answer"] = gold_answer
        result["pred_answer"] = pred_answer
        result["correct"] = (pred_answer == gold_answer)
        results.append(result)
        
        if verbose or (i + 1) % 50 == 0:
            print(f"[{i+1}/{total_samples}] Q: {question[:50]}...")
            print(f"  Pred: {pred_answer} | Gold: {gold_answer} | {'✓' if pred_answer == gold_answer else '✗'}")
            print(f"  Modes: Normal={result['mode_distribution']['normal']}, Latent={result['mode_distribution']['soft']}")
    
    accuracy = compute_accuracy(answers, predictions)
    
    total_tokens = global_mode_counts["normal"] + global_mode_counts["soft"]
    normal_pct = global_mode_counts["normal"] / total_tokens * 100 if total_tokens > 0 else 0
    soft_pct = global_mode_counts["soft"] / total_tokens * 100 if total_tokens > 0 else 0
    
    summary = {
        "dataset": data_name,
        "total_samples": total_samples,
        "accuracy": accuracy,
        "accuracy_pct": accuracy * 100,
        "token_stats": {
            "total_tokens": total_tokens,
            "normal_mode": global_mode_counts["normal"],
            "normal_pct": normal_pct,
            "soft_mode": global_mode_counts["soft"],
            "soft_pct": soft_pct,
        },
        "time_stats": {
            "normal_total_time": global_time_stats.normal_time,
            "normal_steps": global_time_stats.normal_steps,
            "normal_avg_time_ms": global_time_stats.get_avg_normal() * 1000,
            "soft_total_time": global_time_stats.soft_time,
            "soft_steps": global_time_stats.soft_steps,
            "soft_avg_time_ms": global_time_stats.get_avg_soft() * 1000,
        },
        "switch_stats": {
            "total_switches": total_switches,
            "avg_switches_per_sample": total_switches / total_samples,
        },
        "avg_entropy": total_entropy / total_samples,
    }
    
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS: {data_name}")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy*100:.2f}% ({int(accuracy*total_samples)}/{total_samples})")
    print(f"\nToken Distribution:")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Normal mode: {global_mode_counts['normal']} ({normal_pct:.1f}%)")
    print(f"  Latent mode:   {global_mode_counts['soft']} ({soft_pct:.1f}%)")
    print(f"\nTime Statistics:")
    print(f"  Normal mode avg: {global_time_stats.get_avg_normal()*1000:.3f} ms/step")
    print(f"  Latent mode avg:   {global_time_stats.get_avg_soft()*1000:.3f} ms/step")
    print(f"  Normal total:    {global_time_stats.normal_time:.2f} s ({global_time_stats.normal_steps} steps)")
    print(f"  Latent total:      {global_time_stats.soft_time:.2f} s ({global_time_stats.soft_steps} steps)")
    print(f"\nSwitch Statistics:")
    print(f"  Total switches: {total_switches}")
    print(f"  Avg per sample: {total_switches/total_samples:.2f}")
    print(f"\nAvg Entropy: {total_entropy/total_samples:.4f}")
    print(f"{'='*60}\n")
    
    if output_file:
        output_data = {
            "summary": summary,
            "results": results,
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_file}")
    
    return summary


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Adaptive Inference Evaluation")
    
    # 模型参数
    parser.add_argument("--model_type", type=str, default="codi",
                        choices=["coconut", "codi"])
    parser.add_argument("--base_model_path", type=str, default="./CODI/pretrained/Llama-3.2-1B-Instruct")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Coconut checkpoint path (single file)")
    parser.add_argument("--ckpt_dir", type=str, default="./CODI/pretrained/SIM_COT-LLaMA3-CODI-1B", # SIM_COT-LLaMA3-CODI-1B   CODI-llama3.2-1b-Instruct
                        help="CODI checkpoint directory")
    parser.add_argument("--predictor_path", type=str, default=None,
                        help="EntropyPredictor path")
    parser.add_argument("--prj_dim", type=int, default=2048,
                        help="Projection dimension for CODI")
    
    # 数据集参数
    parser.add_argument("--data_name", type=str, default="gsm8k",
                        choices=["gsm8k", "gsm-hard", "multi-arith", "svamp", "commonsense"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to evaluate (None for all)")
    
    # SwiReasoning 参数
    parser.add_argument("--window_e_to_l", type=int, default=5,
                        help="Explicit→Latent 切换需等待的步数")
    parser.add_argument("--window_l_to_e", type=int, default=0,
                        help="Latent→Explicit 切换需等待的步数（通常为0）")
    parser.add_argument("--max_switch_count", type=int, default=5)
    parser.add_argument("--use_predicted_entropy", action="store_true")
    
    # 生成参数
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_latent_steps", type=int, default=6)
    parser.add_argument("--num_latent", type=int, default=6)
    parser.add_argument("--greedy", action="store_true", default=True)
    
    # 其他参数
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output_file", type=str, default=None)

    # 消融实验
    parser.add_argument("--baseline_mode", type=str, default="adaptive", 
                        choices=["adaptive", "random"],
                        help="推理模式: adaptive(默认), random(随机/全显/全隐)")
    parser.add_argument("--random_prob", type=float, default=0.5,
                        help="随机模式下切换到 normal mode 的概率。设置 1.0 为全显, 0.0 为全隐。")
    
    parser.add_argument("--step_start_ids", type=int, nargs='+', default=[2501, 1134],
                        help="Step start token IDs (<<)")
    parser.add_argument("--step_end_id", type=int, default=2511,
                        help="Step end token ID (>>)")
    # 在 main() 函数的 parser 中添加
    parser.add_argument("--k_step_latent_num", type=int, default=1,
                        help="Latent step size k: ensure latent count is multiple of k")

    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载 EntropyPredictor
    predictor = None
    if args.predictor_path and os.path.exists(args.predictor_path):
        print(f"Loading EntropyPredictor from {args.predictor_path}...")
        predictor = EntropyPredictor.load(args.predictor_path, device)
    
    # 创建控制器
    controller = AdaptiveController(
        entropy_predictor=predictor,
        window_e_to_l=args.window_e_to_l,
        window_l_to_e=args.window_l_to_e,
        max_switch_count=args.max_switch_count,
        use_predicted_entropy=args.use_predicted_entropy and predictor is not None,
        baseline_mode=args.baseline_mode,
        random_prob=args.random_prob
    )
    
    # 加载模型
    if args.model_type == "coconut":
        # Coconut 模型加载
        model, tokenizer, latent_id, start_id, end_id = load_coconut_model(
            args.base_model_path,
            args.checkpoint_path,
            device,
            use_bf16=args.bf16,
        )
        
        # Coconut 生成函数
        generate_fn = lambda text: coconut_adaptive_generate_with_timing(
            model, tokenizer, controller, text,
            max_new_tokens=args.max_new_tokens,
            max_latent_steps=args.max_latent_steps,
            device=device,
            verbose=args.verbose,
        )
    else:  # codi
        model, tokenizer = load_codi_model(
            args.base_model_path,
            args.ckpt_dir,
            device,
            num_latent=args.num_latent,
            use_bf16=args.bf16,
            prj_dim=args.prj_dim,
        )
        generate_fn = lambda text: codi_adaptive_generate_with_timing(
            model, tokenizer, controller, text,
            max_new_tokens=args.max_new_tokens,
            max_latent_steps=args.max_latent_steps,
            device=device,
            verbose=args.verbose,
            greedy=args.greedy,
            random_prob=args.random_prob,
            step_start_ids=tuple(args.step_start_ids),  # 新增
            step_end_id=args.step_end_id,  # 新增
            k_step_latent_num=args.k_step_latent_num,  # ← 新增
        )
    
    # 加载数据集
    questions, answers, _, _ = load_eval_dataset(args.data_name)
    
    # 设置输出文件
    if args.output_file is None:
        args.output_file = f"results_{args.model_type}_{args.data_name}_{args.baseline_mode}_{args.random_prob}.json"
    
    # 评估
    summary = evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        controller=controller,
        questions=questions,
        answers=answers,
        data_name=args.data_name,
        generate_fn=generate_fn,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        verbose=args.verbose,
        output_file=args.output_file,
    )
    
    return summary


if __name__ == "__main__":
    main()