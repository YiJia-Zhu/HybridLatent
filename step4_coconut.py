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
from model import CODI, ModelArguments, TrainingArguments

# 导入熵预测器
from entropy_predictor import EntropyPredictor, compute_entropy_swir


# ============================================================================
# 时间统计器
# ============================================================================

@dataclass
class TimeStats:
    """时间统计"""
    normal_time: float = 0.0
    soft_time: float = 0.0
    normal_steps: int = 0
    soft_steps: int = 0
    
    def add_normal(self, duration: float):
        self.normal_time += duration
        self.normal_steps += 1
    
    def add_soft(self, duration: float):
        self.soft_time += duration
        self.soft_steps += 1
    
    def get_avg_normal(self) -> float:
        return self.normal_time / self.normal_steps if self.normal_steps > 0 else 0
    
    def get_avg_soft(self) -> float:
        return self.soft_time / self.soft_steps if self.soft_steps > 0 else 0
    
    def merge(self, other: 'TimeStats'):
        self.normal_time += other.normal_time
        self.soft_time += other.soft_time
        self.normal_steps += other.normal_steps
        self.soft_steps += other.soft_steps


# ============================================================================
# SwiReasoning 风格的模式状态和控制器
# ============================================================================

@dataclass
class SwiRModeState:
    """SwiReasoning 风格的模式状态"""
    mode: torch.Tensor
    mode_stay_steps: torch.Tensor
    ref_entropy: torch.Tensor
    locked_normal: torch.Tensor
    switch_count: torch.Tensor
    answer_locked: torch.Tensor
    
    @classmethod
    def init(cls, batch_size: int, device: torch.device):
        return cls(
            mode=torch.zeros(batch_size, dtype=torch.long, device=device),
            mode_stay_steps=torch.zeros(batch_size, dtype=torch.long, device=device),
            ref_entropy=torch.zeros(batch_size, dtype=torch.float, device=device),
            locked_normal=torch.zeros(batch_size, dtype=torch.bool, device=device),
            switch_count=torch.zeros(batch_size, dtype=torch.long, device=device),
            answer_locked=torch.zeros(batch_size, dtype=torch.bool, device=device),
        )


class SwiRController:
    """SwiReasoning 风格的模式切换控制器"""
    
    def __init__(
        self,         
        window_e_to_l: int = 5,
        window_l_to_e: int = 0,
        max_switch_count: Optional[int] = None
        ):
        self.window_e_to_l = window_e_to_l
        self.window_l_to_e = window_l_to_e
        self.max_switch_count = max_switch_count
    
    def update(
        self,
        state: SwiRModeState,
        cur_entropy: torch.Tensor,
        step: int,
        end_token_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[SwiRModeState, torch.Tensor, torch.Tensor]:
        """更新模式状态"""
        device = cur_entropy.device
        batch_size = cur_entropy.shape[0]
        
        if end_token_mask is not None:
            state.locked_normal = state.locked_normal | end_token_mask
        
        if state.answer_locked.any():
            state.mode = torch.where(state.answer_locked, torch.ones_like(state.mode), state.mode)
            to_normal = torch.zeros(batch_size, dtype=torch.bool, device=device)
            to_soft = torch.zeros(batch_size, dtype=torch.bool, device=device)
            return state, to_normal, to_soft

        if step == 0:
            state.ref_entropy = cur_entropy.clone()
            to_normal = torch.zeros(batch_size, dtype=torch.bool, device=device)
            to_soft = torch.zeros(batch_size, dtype=torch.bool, device=device)
        else:
            state.mode_stay_steps += 1
            allow_l_to_e = (state.mode_stay_steps >= self.window_l_to_e)
            allow_e_to_l = (state.mode_stay_steps >= self.window_e_to_l)
            
            to_normal = (state.mode == 0) & (cur_entropy < state.ref_entropy) & allow_l_to_e
            to_soft = (state.mode == 1) & (cur_entropy > state.ref_entropy) & allow_e_to_l & (~state.locked_normal)

            state.mode = torch.where(to_normal, torch.ones_like(state.mode), state.mode)
            state.mode = torch.where(to_soft, torch.zeros_like(state.mode), state.mode)
            switched = to_normal | to_soft
            state.mode_stay_steps = torch.where(switched, torch.zeros_like(state.mode_stay_steps), state.mode_stay_steps)
            state.ref_entropy = torch.where(switched, cur_entropy, state.ref_entropy)

            if self.max_switch_count is not None:
                state.switch_count = state.switch_count + switched.long()
                limit_reached = (state.switch_count >= self.max_switch_count)
                state.mode = torch.where(limit_reached, torch.ones_like(state.mode), state.mode)
                state.locked_normal = state.locked_normal | limit_reached
        
        return state, to_normal, to_soft
    

class AdaptiveController:
    """结合 EntropyPredictor 和 SwiRController 的自适应控制器"""
    
    def __init__(
        self,
        entropy_predictor: Optional[EntropyPredictor],
        window_e_to_l: int = 5,
        window_l_to_e: int = 0,
        max_switch_count: Optional[int] = None,
        use_predicted_entropy: bool = True,
        baseline_mode: str = "adaptive",
        random_prob: float = 0.5,
    ):
        self.predictor = entropy_predictor
        self.controller = SwiRController(window_e_to_l=window_e_to_l, window_l_to_e=window_l_to_e, max_switch_count=max_switch_count)
        self.use_predicted_entropy = use_predicted_entropy
        self.state = None
        
        self.baseline_mode = baseline_mode
        self.random_prob = random_prob
        self.is_initialized = False
        
    def init_state(self, batch_size: int, device: torch.device):
        """初始化模式状态"""
        self.state = SwiRModeState.init(batch_size, device)
        self.is_initialized = True
        return self.state
    
    def get_entropy(self, hidden_states: torch.Tensor, logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_predicted_entropy and self.predictor is not None:
            with torch.no_grad():
                predicted = self.predictor(hidden_states)
            return predicted.squeeze(-1)
        else:
            if logits is None:
                raise ValueError("logits required when use_predicted_entropy=False")
            return compute_entropy_swir(logits)
    
    def step(
        self,
        hidden_states: torch.Tensor,
        step: int,
        logits: Optional[torch.Tensor] = None,
        end_token_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """更新模式状态并返回当前模式、切换事件和熵"""
        if self.state is None:
            self.init_state(hidden_states.shape[0], hidden_states.device)
            
        device = hidden_states.device
        batch_size = hidden_states.shape[0]
        
        if self.baseline_mode == "random":
            if step == 0 and self.is_initialized:
                random_values = torch.rand(batch_size, device=device)
                self.state.mode = (random_values < self.random_prob).long()
                
            random_switch = torch.rand(batch_size, device=device)
            
            to_normal_new = (random_switch < self.random_prob) & (self.state.mode == 0)
            to_soft_new = (random_switch >= self.random_prob) & (self.state.mode == 1) & (~self.state.locked_normal)
            
            mode = self.state.mode.clone()
            mode = torch.where(to_normal_new, torch.ones_like(mode), mode)
            mode = torch.where(to_soft_new, torch.zeros_like(mode), mode)
            
            self.state.mode = mode
            
            to_normal = to_normal_new
            to_soft = to_soft_new
            cur_entropy = torch.zeros(batch_size, dtype=torch.float, device=device) 
            
            return mode, to_normal, to_soft, cur_entropy
        
        elif self.baseline_mode == "adaptive":
            cur_entropy = self.get_entropy(hidden_states, logits)
            self.state, to_normal, to_soft = self.controller.update(self.state, cur_entropy, step, end_token_mask)
            
            return self.state.mode.clone(), to_normal, to_soft, cur_entropy
        
        else:
            raise ValueError(f"Unknown baseline_mode: {self.baseline_mode}")


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
) -> Tuple[Dict, TimeStats]:
    """修正版：只有进入 latent 时才插入 bot，退出时插入 eot"""
    
    if answer_triggers is None:
        answer_triggers = ["The answer is", "#### ", "the answer is", "Answer:",]
    
    if device is None:
        device = next(model.parameters()).device
    
    time_stats = TimeStats()
    batch = tokenizer(input_text, return_tensors="pt", padding=True)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    batch_size = input_ids.shape[0]
    
    # ========== 删除这部分 ==========
    # bot_tensor = torch.tensor([[model.bot_id]], dtype=torch.long, device=device).expand(batch_size, 1)
    # input_ids = torch.cat([input_ids, bot_tensor], dim=1)
    # attention_mask = torch.cat([attention_mask, torch.ones_like(bot_tensor)], dim=1)
    # ================================
    
    controller.init_state(batch_size, device)
    
    generated_tokens = []
    token_modes = []
    token_entropies = []
    switch_events = []
    
    generated_text_buffer = [""] * batch_size
    consecutive_latent_count = 0
    
    def get_embd(token_ids):
        return model.get_embd(model.codi, model.model_name)(token_ids)
    
    past_key_values = None
    # encode the question
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
    mode, to_normal, to_soft, cur_entropy = controller.step(last_hidden, 0, logits)
    current_mode = "normal" if mode[0].item() == 1 else "soft"
    
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    pred_tokens = [[] for _ in range(batch_size)]
    
    # 预计算 bot 和 eot embeddings
    bot_emb = get_embd(torch.tensor([model.bot_id], device=device)).unsqueeze(0).expand(batch_size, -1, -1)
    eot_emb = get_embd(torch.tensor([model.eot_id], device=device)).unsqueeze(0).expand(batch_size, -1, -1)
    
    next_input_embd = get_embd(torch.argmax(logits, dim=-1)).unsqueeze(1).to(device)
    
    # ========== 新增：如果初始就是 soft 模式，先插入 bot ==========
    if current_mode == "soft":
        with torch.no_grad():
            out = model.codi(
                inputs_embeds=bot_emb,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            past_key_values = out.past_key_values
            last_hidden = out.hidden_states[-1][:, -1, :]
        if verbose:
            print(f"[Init] Starting in soft mode, inserted bot_token")
    # ============================================================

    for step in range(max_new_tokens):
        start_time = time.perf_counter()
        
        if current_mode == "soft" and consecutive_latent_count >= max_latent_steps:
            current_mode = "normal"
            controller.state.mode[0] = 1
            consecutive_latent_count = 0

            if verbose:
                print(f"[Step {step}] Max latent steps ({max_latent_steps}) reached, forcing normal mode")
        
        with torch.no_grad():
            if current_mode == "soft":
                # decode the latent embeddings
                if use_prj and hasattr(model, 'prj'):
                    input_embd = model.prj(last_hidden.unsqueeze(1))
                else:
                    input_embd = last_hidden.unsqueeze(1)
            else:
                input_embd = next_input_embd
            
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
        
        if greedy:
            next_token_ids = torch.argmax(logits, dim=-1)
        else:
            probs = F.softmax(logits / 0.1, dim=-1)
            next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        if current_mode == "normal":
            time_stats.add_normal(step_time)
        else:
            time_stats.add_soft(step_time)
        
        end_token_mask = (next_token_ids == tokenizer.eos_token_id)
        mode, to_normal, to_soft, cur_entropy = controller.step(
            last_hidden, step + 1, logits, end_token_mask
        )
        
        if current_mode == "normal":
            consecutive_latent_count = 0
            
            for b in range(batch_size):
                if not finished[b]:
                    token_text = tokenizer.decode([next_token_ids[b].item()])
                    generated_text_buffer[b] += token_text
                    
                    if not controller.state.answer_locked[b]:
                        for trigger in answer_triggers:
                            if trigger in generated_text_buffer[b]:
                                controller.state.answer_locked[b] = True
                                controller.state.mode[b] = 1
                                mode[b] = 1
                                if verbose:
                                    print(f"[Step {step}] Answer trigger '{trigger}' detected, locking normal mode")
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
            })
        else:
            consecutive_latent_count += 1
            
            generated_tokens.append({
                "token_id": -1,
                "token_text": "<latent>",
                "mode": current_mode,
                "entropy": cur_entropy[0].item(),
                "answer_locked": controller.state.answer_locked[0].item(),
            })
        
        # ========== 修改：处理模式切换时的特殊 token ==========
        if to_normal[0].item():
            # soft -> normal: 插入 eot_token
            next_input_embd = eot_emb
            switch_events.append((step, "soft->normal", cur_entropy[0].item()))
            if verbose:
                print(f"[Step {step}] soft->normal, inserting eot_token")
        elif to_soft[0].item():
            # normal -> soft: 插入 bot_token，然后进行一次前向传播
            switch_events.append((step, "normal->soft", cur_entropy[0].item()))
            if verbose:
                print(f"[Step {step}] normal->soft, inserting bot_token")
            
            with torch.no_grad():
                out = model.codi(
                    inputs_embeds=bot_emb,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values,
                )
                past_key_values = out.past_key_values
                last_hidden = out.hidden_states[-1][:, -1, :]
            # soft 模式下不需要设置 next_input_embd，因为会用 last_hidden
        else:
            next_input_embd = get_embd(next_token_ids).unsqueeze(1)
        # ====================================================
        
        token_modes.append(current_mode)
        token_entropies.append(cur_entropy[0].item())
        
        if controller.state.answer_locked[0]:
            next_mode = "normal"
        elif consecutive_latent_count >= max_latent_steps:
            next_mode = "normal"
        else:
            next_mode = "normal" if mode[0].item() == 1 else "soft"
        
        current_mode = next_mode
        
        if finished.all():
            break
    
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

"""
Coconut 自适应生成 - 修复版

参考:
1. CODI 中 codi_adaptive_generate_with_timing 的 bot/eot 处理逻辑
2. Coconut 官方代码的 generate 方法
"""

import torch
import torch.nn.functional as F
import time
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass



def coconut_adaptive_generate_with_timing(
    model,  # Coconut model
    tokenizer,
    controller,  # AdaptiveController
    input_text: str,
    max_new_tokens: int = 256,
    max_latent_steps: int = 6,
    device: torch.device = None,
    verbose: bool = False,
    answer_triggers: List[str] = None,
    greedy: bool = True,
) -> Tuple[Dict, TimeStats]:
    """
    Coconut 自适应生成 (修复版)
    
    参考 CODI 的 bot/eot 处理逻辑:
    - 初始如果是 soft 模式，先插入 start_id (bot)
    - soft -> normal: 插入 end_id (eot)
    - normal -> soft: 插入 start_id (bot) 并做一次前向传播
    
    Coconut 特殊 token:
    - start_id: <|start-latent|> (对应 CODI 的 bot)
    - end_id: <|end-latent|> (对应 CODI 的 eot)
    - latent_id: <|latent|> (latent placeholder)
    """
    if answer_triggers is None:
        answer_triggers = ["####", "The answer is", "the answer is"]
    
    if device is None:
        device = next(model.parameters()).device
    
    time_stats = TimeStats()
    
    # 获取 Coconut 的特殊 token IDs
    start_id = model.start_id  # bot
    end_id = model.end_id      # eot
    latent_id = model.latent_id
    eos_id = model.eos_id
    
    # 获取 embedding 函数
    def get_embedding(token_ids):
        """获取 token embedding"""
        if hasattr(model, 'embedding'):
            return model.embedding(token_ids)
        elif hasattr(model.base_causallm, 'get_input_embeddings'):
            return model.base_causallm.get_input_embeddings()(token_ids)
        else:
            return model.base_causallm.transformer.wte(token_ids)
    
    # 1. Tokenize 输入
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(device)
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
    
    # 预计算 start (bot) 和 end (eot) 的 embeddings
    start_emb = get_embedding(torch.tensor([start_id], device=device)).unsqueeze(0)  # [1, 1, hidden_dim]
    end_emb = get_embedding(torch.tensor([end_id], device=device)).unsqueeze(0)      # [1, 1, hidden_dim]
    
    # 3. 初始前向传播 - 编码问题
    with torch.no_grad():
        outputs = model.base_causallm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
        )
        past_key_values = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [batch_size, hidden_dim]
    
    logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
    
    # 4. 获取初始模式
    mode, to_normal, to_soft, cur_entropy = controller.step(last_hidden, 0, logits)
    current_mode = "normal" if mode[0].item() == 1 else "soft"
    
    if verbose:
        print(f"[Init] Initial mode: {current_mode}, entropy: {cur_entropy[0].item():.4f}")
    
    # 5. 如果初始就是 soft 模式，先插入 start_id (bot)
    if current_mode == "soft":
        with torch.no_grad():
            out = model.base_causallm(
                inputs_embeds=start_emb,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
            past_key_values = out.past_key_values
            last_hidden = out.hidden_states[-1][:, -1, :]
        if verbose:
            print(f"[Init] Starting in soft mode, inserted start_token (bot)")
    
    # 准备第一个 token 的 embedding
    if greedy:
        next_token_ids = torch.argmax(logits, dim=-1)
    else:
        probs = F.softmax(logits / 0.1, dim=-1)
        next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    next_input_embd = get_embedding(next_token_ids).unsqueeze(1)  # [1, 1, hidden_dim]
    
    finished = False
    pred_tokens = []
    
    # 6. 生成循环
    for step in range(max_new_tokens):
        start_time = time.perf_counter()
        
        # 检查是否达到连续 latent 上限
        if current_mode == "soft" and consecutive_latent_count >= max_latent_steps:
            current_mode = "normal"
            controller.state.mode[0] = 1
            consecutive_latent_count = 0
            if verbose:
                print(f"[Step {step}] Max latent steps ({max_latent_steps}) reached, forcing normal mode")
        
        # 前向传播
        with torch.no_grad():
            if current_mode == "soft":
                # Latent 模式: 使用 hidden states 作为输入
                input_embd = last_hidden.unsqueeze(1)  # [1, 1, hidden_dim]
            else:
                # Normal 模式: 使用 token embedding
                input_embd = next_input_embd
            
            out = model.base_causallm(
                inputs_embeds=input_embd,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
            past_key_values = out.past_key_values
            last_hidden = out.hidden_states[-1][:, -1, :]
        
        step_time = time.perf_counter() - start_time
        logits = out.logits[:, -1, :]
        
        # 获取下一个 token
        if greedy:
            next_token_ids = torch.argmax(logits, dim=-1)
        else:
            probs = F.softmax(logits / 0.1, dim=-1)
            next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        next_token_id = next_token_ids[0].item()
        
        # 记录时间
        if current_mode == "normal":
            time_stats.add_normal(step_time)
        else:
            time_stats.add_soft(step_time)
        
        # 检查是否结束
        end_token_mask = (next_token_ids == eos_id)
        
        # 更新模式
        mode, to_normal, to_soft, cur_entropy = controller.step(
            last_hidden, step + 1, logits, end_token_mask
        )
        
        # 记录当前步骤
        token_modes.append(current_mode)
        token_entropies.append(cur_entropy[0].item())
        
        if current_mode == "normal":
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
            
            pred_tokens.append(next_token_id)
            
            generated_tokens.append({
                "token_id": next_token_id,
                "token_text": token_text,
                "mode": current_mode,
                "entropy": cur_entropy[0].item(),
                "answer_locked": controller.state.answer_locked[0].item(),
            })
            
            if next_token_id == eos_id:
                finished = True
        else:
            # Latent 模式
            consecutive_latent_count += 1
            
            generated_tokens.append({
                "token_id": -1,
                "token_text": "<latent>",
                "mode": current_mode,
                "entropy": cur_entropy[0].item(),
                "answer_locked": controller.state.answer_locked[0].item(),
            })
        
        # 处理模式切换时的特殊 token (参考 CODI 逻辑)
        if to_normal[0].item():
            # soft -> normal: 插入 end_token (eot)，然后用下一个 token 的 embedding
            switch_events.append((step, "soft->normal", cur_entropy[0].item()))
            if verbose:
                print(f"[Step {step}] soft->normal, inserting end_token (eot)")
            
            # 先插入 eot
            with torch.no_grad():
                out = model.base_causallm(
                    inputs_embeds=end_emb,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                )
                past_key_values = out.past_key_values
                # 不更新 last_hidden，因为接下来要用 token embedding
            
            next_input_embd = get_embedding(next_token_ids).unsqueeze(1)
            
        elif to_soft[0].item():
            # normal -> soft: 插入 start_token (bot)，然后用 hidden states
            switch_events.append((step, "normal->soft", cur_entropy[0].item()))
            if verbose:
                print(f"[Step {step}] normal->soft, inserting start_token (bot)")
            
            # 先插入 bot 并更新 hidden states
            with torch.no_grad():
                out = model.base_causallm(
                    inputs_embeds=start_emb,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                )
                past_key_values = out.past_key_values
                last_hidden = out.hidden_states[-1][:, -1, :]
            # soft 模式下不需要设置 next_input_embd，因为会用 last_hidden
            
        else:
            # 没有模式切换，正常更新 next_input_embd
            next_input_embd = get_embedding(next_token_ids).unsqueeze(1)
        
        # 决定下一步模式
        if controller.state.answer_locked[0]:
            next_mode = "normal"
        elif consecutive_latent_count >= max_latent_steps:
            next_mode = "normal"
        else:
            next_mode = "normal" if mode[0].item() == 1 else "soft"
        
        current_mode = next_mode
        
        if finished:
            break
    
    # 7. 提取输出文本
    output_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
    
    # 8. 统计模式分布
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
    parser.add_argument("--ckpt_dir", type=str, default="./CODI/pretrained/CODI-llama3.2-1b-Instruct",
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