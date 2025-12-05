"""
阶段3: 半显半隐式推理 (SwiReasoning风格控制)

参考 SwiReasoning 的模式切换逻辑:
- 维护参考熵 ref_entropy
- 熵下降 (cur < ref) -> 切换到 normal 模式 (显式/explicit)
- 熵上升 (cur > ref) -> 切换到 soft 模式 (隐式/latent)
- 使用 window_size 防止频繁切换

使用方法:
    # Coconut 推理 (参考 Coconut/run.py)
    python step3_adaptive_inference.py \
        --model_type coconut \
        --base_model_path gpt2 \
        --checkpoint_path Coconut/ckpts/gsm-coconut/checkpoint_6 \
        --predictor_path checkpoints/entropy_predictor.pt \
        --input "Question: What is 2+3?"
    
    # CODI 推理 (参考 CODI/test.py)
    python step3_adaptive_inference.py \
        --model_type codi \
        --base_model_path meta-llama/Llama-3.2-1B-Instruct \
        --ckpt_dir CODI/ckpts/xxx \
        --predictor_path checkpoints/entropy_predictor.pt \
        --input "Question: What is 2+3?"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import json
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass

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

# 导入 Coconut (参考 Coconut/run.py)
from coconut import Coconut

# 导入 CODI (参考 CODI/test.py)
from model import CODI, ModelArguments, TrainingArguments

# 导入熵预测器
from entropy_predictor import EntropyPredictor, compute_normalized_entropy, compute_entropy_swir


# ============================================================================
# SwiReasoning 风格的模式状态和控制器
# ============================================================================

@dataclass
class SwiRModeState:
    """SwiReasoning 风格的模式状态"""
    mode: torch.Tensor           # (batch,) 当前模式: 0=soft/latent, 1=normal
    mode_stay_steps: torch.Tensor
    ref_entropy: torch.Tensor
    locked_normal: torch.Tensor
    switch_count: torch.Tensor
    
    @classmethod
    def init(cls, batch_size: int, device: torch.device):
        return cls(
            mode=torch.zeros(batch_size, dtype=torch.long, device=device),
            mode_stay_steps=torch.zeros(batch_size, dtype=torch.long, device=device),
            ref_entropy=torch.zeros(batch_size, dtype=torch.float, device=device),
            locked_normal=torch.zeros(batch_size, dtype=torch.bool, device=device),
            switch_count=torch.zeros(batch_size, dtype=torch.long, device=device),
        )


class SwiRController:
    """SwiReasoning 风格的模式切换控制器"""
    
    def __init__(self, window_size: int = 5, max_switch_count: Optional[int] = None):
        self.window_size = window_size
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
        
        if step == 0:
            state.ref_entropy = cur_entropy.clone()
            to_normal = torch.zeros(batch_size, dtype=torch.bool, device=device)
            to_soft = torch.zeros(batch_size, dtype=torch.bool, device=device)
        else:
            state.mode_stay_steps += 1
            allow_switch = (state.mode_stay_steps >= self.window_size)
            
            # 切换条件: 熵下降->normal, 熵上升->soft
            to_normal = (state.mode == 0) & (cur_entropy < state.ref_entropy)
            to_soft = (state.mode == 1) & (cur_entropy > state.ref_entropy) & allow_switch & (~state.locked_normal)
            
            state.mode = torch.where(to_normal, torch.ones_like(state.mode), state.mode)
            state.mode = torch.where(to_soft, torch.zeros_like(state.mode), state.mode)
            switched = to_normal | to_soft
            state.mode_stay_steps = torch.where(switched, torch.zeros_like(state.mode_stay_steps), state.mode_stay_steps)
            state.ref_entropy = torch.where(switched, cur_entropy, state.ref_entropy)

            if self.max_switch_count is not None:
                state.switch_count = state.switch_count + to_normal.long()
        
        return state, to_normal, to_soft


class AdaptiveController:
    """结合 EntropyPredictor 和 SwiRController 的自适应控制器"""
    
    def __init__(
        self,
        entropy_predictor: Optional[EntropyPredictor],
        window_size: int = 5,
        max_switch_count: Optional[int] = None,
        use_predicted_entropy: bool = True,
    ):
        self.predictor = entropy_predictor
        self.controller = SwiRController(window_size=window_size, max_switch_count=max_switch_count)
        self.use_predicted_entropy = use_predicted_entropy
        self.state = None
    
    def init_state(self, batch_size: int, device: torch.device):
        self.state = SwiRModeState.init(batch_size, device)
        return self.state
    
    def get_entropy(self, hidden_states: torch.Tensor, logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """获取熵值"""
        if self.use_predicted_entropy and self.predictor is not None:
            with torch.no_grad():
                predicted = self.predictor(hidden_states)
                return predicted.squeeze(-1)
        else:
            if logits is None:
                raise ValueError("logits required when use_predicted_entropy=False")
            return compute_normalized_entropy(logits)
    
    def step(
        self,
        hidden_states: torch.Tensor,
        step: int,
        logits: Optional[torch.Tensor] = None,
        end_token_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """执行一步控制"""
        if self.state is None:
            self.init_state(hidden_states.shape[0], hidden_states.device)
        
        cur_entropy = self.get_entropy(hidden_states, logits)
        self.state, to_normal, to_soft = self.controller.update(self.state, cur_entropy, step, end_token_mask)
        
        return self.state.mode.clone(), to_normal, to_soft, cur_entropy


# ============================================================================
# Coconut 模型加载和自适应推理
# ============================================================================

def load_coconut_model(
    base_model_path: str,
    checkpoint_path: str,
    device: torch.device,
    use_bf16: bool = False,
):
    """
    加载 Coconut 模型 (完全参考 Coconut/run.py 第117-192行)
    """
    print(f"Loading Coconut model...")
    print(f"  Base model: {base_model_path}")
    print(f"  Checkpoint: {checkpoint_path}")
    
    # Step 1: 加载基础模型 (run.py 第117行)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    
    # Step 2: 加载 tokenizer 并添加特殊 token (run.py 第124-131行)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    
    # Step 3: 加载 checkpoint 权重 (run.py 第135-163行)
    loaded = False
    saved_weights = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"  Loading checkpoint from {checkpoint_path}")
        saved_weights = torch.load(checkpoint_path, map_location="cpu")
        
        # 检查是否是 Coconut 模型的权重
        if not any([k.startswith("base_causallm") for k in saved_weights.keys()]):
            # 加载的是基础模型权重
            loaded = True
            print(base_model.load_state_dict(saved_weights, strict=False))
    
    # Step 4: resize token embeddings 并初始化新 token (run.py 第165-177行)
    base_model.resize_token_embeddings(len(tokenizer))
    embeddings = base_model.get_input_embeddings()
    target_id = tokenizer.convert_tokens_to_ids("<<")
    for token_id in [latent_id, start_id, end_id]:
        target_embedding = embeddings.weight.data[target_id]
        embeddings.weight.data[token_id] = target_embedding
        if hasattr(base_model, 'lm_head'):
            base_model.lm_head.weight.data[token_id] = base_model.lm_head.weight.data[target_id]
    
    # Step 5: 创建 Coconut 模型 (run.py 第186-187行)
    model = Coconut(base_model, latent_id, start_id, end_id, tokenizer.eos_token_id)
    
    # Step 6: 如果之前没有加载权重，现在加载 Coconut 模型权重 (run.py 第191-192行)
    if saved_weights is not None and not loaded:
        print(model.load_state_dict(saved_weights, strict=False))
    
    # Step 7: 移动到设备
    model = model.to(device)
    if use_bf16:
        model = model.to(torch.bfloat16)
    model.eval()
    
    return model, tokenizer, latent_id, start_id, end_id


def coconut_adaptive_generate(
    model: Coconut,
    tokenizer,
    controller: AdaptiveController,
    input_text: str,
    max_new_tokens: int = 100,
    max_latent_steps: int = 5,
    device: torch.device = None,
    verbose: bool = True,
) -> Dict:
    """
    Coconut 自适应生成
    
    根据熵预测动态选择:
    - soft 模式: 使用 latent embedding 继续推理
    - normal 模式: 使用普通 token 生成
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    batch_size = input_ids.shape[0]
    
    # 初始化控制器
    controller.init_state(batch_size, device)
    
    # 记录
    generated_tokens = []
    token_modes = []
    token_entropies = []
    switch_events = []
    latent_count = 0
    
    # 初始 forward (处理 latent tokens)
    labels = input_ids.clone()
    position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=device).reshape(1, -1)
    
    outputs = model.forward(
        input_ids,
        torch.ones_like(input_ids, device=device),
        labels,
        position_ids,
    )
    inputs_embeds = outputs.inputs_embeds
    
    # 获取初始 hidden states 和 logits
    base_outputs = model.base_causallm(inputs_embeds=inputs_embeds, output_hidden_states=True)
    hidden_states = base_outputs.hidden_states[-1][:, -1, :]  # (batch, hidden_dim)
    logits = base_outputs.logits[:, -1, :]  # (batch, vocab)
    
    # 第一个 token
    next_token = torch.argmax(logits, dim=-1)
    
    # 控制器决策
    mode, to_normal, to_soft, cur_entropy = controller.step(hidden_states, 0, logits)
    current_mode = "normal" if mode[0].item() == 1 else "soft"
    token_modes.append(current_mode)
    token_entropies.append(cur_entropy[0].item())
    
    generated_tokens.append({
        "token_id": next_token[0].item(),
        "token_text": tokenizer.decode([next_token[0].item()]),
        "mode": current_mode,
        "entropy": cur_entropy[0].item(),
    })
    
    # 更新 inputs_embeds
    new_token_embed = model.embedding(next_token).unsqueeze(1)
    new_inputs_embeds = torch.cat([inputs_embeds, new_token_embed], dim=1)
    
    # 生成循环
    for step in range(1, max_new_tokens):
        # Forward
        base_outputs = model.base_causallm(inputs_embeds=new_inputs_embeds, output_hidden_states=True)
        hidden_states = base_outputs.hidden_states[-1][:, -1, :]
        logits = base_outputs.logits[:, -1, :]
        
        # 检查 EOS
        next_token = torch.argmax(logits, dim=-1)
        end_token_mask = (next_token == tokenizer.eos_token_id)
        
        # 控制器决策
        mode, to_normal, to_soft, cur_entropy = controller.step(hidden_states, step, logits, end_token_mask)
        current_mode = "normal" if mode[0].item() == 1 else "soft"
        
        # 记录切换事件
        if to_normal[0].item():
            switch_events.append((step, "soft->normal", cur_entropy[0].item()))
        if to_soft[0].item():
            switch_events.append((step, "normal->soft", cur_entropy[0].item()))
        
        token_modes.append(current_mode)
        token_entropies.append(cur_entropy[0].item())
        
        # 根据模式处理
        is_soft = (mode == 0) & (~controller.state.locked_normal)
        
        if is_soft[0].item() and latent_count < max_latent_steps:
            # Soft/Latent 模式: 使用 hidden state 作为下一步输入
            latent_count += 1
            # 直接使用 hidden state 作为 embedding (Coconut 的核心思想)
            new_token_embed = hidden_states.unsqueeze(1)
            
            generated_tokens.append({
                "token_id": -1,  # latent token
                "token_text": "<latent>",
                "mode": current_mode,
                "entropy": cur_entropy[0].item(),
            })
        else:
            # Normal 模式: 使用普通 token embedding
            new_token_embed = model.embedding(next_token).unsqueeze(1)
            
            generated_tokens.append({
                "token_id": next_token[0].item(),
                "token_text": tokenizer.decode([next_token[0].item()]),
                "mode": current_mode,
                "entropy": cur_entropy[0].item(),
            })
        
        new_inputs_embeds = torch.cat([new_inputs_embeds, new_token_embed], dim=1)
        
        # 检查是否结束
        if next_token[0].item() == tokenizer.eos_token_id:
            break
        
        if verbose and step % 10 == 0:
            print(f"Step {step}: mode={current_mode}, entropy={cur_entropy[0].item():.4f}, latent_count={latent_count}")
    
    # 构建输出
    output_tokens = [t["token_id"] for t in generated_tokens if t["token_id"] != -1]
    output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
    
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
        "latent_count": latent_count,
    }


# ============================================================================
# CODI 模型加载和自适应推理
# ============================================================================

def load_codi_model(
    base_model_path: str,
    ckpt_dir: str,
    device: torch.device,
    num_latent: int = 5,
    use_bf16: bool = True,
    lora_r: int = 128,
    lora_alpha: int = 16,
):
    """
    加载 CODI 模型 (完全参考 CODI/test.py 第87-142行)
    """
    print(f"Loading CODI model...")
    print(f"  Base model: {base_model_path}")
    print(f"  Checkpoint dir: {ckpt_dir}")
    
    # Step 1: 创建 LoRA config (test.py 第88-106行)
    task_type = TaskType.CAUSAL_LM
    if any(name in base_model_path.lower() for name in ["llama", "mistral", "falcon", "qwen"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    elif any(name in base_model_path.lower() for name in ["phi"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
    elif any(name in base_model_path.lower() for name in ["gpt2"]):
        target_modules = ["c_attn", "c_proj", "c_fc"]
    else:
        raise ValueError(f"Only support LLAMA, Mistral, Falcon, Phi-2, GPT2, but got {base_model_path}.")
    
    lora_config = LoraConfig(
        task_type=task_type,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=target_modules,
        init_lora_weights=True,
    )
    
    # Step 2: 创建 model_args 和 training_args (test.py 第110行需要的参数)
    model_args = ModelArguments(
        model_name_or_path=base_model_path,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_init=True,
        train=False,  # 推理模式
    )
    
    training_args = TrainingArguments(
        output_dir="./tmp",
        num_latent=num_latent,
        use_lora=True,
        bf16=use_bf16,
        use_prj=True,
    )
    
    # Step 3: 创建 CODI 模型 (test.py 第110行)
    model = CODI(model_args, training_args, lora_config)
    
    # Step 4: 加载 checkpoint (test.py 第114-122行)
    if ckpt_dir and os.path.exists(ckpt_dir):
        try:
            state_dict = load_file(os.path.join(ckpt_dir, "model.safetensors"))
        except Exception:
            state_dict = torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"), map_location='cpu')
        
        model.load_state_dict(state_dict, strict=False)
        model.codi.tie_weights()
    
    # Step 5: 加载 tokenizer (test.py 第124-138行)
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
    
    # Step 6: 移动到设备 (test.py 第140-142行)
    model = model.to(device)
    if use_bf16:
        model = model.to(torch.bfloat16)
    model.eval()
    
    return model, tokenizer


def codi_adaptive_generate(
    model: CODI,
    tokenizer,
    controller: AdaptiveController,
    input_text: str,
    max_new_tokens: int = 256,
    max_latent_iterations: int = 5,
    device: torch.device = None,
    verbose: bool = True,
    use_prj: bool = True,
    greedy: bool = True,
) -> Dict:
    """
    CODI 自适应生成 (参考 CODI/test.py 第263-388行)
    
    根据熵预测动态选择:
    - soft 模式: 进行 latent iteration
    - normal 模式: 生成普通 token
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Step 1: Tokenize (参考 test.py 第219-238行)
    batch = tokenizer(input_text, return_tensors="pt", padding=True)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    batch_size = input_ids.shape[0]
    
    # 添加 bot token (test.py 第232-236行)
    bot_tensor = torch.tensor([[model.bot_id]], dtype=torch.long, device=device).expand(batch_size, 1)
    input_ids = torch.cat([input_ids, bot_tensor], dim=1)
    attention_mask = torch.cat([attention_mask, torch.ones_like(bot_tensor)], dim=1)

    # 初始化控制器
    controller.init_state(batch_size, device)
    
    # 记录
    generated_tokens = []
    token_modes = []
    token_entropies = []
    switch_events = []
    latent_iterations = 0

    # Step 2: Encode the question (test.py 第267-270行)
    past_key_values = None
    with torch.no_grad():
        outputs = model.codi(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_key_values,
        )
        past_key_values = outputs.past_key_values
        hidden_states_for_controller = outputs.hidden_states[-1][:, -1, :]
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

        if use_prj and hasattr(model, 'prj'):
            latent_embd = model.prj(latent_embd)
    
    # 获取 embedding 函数 (test.py 第370行)
    def get_embd(token_ids):
        return model.get_embd(model.codi, model.model_name)(token_ids)    

    
    # 尝试获取 logits
    try:
        if hasattr(model.codi, 'lm_head'):
            logits = model.codi.lm_head(latent_embd)
        else:
            logits = model.codi.get_base_model().lm_head(latent_embd)
    except:
        logits = None
    
    mode, to_normal, to_soft, cur_entropy = controller.step(hidden_states_for_controller, 0, logits)
    current_mode = "normal" if mode[0].item() == 1 else "soft"
    token_modes.append(current_mode)
    token_entropies.append(cur_entropy[0].item())
    
    # Step 3: Latent iterations (参考 test.py 第286-305行，自适应版本)
    for i in range(max_latent_iterations):
        # 检查是否应该继续 latent iteration
        if current_mode == "normal":
            if verbose:
                print(f"  Stopping latent iterations at step {i} (switched to normal mode)")
            break
        
        latent_iterations += 1
        
        with torch.no_grad():

            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values

            hidden_states_for_controller = outputs.hidden_states[-1][:, -1, :]
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            
            if use_prj and hasattr(model, 'prj'):
                latent_embd = model.prj(latent_embd)
        
        # 更新控制器
        # latent_embd = latent_embd.squeeze(1)
        try:
            if hasattr(model.codi, 'lm_head'):
                logits = model.codi.lm_head(latent_embd)
            else:
                logits = model.codi.get_base_model().lm_head(latent_embd)
        except:
            logits = None
        
        mode, to_normal, to_soft, cur_entropy = controller.step(hidden_states_for_controller, i + 1, logits)
        current_mode = "normal" if mode[0].item() == 1 else "soft"
        
        if to_normal[0].item():
            switch_events.append((i + 1, "soft->normal (latent)", cur_entropy[0].item()))
        
        token_modes.append(current_mode)
        token_entropies.append(cur_entropy[0].item())
        
        if verbose:
            print(f"  Latent iteration {i+1}: mode={current_mode}, entropy={cur_entropy[0].item():.4f}")
    
    # Step 4: 添加 eot token (test.py 第307-314行)
    eot_emb = get_embd(torch.tensor([model.eot_id], dtype=torch.long, device=device)).unsqueeze(0)
    eot_emb = eot_emb.expand(batch_size, -1, -1)
    
    # Step 5: 生成 tokens (test.py 第316-370行)
    output_emb = eot_emb
    pred_tokens = [[] for _ in range(batch_size)]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    step_offset = latent_iterations + 1
    
    for step in range(max_new_tokens):
        with torch.no_grad():
            out = model.codi(
                inputs_embeds=output_emb,
                output_hidden_states=True,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = out.past_key_values
            
            # 获取 logits (test.py 第331行)
            vocab_size = model.codi.config.vocab_size - 1
            logits = out.logits[:, -1, :vocab_size]
            hidden_states = out.hidden_states[-1][:, -1, :]
        
        # 采样 (test.py 第334-356行)
        if greedy:
            next_token_ids = torch.argmax(logits, dim=-1)
        else:
            probs = F.softmax(logits / 0.1, dim=-1)
            next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        # 控制器决策
        end_token_mask = (next_token_ids == tokenizer.eos_token_id)
        mode, to_normal, to_soft, cur_entropy = controller.step(
            hidden_states, step + step_offset, logits, end_token_mask
        )
        current_mode = "normal" if mode[0].item() == 1 else "soft"
        
        if to_normal[0].item():
            switch_events.append((step + step_offset, "soft->normal", cur_entropy[0].item()))
        if to_soft[0].item():
            switch_events.append((step + step_offset, "normal->soft", cur_entropy[0].item()))
        
        token_modes.append(current_mode)
        token_entropies.append(cur_entropy[0].item())
        
        # 记录 token (test.py 第358-363行)
        for b in range(batch_size):
            if not finished[b]:
                pred_tokens[b].append(next_token_ids[b].item())
                if next_token_ids[b] == tokenizer.eos_token_id:
                    finished[b] = True
        
        generated_tokens.append({
            "token_id": next_token_ids[0].item(),
            "token_text": tokenizer.decode([next_token_ids[0].item()]),
            "mode": current_mode,
            "entropy": cur_entropy[0].item(),
        })
        
        if finished.all():
            break
        
        # 准备下一步输入 (test.py 第370行)
        output_emb = get_embd(next_token_ids).unsqueeze(1)
        
        if verbose and step % 20 == 0:
            print(f"  Step {step}: mode={current_mode}, entropy={cur_entropy[0].item():.4f}")
    
    # 构建输出 (test.py 第375行)
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
        "latent_iterations": latent_iterations,
    }


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Adaptive Inference with SwiReasoning-style control")
    
    # 模型参数
    parser.add_argument("--model_type", type=str, default="coconut",
                        choices=["coconut", "codi"],
                        help="Model type: coconut or codi")
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to base model (e.g., gpt2, meta-llama/Llama-3.2-1B-Instruct)")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to Coconut model checkpoint")
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="Path to CODI checkpoint directory (containing model.safetensors)")
    parser.add_argument("--predictor_path", type=str, default=None,
                        help="Path to trained EntropyPredictor")
    
    # 推理参数
    parser.add_argument("--input", type=str, default=None,
                        help="Input text for single inference")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Input file for batch inference")
    parser.add_argument("--output_file", type=str, default="results_adaptive.json",
                        help="Output file for results")
    
    # SwiReasoning 风格参数
    parser.add_argument("--window_size", type=int, default=5,
                        help="Minimum steps before mode switch")
    parser.add_argument("--max_switch_count", type=int, default=None,
                        help="Maximum switch count")
    parser.add_argument("--use_predicted_entropy", action="store_true",
                        help="Use EntropyPredictor instead of real entropy")
    
    # 生成参数
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Maximum new tokens to generate")
    parser.add_argument("--max_latent_steps", type=int, default=8,
                        help="Maximum latent steps (Coconut) or iterations (CODI)")
    parser.add_argument("--num_latent", type=int, default=5,
                        help="Number of latent for CODI")
    
    # 其他参数
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bf16")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    
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
        window_size=args.window_size,
        max_switch_count=args.max_switch_count,
        use_predicted_entropy=args.use_predicted_entropy and predictor is not None,
    )
    
    # 加载模型
    if args.model_type == "coconut":
        model, tokenizer, latent_id, start_id, end_id = load_coconut_model(
            args.base_model_path,
            args.checkpoint_path,
            device,
            use_bf16=args.bf16,
        )
        generate_fn = lambda text: coconut_adaptive_generate(
            model, tokenizer, controller, text,
            max_new_tokens=args.max_new_tokens,
            max_latent_steps=args.max_latent_steps,
            device=device,
            verbose=args.verbose,
        )
    elif args.model_type == "codi":  # codi
        model, tokenizer = load_codi_model(
            args.base_model_path,
            args.ckpt_dir,
            device,
            num_latent=args.num_latent,
            use_bf16=args.bf16,
        )
        generate_fn = lambda text: codi_adaptive_generate(
            model, tokenizer, controller, text,
            max_new_tokens=args.max_new_tokens,
            max_latent_iterations=args.max_latent_steps,
            device=device,
            verbose=args.verbose,
        )
    
    # 准备输入
    if args.input:
        inputs = [args.input]
    elif args.input_file:
        with open(args.input_file, 'r') as f:
            inputs = [line.strip() for line in f if line.strip()]
    else:
        # 交互模式
        print("\n" + "=" * 60)
        print(f"Interactive Mode ({args.model_type.upper()})")
        print("=" * 60)
        print("Commands: exit/quit/q -> Exit")
        print(f"\nSettings:")
        print(f"  window_size: {args.window_size}")
        print(f"  max_latent_steps: {args.max_latent_steps}")
        print(f"  use_predicted_entropy: {args.use_predicted_entropy}")
        print("\nEnter your questions:")
        
        while True:
            try:
                input_text = input("\n> ")
                
                if input_text.lower() in ['quit', 'exit', 'q']:
                    break
                
                # 重新初始化控制器状态
                controller.state = None
                
                result = generate_fn(input_text)
                print(f"\nOutput: {result['output']}")
                print(f"\nMode distribution: {result['mode_distribution']}")
                print(f"Total switches: {result['total_switches']}")
                print(f"Avg entropy: {result['avg_entropy']:.4f}")
                
                if 'latent_count' in result:
                    print(f"Latent count: {result['latent_count']}")
                if 'latent_iterations' in result:
                    print(f"Latent iterations: {result['latent_iterations']}")
                
                if result['switch_events']:
                    print("\nSwitch events:")
                    for step, event, entropy in result['switch_events']:
                        print(f"  Step {step}: {event} (entropy={entropy:.4f})")
                
            except KeyboardInterrupt:
                break
        
        return
    
    # 批量推理
    results = []
    print(f"\nProcessing {len(inputs)} inputs...")
    
    for i, input_text in enumerate(inputs):
        print(f"\n[{i+1}/{len(inputs)}] {input_text[:50]}...")
        
        # 重新初始化控制器状态
        controller.state = None
        
        result = generate_fn(input_text)
        results.append(result)
        
        print(f"  Output: {result['output'][:100]}...")
        print(f"  Mode: {result['mode_distribution']}")
    
    # 保存结果
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output_file}")
    
    # 统计
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    
    total_normal = sum(r["mode_distribution"]["normal"] for r in results)
    total_soft = sum(r["mode_distribution"]["soft"] for r in results)
    total_switches = sum(r["total_switches"] for r in results)
    avg_entropy = sum(r["avg_entropy"] for r in results) / len(results)
    
    print(f"Total tokens: {total_normal + total_soft}")
    print(f"  Normal mode: {total_normal} ({total_normal/(total_normal+total_soft)*100:.1f}%)")
    print(f"  Soft mode: {total_soft} ({total_soft/(total_normal+total_soft)*100:.1f}%)")
    print(f"Total switches: {total_switches}")
    print(f"Average entropy: {avg_entropy:.4f}")


if __name__ == "__main__":
    main()