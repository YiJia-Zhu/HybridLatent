"""
Phase 2: 半显半隐推理训练 - 模型定义

核心改动：
1. 支持混合序列训练：部分显式输出 + 部分隐式循环
2. Loss设计：
   - 显式segment: Cross-Entropy Loss
   - 隐式segment: Distillation Loss (对齐到teacher对应位置)
3. 支持从Phase 1生成的数据训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from peft import get_peft_model, LoraConfig, TaskType
from torch.cuda.amp import autocast
import random
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class HybridModelArguments:
    model_name_or_path: str = field(default="meta-llama/Llama-3.2-1B-Instruct")
    lora_r: int = field(default=128)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.1)
    full_precision: bool = field(default=True)
    lora_init: bool = field(default=True)
    token: Optional[str] = field(default=None)


@dataclass
class HybridTrainingArguments(transformers.TrainingArguments):
    # 基础参数
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(default=512)
    num_latent: int = field(default=6)
    use_lora: bool = field(default=True)
    bf16: bool = field(default=True)
    
    # Projection层参数
    use_prj: bool = field(default=True)
    prj_dim: int = field(default=2048)
    prj_dropout: float = field(default=0.0)
    
    # Loss权重
    ce_loss_weight: float = field(default=1.0)
    distill_loss_weight: float = field(default=0.5)
    ref_loss_weight: float = field(default=0.3)
    
    # Distillation参数
    distill_loss_type: str = field(default="smooth_l1")
    distill_all_layers: bool = field(default=True)
    
    # 训练策略
    max_latent_per_segment: int = field(default=3)  # 每个隐式segment最多几步
    warmup_explicit_ratio: float = field(default=0.3)  # 预热阶段显式比例
    curriculum_learning: bool = field(default=True)  # 是否使用课程学习
    
    # 实验参数
    exp_mode: bool = field(default=False)
    exp_data_num: int = field(default=1000)
    max_token_num: int = field(default=512)
    print_loss: bool = field(default=True)
    expt_name: str = field(default="hybrid_codi")


def print_trainable_parameters(model):
    trainable_parameters = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    print(f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param:.2f}")


class HybridCODI(nn.Module):
    """
    半显半隐CODI模型
    
    支持混合序列训练：
    - 显式token: 正常的next token prediction
    - 隐式token ([L]): 用隐层循环，蒸馏对齐到teacher
    """
    
    def __init__(self, model_args: HybridModelArguments, training_args: HybridTrainingArguments, lora_config=None):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path
        
        # 加载基础模型
        dtype = torch.bfloat16 if training_args.bf16 else torch.float16
        self.codi = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            use_flash_attention_2=False,
            resume_download=True,
        )
        
        # 特殊token IDs
        ori_vocab_size = self.codi.config.vocab_size
        self.pad_token_id = ori_vocab_size
        self.bot_id = ori_vocab_size + 1      # Begin of Thought
        self.eot_id = ori_vocab_size + 2      # End of Thought
        self.latent_id = ori_vocab_size + 3   # Latent token [L]
        
        # Resize embeddings
        self.codi.resize_token_embeddings(ori_vocab_size + 4)
        
        self.dim = self.codi.config.hidden_size
        self.num_latent = training_args.num_latent
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token_id = self.pad_token_id
        
        # 添加特殊token到tokenizer
        special_tokens = {
            'additional_special_tokens': ['<|bot|>', '<|eot|>', '<|latent|>']
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        # LoRA
        if training_args.use_lora and lora_config is not None:
            self.codi = get_peft_model(self.codi, lora_config)
        
        # Projection Layer
        self.use_prj = training_args.use_prj
        if training_args.use_prj:
            self.prj = nn.Sequential(
                nn.Dropout(training_args.prj_dropout),
                nn.Linear(self.dim, training_args.prj_dim),
                nn.GELU(),
                nn.Linear(training_args.prj_dim, self.dim),
                nn.LayerNorm(self.dim),
            )
        
        # Loss functions
        self.ce_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        
        if training_args.distill_loss_type == "smooth_l1":
            self.distill_loss_fct = nn.SmoothL1Loss()
        elif training_args.distill_loss_type == "l2":
            self.distill_loss_fct = nn.MSELoss()
        elif training_args.distill_loss_type == "cosine":
            self.distill_loss_fct = nn.CosineEmbeddingLoss()
        else:
            self.distill_loss_fct = nn.SmoothL1Loss()
        
        # Loss权重
        self.ce_loss_weight = training_args.ce_loss_weight
        self.distill_loss_weight = training_args.distill_loss_weight
        self.ref_loss_weight = training_args.ref_loss_weight
        
        self.print_loss = training_args.print_loss
        
        print_trainable_parameters(self)
    
    def get_embd(self, model, model_name):
        """获取embedding层"""
        try:
            if "gpt2" in model_name.lower():
                try:
                    return model.get_base_model().transformer.wte
                except:
                    return model.transformer.wte
            else:  # llama, mistral, qwen, etc.
                try:
                    return model.get_base_model().model.embed_tokens
                except:
                    return model.model.embed_tokens
        except AttributeError:
            raise NotImplementedError(f"Unknown model: {model_name}")
    
    def forward(
        self,
        encoder_input_ids: torch.LongTensor,           # [B, seq_q] question + bot_id
        decoder_input_ids: torch.LongTensor,           # [B, seq_d] hybrid sequence (含[L] tokens)
        ref_input_ids: torch.LongTensor,               # [B, seq_r] full explicit sequence
        labels: torch.LongTensor,                      # [B, seq_d] decoder labels
        encoder_attention_mask: torch.LongTensor,
        ref_attention_mask: torch.LongTensor,
        ref_labels: torch.LongTensor,                  # [B, seq_r] reference labels
        implicit_mask: torch.BoolTensor,               # [B, seq_d] True for latent positions
        explicit_mask: torch.BoolTensor,               # [B, seq_d] True for explicit positions
        ref_answer_position: torch.LongTensor = None,  # [B] 答案在ref中的位置
        model_answer_position: torch.LongTensor = None,# [B] 答案在decoder中的位置
        step: int = None,
        step_ratio: float = None,
        **kwargs
    ):
        """
        半显半隐前向传播
        
        训练流程：
        1. 编码question
        2. 对于混合序列：
           - 显式token: 正常forward，计算CE loss
           - 隐式token ([L]): 用上一步hidden state，计算distillation loss
        3. 计算reference CE loss（保持显式推理能力）
        """
        batch_size = encoder_input_ids.shape[0]
        device = encoder_input_ids.device
        
        # ========== 1. 获取Teacher的hidden states ==========
        with torch.no_grad():
            ref_outputs = self.codi(
                input_ids=ref_input_ids,
                attention_mask=ref_attention_mask,
                output_hidden_states=True,
            )
        ref_hidden_states = ref_outputs.hidden_states[-1]  # [B, seq_r, dim]
        
        # ========== 2. 编码Question ==========
        with autocast(dtype=torch.bfloat16):
            encoder_outputs = self.codi(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                use_cache=True,
                output_hidden_states=True,
            )
        past_key_values = encoder_outputs.past_key_values
        last_hidden = encoder_outputs.hidden_states[-1][:, -1, :]  # [B, dim]
        
        # ========== 3. 半显半隐生成 ==========
        # 初始化
        all_logits = []
        all_hidden_states = []
        distill_losses = []
        
        # 获取embedding
        get_embd = lambda ids: self.get_embd(self.codi, self.model_name)(ids)
        
        # 当前latent embedding（用于隐式步骤）
        current_latent = last_hidden.unsqueeze(1)  # [B, 1, dim]
        if self.use_prj:
            current_latent = self.prj(current_latent)
        
        seq_len = decoder_input_ids.shape[1]
        
        # 找到ref中对应的位置映射
        # 简化处理：假设隐式token对应ref中的某些步骤
        # 实际中需要更精确的对齐
        
        for t in range(seq_len):
            is_implicit = implicit_mask[:, t]  # [B]
            
            if is_implicit.any():
                # 隐式步骤：使用latent embedding
                # 对于batch中的隐式样本
                with autocast(dtype=torch.bfloat16):
                    implicit_out = self.codi(
                        inputs_embeds=current_latent,
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_hidden_states=True,
                    )
                
                implicit_hidden = implicit_out.hidden_states[-1][:, -1, :]  # [B, dim]
                implicit_logits = implicit_out.logits[:, -1, :]  # [B, vocab]
                
                # 更新latent
                new_latent = implicit_hidden.unsqueeze(1)
                if self.use_prj:
                    new_latent = self.prj(new_latent)
                
                # 计算distillation loss（对齐到ref对应位置）
                # 这里简化处理：对齐到ref_answer_position附近
                if ref_answer_position is not None:
                    # 计算当前步骤应该对齐到ref的哪个位置
                    # 简化：按比例映射
                    ref_pos = (ref_answer_position.float() * t / max(seq_len - 1, 1)).long()
                    ref_pos = ref_pos.clamp(0, ref_hidden_states.shape[1] - 1)
                    
                    ref_target = ref_hidden_states.gather(
                        1, ref_pos.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.dim)
                    ).squeeze(1)  # [B, dim]
                    
                    # 只对implicit位置计算distill loss
                    if is_implicit.sum() > 0:
                        distill_loss = self.distill_loss_fct(
                            implicit_hidden[is_implicit],
                            ref_target[is_implicit].detach()
                        )
                        distill_losses.append(distill_loss)
                
                # 更新past_key_values
                past_key_values = implicit_out.past_key_values
                current_latent = new_latent
                
                all_logits.append(implicit_logits)
                all_hidden_states.append(implicit_hidden)
                
            if (~is_implicit).any():
                # 显式步骤：使用token embedding
                token_ids = decoder_input_ids[:, t:t+1]  # [B, 1]
                token_embeds = get_embd(token_ids)  # [B, 1, dim]
                
                with autocast(dtype=torch.bfloat16):
                    explicit_out = self.codi(
                        inputs_embeds=token_embeds,
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_hidden_states=True,
                    )
                
                explicit_hidden = explicit_out.hidden_states[-1][:, -1, :]
                explicit_logits = explicit_out.logits[:, -1, :]
                
                # 更新past_key_values
                past_key_values = explicit_out.past_key_values
                
                # 更新current_latent为显式输出
                current_latent = explicit_hidden.unsqueeze(1)
                if self.use_prj:
                    current_latent = self.prj(current_latent)
                
                all_logits.append(explicit_logits)
                all_hidden_states.append(explicit_hidden)
        
        # 组装logits
        if all_logits:
            logits = torch.stack(all_logits, dim=1)  # [B, seq, vocab]
        else:
            logits = torch.zeros(batch_size, seq_len, self.codi.config.vocab_size, device=device)
        
        # ========== 4. 计算Loss ==========
        # 4.1 显式部分的CE Loss
        ce_loss = torch.tensor(0.0, device=device)
        if explicit_mask.any():
            # 只对显式位置计算CE
            explicit_labels = labels.clone()
            explicit_labels[~explicit_mask] = -100  # mask掉隐式位置
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = explicit_labels[:, 1:].contiguous()
            
            ce_loss = self.ce_loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        # 4.2 隐式部分的Distillation Loss
        distill_loss = torch.tensor(0.0, device=device)
        if distill_losses:
            distill_loss = torch.stack(distill_losses).mean()
        
        # 4.3 Reference CE Loss（保持显式推理能力）
        ref_ce_loss = torch.tensor(0.0, device=device)
        if self.ref_loss_weight > 0:
            with autocast(dtype=torch.bfloat16):
                ref_out = self.codi(
                    input_ids=ref_input_ids,
                    attention_mask=ref_attention_mask,
                    output_hidden_states=False,
                )
            ref_logits = ref_out.logits
            shift_ref_logits = ref_logits[:, :-1, :].contiguous()
            shift_ref_labels = ref_labels[:, 1:].contiguous()
            
            ref_ce_loss = self.ce_loss_fct(
                shift_ref_logits.view(-1, shift_ref_logits.size(-1)),
                shift_ref_labels.view(-1)
            )
        
        # ========== 5. 总Loss ==========
        total_loss = (
            self.ce_loss_weight * ce_loss +
            self.distill_loss_weight * distill_loss +
            self.ref_loss_weight * ref_ce_loss
        )
        
        if self.print_loss and step is not None and step % 100 == 0:
            print(f"[Step {step}] loss={total_loss.item():.4f}, "
                  f"ce={ce_loss.item():.4f}, distill={distill_loss.item():.4f}, "
                  f"ref_ce={ref_ce_loss.item():.4f}")
        
        return {
            "loss": total_loss,
            "logits": logits,
            "ce_loss": ce_loss.detach(),
            "distill_loss": distill_loss.detach(),
            "ref_ce_loss": ref_ce_loss.detach(),
        }


class SimplifiedHybridCODI(nn.Module):
    """
    简化版半显半隐CODI - 更接近原始CODI结构
    
    关键改动：
    1. 保持CODI的隐式循环结构
    2. 在decoder阶段混合显式token和隐式循环
    3. 兼容原始推理代码
    """
    
    def __init__(self, model_args: HybridModelArguments, training_args: HybridTrainingArguments, lora_config=None):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path
        
        # 加载基础模型
        dtype = torch.bfloat16 if training_args.bf16 else torch.float16
        self.codi = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            use_flash_attention_2=False,
        )
        
        # 特殊token
        ori_vocab_size = self.codi.config.vocab_size
        self.pad_token_id = ori_vocab_size
        self.bot_id = ori_vocab_size + 1
        self.eot_id = ori_vocab_size + 2
        self.latent_id = ori_vocab_size + 3
        
        self.codi.resize_token_embeddings(ori_vocab_size + 4)
        
        self.dim = self.codi.config.hidden_size
        self.num_latent = training_args.num_latent
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = '[PAD]'
            self.tokenizer.pad_token_id = self.pad_token_id
        
        # LoRA
        if training_args.use_lora and lora_config:
            self.codi = get_peft_model(self.codi, lora_config)
        
        # Projection
        self.use_prj = training_args.use_prj
        if training_args.use_prj:
            self.prj = nn.Sequential(
                nn.Dropout(training_args.prj_dropout),
                nn.Linear(self.dim, training_args.prj_dim),
                nn.GELU(),
                nn.Linear(training_args.prj_dim, self.dim),
                nn.LayerNorm(self.dim),
            )
        
        # Loss
        self.ce_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.distill_loss_fct = nn.SmoothL1Loss()
        
        self.ce_weight = training_args.ce_loss_weight
        self.distill_weight = training_args.distill_loss_weight
        self.ref_weight = training_args.ref_loss_weight
        
        print_trainable_parameters(self)
    
    def get_embd(self, model, model_name):
        try:
            if "gpt2" in model_name.lower():
                try:
                    return model.get_base_model().transformer.wte
                except:
                    return model.transformer.wte
            else:
                try:
                    return model.get_base_model().model.embed_tokens
                except:
                    return model.model.embed_tokens
        except:
            raise NotImplementedError
    
    def forward(
        self,
        encoder_input_ids: torch.LongTensor,
        decoder_input_ids: torch.LongTensor,
        ref_input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        encoder_attention_mask: torch.LongTensor,
        ref_attention_mask: torch.LongTensor,
        ref_labels: torch.LongTensor,
        implicit_mask: torch.BoolTensor,
        explicit_mask: torch.BoolTensor,
        num_implicit_steps: torch.LongTensor = None,
        ref_answer_position: torch.LongTensor = None,
        model_answer_position: torch.LongTensor = None,
        step: int = None,
        step_ratio: float = None,
        **kwargs
    ):
        """
        简化的半显半隐前向传播
        
        核心思路：
        1. 编码question，进入隐式循环
        2. 隐式循环中，遇到显式标记就输出token
        3. 最后输出答案
        """
        device = encoder_input_ids.device
        batch_size = encoder_input_ids.shape[0]
        
        # 1. Teacher forward (for distillation target)
        with torch.no_grad():
            ref_outputs = self.codi(
                input_ids=ref_input_ids,
                attention_mask=ref_attention_mask,
                output_hidden_states=True,
            )
        ref_hidden = ref_outputs.hidden_states[-1]
        
        # 2. Encode question
        with autocast(dtype=torch.bfloat16):
            enc_out = self.codi(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                use_cache=True,
                output_hidden_states=True,
            )
        past_kv = enc_out.past_key_values
        latent_embd = enc_out.hidden_states[-1][:, -1:, :]  # [B, 1, dim]
        
        if self.use_prj:
            latent_embd = self.prj(latent_embd)
        
        # 3. 计算隐式循环次数
        # implicit_mask中True的数量决定隐式步数
        num_latent = self.num_latent
        
        # 4. 隐式循环 + 蒸馏
        distill_loss = torch.tensor(0.0, device=device)
        
        for i in range(num_latent):
            with autocast(dtype=torch.bfloat16):
                out = self.codi(
                    inputs_embeds=latent_embd,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_hidden_states=True,
                )
            past_kv = out.past_key_values
            latent_embd = out.hidden_states[-1][:, -1:, :]
            
            if self.use_prj:
                latent_embd = self.prj(latent_embd)
            
            # Distillation: 对齐到ref的对应位置
            if ref_answer_position is not None:
                # 简化：每个latent步骤对齐到ref的某个位置
                # 可以改进为更精确的对齐
                target_pos = ref_answer_position - (num_latent - i)
                target_pos = target_pos.clamp(0, ref_hidden.shape[1] - 1)
                
                ref_target = ref_hidden.gather(
                    1, target_pos.view(-1, 1, 1).expand(-1, 1, self.dim)
                )  # [B, 1, dim]
                
                d_loss = self.distill_loss_fct(latent_embd, ref_target.detach())
                distill_loss = distill_loss + d_loss
        
        distill_loss = distill_loss / max(num_latent, 1)
        
        # 5. Decode answer
        get_embd = lambda ids: self.get_embd(self.codi, self.model_name)(ids)
        answer_embds = get_embd(decoder_input_ids)
        
        with autocast(dtype=torch.bfloat16):
            dec_out = self.codi(
                inputs_embeds=answer_embds,
                past_key_values=past_kv,
                use_cache=False,
                output_hidden_states=True,
            )
        
        logits = dec_out.logits
        
        # 6. CE Loss for decoder
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        ce_loss = self.ce_loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # 7. Reference CE Loss
        ref_ce_loss = torch.tensor(0.0, device=device)
        if self.ref_weight > 0:
            with autocast(dtype=torch.bfloat16):
                ref_out_grad = self.codi(
                    input_ids=ref_input_ids,
                    attention_mask=ref_attention_mask,
                )
            ref_logits = ref_out_grad.logits
            shift_ref = ref_logits[:, :-1, :].contiguous()
            shift_ref_labels = ref_labels[:, 1:].contiguous()
            ref_ce_loss = self.ce_loss_fct(
                shift_ref.view(-1, shift_ref.size(-1)),
                shift_ref_labels.view(-1)
            )
        
        # 8. Total loss
        total_loss = (
            self.ce_weight * ce_loss +
            self.distill_weight * distill_loss +
            self.ref_weight * ref_ce_loss
        )
        
        if step is not None and step % 100 == 0:
            print(f"[Step {step}] loss={total_loss.item():.4f}, "
                  f"ce={ce_loss.item():.4f}, distill={distill_loss.item():.4f}, "
                  f"ref_ce={ref_ce_loss.item():.4f}")
        
        return {
            "loss": total_loss,
            "logits": logits,
            "ce_loss": ce_loss.detach(),
            "distill_loss": distill_loss.detach(),
            "ref_ce_loss": ref_ce_loss.detach(),
        }
