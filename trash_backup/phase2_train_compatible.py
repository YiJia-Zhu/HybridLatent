"""
Phase 2: 半显半隐训练 - 与原始CODI兼容版本

这个版本直接使用原始的CODI模型结构(model_hybrid.py)，
只修改数据处理和训练流程，确保训练好的模型可以直接用step4_adaptive_eval.py推理。

关键设计：
1. 使用原始CODI模型，保持结构不变
2. 混合训练数据：
   - 熵引导样本：学会在合适位置用隐式
   - 随机样本：鲁棒性
   - 纯显式样本：保持显式推理能力
3. 渐进式训练：先学显式，再学隐式

使用方法:
    python phase2_train_compatible.py \
        --model_name_or_path ./CODI/pretrained/Llama-3.2-1B-Instruct \
        --hybrid_data_path ./data/hybrid_training/hybrid_training_data.json \
        --output_dir ./checkpoints/hybrid_codi_v2 \
        --num_train_epochs 5 \
        --per_device_train_batch_size 4 \
        --learning_rate 2e-5 \
        --bf16
"""

import os
import sys
import json
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import transformers
from transformers import Trainer, AutoTokenizer
from peft import LoraConfig, TaskType
from tqdm import tqdm
from datasets import load_dataset

# 添加CODI路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, 'CODI'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'CODI', 'src'))

# 导入原始CODI模型
from src.model_hybrid import (
    CODI,
    ModelArguments,
    TrainingArguments,
    DataArguments,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


# ============================================================================
# 辅助函数
# ============================================================================

def _to_scalar(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().float().mean().item()
    return float(x)


def _tokenize_fn(strings: Sequence[str], tokenizer, max_length: int = 256) -> Dict:
    """Tokenize字符串列表"""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
            return_attention_mask=False
        )
        for text in strings
    ]
    input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
    return {"input_ids": input_ids}


# ============================================================================
# 混合数据集
# ============================================================================

class HybridCODIDataset(Dataset):
    """
    半显半隐CODI训练数据集
    
    利用Phase 1生成的token_modes信息:
    - token_modes: 每个token的模式 (0=latent, 1=normal)
    - 根据实际隐式token数量动态计算num_latent
    
    输出格式与原始CODI兼容
    """
    
    def __init__(
        self,
        hybrid_data_path: str,
        tokenizer,
        bot_id: int,
        eot_id: int,
        num_latent: int = 6,
        max_token_num: int = 512,
        use_dynamic_latent: bool = True,  # 是否根据样本动态调整latent数量
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.bot_id = bot_id
        self.eot_id = eot_id
        self.num_latent = num_latent
        self.max_token_num = max_token_num
        self.use_dynamic_latent = use_dynamic_latent
        
        # 加载混合数据
        logger.info(f"Loading hybrid data from {hybrid_data_path}")
        with open(hybrid_data_path, 'r', encoding='utf-8') as f:
            hybrid_data = json.load(f)
        
        # 处理数据
        self.data_dict = self._preprocess(hybrid_data)
        self.keys = list(self.data_dict.keys())
        
        logger.info(f"Loaded {len(self.data_dict['encoder_input_ids'])} training samples")
    
    def _get_answer_token_position(self, tokens, answer_prompts):
        """找到answer prompt的位置"""
        for answer_prompt in answer_prompts:
            try:
                match_indices = (tokens.unfold(0, len(answer_prompt), 1) == answer_prompt).all(dim=1).nonzero(as_tuple=True)[0].item()
                return match_indices + len(answer_prompt)
            except:
                continue
        # 如果找不到，返回一个合理的默认位置
        return len(tokens) - 5
    
    def _compute_effective_latent(self, token_modes: List[int]) -> int:
        """
        根据token_modes计算有效的latent步数
        
        策略：计算连续latent段的数量，而不是总latent token数
        这样更符合CODI的隐式循环概念
        """
        if not token_modes:
            return 0
        
        # 计算连续latent段的数量
        num_latent_segments = 0
        in_latent = False
        
        for mode in token_modes:
            if mode == 0 and not in_latent:  # 进入latent段
                num_latent_segments += 1
                in_latent = True
            elif mode == 1:
                in_latent = False
        
        # 限制在合理范围内
        return min(max(num_latent_segments, 1), self.num_latent)
    
    def _preprocess(self, hybrid_data: List[Dict]) -> Dict:
        """预处理数据为CODI格式"""
        
        questions = []
        cots = []
        answers = []
        pattern_types = []
        effective_latents = []  # 每个样本的有效latent步数
        
        # 统计信息
        total_latent_tokens = 0
        total_explicit_tokens = 0
        
        for sample in tqdm(hybrid_data, desc="Processing samples"):
            question = sample["question"].strip()
            full_cot = sample["full_cot"].strip()
            answer = sample["answer"]
            pattern_type = sample.get("pattern_type", "entropy_guided")
            token_modes = sample.get("token_modes", [])
            
            # 检查长度
            total_len = len(self.tokenizer.encode(question + full_cot + str(answer)))
            if total_len > self.max_token_num:
                continue
            
            # 处理answer格式
            if not str(answer).startswith("The answer is"):
                answer = f"The answer is: {answer}"
            answer = answer.replace("####", "").strip()
            
            # 计算有效latent步数
            if self.use_dynamic_latent and token_modes:
                eff_latent = self._compute_effective_latent(token_modes)
                # 统计
                total_latent_tokens += sum(1 for m in token_modes if m == 0)
                total_explicit_tokens += sum(1 for m in token_modes if m == 1)
            else:
                eff_latent = self.num_latent
            
            questions.append(question)
            cots.append(full_cot)
            answers.append(answer)
            pattern_types.append(pattern_type)
            effective_latents.append(eff_latent)
        
        # 打印统计
        if total_latent_tokens + total_explicit_tokens > 0:
            latent_ratio = total_latent_tokens / (total_latent_tokens + total_explicit_tokens)
            logger.info(f"Token statistics: latent={total_latent_tokens}, explicit={total_explicit_tokens}, ratio={latent_ratio:.2%}")
        
        logger.info(f"Processed {len(questions)} valid samples")
        
        # Tokenize
        logger.info("Tokenizing...")
        sources_id = _tokenize_fn(questions, self.tokenizer)["input_ids"]
        cot_id = _tokenize_fn(cots, self.tokenizer)["input_ids"]
        answers_id = _tokenize_fn(answers, self.tokenizer)["input_ids"]
        
        # 添加eos token
        sources_id = [torch.tensor(x.numpy().tolist() + [self.tokenizer.eos_token_id], dtype=torch.long) for x in sources_id]
        cot_id = [torch.tensor(x.numpy().tolist() + [self.tokenizer.eos_token_id], dtype=torch.long) for x in cot_id]
        answers_id = [torch.tensor(x.numpy().tolist() + [self.tokenizer.eos_token_id], dtype=torch.long) for x in answers_id]
        
        # 处理bos token
        if cot_id[0][0] == self.tokenizer.bos_token_id:
            cot_id = [x[1:] for x in cot_id]
            answers_id = [x[1:] for x in answers_id]
        
        # 构建ref序列 (完整显式序列，用于蒸馏)
        ref_input_ids = [torch.cat([x, y, z]).to(torch.long) for x, y, z in zip(sources_id, cot_id, answers_id)]
        
        # ref_labels
        ref_labels = []
        for x, y in zip(ref_input_ids, sources_id):
            z = x.clone()
            z[:len(y)] = IGNORE_INDEX
            ref_labels.append(z)
        
        # 添加bot_id到source
        sources_id = [torch.tensor(x.numpy().tolist() + [self.bot_id], dtype=torch.long) for x in sources_id]
        
        # decoder序列
        answers_id = [torch.tensor([self.eot_id, self.tokenizer.eos_token_id] + x.numpy().tolist(), dtype=torch.long) for x in answers_id]
        
        # 找answer position
        answer_prompts = [
            torch.tensor(self.tokenizer.encode("The answer is:")),
            torch.tensor(self.tokenizer.encode("The next step result is:"))
        ]
        if answer_prompts[0][0] == self.tokenizer.bos_token_id:
            answer_prompts = [x[1:] for x in answer_prompts]
        
        ref_answer_position = [self._get_answer_token_position(x, answer_prompts) for x in ref_input_ids]
        model_answer_position = [self._get_answer_token_position(x, answer_prompts) for x in answers_id]
        
        return {
            "encoder_input_ids": sources_id,
            "decoder_input_ids": answers_id,
            "ref_input_ids": ref_input_ids,
            "labels": answers_id,
            "ref_answer_position": ref_answer_position,
            "model_answer_position": model_answer_position,
            "ref_labels": ref_labels,
            "pattern_type": pattern_types,
            "effective_latent": effective_latents,  # 新增：每个样本的有效latent步数
        }
    
    def __len__(self):
        return len(self.data_dict["encoder_input_ids"])
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        result = {key: self.data_dict[key][i] for key in self.keys if key not in ["pattern_type"]}
        # 将effective_latent转为tensor
        if "effective_latent" in result:
            result["effective_latent"] = torch.tensor(result["effective_latent"], dtype=torch.long)
        return result


@dataclass
class HybridDataCollator:
    """数据Collator - 与原始CODI兼容"""
    tokenizer: transformers.PreTrainedTokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        encoder_input_ids = [instance["encoder_input_ids"] for instance in instances]
        decoder_input_ids = [instance["decoder_input_ids"] for instance in instances]
        ref_input_ids = [instance["ref_input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        ref_answer_position = [instance["ref_answer_position"] for instance in instances]
        model_answer_position = [instance["model_answer_position"] for instance in instances]
        ref_labels = [instance["ref_labels"] for instance in instances]
        
        # 新增：effective_latent
        effective_latent = None
        if "effective_latent" in instances[0]:
            effective_latent = torch.stack([instance["effective_latent"] for instance in instances])
        
        # Left pad encoder
        reversed_input_ids = [seq.flip(0) for seq in encoder_input_ids]
        encoder_input_ids = torch.nn.utils.rnn.pad_sequence(
            reversed_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).flip(1)
        
        # Pad others
        ref_input_ids = torch.nn.utils.rnn.pad_sequence(
            ref_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        ref_labels = torch.nn.utils.rnn.pad_sequence(
            ref_labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        decoder_input_ids = torch.nn.utils.rnn.pad_sequence(
            decoder_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        
        result = {
            "encoder_input_ids": encoder_input_ids,
            "decoder_input_ids": decoder_input_ids,
            "ref_input_ids": ref_input_ids,
            "labels": labels,
            "encoder_attention_mask": encoder_input_ids.ne(self.tokenizer.pad_token_id),
            "ref_answer_position": torch.tensor(ref_answer_position, dtype=torch.long),
            "model_answer_position": torch.tensor(model_answer_position, dtype=torch.long),
            "ref_attention_mask": ref_input_ids.ne(self.tokenizer.pad_token_id),
            "ref_labels": ref_labels,
        }
        
        if effective_latent is not None:
            result["effective_latent"] = effective_latent
        
        return result


# ============================================================================
# 自定义Trainer
# ============================================================================

class HybridCODITrainer(Trainer):
    """半显半隐CODI Trainer"""
    
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        step = self.state.global_step
        
        # 计算总步数
        batch_size = self.args.per_device_train_batch_size
        gradient_accumulation_steps = self.args.gradient_accumulation_steps
        num_epochs = self.args.num_train_epochs
        dataset_size = len(self.train_dataset)
        
        effective_batch_size = batch_size * self.args.world_size * gradient_accumulation_steps
        total_steps = ceil(dataset_size / effective_batch_size) * num_epochs
        
        # 添加step信息
        inputs["step_ratio"] = step / max(total_steps, 1)
        inputs["step"] = step
        
        # 移除原始CODI模型不接受的参数
        # effective_latent是我们添加的，但原始CODI.forward()不接受
        inputs_for_model = {k: v for k, v in inputs.items() if k != "effective_latent"}
        
        # Forward
        outputs = model(**inputs_for_model)
        loss = outputs["loss"]
        
        # Logging
        if step % self.args.logging_steps == 0:
            logs = {
                "loss": _to_scalar(loss),
                "ce_loss": _to_scalar(outputs.get("ce_loss")),
                "distill_loss": _to_scalar(outputs.get("distill_loss")),
                "ref_ce_loss": _to_scalar(outputs.get("ref_ce_loss")),
            }
            self.log(logs)
        
        if return_outputs:
            return loss, outputs
        return loss


# ============================================================================
# 主函数
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 2: Hybrid CODI Training (Compatible)")
    
    # 模型参数
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    # 数据参数
    parser.add_argument("--hybrid_data_path", type=str, required=True,
                        help="Path to Phase 1 generated hybrid data")
    parser.add_argument("--max_token_num", type=int, default=512)
    
    # 训练参数
    parser.add_argument("--output_dir", type=str, default="./checkpoints/hybrid_codi")
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    
    # CODI参数
    parser.add_argument("--num_latent", type=int, default=6)
    parser.add_argument("--use_prj", action="store_true", default=True)
    parser.add_argument("--prj_dim", type=int, default=2048)
    parser.add_argument("--bf16", action="store_true", default=True)
    
    # Loss参数
    parser.add_argument("--distill_loss_factor", type=float, default=1.0)
    parser.add_argument("--ref_loss_factor", type=float, default=1.0)
    parser.add_argument("--hybrid_cot_only_ratio", type=float, default=0.2,
                        help="比例的样本只用CoT loss（保持显式推理能力）")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--restore_from", type=str, default="",
                        help="从现有CODI checkpoint继续训练")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # LoRA配置
    if "llama" in args.model_name_or_path.lower() or "mistral" in args.model_name_or_path.lower() or "qwen" in args.model_name_or_path.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    elif "phi" in args.model_name_or_path.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
    elif "gpt2" in args.model_name_or_path.lower():
        target_modules = ["c_attn", "c_proj", "c_fc"]
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=target_modules,
    )
    
    # 模型参数 (使用原始CODI的参数类)
    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_init=True,
        train=True,
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        num_latent=args.num_latent,
        use_lora=True,
        use_prj=args.use_prj,
        prj_dim=args.prj_dim,
        distill_loss_factor=args.distill_loss_factor,
        ref_loss_factor=args.ref_loss_factor,
        hybrid_cot_only_ratio=args.hybrid_cot_only_ratio,
        max_token_num=args.max_token_num,
        remove_unused_columns=False,
        restore_from=args.restore_from,
        expt_name="hybrid_codi",
    )
    
    # 创建模型 (使用原始CODI)
    logger.info("Creating CODI model...")
    model = CODI(model_args, training_args, lora_config)
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id
    
    # 创建数据集
    logger.info("Creating dataset...")
    train_dataset = HybridCODIDataset(
        hybrid_data_path=args.hybrid_data_path,
        tokenizer=tokenizer,
        bot_id=model.bot_id,
        eot_id=model.eot_id,
        num_latent=args.num_latent,
        max_token_num=args.max_token_num,
    )
    
    # Data collator
    data_collator = HybridDataCollator(tokenizer=tokenizer)
    
    # Trainer
    trainer = HybridCODITrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 训练
    logger.info("Starting training...")
    logger.info(f"  Total samples: {len(train_dataset)}")
    logger.info(f"  Epochs: {args.num_train_epochs}")
    logger.info(f"  Batch size: {args.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Hybrid CoT-only ratio: {args.hybrid_cot_only_ratio}")
    
    trainer.train()
    
    # 保存
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    trainer.save_state()
    
    # 保存配置信息
    config_info = {
        "model_name_or_path": args.model_name_or_path,
        "num_latent": args.num_latent,
        "prj_dim": args.prj_dim,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "hybrid_cot_only_ratio": args.hybrid_cot_only_ratio,
    }
    with open(os.path.join(args.output_dir, "hybrid_config.json"), 'w') as f:
        json.dump(config_info, f, indent=2)
    
    logger.info("Training complete!")
    logger.info(f"\n模型已保存到: {args.output_dir}")
    logger.info(f"可以使用以下命令进行推理:")
    logger.info(f"  python step4_adaptive_eval.py \\")
    logger.info(f"      --model_type codi \\")
    logger.info(f"      --base_model_path {args.model_name_or_path} \\")
    logger.info(f"      --ckpt_dir {args.output_dir} \\")
    logger.info(f"      --prj_dim {args.prj_dim} \\")
    logger.info(f"      --data_name gsm8k \\")
    logger.info(f"      --bf16")


if __name__ == "__main__":
    main()
