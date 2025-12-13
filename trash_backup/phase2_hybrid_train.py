"""
Phase 2: 半显半隐推理训练脚本

使用Phase 1生成的数据训练CODI模型，使其学会：
1. 显式生成：正常输出文本token
2. 隐式压缩：将若干显式步骤压缩到隐层循环
3. 无缝切换：从隐层状态能顺畅接续显式生成

使用方法:
    python phase2_hybrid_train.py \
        --model_name_or_path ./CODI/pretrained/Llama-3.2-1B-Instruct \
        --data_path ./data/hybrid_training/hybrid_training_data.json \
        --output_dir ./checkpoints/hybrid_codi \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 \
        --learning_rate 2e-5 \
        --bf16
"""

import os
import sys
import json
import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
from math import ceil

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import Trainer, TrainingArguments as HFTrainingArguments
from peft import LoraConfig, TaskType
from tqdm import tqdm
from datasets import load_dataset

# 导入模型
from phase2_model_hybrid import (
    HybridCODI,
    SimplifiedHybridCODI,
    HybridModelArguments,
    HybridTrainingArguments,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


# ============================================================================
# 数据处理
# ============================================================================

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


class HybridTrainingDataset(Dataset):
    """半显半隐训练数据集"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        bot_id: int,
        eot_id: int,
        latent_id: int,
        max_length: int = 512,
        latent_token: str = "[L]",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.bot_id = bot_id
        self.eot_id = eot_id
        self.latent_id = latent_id
        self.max_length = max_length
        self.latent_token = latent_token
        
        # 加载数据
        logger.info(f"Loading data from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        logger.info(f"Processing {len(raw_data)} samples...")
        self.data = self._preprocess(raw_data)
        logger.info(f"Processed {len(self.data)} valid samples")
    
    def _preprocess(self, raw_data: List[Dict]) -> List[Dict]:
        """预处理数据"""
        processed = []
        
        for sample in tqdm(raw_data, desc="Preprocessing"):
            try:
                item = self._process_sample(sample)
                if item is not None:
                    processed.append(item)
            except Exception as e:
                logger.warning(f"Error processing sample: {e}")
                continue
        
        return processed
    
    def _process_sample(self, sample: Dict) -> Optional[Dict]:
        """处理单个样本"""
        question = sample["question"]
        hybrid_seq = sample["hybrid_sequence"]
        full_cot = sample["full_cot"]
        answer = sample["answer"]
        
        # 1. Tokenize question
        question_tokens = self.tokenizer.encode(question, add_special_tokens=True)
        
        # 2. 处理hybrid sequence - 识别[L]位置
        # 将[L]替换为特殊标记，并记录位置
        parts = hybrid_seq.split(self.latent_token)
        
        hybrid_tokens = []
        implicit_positions = []
        
        for i, part in enumerate(parts):
            if part.strip():
                part_tokens = self.tokenizer.encode(part.strip(), add_special_tokens=False)
                hybrid_tokens.extend(part_tokens)
            
            # 在[L]位置添加latent token
            if i < len(parts) - 1:
                implicit_positions.append(len(hybrid_tokens))
                hybrid_tokens.append(self.latent_id)
        
        # 3. Tokenize answer
        answer_text = f"The answer is: {answer}"
        answer_tokens = self.tokenizer.encode(answer_text, add_special_tokens=False)
        
        # 4. Tokenize full CoT (for distillation)
        full_cot_tokens = self.tokenizer.encode(full_cot, add_special_tokens=False)
        
        # 5. 构建序列
        # encoder_input: question + bot_id
        encoder_input_ids = question_tokens + [self.bot_id]
        
        # decoder_input: eot_id + hybrid + answer + eos
        decoder_input_ids = [self.eot_id] + hybrid_tokens + answer_tokens
        if self.tokenizer.eos_token_id:
            decoder_input_ids.append(self.tokenizer.eos_token_id)
        
        # ref_input: question + full_cot + answer + eos
        ref_input_ids = question_tokens + full_cot_tokens + answer_tokens
        if self.tokenizer.eos_token_id:
            ref_input_ids.append(self.tokenizer.eos_token_id)
        
        # 6. 创建mask
        # implicit_mask: 标记decoder中的latent位置
        implicit_mask = [False] * len(decoder_input_ids)
        for pos in implicit_positions:
            if pos + 1 < len(implicit_mask):  # +1 因为decoder_input前面加了eot_id
                implicit_mask[pos + 1] = True
        
        explicit_mask = [not m for m in implicit_mask]
        
        # 7. Labels
        labels = decoder_input_ids.copy()
        ref_labels = ref_input_ids.copy()
        # mask掉question部分
        ref_labels[:len(question_tokens)] = [IGNORE_INDEX] * len(question_tokens)
        
        # 8. 找answer position
        # 简化处理：answer position是decoder中answer开始的位置
        answer_start = 1 + len(hybrid_tokens)  # eot + hybrid
        ref_answer_start = len(question_tokens) + len(full_cot_tokens)
        
        # 长度检查
        if len(encoder_input_ids) > self.max_length or len(decoder_input_ids) > self.max_length:
            return None
        
        return {
            "encoder_input_ids": torch.tensor(encoder_input_ids, dtype=torch.long),
            "decoder_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long),
            "ref_input_ids": torch.tensor(ref_input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "ref_labels": torch.tensor(ref_labels, dtype=torch.long),
            "implicit_mask": torch.tensor(implicit_mask, dtype=torch.bool),
            "explicit_mask": torch.tensor(explicit_mask, dtype=torch.bool),
            "model_answer_position": torch.tensor(answer_start, dtype=torch.long),
            "ref_answer_position": torch.tensor(ref_answer_start, dtype=torch.long),
            "num_implicit_steps": torch.tensor(len(implicit_positions), dtype=torch.long),
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class HybridDataCollator:
    """数据Collator"""
    
    def __init__(self, tokenizer, pad_token_id: int = 0):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
    
    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        # Pad encoder (left padding)
        encoder_input_ids = [s["encoder_input_ids"].flip(0) for s in instances]
        encoder_input_ids = torch.nn.utils.rnn.pad_sequence(
            encoder_input_ids, batch_first=True, padding_value=self.pad_token_id
        ).flip(1)
        
        # Pad decoder
        decoder_input_ids = [s["decoder_input_ids"] for s in instances]
        decoder_input_ids = torch.nn.utils.rnn.pad_sequence(
            decoder_input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        
        # Pad ref
        ref_input_ids = [s["ref_input_ids"] for s in instances]
        ref_input_ids = torch.nn.utils.rnn.pad_sequence(
            ref_input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        
        # Pad labels
        labels = [s["labels"] for s in instances]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        
        ref_labels = [s["ref_labels"] for s in instances]
        ref_labels = torch.nn.utils.rnn.pad_sequence(
            ref_labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        
        # Pad masks
        implicit_mask = [s["implicit_mask"] for s in instances]
        implicit_mask = torch.nn.utils.rnn.pad_sequence(
            implicit_mask, batch_first=True, padding_value=False
        )
        
        explicit_mask = [s["explicit_mask"] for s in instances]
        explicit_mask = torch.nn.utils.rnn.pad_sequence(
            explicit_mask, batch_first=True, padding_value=True
        )
        
        # Stack positions
        model_answer_position = torch.stack([s["model_answer_position"] for s in instances])
        ref_answer_position = torch.stack([s["ref_answer_position"] for s in instances])
        num_implicit_steps = torch.stack([s["num_implicit_steps"] for s in instances])
        
        return {
            "encoder_input_ids": encoder_input_ids,
            "decoder_input_ids": decoder_input_ids,
            "ref_input_ids": ref_input_ids,
            "labels": labels,
            "ref_labels": ref_labels,
            "encoder_attention_mask": encoder_input_ids.ne(self.pad_token_id),
            "ref_attention_mask": ref_input_ids.ne(self.pad_token_id),
            "implicit_mask": implicit_mask,
            "explicit_mask": explicit_mask,
            "model_answer_position": model_answer_position,
            "ref_answer_position": ref_answer_position,
            "num_implicit_steps": num_implicit_steps,
        }


# ============================================================================
# 自定义Trainer
# ============================================================================

def _to_scalar(x):
    """转换为标量"""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().float().mean().item()
    return float(x)


class HybridTrainer(Trainer):
    """半显半隐训练Trainer"""
    
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
        inputs["step"] = step
        inputs["step_ratio"] = step / max(total_steps, 1)
        
        # Forward
        outputs = model(**inputs)
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
# 主训练函数
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 2: Hybrid CODI Training")
    
    # 模型参数
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    # 数据参数
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to Phase 1 generated data")
    parser.add_argument("--max_length", type=int, default=512)
    
    # 训练参数
    parser.add_argument("--output_dir", type=str, default="./checkpoints/hybrid_codi")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    
    # 模型配置
    parser.add_argument("--num_latent", type=int, default=6)
    parser.add_argument("--use_prj", action="store_true", default=True)
    parser.add_argument("--prj_dim", type=int, default=2048)
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--bf16", action="store_true", default=True)
    
    # Loss权重
    parser.add_argument("--ce_loss_weight", type=float, default=1.0)
    parser.add_argument("--distill_loss_weight", type=float, default=0.5)
    parser.add_argument("--ref_loss_weight", type=float, default=0.3)
    
    # 其他
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_simplified_model", action="store_true", default=True,
                        help="使用简化版模型（推荐，兼容原始推理代码）")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # LoRA配置
    if "llama" in args.model_name_or_path.lower() or "mistral" in args.model_name_or_path.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
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
    
    # 模型参数
    model_args = HybridModelArguments(
        model_name_or_path=args.model_name_or_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_init=True,
    )
    
    # 训练参数
    training_args = HybridTrainingArguments(
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
        use_lora=args.use_lora,
        use_prj=args.use_prj,
        prj_dim=args.prj_dim,
        ce_loss_weight=args.ce_loss_weight,
        distill_loss_weight=args.distill_loss_weight,
        ref_loss_weight=args.ref_loss_weight,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
    )
    
    # 创建模型
    logger.info("Creating model...")
    if args.use_simplified_model:
        model = SimplifiedHybridCODI(model_args, training_args, lora_config)
    else:
        model = HybridCODI(model_args, training_args, lora_config)
    
    # 创建数据集
    logger.info("Creating dataset...")
    train_dataset = HybridTrainingDataset(
        data_path=args.data_path,
        tokenizer=model.tokenizer,
        bot_id=model.bot_id,
        eot_id=model.eot_id,
        latent_id=model.latent_id,
        max_length=args.max_length,
    )
    
    # Data collator
    data_collator = HybridDataCollator(
        tokenizer=model.tokenizer,
        pad_token_id=model.pad_token_id,
    )
    
    # Trainer
    trainer = HybridTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=model.tokenizer,
    )
    
    # 训练
    logger.info("Starting training...")
    trainer.train()
    
    # 保存
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    trainer.save_state()
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
