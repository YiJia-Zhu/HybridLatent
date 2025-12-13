"""
Phase 1 数据格式转换器

将生成的半显半隐训练数据转换为CODI可用的格式

CODI 训练数据格式:
- encoder_input_ids: question tokens + bot_id
- decoder_input_ids: [eot_id] + answer tokens
- ref_input_ids: question + cot + answer (完整显式序列)
- ref_labels: 用于distillation的标签

半显半隐格式扩展:
- 混合序列中的 [L] 标记转换为隐式循环步数
- 保留完整CoT用于蒸馏目标
"""

import torch
import json
import argparse
import os
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# 数据格式转换
# ============================================================================

@dataclass
class CODIHybridSample:
    """CODI半显半隐训练样本格式"""
    encoder_input_ids: torch.Tensor      # question + bot_id
    decoder_input_ids: torch.Tensor      # hybrid sequence tokens
    ref_input_ids: torch.Tensor          # full explicit sequence
    labels: torch.Tensor                 # decoder labels
    ref_labels: torch.Tensor             # reference labels for distillation
    implicit_mask: torch.Tensor          # 标记哪些位置是隐式的
    explicit_mask: torch.Tensor          # 标记哪些位置是显式的
    num_latent_steps: int                # 隐式步数总计


class HybridDataConverter:
    """半显半隐数据格式转换器"""
    
    LATENT_TOKEN = "[L]"
    
    def __init__(
        self,
        tokenizer_path: str,
        bot_token: str = "<|bot|>",
        eot_token: str = "<|eot|>",
        latent_token: str = "<|latent|>",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # 添加特殊token
        special_tokens = []
        if bot_token not in self.tokenizer.get_vocab():
            special_tokens.append(bot_token)
        if eot_token not in self.tokenizer.get_vocab():
            special_tokens.append(eot_token)
        if latent_token not in self.tokenizer.get_vocab():
            special_tokens.append(latent_token)
        
        if special_tokens:
            self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        
        # 获取token IDs
        self.bot_id = self.tokenizer.convert_tokens_to_ids(bot_token)
        self.eot_id = self.tokenizer.convert_tokens_to_ids(eot_token)
        self.latent_id = self.tokenizer.convert_tokens_to_ids(latent_token)
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Tokenizer initialized:")
        logger.info(f"  bot_id: {self.bot_id}, eot_id: {self.eot_id}, latent_id: {self.latent_id}")
    
    def convert_sample(
        self,
        question: str,
        hybrid_sequence: str,
        full_cot: str,
        answer: str,
        max_length: int = 512,
    ) -> Dict:
        """转换单个样本为CODI训练格式"""
        
        # 1. Tokenize question
        question_tokens = self.tokenizer.encode(question, add_special_tokens=True)
        
        # 2. 处理混合序列 - 将 [L] 替换为 latent token
        hybrid_text = hybrid_sequence.replace(self.LATENT_TOKEN, f" {self.tokenizer.convert_ids_to_tokens(self.latent_id)} ")
        hybrid_tokens = self.tokenizer.encode(hybrid_text, add_special_tokens=False)
        
        # 3. Tokenize answer
        answer_text = f"The answer is: {answer}"
        answer_tokens = self.tokenizer.encode(answer_text, add_special_tokens=False)
        
        # 4. Tokenize full CoT (用于distillation)
        full_cot_tokens = self.tokenizer.encode(full_cot, add_special_tokens=False)
        
        # 5. 构建各个序列
        # encoder_input: question + bot_id
        encoder_input_ids = question_tokens + [self.bot_id]
        
        # decoder_input: eot_id + hybrid + answer + eos
        decoder_input_ids = [self.eot_id] + hybrid_tokens + answer_tokens + [self.tokenizer.eos_token_id]
        
        # ref_input: question + full_cot + answer + eos (完整显式序列)
        ref_input_ids = question_tokens + full_cot_tokens + answer_tokens + [self.tokenizer.eos_token_id]
        
        # 6. 创建mask
        # implicit_mask: 标记latent token位置
        implicit_mask = [1 if t == self.latent_id else 0 for t in decoder_input_ids]
        explicit_mask = [1 if t != self.latent_id else 0 for t in decoder_input_ids]
        
        # 7. Labels
        labels = decoder_input_ids.copy()
        ref_labels = ref_input_ids.copy()
        # Mask掉question部分
        ref_labels[:len(question_tokens)] = [-100] * len(question_tokens)
        
        return {
            "encoder_input_ids": encoder_input_ids,
            "decoder_input_ids": decoder_input_ids,
            "ref_input_ids": ref_input_ids,
            "labels": labels,
            "ref_labels": ref_labels,
            "implicit_mask": implicit_mask,
            "explicit_mask": explicit_mask,
            "num_latent_tokens": sum(implicit_mask),
        }
    
    def convert_dataset(
        self,
        samples: List[Dict],
        max_length: int = 512,
    ) -> List[Dict]:
        """转换整个数据集"""
        converted = []
        
        for sample in tqdm(samples, desc="Converting samples"):
            try:
                converted_sample = self.convert_sample(
                    sample["question"],
                    sample["hybrid_sequence"],
                    sample["full_cot"],
                    sample["answer"],
                    max_length,
                )
                converted_sample["pattern_type"] = sample.get("pattern_type", "unknown")
                converted_sample["metadata"] = sample.get("metadata", {})
                converted.append(converted_sample)
            except Exception as e:
                logger.warning(f"Error converting sample: {e}")
                continue
        
        return converted


# ============================================================================
# 数据集类 (用于PyTorch训练)
# ============================================================================

class HybridCODIDataset(torch.utils.data.Dataset):
    """半显半隐CODI训练数据集"""
    
    def __init__(self, data_path: str, tokenizer_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "encoder_input_ids": torch.tensor(sample["encoder_input_ids"], dtype=torch.long),
            "decoder_input_ids": torch.tensor(sample["decoder_input_ids"], dtype=torch.long),
            "ref_input_ids": torch.tensor(sample["ref_input_ids"], dtype=torch.long),
            "labels": torch.tensor(sample["labels"], dtype=torch.long),
            "ref_labels": torch.tensor(sample["ref_labels"], dtype=torch.long),
            "implicit_mask": torch.tensor(sample["implicit_mask"], dtype=torch.bool),
            "explicit_mask": torch.tensor(sample["explicit_mask"], dtype=torch.bool),
        }


class HybridDataCollator:
    """半显半隐数据Collator"""
    
    def __init__(self, tokenizer, pad_token_id: int = 0, ignore_index: int = -100):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
    
    def __call__(self, batch):
        # Pad encoder inputs (left padding)
        encoder_input_ids = [s["encoder_input_ids"].flip(0) for s in batch]
        encoder_input_ids = torch.nn.utils.rnn.pad_sequence(
            encoder_input_ids, batch_first=True, padding_value=self.pad_token_id
        ).flip(1)
        
        # Pad decoder inputs
        decoder_input_ids = [s["decoder_input_ids"] for s in batch]
        decoder_input_ids = torch.nn.utils.rnn.pad_sequence(
            decoder_input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        
        # Pad ref inputs
        ref_input_ids = [s["ref_input_ids"] for s in batch]
        ref_input_ids = torch.nn.utils.rnn.pad_sequence(
            ref_input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        
        # Pad labels
        labels = [s["labels"] for s in batch]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.ignore_index
        )
        
        ref_labels = [s["ref_labels"] for s in batch]
        ref_labels = torch.nn.utils.rnn.pad_sequence(
            ref_labels, batch_first=True, padding_value=self.ignore_index
        )
        
        # Pad masks
        implicit_mask = [s["implicit_mask"] for s in batch]
        implicit_mask = torch.nn.utils.rnn.pad_sequence(
            implicit_mask, batch_first=True, padding_value=False
        )
        
        explicit_mask = [s["explicit_mask"] for s in batch]
        explicit_mask = torch.nn.utils.rnn.pad_sequence(
            explicit_mask, batch_first=True, padding_value=False
        )
        
        return {
            "encoder_input_ids": encoder_input_ids,
            "decoder_input_ids": decoder_input_ids,
            "ref_input_ids": ref_input_ids,
            "labels": labels,
            "ref_labels": ref_labels,
            "implicit_mask": implicit_mask,
            "explicit_mask": explicit_mask,
            "encoder_attention_mask": encoder_input_ids.ne(self.pad_token_id),
            "ref_attention_mask": ref_input_ids.ne(self.pad_token_id),
        }


# ============================================================================
# 数据可视化工具
# ============================================================================

def visualize_entropy_distribution(stats_file: str, output_dir: str):
    """可视化熵分布"""
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Pattern分布饼图
    ax1 = axes[0, 0]
    pattern_dist = stats.get("pattern_distribution", {})
    if pattern_dist:
        labels = list(pattern_dist.keys())
        sizes = list(pattern_dist.values())
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.set_title("Pattern Distribution")
    
    # 2. 熵统计信息
    ax2 = axes[0, 1]
    entropy_stats = stats.get("entropy_stats", {})
    if entropy_stats:
        metrics = ["mean", "std", "min", "max"]
        values = [entropy_stats.get(m, 0) for m in metrics]
        ax2.bar(metrics, values, color=['blue', 'orange', 'green', 'red'])
        ax2.set_title("Entropy Statistics")
        ax2.set_ylabel("Value")
    
    # 3. 隐式步数分布
    ax3 = axes[1, 0]
    implicit_stats = stats.get("implicit_steps", {})
    if implicit_stats:
        ax3.text(0.5, 0.5, f"Mean implicit steps: {implicit_stats.get('mean', 0):.2f}\n"
                          f"Max implicit steps: {implicit_stats.get('max', 0)}",
                ha='center', va='center', fontsize=14,
                transform=ax3.transAxes)
        ax3.set_title("Implicit Steps Statistics")
        ax3.axis('off')
    
    # 4. 总步数统计
    ax4 = axes[1, 1]
    total_steps = stats.get("total_steps", {})
    if total_steps:
        ax4.text(0.5, 0.5, f"Mean total steps: {total_steps.get('mean', 0):.2f}\n"
                          f"Entropy threshold: {stats.get('entropy_threshold', 0):.4f}",
                ha='center', va='center', fontsize=14,
                transform=ax4.transAxes)
        ax4.set_title("Generation Statistics")
        ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "generation_stats.png"), dpi=150)
    plt.close()
    
    logger.info(f"Visualization saved to {output_dir}/generation_stats.png")


def visualize_sample(sample: Dict, tokenizer, output_file: str):
    """可视化单个样本的显式/隐式分布"""
    hybrid_seq = sample["hybrid_sequence"]
    
    # 分析显式和隐式部分
    parts = hybrid_seq.split("[L]")
    
    fig, ax = plt.subplots(figsize=(14, 3))
    
    current_pos = 0
    colors = []
    texts = []
    
    for i, part in enumerate(parts):
        if part.strip():
            # 显式部分
            texts.append(part.strip()[:20] + "...")
            colors.append('green')
            current_pos += len(part)
        
        if i < len(parts) - 1:
            # 隐式token
            texts.append("[L]")
            colors.append('red')
    
    # 绘制
    y_pos = 0
    for i, (text, color) in enumerate(zip(texts, colors)):
        ax.barh(y_pos, 1, left=i, color=color, alpha=0.7, edgecolor='black')
        ax.text(i + 0.5, y_pos, text, ha='center', va='center', fontsize=8, rotation=45)
    
    ax.set_xlim(0, len(texts))
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("Sequence Position")
    ax.set_title(f"Sample Visualization (Pattern: {sample.get('pattern_type', 'unknown')})")
    ax.legend(['Explicit', 'Implicit'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Convert hybrid training data to CODI format")
    
    parser.add_argument("--input_file", type=str, required=True,
                        help="Input JSON file from phase1_data_generation.py")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output JSON file for CODI training")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to tokenizer")
    parser.add_argument("--stats_file", type=str, default=None,
                        help="Stats file for visualization")
    parser.add_argument("--viz_output_dir", type=str, default="./visualizations",
                        help="Directory for visualization outputs")
    parser.add_argument("--max_length", type=int, default=512)
    
    args = parser.parse_args()
    
    # 加载数据
    logger.info(f"Loading data from {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    logger.info(f"Loaded {len(samples)} samples")
    
    # 转换数据
    converter = HybridDataConverter(args.tokenizer_path)
    converted_samples = converter.convert_dataset(samples, args.max_length)
    
    # 保存转换后的数据
    logger.info(f"Saving converted data to {args.output_file}")
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_samples, f, indent=2)
    
    # 可视化
    if args.stats_file and os.path.exists(args.stats_file):
        logger.info("Generating visualizations...")
        visualize_entropy_distribution(args.stats_file, args.viz_output_dir)
        
        # 可视化几个样本
        for i, sample in enumerate(samples[:3]):
            viz_file = os.path.join(args.viz_output_dir, f"sample_{i}.png")
            visualize_sample(sample, converter.tokenizer, viz_file)
    
    # 打印统计
    pattern_counts = {}
    total_latent = 0
    for s in converted_samples:
        pt = s.get("pattern_type", "unknown")
        pattern_counts[pt] = pattern_counts.get(pt, 0) + 1
        total_latent += s.get("num_latent_tokens", 0)
    
    print("\n" + "="*60)
    print("CONVERSION STATISTICS")
    print("="*60)
    print(f"Total converted samples: {len(converted_samples)}")
    print(f"Pattern distribution: {pattern_counts}")
    print(f"Average latent tokens per sample: {total_latent / len(converted_samples):.2f}")
    print("="*60)


if __name__ == "__main__":
    main()
