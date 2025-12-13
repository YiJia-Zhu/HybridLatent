"""
Phase 1: 半显半隐推理训练数据生成 (SwiReasoning风格)

核心改动：
1. 对每个样本的每个token位置计算熵
2. 直接使用step4_adaptive_eval.py中的SwiRController切换逻辑
3. 根据熵序列动态决定每个位置是显式还是隐式

使用方法:
    python phase1_data_generation.py \
        --model_path ./CODI/pretrained/Llama-3.2-1B-Instruct \
        --output_dir ./data/hybrid_training \
        --data_name icot \
        --window_e_to_l 5 \
        --window_l_to_e 0 \
        --max_switch_count 5 \
        --max_latent_steps 6 \
        --bf16
"""

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import sys
import json
import re
import random
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加路径以导入step4的组件
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

# 从step4导入SwiRController相关组件
from step4_adaptive_eval import SwiRModeState, SwiRController


# ============================================================================
# 离线模拟SwiRController (用于数据生成)
# ============================================================================

class OfflineSwiRSimulator:
    """
    离线模拟SwiRController的切换逻辑
    
    与step4中的在线版本使用相同的参数和逻辑，
    但可以一次性处理整个熵序列
    """
    
    def __init__(
        self,
        window_e_to_l: int = 5,
        window_l_to_e: int = 0,
        max_switch_count: Optional[int] = None,
        max_latent_steps: int = 6,
    ):
        self.controller = SwiRController(
            window_e_to_l=window_e_to_l,
            window_l_to_e=window_l_to_e,
            max_switch_count=max_switch_count,
        )
        self.max_latent_steps = max_latent_steps
    
    def simulate(
        self,
        entropies: List[float],
        answer_trigger_idx: Optional[int] = None,
    ) -> Tuple[List[int], List[Dict]]:
        """
        模拟SwiReasoning切换逻辑
        
        Args:
            entropies: 每个token位置的熵值列表
            answer_trigger_idx: 答案触发位置（之后锁定normal）
            
        Returns:
            modes: 每个位置的模式 (0=latent, 1=normal)
            events: 切换事件列表
        """
        device = torch.device('cpu')
        batch_size = 1
        
        # 初始化状态
        state = SwiRModeState.init(batch_size, device)
        
        modes = []
        events = []
        consecutive_latent = 0
        
        for step, cur_entropy in enumerate(entropies):
            cur_entropy_tensor = torch.tensor([cur_entropy], device=device)
            
            # 检查是否进入答案阶段
            if answer_trigger_idx is not None and step >= answer_trigger_idx:
                state.answer_locked[0] = True
            
            # 检查连续latent上限
            if state.mode[0].item() == 0 and consecutive_latent >= self.max_latent_steps:
                state.mode[0] = 1
                state.mode_stay_steps[0] = 0
                state.ref_entropy[0] = cur_entropy
                consecutive_latent = 0
                events.append({
                    "step": step,
                    "event": "latent->normal (max_consecutive)",
                    "entropy": cur_entropy
                })
            
            # 调用controller的update方法
            state, to_normal, to_soft = self.controller.update(
                state, cur_entropy_tensor, step, None
            )
            
            current_mode = state.mode[0].item()
            
            # 记录切换事件
            if to_normal[0].item():
                events.append({
                    "step": step,
                    "event": "latent->normal",
                    "entropy": cur_entropy
                })
                consecutive_latent = 0
            elif to_soft[0].item():
                events.append({
                    "step": step,
                    "event": "normal->latent",
                    "entropy": cur_entropy
                })
                consecutive_latent = 1
            else:
                if current_mode == 0:
                    consecutive_latent += 1
                else:
                    consecutive_latent = 0
            
            modes.append(current_mode)
        
        return modes, events


# ============================================================================
# 熵计算
# ============================================================================

def compute_token_entropy(logits: torch.Tensor) -> float:
    """计算单个位置的熵值"""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy.item()


def compute_entropy_sequence(
    model,
    tokenizer,
    full_text: str,
    question_len: int,
    device: torch.device,
) -> Tuple[List[float], List[int]]:
    """
    计算完整序列中每个CoT token的熵值
    
    Args:
        model: 语言模型
        tokenizer: tokenizer
        full_text: question + cot + answer 的完整文本
        question_len: question部分的token长度
        device: 设备
        
    Returns:
        entropies: CoT部分每个token的熵值
        token_ids: CoT部分的token ids
    """
    # Tokenize
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    
    seq_len = input_ids.shape[1]
    
    # 前向传播获取所有logits
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=False)
        logits = outputs.logits[0]  # [seq_len, vocab_size]
    
    # 计算每个位置的熵
    # logits[i] 预测的是 position i+1 的token
    # 所以我们需要从 question_len-1 开始
    entropies = []
    for i in range(question_len - 1, seq_len - 1):
        ent = compute_token_entropy(logits[i])
        entropies.append(ent)
    
    # 获取对应的token ids (从question_len开始)
    token_ids = input_ids[0, question_len:].tolist()
    
    # 确保长度一致
    min_len = min(len(entropies), len(token_ids))
    entropies = entropies[:min_len]
    token_ids = token_ids[:min_len]
    
    return entropies, token_ids


# ============================================================================
# 数据处理
# ============================================================================

@dataclass
class HybridSample:
    """半显半隐训练样本"""
    question: str
    full_cot: str                    # 完整显式CoT
    answer: str
    hybrid_sequence: str             # 混合序列 (显式text + [L] markers)
    token_modes: List[int]           # 每个token的模式 (0=latent, 1=normal)
    entropies: List[float]           # 每个token的熵值
    switch_events: List[Dict]        # 切换事件
    num_latent_tokens: int           # latent token数量
    num_explicit_tokens: int         # explicit token数量
    pattern_type: str                # 生成模式


def find_answer_trigger_position(
    text: str,
    tokenizer,
    triggers: List[str] = None,
) -> Optional[int]:
    """找到答案触发位置"""
    if triggers is None:
        triggers = ["The answer is", "#### ", "the answer is", "Answer:"]
    
    for trigger in triggers:
        pos = text.find(trigger)
        if pos != -1:
            # 找到trigger在token序列中的位置
            prefix = text[:pos]
            prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
            return len(prefix_tokens)
    
    return None


def generate_hybrid_sequence(
    tokens: List[int],
    modes: List[int],
    tokenizer,
) -> str:
    """
    根据模式序列生成混合文本
    
    Args:
        tokens: token id列表
        modes: 每个token的模式 (0=latent, 1=normal)
        tokenizer: tokenizer
        
    Returns:
        hybrid_sequence: 混合序列，latent位置用[L]替代
    """
    parts = []
    current_explicit_tokens = []
    
    for token_id, mode in zip(tokens, modes):
        if mode == 1:  # normal/explicit
            current_explicit_tokens.append(token_id)
        else:  # latent
            # 先输出累积的explicit tokens
            if current_explicit_tokens:
                text = tokenizer.decode(current_explicit_tokens, skip_special_tokens=True)
                parts.append(text)
                current_explicit_tokens = []
            # 添加latent marker
            parts.append("[L]")
    
    # 输出剩余的explicit tokens
    if current_explicit_tokens:
        text = tokenizer.decode(current_explicit_tokens, skip_special_tokens=True)
        parts.append(text)
    
    return "".join(parts)


def process_sample_with_swir(
    model,
    tokenizer,
    question: str,
    cot: str,
    answer: str,
    simulator: OfflineSwiRSimulator,
    device: torch.device,
    pattern_type: str = "entropy_guided",
) -> Optional[HybridSample]:
    """
    使用SwiReasoning逻辑处理单个样本
    
    Args:
        model: 用于计算熵的模型
        tokenizer: tokenizer
        question: 问题
        cot: Chain-of-Thought
        answer: 答案
        simulator: SwiR模拟器
        device: 设备
        pattern_type: 模式类型
        
    Returns:
        HybridSample 或 None
    """
    # 构造完整文本
    answer_text = f"The answer is: {answer}"
    full_text = f"{question}{cot}{answer_text}"
    
    # 获取question长度
    question_tokens = tokenizer.encode(question, add_special_tokens=True)
    question_len = len(question_tokens)
    
    # 计算熵序列
    try:
        entropies, token_ids = compute_entropy_sequence(
            model, tokenizer, full_text, question_len, device
        )
    except Exception as e:
        logger.warning(f"Error computing entropy: {e}")
        return None
    
    if len(entropies) == 0:
        return None
    
    # 找到答案触发位置
    cot_answer = f"{cot}{answer_text}"
    answer_trigger_idx = find_answer_trigger_position(cot_answer, tokenizer)
    
    # 使用SwiR模拟器决定每个位置的模式
    if pattern_type == "entropy_guided":
        modes, events = simulator.simulate(entropies, answer_trigger_idx)
    elif pattern_type == "random":
        # 随机模式：随机决定每个位置
        modes = [random.choice([0, 1]) for _ in entropies]
        # 答案部分锁定为normal
        if answer_trigger_idx is not None:
            for i in range(answer_trigger_idx, len(modes)):
                modes[i] = 1
        events = []
    elif pattern_type == "full_explicit":
        # 全显式
        modes = [1] * len(entropies)
        events = []
    else:
        modes, events = simulator.simulate(entropies, answer_trigger_idx)
    
    # 生成混合序列
    hybrid_sequence = generate_hybrid_sequence(token_ids, modes, tokenizer)
    
    # 统计
    num_latent = sum(1 for m in modes if m == 0)
    num_explicit = sum(1 for m in modes if m == 1)
    
    return HybridSample(
        question=question,
        full_cot=cot,
        answer=answer,
        hybrid_sequence=hybrid_sequence,
        token_modes=modes,
        entropies=entropies,
        switch_events=events,
        num_latent_tokens=num_latent,
        num_explicit_tokens=num_explicit,
        pattern_type=pattern_type,
    )


# ============================================================================
# 数据集加载
# ============================================================================

def load_icot_dataset(data_name: str = "icot", max_samples: Optional[int] = None):
    """加载icot数据集"""
    logger.info(f"Loading dataset: {data_name}")
    
    if "full" in data_name:
        dataset = load_dataset("zen-E/GSM8k-Aug-NL")["train"]
    else:
        dataset = load_dataset("zen-E/GSM8k-Aug")["train"]
    
    samples = []
    for i, example in enumerate(tqdm(dataset, desc="Loading data")):
        if max_samples and i >= max_samples:
            break
        
        # 处理cot字段
        if 'cot' not in example:
            if 'steps' in example:
                example['cot'] = ' '.join(example['steps'])
            else:
                continue
        
        question = example['question'].strip()
        cot = example['cot'].strip()
        
        # 提取答案
        if 'answer' in example and example['answer']:
            answer = str(example['answer']).split()[-1].replace("####", "").strip()
        else:
            continue
        
        samples.append({
            "question": question,
            "cot": cot,
            "answer": answer,
        })
    
    logger.info(f"Loaded {len(samples)} samples from {data_name}")
    return samples


# ============================================================================
# 主处理流程
# ============================================================================

class Phase1DataGenerator:
    """Phase 1 数据生成器"""
    
    def __init__(
        self,
        model_path: str,
        device: torch.device,
        use_bf16: bool = True,
        window_e_to_l: int = 5,
        window_l_to_e: int = 0,
        max_switch_count: Optional[int] = None,
        max_latent_steps: int = 6,
    ):
        self.device = device
        
        # 创建SwiR模拟器
        self.simulator = OfflineSwiRSimulator(
            window_e_to_l=window_e_to_l,
            window_l_to_e=window_l_to_e,
            max_switch_count=max_switch_count,
            max_latent_steps=max_latent_steps,
        )
        
        # 加载模型
        logger.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = self.model.to(device)
        if use_bf16:
            self.model = self.model.to(torch.bfloat16)
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def process_dataset(
        self,
        raw_samples: List[Dict],
        entropy_guided_ratio: float = 0.5,
        random_ratio: float = 0.3,
        explicit_ratio: float = 0.2,
    ) -> Tuple[List[Dict], Dict]:
        """处理数据集"""
        
        logger.info("Processing samples with SwiReasoning logic...")
        
        processed_samples = []
        stats = defaultdict(int)
        all_entropies = []
        all_latent_ratios = []
        
        for i, sample in enumerate(tqdm(raw_samples, desc="Processing")):
            # 随机决定pattern类型
            rand_val = random.random()
            if rand_val < entropy_guided_ratio:
                pattern_type = "entropy_guided"
            elif rand_val < entropy_guided_ratio + random_ratio:
                pattern_type = "random"
            else:
                pattern_type = "full_explicit"
            
            try:
                result = process_sample_with_swir(
                    self.model,
                    self.tokenizer,
                    sample["question"],
                    sample["cot"],
                    sample["answer"],
                    self.simulator,
                    self.device,
                    pattern_type,
                )
                
                if result is None:
                    continue
                
                # 转换为dict格式
                sample_dict = {
                    "question": result.question,
                    "full_cot": result.full_cot,
                    "answer": result.answer,
                    "hybrid_sequence": result.hybrid_sequence,
                    "token_modes": result.token_modes,
                    "entropies": result.entropies,
                    "switch_events": result.switch_events,
                    "num_latent_tokens": result.num_latent_tokens,
                    "num_explicit_tokens": result.num_explicit_tokens,
                    "pattern_type": result.pattern_type,
                }
                
                processed_samples.append(sample_dict)
                
                # 统计
                stats[pattern_type] += 1
                all_entropies.extend(result.entropies)
                
                total_tokens = result.num_latent_tokens + result.num_explicit_tokens
                if total_tokens > 0:
                    all_latent_ratios.append(result.num_latent_tokens / total_tokens)
                
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue
        
        # 计算统计信息
        summary_stats = {
            "total_samples": len(processed_samples),
            "pattern_distribution": dict(stats),
            "entropy_stats": {
                "mean": float(np.mean(all_entropies)) if all_entropies else 0,
                "std": float(np.std(all_entropies)) if all_entropies else 0,
                "min": float(np.min(all_entropies)) if all_entropies else 0,
                "max": float(np.max(all_entropies)) if all_entropies else 0,
                "median": float(np.median(all_entropies)) if all_entropies else 0,
            },
            "latent_ratio_stats": {
                "mean": float(np.mean(all_latent_ratios)) if all_latent_ratios else 0,
                "std": float(np.std(all_latent_ratios)) if all_latent_ratios else 0,
                "min": float(np.min(all_latent_ratios)) if all_latent_ratios else 0,
                "max": float(np.max(all_latent_ratios)) if all_latent_ratios else 0,
            },
            "swir_config": {
                "window_e_to_l": self.simulator.controller.window_e_to_l,
                "window_l_to_e": self.simulator.controller.window_l_to_e,
                "max_switch_count": self.simulator.controller.max_switch_count,
                "max_latent_steps": self.simulator.max_latent_steps,
            }
        }
        
        return processed_samples, summary_stats


def save_training_data(
    samples: List[Dict],
    stats: Dict,
    output_dir: str,
):
    """保存训练数据"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存所有样本
    samples_file = os.path.join(output_dir, "hybrid_training_data.json")
    with open(samples_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(samples)} training samples to {samples_file}")
    
    # 保存统计信息
    stats_file = os.path.join(output_dir, "generation_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved statistics to {stats_file}")
    
    # 按pattern类型分别保存
    for pattern_type in ["entropy_guided", "random", "full_explicit"]:
        pattern_samples = [s for s in samples if s["pattern_type"] == pattern_type]
        if pattern_samples:
            pattern_file = os.path.join(output_dir, f"hybrid_training_{pattern_type}.json")
            with open(pattern_file, 'w', encoding='utf-8') as f:
                json.dump(pattern_samples, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(pattern_samples)} {pattern_type} samples")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Hybrid Training Data Generation (SwiR Style)")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model for entropy computation")
    
    # 数据参数
    parser.add_argument("--data_name", type=str, default="icot",
                        choices=["icot", "icot-full"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./data/hybrid_training")
    
    # SwiReasoning 参数 (与step4一致)
    parser.add_argument("--window_e_to_l", type=int, default=5,
                        help="Explicit→Latent 切换需等待的步数")
    parser.add_argument("--window_l_to_e", type=int, default=0,
                        help="Latent→Explicit 切换需等待的步数")
    parser.add_argument("--max_switch_count", type=int, default=5,
                        help="最大切换次数")
    parser.add_argument("--max_latent_steps", type=int, default=6,
                        help="最大连续latent步数")
    
    # 数据比例参数
    parser.add_argument("--entropy_guided_ratio", type=float, default=0.5)
    parser.add_argument("--random_ratio", type=float, default=0.3)
    parser.add_argument("--explicit_ratio", type=float, default=0.2)
    
    # 其他参数
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 加载原始数据
    raw_samples = load_icot_dataset(args.data_name, args.max_samples)
    
    # 初始化生成器
    generator = Phase1DataGenerator(
        model_path=args.model_path,
        device=device,
        use_bf16=args.bf16,
        window_e_to_l=args.window_e_to_l,
        window_l_to_e=args.window_l_to_e,
        max_switch_count=args.max_switch_count,
        max_latent_steps=args.max_latent_steps,
    )
    
    # 处理数据
    training_samples, stats = generator.process_dataset(
        raw_samples,
        entropy_guided_ratio=args.entropy_guided_ratio,
        random_ratio=args.random_ratio,
        explicit_ratio=args.explicit_ratio,
    )
    
    # 保存数据
    save_training_data(training_samples, stats, args.output_dir)
    
    # 打印统计
    print("\n" + "="*60)
    print("DATA GENERATION STATISTICS (SwiReasoning Style)")
    print("="*60)
    print(f"Total samples: {stats['total_samples']}")
    print(f"\nPattern distribution:")
    for pattern, count in stats['pattern_distribution'].items():
        pct = count / stats['total_samples'] * 100 if stats['total_samples'] > 0 else 0
        print(f"  {pattern}: {count} ({pct:.1f}%)")
    print(f"\nEntropy stats:")
    print(f"  mean: {stats['entropy_stats']['mean']:.4f}")
    print(f"  std:  {stats['entropy_stats']['std']:.4f}")
    print(f"  median: {stats['entropy_stats']['median']:.4f}")
    print(f"\nLatent ratio stats:")
    print(f"  mean: {stats['latent_ratio_stats']['mean']:.2%}")
    print(f"  std:  {stats['latent_ratio_stats']['std']:.2%}")
    print(f"\nSwiR config:")
    print(f"  window_e_to_l: {stats['swir_config']['window_e_to_l']}")
    print(f"  window_l_to_e: {stats['swir_config']['window_l_to_e']}")
    print(f"  max_switch_count: {stats['swir_config']['max_switch_count']}")
    print(f"  max_latent_steps: {stats['swir_config']['max_latent_steps']}")
    print("="*60)


if __name__ == "__main__":
    main()
