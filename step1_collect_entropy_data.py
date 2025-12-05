"""
阶段1: 数据收集脚本 (SwiReasoning风格熵计算)

用普通 CoT 模型推理，收集 (hidden_states, entropy) 数据对
熵计算方式参考 SwiReasoning：使用原始熵值，并记录熵的相对变化

使用方法:
    python step1_collect_entropy_data.py \
        --model_path <pretrained_model> \
        --data_name icot \
        --output_path data/entropy_data.pt \
        --max_samples 10000
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
from tqdm import tqdm
import json
import logging
from typing import Dict, Sequence, Tuple, Optional
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# 导入本地模块
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from entropy_predictor import EntropyPredictor, EntropyDataset

IGNORE_INDEX = -100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# SwiReasoning 风格的熵计算
# ============================================================================

def compute_entropy_swir(logits: torch.Tensor) -> torch.Tensor:
    """
    SwiReasoning 风格的熵计算 (非归一化)
    """
    # 转换为 float32 计算，避免 float16 精度问题
    logits_f32 = logits.float()
    probs = F.softmax(logits_f32, dim=-1)
    entropy = -(probs * probs.clamp(min=1e-12).log()).sum(dim=-1)
    # 转回原始 dtype
    return entropy.to(logits.dtype)

def compute_normalized_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    归一化熵计算 (0-1 范围)
    用于 EntropyPredictor 的训练目标，方便设置阈值
    """
    vocab_size = logits.size(-1)
    max_entropy = torch.log(torch.tensor(vocab_size, dtype=logits.dtype, device=logits.device))
    entropy = compute_entropy_swir(logits)
    return entropy / max_entropy


def compute_entropy_features(
    logits: torch.Tensor,
    return_both: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    计算熵相关特征，同时返回原始熵和归一化熵
    
    Args:
        logits: (batch, seq_len, vocab_size)
        return_both: 是否同时返回原始和归一化熵
    
    Returns:
        raw_entropy: 原始熵值 (SwiReasoning风格)
        normalized_entropy: 归一化熵 (0-1范围，用于训练)
    """
    raw_entropy = compute_entropy_swir(logits)
    
    if return_both:
        vocab_size = logits.size(-1)
        max_entropy = torch.log(torch.tensor(vocab_size, dtype=logits.dtype, device=logits.device))
        normalized_entropy = raw_entropy / max_entropy
        return raw_entropy, normalized_entropy
    
    return raw_entropy, None


def compute_entropy_dynamics(
    logits_sequence: torch.Tensor,
    window_size: int = 5
) -> Dict[str, torch.Tensor]:
    """
    计算熵的动态特征，参考 SwiReasoning 的模式切换逻辑
    
    SwiReasoning 的切换逻辑:
    - to_normal: (mode == 0) & (cur_entropy < cur_ref_entropy) 
    - to_soft: (mode == 1) & (cur_entropy > cur_ref_entropy) & allow_switch
    
    Args:
        logits_sequence: (batch, seq_len, vocab_size)
        window_size: 计算移动平均的窗口大小
    
    Returns:
        dict containing:
        - raw_entropy: 原始熵
        - normalized_entropy: 归一化熵
        - entropy_delta: 熵变化量 (current - previous)
        - entropy_trend: 熵趋势 (1=上升/应使用soft, -1=下降/应使用normal, 0=稳定)
        - moving_avg: 移动平均熵
    """
    batch_size, seq_len, vocab_size = logits_sequence.shape
    
    # 计算每个位置的熵
    raw_entropy, normalized_entropy = compute_entropy_features(logits_sequence)
    
    # 计算熵变化量 delta (SwiReasoning 用 cur_entropy vs ref_entropy)
    entropy_delta = torch.zeros_like(raw_entropy)
    entropy_delta[:, 1:] = raw_entropy[:, 1:] - raw_entropy[:, :-1]
    
    # 熵趋势标记
    # 熵下降 -> -1 (应该用 normal/显式)
    # 熵上升 -> +1 (应该用 soft/隐式)  
    entropy_trend = torch.zeros_like(raw_entropy)
    entropy_trend[entropy_delta > 0] = 1   # 熵上升
    entropy_trend[entropy_delta < 0] = -1  # 熵下降
    
    # 移动平均（参考 SwiReasoning 的 window_size 概念）
    # 注意：SwiReasoning 的 window_size 实际上是指模式保持步数，不是平滑窗口
    # 这里的移动平均只是辅助特征，不影响核心功能
    if seq_len >= window_size and window_size > 1:
        # 在 float32 下计算避免精度问题
        raw_float = raw_entropy.float()
        
        # 用 avg_pool1d 计算移动平均
        # 维度变换：(batch, seq) -> (batch, 1, seq) -> avg_pool -> (batch, seq)
        raw_3d = raw_float.unsqueeze(1)  # (batch, 1, seq)
        padded_3d = F.pad(raw_3d, (window_size - 1, 0), mode='replicate')
        moving_avg = F.avg_pool1d(padded_3d, kernel_size=window_size, stride=1).squeeze(1)
        
        moving_avg = moving_avg.to(raw_entropy.dtype)
    else:
        moving_avg = raw_entropy.clone()
    
    return {
        "raw_entropy": raw_entropy,
        "normalized_entropy": normalized_entropy,
        "entropy_delta": entropy_delta,
        "entropy_trend": entropy_trend,
        "moving_avg": moving_avg,
    }


# ============================================================================
# 数据集和数据处理（与原版相同）
# ============================================================================

def _tokenize_fn(strings: Sequence[str], tokenizer, max_length: int = 512) -> Dict:
    """Tokenize a list of strings."""
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
    input_ids_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        input_ids_lens=input_ids_lens,
    )


class CoTDataset(Dataset):
    """CoT数据集类"""
    QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"
    
    def __init__(
        self, 
        data_name: str, 
        raw_data, 
        tokenizer, 
        max_length: int = 512,
        max_token_num: int = 1024,
        max_samples: int = None,
        include_last_cot: bool = True,
    ):
        super().__init__()
        logging.warning("Formatting inputs...")
        
        self.data_name = data_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.questions = []
        self.cots = []
        self.answers = []
        self.full_texts = []
        
        for num_iter, example in tqdm(enumerate(raw_data), desc="Processing data"):
            if max_samples and num_iter >= max_samples:
                break
                
            if 'cot' not in example:
                if 'steps' in example:
                    example['cot'] = ' '.join(example['steps']) if isinstance(example['steps'], list) else example['steps']
                else:
                    continue
            
            try:
                processed = self._process_example(example, data_name, tokenizer, max_token_num, include_last_cot)
                if processed:
                    question, cot, answer = processed
                    self.questions.append(question)
                    self.cots.append(cot)
                    self.answers.append(answer)
                    full_text = question + cot + answer
                    self.full_texts.append(full_text)
            except Exception as e:
                logging.warning(f"Error processing example {num_iter}: {e}")
                continue
        
        print(f"{len(self.full_texts)} data in total...")
        
        logging.warning("Tokenizing inputs...")
        self.tokenized = _tokenize_fn(self.full_texts, tokenizer, max_length)
        self.cot_positions = self._compute_cot_positions(tokenizer)
    
    def _process_example(self, example, data_name, tokenizer, max_token_num, include_last_cot):
        """处理单个样本"""
        if "icot" in data_name and "full" in data_name:
            if example.get("answer") is None or example.get("response") is None:
                return None
            question = f"{example['question']}"
            token_num = len(tokenizer.encode(example["question"] + example["cot"] + str(example["answer"])))
            if token_num > max_token_num:
                return None
            cot = f"{example['cot']}".split(". ")
            if not include_last_cot:
                cot = cot[:-1]
            answer = f"The answer is: {example['answer'].split(' ')[-1]}"
            answer = answer.replace("####", "")
            return question, ". ".join(cot) + ".\n", answer
            
        elif "icot" in data_name:
            token_num = len(tokenizer.encode(example["question"] + example["cot"] + str(example["answer"])))
            if token_num > max_token_num:
                return None
            question = f"{example['question']}"
            cot = f"{example['cot']}".split(" ")
            if not include_last_cot:
                cot = cot[:-1]
            answer = str(example['answer']).split(' ')[-1]
            if not answer[0].isdigit() and answer[0] != '-':
                return None
            answer = f"The answer is: {answer}"
            answer = answer.replace("####", "")
            return question, " ".join(cot), answer
            
        elif "commonsense" in data_name or "strategy" in data_name:
            question = example['question'].strip() + '\n'
            cot = example['cot'].strip() + "\n"
            answer = f"The answer is: {str(example['answer']).strip()}"
            token_num = len(tokenizer.encode(question + " " + cot + " " + answer))
            if token_num > max_token_num:
                return None
            return question, cot, answer
            
        elif "prontoqa" in data_name:
            question = example['question'].strip() + '\n'
            steps = example.get('steps', [])
            cot = '\n'.join(steps[:-1]) + "\n" if len(steps) > 1 else ""
            answer = f"The answer is: {str(example['answer']).strip()}"
            token_num = len(tokenizer.encode(question + " " + cot + " " + answer))
            if token_num > max_token_num:
                return None
            return question, cot, answer
            
        else:
            question = str(example.get('question', example.get('input', '')))
            cot = str(example.get('cot', example.get('reasoning', '')))
            answer = str(example.get('answer', example.get('output', '')))
            if not question or not answer:
                return None
            token_num = len(tokenizer.encode(question + " " + cot + " " + answer))
            if token_num > max_token_num:
                return None
            return question, cot, answer
    
    def _compute_cot_positions(self, tokenizer):
        """计算每个样本中 CoT 部分的起始和结束位置"""
        positions = []
        for i, (question, cot, answer) in enumerate(zip(self.questions, self.cots, self.answers)):
            q_tokens = tokenizer.encode(question, add_special_tokens=True)
            cot_tokens = tokenizer.encode(cot, add_special_tokens=False)
            cot_start = len(q_tokens)
            cot_end = cot_start + len(cot_tokens)
            positions.append({
                'cot_start': cot_start,
                'cot_end': cot_end,
                'question_len': len(q_tokens),
                'cot_len': len(cot_tokens),
            })
        return positions
    
    def __len__(self):
        return len(self.full_texts)
    
    def __getitem__(self, i) -> Dict:
        return {
            'input_ids': self.tokenized['input_ids'][i],
            'full_text': self.full_texts[i],
            'question': self.questions[i],
            'cot': self.cots[i],
            'answer': self.answers[i],
            'cot_positions': self.cot_positions[i],
        }


@dataclass
class DataCollatorForCollection:
    """用于数据收集的Collator"""
    tokenizer: object
    
    def __call__(self, instances: Sequence[Dict]) -> Dict:
        input_ids = [instance['input_ids'] for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'cot_positions': [inst['cot_positions'] for inst in instances],
        }


def load_cot_dataset(data_name: str, data_path: str = None):
    """加载 CoT 数据集"""
    if "icot" in data_name:
        if 'full' in data_name:
            dataset = load_dataset("zen-E/GSM8k-Aug-NL")["train"]
        else:
            dataset = load_dataset("zen-E/GSM8k-Aug")["train"]
    elif "strategy" in data_name:
        dataset = load_dataset("zen-E/StrategyQA_CoT_GPT4o")["train"]
    elif "commonsense" in data_name:
        dataset = load_dataset("zen-E/CommonsenseQA-GPT4omini")["train"]
    elif "prontoqa" in data_name:
        if data_path and os.path.exists(data_path):
            with open(data_path) as f:
                dataset = json.load(f)
        else:
            raise ValueError(f"ProntoQA requires a data_path, got: {data_path}")
    elif data_path:
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                dataset = json.load(f)
        elif data_path.endswith('.jsonl'):
            dataset = []
            with open(data_path, 'r') as f:
                for line in f:
                    dataset.append(json.loads(line))
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
    else:
        raise ValueError(f"Dataset {data_name} is not supported and no data_path provided.")
    
    return dataset


# ============================================================================
# 扩展的 EntropyDataset，支持额外特征
# ============================================================================

class EntropyDatasetExtended(EntropyDataset):
    """
    扩展的熵数据集，支持存储额外特征:
    - raw_entropy: 原始熵值 (SwiReasoning风格)
    - normalized_entropy: 归一化熵 (用于训练)
    - entropy_delta: 熵变化量
    - entropy_trend: 熵趋势 (-1, 0, 1)
    """
    
    def __init__(self, data_path: str = None):
        super().__init__(data_path)
        self.raw_entropies = []
        self.entropy_deltas = []
        self.entropy_trends = []
    
    def add_sample_extended(
        self,
        hidden_state: torch.Tensor,
        normalized_entropy: torch.Tensor,
        raw_entropy: torch.Tensor = None,
        entropy_delta: torch.Tensor = None,
        entropy_trend: torch.Tensor = None,
    ):
        """添加带有扩展特征的样本"""
        self.add_sample(hidden_state, normalized_entropy)
        
        if raw_entropy is not None:
            self.raw_entropies.append(raw_entropy)
        if entropy_delta is not None:
            self.entropy_deltas.append(entropy_delta)
        if entropy_trend is not None:
            self.entropy_trends.append(entropy_trend)
    
    def save(self, path: str):
        """保存数据（包括扩展特征）"""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        
        data = {
            'hidden_states': torch.stack(self.hidden_states) if self.hidden_states else None,
            'entropies': torch.stack(self.entropies) if self.entropies else None,
        }
        
        if self.raw_entropies:
            data['raw_entropies'] = torch.stack(self.raw_entropies)
        if self.entropy_deltas:
            data['entropy_deltas'] = torch.stack(self.entropy_deltas)
        if self.entropy_trends:
            data['entropy_trends'] = torch.stack(self.entropy_trends)
        
        torch.save(data, path)
        print(f"Saved {len(self)} samples to {path}")
    
    @classmethod
    def load_extended(cls, path: str):
        """加载扩展数据集"""
        data = torch.load(path)
        dataset = cls()
        
        if data.get('hidden_states') is not None:
            dataset.hidden_states = list(data['hidden_states'])
        if data.get('entropies') is not None:
            dataset.entropies = list(data['entropies'])
        if data.get('raw_entropies') is not None:
            dataset.raw_entropies = list(data['raw_entropies'])
        if data.get('entropy_deltas') is not None:
            dataset.entropy_deltas = list(data['entropy_deltas'])
        if data.get('entropy_trends') is not None:
            dataset.entropy_trends = list(data['entropy_trends'])
        
        return dataset


# ============================================================================
# 数据收集
# ============================================================================

def collect_entropy_data(
    model,
    tokenizer,
    dataset: CoTDataset,
    device: torch.device,
    batch_size: int = 1,
    collect_positions: str = "all",
    compute_dynamics: bool = True,
    window_size: int = 5,
) -> EntropyDatasetExtended:
    """
    收集 (hidden_states, entropy) 数据，使用 SwiReasoning 风格的熵计算
    
    Args:
        model: 预训练的模型
        tokenizer: tokenizer
        dataset: CoTDataset 实例
        device: 设备
        batch_size: batch 大小
        collect_positions: 收集哪些位置的数据 ("all", "cot", "answer")
        compute_dynamics: 是否计算熵动态特征
        window_size: 计算移动平均的窗口大小
    
    Returns:
        EntropyDatasetExtended: 包含 (hidden_state, entropy) pairs 及扩展特征
    """
    model.eval()
    entropy_dataset = EntropyDatasetExtended()
    
    data_collator = DataCollatorForCollection(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=data_collator,
        shuffle=False
    )
    
    print(f"Collecting entropy data (positions={collect_positions}, dynamics={compute_dynamics})...")
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        cot_positions_batch = batch['cot_positions']
        
        if input_ids.shape[1] < 2:
            continue
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            
            hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
            logits = outputs.logits  # (batch, seq_len, vocab_size)
            
            # 使用 SwiReasoning 风格计算熵
            if compute_dynamics:
                entropy_features = compute_entropy_dynamics(logits, window_size)
                raw_entropy = entropy_features["raw_entropy"]
                normalized_entropy = entropy_features["normalized_entropy"]
                entropy_delta = entropy_features["entropy_delta"]
                entropy_trend = entropy_features["entropy_trend"]
            else:
                raw_entropy, normalized_entropy = compute_entropy_features(logits)
                entropy_delta = None
                entropy_trend = None
        
        # 根据收集策略添加数据
        for i in range(input_ids.shape[0]):
            seq_len = attention_mask[i].sum().item()
            cot_pos = cot_positions_batch[i]
            
            if collect_positions == "all":
                positions = range(seq_len)
            elif collect_positions == "cot":
                cot_start = cot_pos['cot_start']
                cot_end = min(cot_pos['cot_end'], seq_len)
                positions = range(cot_start, cot_end)
            elif collect_positions == "answer":
                cot_end = cot_pos['cot_end']
                positions = range(cot_end, seq_len)
            else:
                positions = range(seq_len)
            
            for pos in positions:
                if pos < seq_len and attention_mask[i, pos] > 0:
                    entropy_dataset.add_sample_extended(
                        hidden_state=hidden_states[i, pos].cpu(),
                        normalized_entropy=normalized_entropy[i, pos].cpu(),
                        raw_entropy=raw_entropy[i, pos].cpu() if raw_entropy is not None else None,
                        entropy_delta=entropy_delta[i, pos].cpu() if entropy_delta is not None else None,
                        entropy_trend=entropy_trend[i, pos].cpu() if entropy_trend is not None else None,
                    )
        
        if (batch_idx + 1) % 100 == 0:
            print(f"Processed {(batch_idx + 1) * batch_size} samples, collected {len(entropy_dataset)} data points")
    
    print(f"Total collected: {len(entropy_dataset)} data points")
    return entropy_dataset


def main():
    parser = argparse.ArgumentParser(description="Collect entropy data (SwiReasoning style)")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to pretrained CoT model")
    parser.add_argument("--data_name", type=str, default="icot",
                        choices=["icot", "icot-full", "strategy", "commonsense", "prontoqa", "custom"],
                        help="Name of the dataset to use")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to local data file")
    parser.add_argument("--output_path", type=str, default="data/entropy_data_swir.pt",
                        help="Output path for collected data")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--max_token_num", type=int, default=1024,
                        help="Maximum token number for filtering")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for data collection")
    parser.add_argument("--collect_positions", type=str, default="cot",
                        choices=["all", "cot", "answer"],
                        help="Which positions to collect data from")
    parser.add_argument("--compute_dynamics", action="store_true",
                        help="Compute entropy dynamics (delta, trend)")
    parser.add_argument("--window_size", type=int, default=5,
                        help="Window size for moving average")
    parser.add_argument("--include_last_cot", action="store_true",
                        help="Include the last CoT step")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载模型和 tokenizer
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
        model.resize_token_embeddings(len(tokenizer))
    
    # 加载数据
    print(f"Loading dataset: {args.data_name}...")
    raw_data = load_cot_dataset(args.data_name, args.data_path)
    
    # 创建数据集
    dataset = CoTDataset(
        data_name=args.data_name,
        raw_data=raw_data,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_token_num=args.max_token_num,
        max_samples=args.max_samples,
        include_last_cot=args.include_last_cot,
    )
    
    # 收集熵数据
    entropy_dataset = collect_entropy_data(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        collect_positions=args.collect_positions,
        compute_dynamics=args.compute_dynamics,
        window_size=args.window_size,
    )
    
    # 保存
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    entropy_dataset.save(args.output_path)
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("Data Collection Summary (SwiReasoning Style)")
    print("=" * 60)
    print(f"Dataset: {args.data_name}")
    print(f"Total samples in dataset: {len(dataset)}")
    print(f"Collected data points: {len(entropy_dataset)}")
    print(f"Collect positions: {args.collect_positions}")
    print(f"Compute dynamics: {args.compute_dynamics}")
    
    if len(entropy_dataset) > 0:
        # 归一化熵统计
        all_entropies = torch.stack(entropy_dataset.entropies)
        print(f"\nNormalized Entropy (for training):")
        print(f"  Mean: {all_entropies.mean().item():.4f}")
        print(f"  Std:  {all_entropies.std().item():.4f}")
        print(f"  Min:  {all_entropies.min().item():.4f}")
        print(f"  Max:  {all_entropies.max().item():.4f}")
        
        # 原始熵统计 (SwiReasoning风格)
        if entropy_dataset.raw_entropies:
            raw_entropies = torch.stack(entropy_dataset.raw_entropies)
            print(f"\nRaw Entropy (SwiReasoning style):")
            print(f"  Mean: {raw_entropies.mean().item():.4f}")
            print(f"  Std:  {raw_entropies.std().item():.4f}")
            print(f"  Min:  {raw_entropies.min().item():.4f}")
            print(f"  Max:  {raw_entropies.max().item():.4f}")
        
        # 熵趋势统计
        if entropy_dataset.entropy_trends:
            trends = torch.stack(entropy_dataset.entropy_trends)
            up_count = (trends > 0).sum().item()
            down_count = (trends < 0).sum().item()
            stable_count = (trends == 0).sum().item()
            total = len(trends)
            print(f"\nEntropy Trend Distribution:")
            print(f"  Up (should use soft/latent): {up_count} ({up_count/total*100:.1f}%)")
            print(f"  Down (should use normal): {down_count} ({down_count/total*100:.1f}%)")
            print(f"  Stable: {stable_count} ({stable_count/total*100:.1f}%)")
        
        # 分布
        print(f"\nNormalized Entropy Distribution:")
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        for i in range(len(bins) - 1):
            count = ((all_entropies >= bins[i]) & (all_entropies < bins[i+1])).sum().item()
            pct = count / len(all_entropies) * 100
            print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {count} ({pct:.1f}%)")
    
    print(f"\nData saved to: {args.output_path}")
    
    # 阈值建议（参考SwiReasoning的切换逻辑）
    print("\n" + "=" * 60)
    print("Threshold Recommendations (SwiReasoning Style)")
    print("=" * 60)
    print("SwiReasoning uses relative entropy comparison:")
    print("  - cur_entropy < ref_entropy -> Switch to NORMAL (explicit)")
    print("  - cur_entropy > ref_entropy -> Switch to SOFT (latent)")
    print("\nFor absolute thresholds, consider percentiles:")
    
    if len(entropy_dataset) > 0:
        percentiles = [25, 50, 75]
        for p in percentiles:
            # val = torch.tensor(all_entropies).quantile(p / 100).item()
            val = all_entropies.float().quantile(p / 100).item()

            print(f"  {p}th percentile: {val:.4f}")


if __name__ == "__main__":
    main()