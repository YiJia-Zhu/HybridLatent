"""
阶段1: 数据收集脚本 (支持 Llama-3.2 Instruct 格式)

修改说明:
1. 添加 Llama-3.2 Instruct 的 prompt 格式
2. 添加调试打印功能
3. 修复归一化熵计算
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

IGNORE_INDEX = -100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# Llama-3.2 Instruct 格式模板
# ============================================================================

LLAMA32_INSTRUCT_TEMPLATE = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{response}"""

LLAMA32_QUESTION_ONLY = """<|begin_of_text|><|start_header_id|>user<|end_header_id|}

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

def format_llama32_instruct(question: str, cot: str, answer: str) -> Tuple[str, str, str]:
    """格式化为 Llama-3.2 Instruct 格式"""
    formatted_question = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    # CoT 和 answer 保持原样，它们是 assistant 的回复部分
    return formatted_question, cot, answer


# ============================================================================
# 熵计算函数
# ============================================================================

def compute_entropy_swir(logits: torch.Tensor) -> torch.Tensor:
    """SwiReasoning 风格的熵计算 (非归一化)"""
    logits_f32 = logits.float()
    probs = F.softmax(logits_f32, dim=-1)
    entropy = -(probs * probs.clamp(min=1e-12).log()).sum(dim=-1)
    return entropy.to(logits.dtype)


def compute_entropy_features(
    logits: torch.Tensor,
    return_both: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """计算熵特征"""
    raw_entropy = compute_entropy_swir(logits)
    
    if return_both:
        vocab_size = logits.size(-1)
        max_entropy = torch.log(torch.tensor(vocab_size, dtype=torch.float32, device=logits.device))
        normalized_entropy = raw_entropy.float() / max_entropy
        return raw_entropy, normalized_entropy.to(logits.dtype)
    
    return raw_entropy, None


def compute_entropy_dynamics(
    logits_sequence: torch.Tensor,
    window_size: int = 5
) -> Dict[str, torch.Tensor]:
    """计算熵的动态特征"""
    batch_size, seq_len, vocab_size = logits_sequence.shape
    
    raw_entropy, normalized_entropy = compute_entropy_features(logits_sequence)
    
    entropy_delta = torch.zeros_like(raw_entropy)
    entropy_delta[:, 1:] = raw_entropy[:, 1:] - raw_entropy[:, :-1]
    
    entropy_trend = torch.zeros_like(raw_entropy)
    entropy_trend[entropy_delta > 0] = 1
    entropy_trend[entropy_delta < 0] = -1
    
    if seq_len >= window_size and window_size > 1:
        raw_float = raw_entropy.float()
        raw_3d = raw_float.unsqueeze(1)
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
# 数据集类 (支持 Llama-3.2 格式)
# ============================================================================

def _tokenize_fn(strings: Sequence[str], tokenizer, max_length: int = 8192) -> Dict:
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


class CoTDatasetLlama(Dataset):
    """支持 Llama-3.2 Instruct 格式的 CoT 数据集"""
    
    def __init__(
        self, 
        data_name: str, 
        raw_data, 
        tokenizer, 
        max_length: int = 8192,
        max_token_num: int = 1024,
        max_samples: int = None,
        include_last_cot: bool = True,
        use_instruct_format: bool = True,  # 是否使用 Instruct 格式
        debug_print: int = 3,  # 打印前 N 个样本用于调试
    ):
        super().__init__()
        logging.warning("Formatting inputs...")
        
        self.data_name = data_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_instruct_format = use_instruct_format
        
        self.questions = []
        self.cots = []
        self.answers = []
        self.full_texts = []
        
        debug_count = 0
        
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
                    
                    # 应用 Llama-3.2 Instruct 格式
                    if use_instruct_format:
                        question, cot, answer = format_llama32_instruct(question, cot, answer)
                    
                    self.questions.append(question)
                    self.cots.append(cot)
                    self.answers.append(answer)
                    full_text = question + cot + answer
                    self.full_texts.append(full_text)
                    
                    # 调试打印
                    if debug_count < debug_print:
                        print(f"\n{'='*60}")
                        print(f"Sample {debug_count + 1}:")
                        print(f"{'='*60}")
                        print(f"[Full Text]:\n{full_text}")
                        print(f"\n[Tokenized length]: {len(tokenizer.encode(full_text))}")
                        debug_count += 1
                        
            except Exception as e:
                logging.warning(f"Error processing example {num_iter}: {e}")
                continue
        
        print(f"\n{len(self.full_texts)} data in total...")
        
        logging.warning("Tokenizing inputs...")
        self.tokenized = _tokenize_fn(self.full_texts, tokenizer, max_length)
        self.cot_positions = self._compute_cot_positions(tokenizer)
        
        # 打印 cot_positions 示例
        if len(self.cot_positions) > 0:
            print(f"\n[CoT Positions Example]:")
            print(f"  First sample: {self.cot_positions[0]}")
    
    def _process_example(self, example, data_name, tokenizer, max_token_num, include_last_cot):
        """处理单个样本 (返回原始格式，格式化在外部完成)"""
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
            # 对于 Llama-3.2 格式，question 已包含特殊标记
            q_tokens = tokenizer.encode(question, add_special_tokens=False)
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
    """用于数据收集的 Collator"""
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
            'full_texts': [inst['full_text'] for inst in instances],
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
        raise ValueError(f"Dataset {data_name} not supported and no data_path provided.")
    
    return dataset


# ============================================================================
# 扩展的 EntropyDataset
# ============================================================================

class EntropyDatasetExtended:
    """扩展的熵数据集"""
    
    def __init__(self, data_path: str = None):
        self.hidden_states = []
        self.entropies = []
        self.raw_entropies = []
        self.entropy_deltas = []
        self.entropy_trends = []
        
        if data_path and os.path.exists(data_path):
            self.load(data_path)
    
    def __len__(self):
        return len(self.hidden_states)
    
    def add_sample(self, hidden_state, normalized_entropy):
        self.hidden_states.append(hidden_state)
        self.entropies.append(normalized_entropy)
    
    def add_sample_extended(
        self,
        hidden_state: torch.Tensor,
        normalized_entropy: torch.Tensor,
        raw_entropy: torch.Tensor = None,
        entropy_delta: torch.Tensor = None,
        entropy_trend: torch.Tensor = None,
    ):
        self.add_sample(hidden_state, normalized_entropy)
        if raw_entropy is not None:
            self.raw_entropies.append(raw_entropy)
        if entropy_delta is not None:
            self.entropy_deltas.append(entropy_delta)
        if entropy_trend is not None:
            self.entropy_trends.append(entropy_trend)
    
    def save(self, path: str):
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
    
    def load(self, path: str):
        data = torch.load(path)
        if data.get('hidden_states') is not None:
            self.hidden_states = list(data['hidden_states'])
        if data.get('entropies') is not None:
            self.entropies = list(data['entropies'])
        if data.get('raw_entropies') is not None:
            self.raw_entropies = list(data['raw_entropies'])
        if data.get('entropy_deltas') is not None:
            self.entropy_deltas = list(data['entropy_deltas'])
        if data.get('entropy_trends') is not None:
            self.entropy_trends = list(data['entropy_trends'])


# ============================================================================
# 数据收集 (带调试功能)
# ============================================================================

def collect_entropy_data(
    model,
    tokenizer,
    dataset,
    device: torch.device,
    batch_size: int = 1,
    collect_positions: str = "all",
    compute_dynamics: bool = True,
    window_size: int = 5,
    debug_batches: int = 2,  # 调试打印前 N 个 batch
) -> EntropyDatasetExtended:
    """收集熵数据，带调试功能"""
    model.eval()
    entropy_dataset = EntropyDatasetExtended()
    
    data_collator = DataCollatorForCollection(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=data_collator,
        shuffle=False
    )
    
    print(f"\nCollecting entropy data (positions={collect_positions}, dynamics={compute_dynamics})...")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Max entropy (log vocab): {torch.log(torch.tensor(float(tokenizer.vocab_size))).item():.4f}")
    
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
            
            hidden_states = outputs.hidden_states[-1]
            logits = outputs.logits
            
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
        
        # 调试打印
        if batch_idx < debug_batches:
            print(f"\n{'='*60}")
            print(f"Debug Batch {batch_idx + 1}:")
            print(f"{'='*60}")
            print(f"Input shape: {input_ids.shape}")
            print(f"Logits shape: {logits.shape}")
            
            for i in range(min(2, input_ids.shape[0])):
                seq_len = attention_mask[i].sum().item()
                print(f"\n  Sample {i+1} (seq_len={seq_len}):")
                print(f"    Raw entropy - mean: {raw_entropy[i, :seq_len].mean():.4f}, "
                      f"std: {raw_entropy[i, :seq_len].std():.4f}, "
                      f"min: {raw_entropy[i, :seq_len].min():.4f}, "
                      f"max: {raw_entropy[i, :seq_len].max():.4f}")
                print(f"    Normalized entropy - mean: {normalized_entropy[i, :seq_len].mean():.4f}, "
                      f"std: {normalized_entropy[i, :seq_len].std():.4f}, "
                      f"min: {normalized_entropy[i, :seq_len].min():.4f}, "
                      f"max: {normalized_entropy[i, :seq_len].max():.4f}")
                
                # 打印几个 token 的详细信息
                print(f"    First 10 tokens entropy:")
                for j in range(min(10, seq_len)):
                    token = tokenizer.decode([input_ids[i, j].item()])
                    print(f"      pos {j}: '{token}' -> raw={raw_entropy[i,j]:.4f}, norm={normalized_entropy[i,j]:.4f}")
        
        # 收集数据
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
    parser = argparse.ArgumentParser(description="Collect entropy data (Llama-3.2 support)")
    
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_name", type=str, default="icot",
                        choices=["icot", "icot-full", "strategy", "commonsense", "prontoqa", "custom"])
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="data/entropy_data_llama.pt")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=10240)
    parser.add_argument("--max_token_num", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--collect_positions", type=str, default="cot",
                        choices=["all", "cot", "answer"])
    parser.add_argument("--compute_dynamics", action="store_true")
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--include_last_cot", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_instruct_format", action="store_true", default=True,
                        help="Use Llama-3.2 Instruct format")
    parser.add_argument("--no_instruct_format", action="store_true",
                        help="Disable Instruct format")
    parser.add_argument("--debug_samples", type=int, default=3,
                        help="Number of samples to print for debugging")
    
    args = parser.parse_args()
    
    # 处理 instruct format 参数
    use_instruct = args.use_instruct_format and not args.no_instruct_format
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Use Instruct format: {use_instruct}")
    
    # 加载模型和 tokenizer
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).to(device)
    
    # 设置 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Pad token: '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")
    print(f"EOS token: '{tokenizer.eos_token}' (id={tokenizer.eos_token_id})")
    
    # 加载数据
    print(f"\nLoading dataset: {args.data_name}...")
    raw_data = load_cot_dataset(args.data_name, args.data_path)
    
    # 创建数据集
    dataset = CoTDatasetLlama(
        data_name=args.data_name,
        raw_data=raw_data,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_token_num=args.max_token_num,
        max_samples=args.max_samples,
        include_last_cot=args.include_last_cot,
        use_instruct_format=use_instruct,
        debug_print=args.debug_samples,
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
        debug_batches=2,
    )
    
    # 保存
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    entropy_dataset.save(args.output_path)
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("Data Collection Summary")
    print("=" * 60)
    print(f"Dataset: {args.data_name}")
    print(f"Use Instruct format: {use_instruct}")
    print(f"Total samples in dataset: {len(dataset)}")
    print(f"Collected data points: {len(entropy_dataset)}")
    
    if len(entropy_dataset) > 0:
        all_entropies = torch.stack(entropy_dataset.entropies)
        print(f"\nNormalized Entropy:")
        print(f"  Mean: {all_entropies.mean().item():.4f}")
        print(f"  Std:  {all_entropies.std().item():.4f}")
        print(f"  Min:  {all_entropies.min().item():.4f}")
        print(f"  Max:  {all_entropies.max().item():.4f}")
        
        if entropy_dataset.raw_entropies:
            raw_entropies = torch.stack(entropy_dataset.raw_entropies)
            print(f"\nRaw Entropy:")
            print(f"  Mean: {raw_entropies.mean().item():.4f}")
            print(f"  Std:  {raw_entropies.std().item():.4f}")
            print(f"  Min:  {raw_entropies.min().item():.4f}")
            print(f"  Max:  {raw_entropies.max().item():.4f}")
        
        print(f"\nNormalized Entropy Distribution:")
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for i in range(len(bins) - 1):
            count = ((all_entropies >= bins[i]) & (all_entropies < bins[i+1])).sum().item()
            pct = count / len(all_entropies) * 100
            bar = '█' * int(pct / 2)
            print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {count:6d} ({pct:5.1f}%) {bar}")
        
        print(f"\nPercentiles:")
        for p in [10, 25, 50, 75, 90]:
            val = all_entropies.float().quantile(p / 100).item()
            print(f"  {p:2d}th: {val:.4f}")
    
    print(f"\nData saved to: {args.output_path}")


if __name__ == "__main__":
    main()