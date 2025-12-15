"""
Normal Model Evaluation with vLLM

使用 vLLM 进行标准模型的批量评估，支持多种数据集和模型类型。

使用方法:

# Llama 模型
python step0_baseline_eval.py \
    --model_path ./CODI/pretrained/Llama-3.2-1B-Instruct \
    --data_name gsm8k

# GPT-2 模型
python step0_baseline_eval.py \
    --model_path gpt2 \
    --data_name gsm8k \
    --model_type gpt2

# 限制样本数（测试用）
python step0_baseline_eval.py \
    --model_path gpt2 \
    --data_name gsm8k \
    --model_type gpt2 \
    --max_samples 100 \
    --verbose
"""

import argparse
import json
import re
import time
from typing import List, Tuple, Optional, Dict
from datasets import load_dataset, concatenate_datasets
from vllm import LLM, SamplingParams


# ============================================================================
# 数据集加载
# ============================================================================

def load_eval_dataset(data_name: str) -> Tuple[List[str], List, str, str]:
    """
    加载评估数据集
    返回: (questions, answers, question_key, answer_key)
    """
    print(f"Loading dataset: {data_name}")
    
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
    
    # 提取问题
    questions = [f"{example[question_name].strip().replace('  ', ' ')}" for example in test_set]
    
    # 提取答案
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
        
        # 数值答案
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
    
    print(f"Loaded {len(questions)} examples from {data_name}")
    return questions, answers, question_name, answer_name


# ============================================================================
# 答案提取
# ============================================================================

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


# ============================================================================
# 精度计算
# ============================================================================

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
# Prompt 格式化
# ============================================================================

def format_prompt(question: str, model_type: str = "llama") -> str:
    """格式化输入 prompt"""
    if model_type == "llama":
        # Llama-3 Instruct 格式
        return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{question} Let's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    elif model_type == "gpt2":
        # GPT-2 格式 - 简单的 few-shot 或 completion 格式
        # GPT-2 是基础语言模型，不是指令模型，需要用更简单的格式
        return f"""Q: {question}
A: Let me solve this step by step."""
    elif model_type == "gpt2-simple":
        # GPT-2 更简单的格式 - 直接续写
        return f"Question: {question}\nAnswer:"
    elif model_type == "gpt2-fewshot":
        # GPT-2 few-shot 格式 - 提供示例帮助模型理解任务
        return f"""Q: What is 2 + 3?
A: 2 + 3 = 5. The answer is 5.

Q: What is 10 - 4?
A: 10 - 4 = 6. The answer is 6.

Q: {question}
A:"""
    else:
        # 通用格式
        return f"Question: {question}\nAnswer:"


def detect_model_type(model_path: str, specified_type: Optional[str] = None) -> str:
    """自动检测模型类型"""
    if specified_type:
        return specified_type
    
    model_path_lower = model_path.lower()
    
    if "gpt2" in model_path_lower or "gpt-2" in model_path_lower:
        return "gpt2-fewshot"  # GPT-2 默认使用 few-shot 格式
    elif "llama" in model_path_lower:
        return "llama"
    elif "qwen" in model_path_lower:
        return "qwen"
    elif "mistral" in model_path_lower:
        return "mistral"
    else:
        return "default"


# ============================================================================
# vLLM 批量推理
# ============================================================================

def batch_generate(
    model: LLM,
    prompts: List[str],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    stop_tokens: Optional[List[str]] = None,
) -> Tuple[List[str], Dict[str, float]]:
    """
    批量生成
    返回: (outputs, token_stats)
    token_stats 包含: total_tokens, avg_tokens, min_tokens, max_tokens
    """
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop_tokens,
    )
    
    outputs = model.generate(prompts, sampling_params)
    
    # 按照输入顺序排序输出
    results = [None] * len(prompts)
    token_counts = [None] * len(prompts)
    
    for output in outputs:
        idx = output.request_id
        # vLLM 返回的 request_id 可能是字符串形式的索引
        if isinstance(idx, str):
            idx = int(idx)
        results[idx] = output.outputs[0].text
        # 统计生成的 token 数量
        token_counts[idx] = len(output.outputs[0].token_ids)
    
    # 如果排序失败，按原始顺序返回
    if any(r is None for r in results):
        results = [output.outputs[0].text for output in outputs]
        token_counts = [len(output.outputs[0].token_ids) for output in outputs]
    
    # 计算 token 统计信息
    total_tokens = sum(token_counts)
    avg_tokens = total_tokens / len(token_counts) if token_counts else 0
    min_tokens = min(token_counts) if token_counts else 0
    max_tokens = max(token_counts) if token_counts else 0
    
    token_stats = {
        "total_tokens": total_tokens,
        "avg_tokens": avg_tokens,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "token_counts": token_counts,  # 每个样本的 token 数
    }
    
    return results, token_stats


# ============================================================================
# 主评估函数
# ============================================================================

def evaluate(
    model_path: str,
    data_name: str,
    model_type: Optional[str] = None,
    max_samples: Optional[int] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    gpu_memory_utilization: float = 0.8,
    output_file: Optional[str] = None,
    verbose: bool = False,
):
    """评估模型"""
    
    # 加载数据集
    questions, answers, _, _ = load_eval_dataset(data_name)
    
    if max_samples is not None:
        questions = questions[:max_samples]
        answers = answers[:max_samples]
    
    total_samples = len(questions)
    
    # 检测模型类型
    detected_model_type = detect_model_type(model_path, model_type)
    
    print(f"\n{'='*60}")
    print(f"Evaluating {data_name}: {total_samples} samples")
    print(f"Model: {model_path}")
    print(f"Model type: {detected_model_type}")
    print(f"{'='*60}")
    
    # 加载 vLLM 模型
    print("\nLoading model with vLLM...")
    start_load = time.time()
    
    # GPT-2 使用 float32 或 float16，不支持 bfloat16
    if "gpt2" in detected_model_type or "gpt2" in model_path.lower():
        dtype = "float16"
    else:
        dtype = "bfloat16"
    
    model = LLM(
        model=model_path,
        dtype=dtype,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f}s")
    
    # 格式化 prompts
    prompts = [format_prompt(q, detected_model_type) for q in questions]
    
    # 设置停止 token
    stop_tokens = None
    if "gpt2" in detected_model_type:
        stop_tokens = ["\n\n", "Q:", "Question:"]
    
    # 批量生成
    print(f"\nGenerating responses for {total_samples} samples...")
    start_gen = time.time()
    outputs, token_stats = batch_generate(
        model,
        prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        stop_tokens=stop_tokens,
    )
    gen_time = time.time() - start_gen
    print(f"Generation completed in {gen_time:.2f}s")
    print(f"Average time per sample: {gen_time/total_samples*1000:.2f}ms")
    
    # 提取预测答案并计算精度
    predictions = []
    results = []
    correct_count = 0
    
    for i, (question, output, gold_answer) in enumerate(zip(questions, outputs, answers)):
        pred_answer = extract_answer_number(output, data_name)
        predictions.append(pred_answer)
        
        is_correct = (pred_answer == gold_answer)
        if is_correct:
            correct_count += 1
        
        result = {
            "question": question,
            "output": output,
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
            "correct": is_correct,
            "output_tokens": token_stats["token_counts"][i],
        }
        results.append(result)
        
        if verbose:
            print(f"\n[{i+1}/{total_samples}]")
            print(f"Q: {question[:80]}...")
            print(f"A: {output[:200]}...")
            print(f"Pred: {pred_answer} | Gold: {gold_answer} | {'✓' if is_correct else '✗'}")
            print(f"Output tokens: {token_stats['token_counts'][i]}")
    
    # 计算精度
    accuracy = compute_accuracy(answers, predictions)
    
    # 打印结果
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS: {data_name}")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Model type: {detected_model_type}")
    print(f"Total samples: {total_samples}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"\nToken Statistics:")
    print(f"  Total output tokens: {token_stats['total_tokens']}")
    print(f"  Average output tokens: {token_stats['avg_tokens']:.2f}")
    print(f"  Min output tokens: {token_stats['min_tokens']}")
    print(f"  Max output tokens: {token_stats['max_tokens']}")
    print(f"\nTiming:")
    print(f"  Model load time: {load_time:.2f}s")
    print(f"  Generation time: {gen_time:.2f}s")
    print(f"  Avg per sample: {gen_time/total_samples*1000:.2f}ms")
    print(f"  Tokens per second: {token_stats['total_tokens']/gen_time:.2f}")
    print(f"{'='*60}\n")
    
    # 保存结果
    summary = {
        "model_path": model_path,
        "model_type": detected_model_type,
        "dataset": data_name,
        "total_samples": total_samples,
        "correct": correct_count,
        "accuracy": accuracy,
        "accuracy_pct": accuracy * 100,
        "token_stats": {
            "total_output_tokens": token_stats["total_tokens"],
            "avg_output_tokens": token_stats["avg_tokens"],
            "min_output_tokens": token_stats["min_tokens"],
            "max_output_tokens": token_stats["max_tokens"],
        },
        "timing": {
            "model_load_time": load_time,
            "generation_time": gen_time,
            "avg_time_per_sample_ms": gen_time / total_samples * 1000,
            "tokens_per_second": token_stats["total_tokens"] / gen_time,
        },
    }
    
    if output_file is None:
        output_file = f"results_normal_{data_name}.json"
    
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
    parser = argparse.ArgumentParser(description="Normal Model Evaluation with vLLM")
    
    parser.add_argument("--model_path", type=str, default="/storage/zyj_data/swilatent/SIM-CoT/CODI/ckpts/sft_cot_llama1b/Llama-3.2-1B-Instruct/ep_3/lr_0.0008/seed_11",
                        help="Path to the model (e.g., gpt2, gpt2-medium, gpt2-large, gpt2-xl)")
    parser.add_argument("--data_name", type=str, default="gsm8k",
                        choices=["gsm8k", "gsm-hard", "multi-arith", "svamp", "commonsense"],
                        help="Dataset to evaluate")
    parser.add_argument("--model_type", type=str, default="llama",
                        choices=["llama", "gpt2", "gpt2-simple", "gpt2-fewshot", "default"],
                        help="Model type for prompt formatting (auto-detected if not specified)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to evaluate (None for all)")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 for greedy)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8,
                        help="GPU memory utilization for vLLM")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file path")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed results")
    
    args = parser.parse_args()
    
    evaluate(
        model_path=args.model_path,
        data_name=args.data_name,
        model_type=args.model_type,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        gpu_memory_utilization=args.gpu_memory_utilization,
        output_file=args.output_file,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()