# Modified from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
import copy
import logging
import os
import re
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Iterator
import torch
import json
import transformers
from torch.utils.data import Dataset, Sampler, DataLoader
from transformers import Trainer
from safetensors.torch import load_file
from tqdm import tqdm
from math import ceil
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from datasets import load_dataset
from functools import partial
from collections import defaultdict, Counter
from src.model_adaptive import (
    CODI,
    ModelArguments,
    DataArguments,
    TrainingArguments,
    freeze_model
)


def _to_scalar(x):
    """Convert Tensor/number/None to python float (mean-reduced if needed)."""
    import torch
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().float().mean().item()
    return float(x)


def read_json(file_path):
    """从指定路径读取JSON文件并返回对应的Python对象。"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except Exception as e:
        print(f"读取JSON文件时出错: {e}")
        return None


IGNORE_INDEX = -100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# ============================================================================
# Step-Grouped Batch Sampler
# ============================================================================

class StepGroupedBatchSampler(Sampler):
    """
    按step数量分组的BatchSampler
    确保每个batch内的样本具有相同的step数量
    """
    def __init__(self, num_steps_list, batch_size, drop_last=True, shuffle=True):
        self.num_steps_list = num_steps_list
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        # 按step数量分组
        self.step_to_indices = defaultdict(list)
        for idx, num_steps in enumerate(num_steps_list):
            self.step_to_indices[num_steps].append(idx)
        
        # 打印分组统计
        print("\n" + "="*80)
        print("Step-Grouped Batch Sampling Statistics:")
        print("="*80)
        total_samples = 0
        total_batches = 0
        for num_steps in sorted(self.step_to_indices.keys()):
            indices = self.step_to_indices[num_steps]
            num_samples = len(indices)
            num_batches = num_samples // batch_size
            if not drop_last and num_samples % batch_size > 0:
                num_batches += 1
            dropped = num_samples % batch_size if drop_last else 0
            total_samples += (num_samples - dropped)
            total_batches += num_batches
            print(f"  Step {num_steps:2d}: {num_samples:5d} samples -> {num_batches:4d} batches (dropped: {dropped})")
        
        print(f"\nTotal: {total_samples} samples -> {total_batches} batches")
        print("="*80 + "\n")
        
    def __iter__(self) -> Iterator[list]:
        # 对每组内的indices进行shuffle
        batches = []
        
        for num_steps, indices in self.step_to_indices.items():
            indices = indices.copy()
            if self.shuffle:
                random.shuffle(indices)
            
            # 创建batch
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        
        # 打乱所有batch的顺序
        if self.shuffle:
            random.shuffle(batches)
        
        for batch in batches:
            yield batch
    
    def __len__(self):
        total = 0
        for indices in self.step_to_indices.values():
            num_batches = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size > 0:
                num_batches += 1
            total += num_batches
        return total


# ============================================================================
# Custom Trainer
# ============================================================================

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch):
        # Extract the global step from the optimizer
        step = self.state.global_step

        # Get total training steps
        batch_size = self.args.per_device_train_batch_size
        gradient_accumulation_steps = self.args.gradient_accumulation_steps
        num_epochs = self.args.num_train_epochs
        dataset_size = len(self.train_dataset)

        effective_batch_size = batch_size * self.args.world_size * gradient_accumulation_steps
        total_steps = ceil(dataset_size / effective_batch_size) * num_epochs

        # Add the step information to the inputs dictionary
        inputs["step_ratio"] = step / total_steps
        inputs["step"] = step
        
        # Call the model's forward method
        outputs = model(**inputs)
        loss = outputs["loss"]
        
        if step % self.args.logging_steps == 0:
            logs = {
                "loss": _to_scalar(outputs.get("loss")),
                "ce_loss": _to_scalar(outputs.get("ce_loss")),
                "distill_loss": _to_scalar(outputs.get("distill_loss")),
                "ref_ce_loss": _to_scalar(outputs.get("ref_ce_loss")),
                "explain_loss": _to_scalar(outputs.get("explain_loss")),
                "align_loss": _to_scalar(outputs.get("align_loss")),
            }
            if not hasattr(self, "is_global_zero") or self.is_global_zero:
                self.log(logs)

        return loss

    def log(self, logs, start_time=None):
        if self.state.global_step is not None:
            for k, v in logs.items():
                super().log({k: v})


# ============================================================================
# Tokenization Functions
# ============================================================================

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=256,
            truncation=True,
            return_attention_mask=False
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    segment = [sentence]
    if len(segment) > 1:
        pred_answer = segment[1]
        pred_answer = [s for s in re.findall(r'-?\d+\.?\d*', pred_answer)]
        if len(pred_answer) > 0:
            pred_answer = pred_answer[0]
        else:
            pred_answer = float(pred[-1])
    else:
        pred_answer = float(pred[-1])

    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer


def get_answer_token_position(tokens, answer_prompts, tokenizer):
    try:
        match_indices = (tokens.unfold(0, len(answer_prompts[0]), 1) == answer_prompts[0]).all(dim=1).nonzero(as_tuple=True)[0].item()
        answer_token_id = match_indices + len(answer_prompts[0])
        return answer_token_id
    except Exception:
        breakpoint()


def preprocess(
    sources: Sequence[str], 
    targets: Sequence[str], 
    answers: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer, 
    bot_id: int,
    eot_id: int,
    training_args,
) -> Dict:
    print("Tokenizing inputs... This may take some time...")
    sources_id = _tokenize_fn(sources, tokenizer)["input_ids"]
    cot_id = _tokenize_fn(targets, tokenizer)["input_ids"]
    answers_id = _tokenize_fn(answers, tokenizer)["input_ids"]

    # add eos token to accommodate pretrained model's format
    if not training_args.remove_eos:
        sources_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in sources_id]
        cot_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in cot_id]
    answers_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in answers_id]

    if cot_id[0][0] == tokenizer.bos_token_id:
        cot_id = [x[1:] for x in cot_id]
        answers_id = [x[1:] for x in answers_id]

    ref_input_ids = [torch.cat([x, y, z]).to(torch.long) for x, y, z in zip(sources_id, cot_id, answers_id)]
    ref_labels = []
    for x, y in zip(ref_input_ids, sources_id):
        z = x.clone()
        z[:len(y)] = -100
        ref_labels.append(z)
    
    if training_args.remove_eos:
        pass
    else:
        pass

    answer_prompts = [torch.tensor(tokenizer.encode("The answer is:")), torch.tensor(tokenizer.encode("The next step result is:"))]
    if answer_prompts[0][0] == tokenizer.bos_token_id:
        answer_prompts[0] = answer_prompts[0][1:]
        answer_prompts[1] = answer_prompts[1][1:]
    
    ref_answer_position = [get_answer_token_position(x, answer_prompts, tokenizer) for i, x in enumerate(ref_input_ids)]
    model_answer_position = [get_answer_token_position(x, answer_prompts, tokenizer) for x in answers_id]

    ref_eos_position = [len(x)-1 for x in ref_input_ids]
    model_eos_position = [len(x)-1 for x in answers_id]
    
    return dict(
        encoder_input_ids=sources_id,
        decoder_input_ids=answers_id,
        ref_input_ids=ref_input_ids, 
        labels=answers_id,
        ref_answer_position=ref_answer_position, 
        model_answer_position=model_answer_position,
        ref_eos_position=ref_eos_position, 
        model_eos_position=model_eos_position, 
        ref_labels=ref_labels
    )


# ============================================================================
# Dataset
# ============================================================================

class SupervisedDataset(Dataset):
    QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"
    QUESTION_DA_PROMPT = "\nAnswer the above question. Answer the final number directly in one number.\n"
    
    def __init__(self, data_name, raw_data, tokenizer, bot, eot, training_args):
        super(SupervisedDataset, self).__init__()
        logging.warning("Formatting inputs...")
        
        self.data_name = data_name
        self.training_args = training_args
        questions, cots, answers = [], [], []
        num_steps_list = []  # 新增：记录每个样本的step数量

        token_nums = []
        
        for num_iter, example in tqdm(enumerate(raw_data)):
            if 'cot' not in example: 
                example['cot'] = example['steps']
                example['cot'] = ' '.join(example['cot'])
            if training_args.exp_mode and num_iter > training_args.exp_data_num:
                break
                
            question = f"{example['question']}"
            
            if "icot" in self.data_name and "full" in self.data_name:
                # bad data
                if example["answer"] is None:
                    continue
                
                # avoid OOM: remove very long data
                token_num = len(tokenizer.encode(example["question"] + example["cot"] + example["answer"]))
                if token_num > training_args.max_token_num:
                    continue
 
                cot = f"{example['cot']}".split(". ")
                if not (training_args.include_last_cot):
                    cot = cot[:-1]
                
                # 计算step数量
                num_steps = len(cot)
                
                answer = f"The answer is: {example['answer'].split(' ')[-1]}"
                answer = answer.replace("####", "")
                questions.append(question)
                cots.append(". ".join(cot)+".\n")
                answers.append(answer)
                num_steps_list.append(num_steps)
                
            elif "icot" in self.data_name:
                # avoid OOM: remove very long data
                token_num = len(tokenizer.encode(example["question"] + example["cot"] + example["answer"]))
                if token_num > training_args.max_token_num:
                    continue
 
                cot = f"{example['cot']}".split(" ")
                if not training_args.include_last_cot:
                    cot = cot[:-1]
                
                # 计算step数量（根据特殊标记）
                # GSM8k-Aug 格式中，通常用 << >> 标记step
                full_cot = " ".join(cot)
                num_steps = full_cot.count('<<')
                if num_steps == 0:  # 如果没有明确标记，使用默认值
                    num_steps = min(training_args.num_latent, len(cot) // 10 + 1)
                    
                answer = example['answer'].split(' ')[-1]
                
                # some answers start with the negative sign (-), bringing distillation problems for LLaMA
                if not answer[0].isdigit():
                    continue

                answer = f"The answer is: {answer}" 
                answer = answer.replace("####", "")
                questions.append(question)
                cots.append(" ".join(cot))
                answers.append(answer)
                num_steps_list.append(num_steps)
                
            elif "commonsense" in self.data_name or "strategy" in self.data_name:
                question = example['question'].strip() + '\n'
                cot = example['cot'].strip() + "\n"
                answer = f"The answer is: {str(example['answer']).strip()}"
                
                # avoid OOM: remove very long data
                token_num = len(tokenizer.encode(question + " " + cot + " " + answer))
                if token_num > training_args.max_token_num: 
                    continue
                
                # 计算step数量（按句子分）
                num_steps = len([s for s in cot.split('.') if s.strip()])
                
                questions.append(question)
                cots.append(cot)
                answers.append(answer)
                num_steps_list.append(num_steps)
                
            elif "prontoqa" in self.data_name:
                question = example['question'].strip() + '\n'
                cot = '\n'.join(example['steps'][:-1]) + "\n"
                answer = f"The answer is: {str(example['answer']).strip()}"
                
                # avoid OOM: remove very long data
                token_num = len(tokenizer.encode(question + " " + cot + " " + answer))
                if token_num > training_args.max_token_num: 
                    continue
                
                # 计算step数量
                num_steps = len(example['steps']) - 1  # 减1因为排除了最后一步
                
                questions.append(question)
                cots.append(cot)
                answers.append(answer)
                num_steps_list.append(num_steps)
        
        if training_args.exp_mode:
            questions = questions[:training_args.exp_data_num]
            cots = cots[:training_args.exp_data_num]
            answers = answers[:training_args.exp_data_num]
            num_steps_list = num_steps_list[:training_args.exp_data_num]
        
        print(f"{len(cots)} data in total...")
        
        # 按step数量分组并重排序
        print("\nGrouping data by number of steps...")
        data_with_steps = list(zip(questions, cots, answers, num_steps_list))
        
        # 按step数量排序
        data_with_steps.sort(key=lambda x: x[3])
        
        # 打印统计信息
        step_counter = Counter(num_steps_list)
        print(f"\nStep distribution before sorting:")
        for num_steps in sorted(step_counter.keys()):
            print(f"  {num_steps} steps: {step_counter[num_steps]} samples")
        
        # 解包排序后的数据
        questions, cots, answers, num_steps_list = zip(*data_with_steps)
        questions = list(questions)
        cots = list(cots)
        answers = list(answers)
        num_steps_list = list(num_steps_list)
        
        # 存储step信息供后续使用
        self.num_steps_list = num_steps_list
        
        logging.warning("Tokenizing inputs... This may take some time...")
        self.data_dict = preprocess(questions, cots, answers, tokenizer, bot, eot, training_args)
        self.keys = list(self.data_dict.keys())
        
        # 将step信息添加到data_dict
        self.data_dict['num_steps'] = num_steps_list

    def __len__(self):
        return len(self.data_dict["encoder_input_ids"])

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = {key: self.data_dict[key][i] for key in self.keys}
        item['num_steps'] = self.num_steps_list[i]
        return item


# ============================================================================
# Data Collator
# ============================================================================

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        encoder_input_ids, decoder_input_ids, ref_input_ids, labels, ref_answer_position, model_answer_position, ref_labels, num_steps = \
            tuple([instance[key] for instance in instances] for key in (
                "encoder_input_ids", "decoder_input_ids", "ref_input_ids", 
                "labels", "ref_answer_position", "model_answer_position", 
                "ref_labels", "num_steps"
            ))
        
        # 验证batch内step数量一致性
        unique_steps = set(num_steps)
        if len(unique_steps) > 1:
            print(f"Warning: Batch contains different step counts: {unique_steps}")
        
        # pad left
        reversed_input_ids = [seq.flip(0) for seq in encoder_input_ids]
        encoder_input_ids = torch.nn.utils.rnn.pad_sequence(
            reversed_input_ids, batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        ).flip(1)
        
        # pad
        ref_input_ids = torch.nn.utils.rnn.pad_sequence(
            ref_input_ids, batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        )
        ref_labels = torch.nn.utils.rnn.pad_sequence(
            ref_labels, batch_first=True, 
            padding_value=IGNORE_INDEX
        ) 

        decoder_input_ids = torch.nn.utils.rnn.pad_sequence(
            decoder_input_ids, batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, 
            padding_value=IGNORE_INDEX
        )
      
        return dict(
            encoder_input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            ref_input_ids=ref_input_ids,
            labels=labels,
            encoder_attention_mask=encoder_input_ids.ne(self.tokenizer.pad_token_id),
            ref_answer_position=torch.tensor(ref_answer_position, dtype=torch.long),
            model_answer_position=torch.tensor(model_answer_position, dtype=torch.long),
            ref_attention_mask=ref_input_ids.ne(self.tokenizer.pad_token_id),
            ref_labels=ref_labels,
            num_steps=torch.tensor(num_steps, dtype=torch.long),
        )


# ============================================================================
# Make Dataset Module
# ============================================================================

def make_supervised_data_module(tokenizer, data_args, training_args, model) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    logging.warning("Downloading Data")
    
    if "icot" in data_args.data_name:
        if 'full' in data_args.data_name:
            dataset = load_dataset("zen-E/GSM8k-Aug-NL")["train"]
        else:
            dataset = load_dataset("zen-E/GSM8k-Aug")["train"]
        train_dataset = SupervisedDataset(
            data_name=data_args.data_name, 
            raw_data=dataset, 
            tokenizer=tokenizer, 
            bot=model.bot_id, 
            eot=model.eot_id,
            training_args=training_args
        )
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        
        # 创建step-grouped batch sampler
        batch_sampler = StepGroupedBatchSampler(
            num_steps_list=train_dataset.num_steps_list,
            batch_size=training_args.per_device_train_batch_size,
            drop_last=True,
            shuffle=True
        )
        
        return dict(
            train_dataset=train_dataset, 
            eval_dataset=None, 
            data_collator=data_collator,
            batch_sampler=batch_sampler
        )
        
    elif "strategy" in data_args.data_name:
        dataset = load_dataset("zen-E/StrategyQA_CoT_GPT4o")["train"]
        train_dataset = SupervisedDataset(
            data_name=data_args.data_name, 
            raw_data=dataset, 
            tokenizer=tokenizer, 
            bot=model.bot_id, 
            eot=model.eot_id,
            training_args=training_args
        )
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        
        batch_sampler = StepGroupedBatchSampler(
            num_steps_list=train_dataset.num_steps_list,
            batch_size=training_args.per_device_train_batch_size,
            drop_last=True,
            shuffle=True
        )
        
        return dict(
            train_dataset=train_dataset, 
            eval_dataset=None, 
            data_collator=data_collator,
            batch_sampler=batch_sampler
        )
        
    elif "commonsense" in data_args.data_name:
        dataset = load_dataset("zen-E/CommonsenseQA-GPT4omini")["train"]
        train_dataset = SupervisedDataset(
            data_name=data_args.data_name, 
            raw_data=dataset, 
            tokenizer=tokenizer, 
            bot=model.bot_id, 
            eot=model.eot_id,
            training_args=training_args
        )
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        
        batch_sampler = StepGroupedBatchSampler(
            num_steps_list=train_dataset.num_steps_list,
            batch_size=training_args.per_device_train_batch_size,
            drop_last=True,
            shuffle=True
        )
        
        return dict(
            train_dataset=train_dataset, 
            eval_dataset=None, 
            data_collator=data_collator,
            batch_sampler=batch_sampler
        )
        
    elif "prontoqa" in data_args.data_name:
        with open("/home/ubuntu/coconut/data/prontoqa_train.json") as f:
            dataset = json.load(f)
        train_dataset = SupervisedDataset(
            data_name=data_args.data_name, 
            raw_data=dataset, 
            tokenizer=tokenizer, 
            bot=model.bot_id, 
            eot=model.eot_id,
            training_args=training_args
        )
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        
        batch_sampler = StepGroupedBatchSampler(
            num_steps_list=train_dataset.num_steps_list,
            batch_size=training_args.per_device_train_batch_size,
            drop_last=True,
            shuffle=True
        )
        
        return dict(
            train_dataset=train_dataset, 
            eval_dataset=None, 
            data_collator=data_collator,
            batch_sampler=batch_sampler
        )
    else:
        raise NotImplementedError(f"Dataset {data_args.data_name} is not supported.")


# ============================================================================
# Main Training Function
# ============================================================================

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    ##########################
    #       Peft Model       #
    ##########################
    if model_args.lora_init:
        task_type = TaskType.CAUSAL_LM
        if any(name in model_args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon", "qwen"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["phi"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["gpt2", "gsm-cot"]):
            target_modules = ["c_attn", "c_proj", 'c_fc']
        else:
            raise ValueError(f"Only support LLAMA, Mistral, Falcon, Phi-2, but got {model_args.model_name_or_path}.")
        
        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
            init_lora_weights=True,
        )

    model = CODI(model_args, training_args, lora_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.token,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    training_args.output_dir = os.path.join(
        training_args.output_dir,
        training_args.expt_name,
        model_args.model_name_or_path.split('/')[-1],
        f"ep_{int(training_args.num_train_epochs)}",
        f"lr_{training_args.learning_rate}",
        f"seed_{training_args.seed}",
    )

    data_module = make_supervised_data_module(
        tokenizer=tokenizer, 
        data_args=data_args, 
        training_args=training_args,
        model=model
    )
    
    # 创建trainer但不传入batch_sampler（因为HF Trainer不直接支持）
    trainer = CustomTrainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        train_dataset=data_module['train_dataset'],
        eval_dataset=data_module['eval_dataset'],
        data_collator=data_module['data_collator'],
    )
    
    # 覆盖 get_train_dataloader 方法以使用自定义batch_sampler
    original_get_train_dataloader = trainer.get_train_dataloader
    
    def get_train_dataloader_with_step_grouping():
        """使用step-grouped batch sampler的自定义dataloader，支持断点续训"""
        from transformers.trainer_pt_utils import IterableDatasetShard
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            data_module['train_dataset'],
            batch_sampler=data_module['batch_sampler'],
            collate_fn=data_module['data_collator'],
            num_workers=training_args.dataloader_num_workers,
            pin_memory=True,
        )
        
        return dataloader
    
    trainer.get_train_dataloader = get_train_dataloader_with_step_grouping
        
    resume_ckpt = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=resume_ckpt)

    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()