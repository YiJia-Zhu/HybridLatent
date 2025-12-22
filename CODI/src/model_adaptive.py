"""
CODI模型 - 集成step4的动态显隐控制器进行训练

关键改动：
1. 导入step4中的SwiRModeState和SwiRController
2. 在每个step末尾基于熵判断是否切换模式
3. 显式(S): CE loss预测step tokens
4. 隐式(L): bot→latent→eot + alignment loss (对齐下一step开头) + explain loss (decoder预测)
"""

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Sequence, Iterable, Union
from peft import get_peft_model, PeftModel, PeftConfig
from safetensors.torch import load_file
from torch.amp import autocast
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import sys
sys.path.append('..')  # 或使用绝对路径
from Swicontrol import SwiRController, SwiRModeState


def compute_entropy_swir(logits: torch.Tensor) -> torch.Tensor:
    """
    SwiReasoning风格的熵计算（非归一化）
    
    Args:
        logits: [batch, seq_len, vocab_size] 或 [batch, vocab_size]
    Returns:
        entropy: [batch, seq_len] 或 [batch]
    """
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * probs.clamp(min=1e-12).log()).sum(dim=-1)
    return entropy



# ============================================================================
# 原model.py的基础类
# ============================================================================

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="mistralai/Mistral-7B-Instruct-v0.2")
    separate_decoder_name: str = field(default="")
    lora_r: int = field(default=128, metadata={"help": "lora rank"})
    lora_dropout: float = field(default=0.05, metadata={"help": "lora dropout"})
    full_precision: bool = field(default=True, metadata={"help": "whether use int4 for the base model"})
    use_decoder: bool = field(default=False, metadata={"help": 'use decoder'})
    decoder_path: str = field(default=None)
    soft_weight: float = field(default=None, metadata={"help": "soft weight"})
    save_ablation: bool = field(
        default=False,
        metadata={"help": "Save ablation results. Only True when specified in command line."},
    )
    train: bool = field(
        default=True,
        metadata={
            "help": "if true, the model ckpt will be initialized for training; else, it's for inference"
        },
    )
    lora_init: bool = field(
        default=False,
        metadata={"help": "True: Use zero and gaussian initialization; False: Load adapters from LoftQ in HF hub."},
    )
    token: Optional[str] = field(
        default=None,
        metadata={"help": "HF token to access to private models, e.g., meta-llama"},
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the LoRA adapter. Used in evaluation or resuming from the checkpoint."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoftQ does not require this config. Used for QLoRA."},
    )
    ckpt_dir: Optional[str] = field(default=None, metadata={"help": "checkpoint dir for inference."})

@dataclass
class DataArguments:
    data_name: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    debug_data: bool = field(
        default=False,
        metadata={
            "help": "Enable debug dataset to quickly verify the training process"
        },
    )
    batch_size: int = field(default=1, metadata={"help": "batch size during inference"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=28000,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    restore_from: str = field(
        default="",
        metadata={
            "help": "The checkpoint that should be restored from for fine-tuning"
        },
    )
    per_device_train_batch_size: int = field(
        default=1,
    )
    per_device_eval_batch_size: int = field(
        default=1,
    )
    expt_name: str = field(
        default="default",
        metadata={"help": "Experiment name"},
    )
    icot_train_path: str = field(default="/users/k24020023/efficient_cot/icae/code/coconut/icot_gsm8k/train.txt", metadata={"help":"The training data path"})
    num_latent: int = field(default=10, metadata={"help": "The number of latent for training or inference."})
    use_lora: bool = field(default=True, metadata={"help": "Use lora or not."})
    greedy: bool = field(default=False, metadata={"help": "Greedy decoding during inference."})
    exp_mode: bool = field(default=False, metadata={"help": "Use partial number of data. for debugging."})
    exp_data_num: int = field(default=10000, metadata={"help": "The number of data used in exp mode"}) 
    use_prj: bool = field(default=False, metadata={"help": "Use a prj module after the llm for latent generation."}) 
    prj_dim: int = field(default=2048, metadata={"help": "The hidden dim of the projection module."})
    prj_dropout: float = field(default=0.0, metadata={"help": "Dropout ratio of the projection module."})
    prj_no_ln: bool = field(default=False, metadata={"help": "Remove the Layer Norm layer for the projection module."})
    distill_loss_div_std: bool = field(default=False, metadata={"help": "Divide the distillation loss by a std for normallisation."})
    distill_loss_type: str = field(default="smooth_l1", metadata={"help": "Specify the distillation loss. Use smoothL1 by default."})
    distill_loss_factor: float = field(default=1.0, metadata={"help": "A multiplier of the distillation loss."})
    explain_loss_factor: float = field(default=1.0, metadata={"help": "A multiplier of the explain loss."})
    ref_loss_factor: float = field(default=1.0, metadata={"help": "A multiplier of the distillation loss."})
    inf_latent_iterations: int = field(default=1, metadata={"help": ""})
    inf_num_iterations: int = field(default=5, metadata={"help": "Run multiple times during inference"})
    remove_eos: bool = field(default=False, metadata={"help": "Do not add <eos> as a delimiter to split QA."})
    print_ref_model_stats: bool = field(default=False, metadata={"help": "Print some stats for the teacher task."})
    include_last_cot: bool = field(default=False, metadata={"help": "Include the last CoT step in the training data."})
    fix_attn_mask: bool = field(default=False, metadata={"help": "Correct a bug about attention mask."})
    log_full: bool = field(default=False, metadata={"help": "Log all losses."})
    print_loss: bool = field(default=True)
    max_token_num: int = field(default=1000)
    # ===== 新增: 动态显隐训练参数 (与step4对齐) =====
    adaptive_training: bool = field(default=False, metadata={"help": "启用动态显隐训练"})
    window_e_to_l: int = field(default=5, metadata={"help": "Explicit→Latent需等待的步数"})
    window_l_to_e: int = field(default=0, metadata={"help": "Latent→Explicit需等待的步数"})
    max_switch_count: int = field(default=5, metadata={"help": "最大切换次数"})
    align_loss_factor: float = field(default=1.0, metadata={"help": "对齐loss权重"})
    ce_loss_factor: float = field(default=1.0)
    # 在 TrainingArguments dataclass 中添加:
    baseline_mode: str = field(default="adaptive", metadata={"help": "训练模式: adaptive 或 random"})
    random_prob: float = field(default=0.5, metadata={"help": "random模式下切换到normal的概率"})

def print_trainable_parameters(model):
    trainable_parameters = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    print(
        f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}"
    )
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)


def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False

def get_steps(
    ref_input_ids: Union[torch.Tensor, Sequence[Sequence[int]]],
    latent_num: int = 2,
    start_ids: Iterable[int] = (2501, 1134),
    end_id: int = 2511,
    eos_id: int = 128009,
    pad_id: int = 128256,
    stop_ids: Iterable[int] = (128009, 128256),
    trim_at_first_stop: bool = True,
) -> List[List[List[int]]]:
    if isinstance(ref_input_ids, torch.Tensor):
        assert ref_input_ids.dim() == 2, "ref_input_ids 应为 [B, T] 的二维张量"
        batch = ref_input_ids
        B, T = batch.size()
        as_lists = batch.detach().cpu().tolist()
    else:
        as_lists = ref_input_ids
        B = len(as_lists)

    start_set = set(start_ids)
    stop_set = set(stop_ids)

    result: List[List[List[int]]] = []
    for b in range(B):
        seq: List[int] = list(as_lists[b])
        if trim_at_first_stop:
            for k, tok in enumerate(seq):
                if tok in stop_set:
                    seq = seq[:k]
                    break

        steps_for_sample: List[List[int]] = []
        i = 0
        n = len(seq)
        while i < n:
            tok = seq[i]
            if tok in start_set:
                j = i + 1
                end_pos: Optional[int] = None
                while j < n:
                    if seq[j] == end_id:
                        end_pos = j
                        break
                    if seq[j] in stop_set:
                        break
                    j += 1
                if end_pos is not None:
                    # steps_for_sample.append(seq[i:end_pos + 1] + [eos_id])
                    steps_for_sample.append(seq[i:end_pos + 1])
                    i = end_pos + 1
                    continue
            i += 1
        # >
        max_steps = latent_num
        if len(steps_for_sample) > max_steps:
            kept = steps_for_sample[:max_steps - 1]
            merged: List[int] = []
            for s in steps_for_sample[max_steps - 1:]:
                if len(s) > 0 and s[-1] == eos_id:
                    merged.extend(s[:-1])
                else:
                    merged.extend(s)
            merged.append(eos_id)
            kept.append(merged)
            steps_for_sample = kept
        # <
        elif len(steps_for_sample) < max_steps:
            while len(steps_for_sample) < max_steps:
                steps_for_sample.append([pad_id])
        else:
        # =
            ...

        result.append(steps_for_sample)

    return result

def pad_steps(
    step_list,
    pad_id: int = 128256
):
    max_len = max(len(step) for steps in step_list for step in steps)
    # 最大的 step 数量
    S_max = max(len(steps) for steps in step_list)
    # 全局最长的 step 长度
    L_max = max(len(step) for steps in step_list for step in steps)
    
    result: List[List[List[int]]] = []
    for steps in step_list:
        padded_steps: List[List[int]] = []
        for step in steps:
            cur = list(step)
            pad_len = L_max - len(cur)
            if pad_len > 0:
                cur = cur + [pad_id] * pad_len
            padded_steps.append(cur)
        while len(padded_steps) < S_max:
            padded_steps.append([pad_id] * L_max)
        result.append(padded_steps)

    return result

def dedup_trailing_pads(explain_embds_list, pad_id=128256):
    if not explain_embds_list:
        return []

    max_len = len(explain_embds_list[0])

    while max_len > 1:
        if all(row[max_len - 2] == pad_id for row in explain_embds_list):
            max_len -= 1
        else:
            break

    return [row[:max_len] for row in explain_embds_list]

class LowRankProjector(nn.Module):
    def __init__(self, input_dim, output_dim, rank=64):
        super(LowRankProjector, self).__init__()
        self.rank = rank
        self.U = nn.Parameter(torch.randn(input_dim, rank))
        self.V = nn.Parameter(torch.randn(rank, output_dim))

    def forward(self, x):
        return torch.matmul(torch.matmul(x, self.U), self.V)


# ============================================================================
# CODI模型 - 集成动态显隐训练
# ============================================================================

class CODI(torch.nn.Module):
    def __init__(self, model_args, training_args, lora_config):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path
        # import pdb; pdb.set_trace()
        model_wrapper_class = AutoModelForCausalLM 
        if model_args.full_precision:
            self.codi = model_wrapper_class.from_pretrained(
                    self.model_name,
                    torch_dtype=(
                        torch.float16 if training_args.bf16 is False else torch.bfloat16
                    ),
                    use_flash_attention_2=False,
                    resume_download=True,
                )
        else:
            self.codi = model_wrapper_class.from_pretrained(
                    self.model_name,
                    torch_dtype=(
                        torch.float16 if training_args.bf16 is False else torch.bfloat16
                    ),
                    use_flash_attention_2=False,
                    resume_download=True,
                    quantization_config=transformers.BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=False,
                        bnb_4bit_quant_type='nf4',
                    )
                )
        # import pdb; pdb.set_trace()
        if model_args.use_decoder:
            if model_args.decoder_path:
                self.decoder = model_wrapper_class.from_pretrained(
                    model_args.decoder_path,
                    torch_dtype=(
                        torch.float16 if training_args.bf16 is False else torch.bfloat16
                    ),
                    use_flash_attention_2=False,
                    resume_download=True,
                )
                if self.codi.lm_head.in_features == self.decoder.lm_head.in_features:
                    self.pj_in = nn.Identity()
                else:
                    self.pj_in = nn.Linear(self.codi.lm_head.in_features, self.decoder.lm_head.in_features)
                # self.pj_out = nn.Linear(self.decoder.lm_head.out_features, self.codi.lm_head.out_features)
                input_dim = self.decoder.lm_head.out_features
                output_dim = self.codi.lm_head.out_features
                if input_dim == output_dim:
                    self.pj_out = nn.Identity()
                else:
                    self.pj_out = LowRankProjector(input_dim, output_dim, rank=input_dim // 4)
            else:
                self.decoder = model_wrapper_class.from_pretrained(
                        self.model_name,
                        torch_dtype=(
                            torch.float16 if training_args.bf16 is False else torch.bfloat16
                        ),
                        use_flash_attention_2=False,
                        resume_download=True,
                    )
        
        # import pdb; pdb.set_trace()

        # saved_weights = torch.load(
        #     '/fs-computility/mllm/shared/weixilin/coconut/ckpts/gsm_cot/gsm-cot/checkpoint_13', map_location=torch.device(self.codi.device)
        # )
        # self.codi.load_state_dict(saved_weights, strict=False)
        

        ori_vocab_size = self.codi.config.vocab_size
        self.training = self.model_args.train

        # special tokens to enclose the latent embeddings
        self.pad_token_id = ori_vocab_size
        self.bot_id = ori_vocab_size + 1
        self.eot_id = ori_vocab_size + 2

        self.codi.resize_token_embeddings(
            ori_vocab_size + 3
        )  # dummy values for mem tokens

        self.dim = self.codi.config.hidden_size
        self.num_latent = training_args.num_latent
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

        # LoRA
        if training_args.use_lora:
            self.codi = get_peft_model(self.codi, lora_config)

        # Projection Layer
        self.use_prj = training_args.use_prj
        self.prj_no_ln = training_args.prj_no_ln
        if training_args.use_prj:
            self.prj = nn.Sequential(
                nn.Dropout(training_args.prj_dropout),
                nn.Linear(self.dim, training_args.prj_dim),
                nn.GELU(),
                nn.Linear(training_args.prj_dim, self.dim),
            )
            if not self.prj_no_ln:
                self.prj.add_module("ln", nn.LayerNorm(self.dim))

            codi_dtype = next(self.codi.parameters()).dtype
            self.prj = self.prj.to(codi_dtype)     

        # Losses
        self.print_loss = training_args.print_loss
        self.ref_loss_factor = training_args.ref_loss_factor

        # Cross Entropy Loss
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100) 
        self.loss_fct_sum = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')

        # Distillation Loss
        self.distill_loss_div_std = training_args.distill_loss_div_std
        self.distill_loss_type = training_args.distill_loss_type
        self.distill_loss_factor = training_args.distill_loss_factor
        if self.distill_loss_type == "smooth_l1":
            self.distill_loss_fct = nn.SmoothL1Loss()
        elif self.distill_loss_type == "l2":
            self.distill_loss_fct = nn.MSELoss()
        else:
            raise NotImplementedError

        # Explain Loss
        self.explain_loss_factor = training_args.explain_loss_factor
        # general 
        self.fix_attn_mask = training_args.fix_attn_mask

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token_id = self.pad_token_id

        # ===== 新增: 动态显隐训练 =====
        self.adaptive_training = getattr(training_args, 'adaptive_training', False)
        self.align_loss_factor = getattr(training_args, 'align_loss_factor', 1.0)
        self.ce_loss_factor = getattr(training_args, 'ce_loss_factor', 1.0)

        self.baseline_mode = getattr(training_args, 'baseline_mode', 'adaptive')
        self.random_prob = getattr(training_args, 'random_prob', 0.5)
        
        # 初始化SwiR控制器 (与step4对齐)
        self.swir_controller = SwiRController(
            window_e_to_l=getattr(training_args, 'window_e_to_l', 5),
            window_l_to_e=getattr(training_args, 'window_l_to_e', 0),
            max_switch_count=getattr(training_args, 'max_switch_count', 5),
        )

        if self.training:
            self.init()

    def get_embd(self, model, model_name):
        try:
            if "pythia" in model_name.lower():
                return model.get_base_model().gpt_neox.embed_in
            elif "gpt2" in model_name.lower():
                try:
                    return model.get_base_model().transformer.wte
                except Exception: # no lora
                    return model.transformer.wte
            else:
                try:
                    return model.get_base_model().model.embed_tokens
                except Exception: # no lora
                    return model.model.embed_tokens
        except AttributeError:
            if "pythia" in model_name:
                return model.gpt_neox.embed_in
            raise NotImplementedError

    def init(self):
        print_trainable_parameters(self)
        if (
            self.training_args.restore_from is not None
            and self.training_args.restore_from != ""
        ):
            print(
                f"Loading from the pretrained checkpoint: {self.training_args.restore_from}..."
            )
            
            restore_path = self.training_args.restore_from
            if restore_path.endswith('.safetensors'):
                state_dict = load_file(restore_path)
            else:
                state_dict = torch.load(restore_path, map_location='cpu', weights_only=False)
            
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"Missing keys: {len(missing)}")
            if unexpected:
                print(f"Unexpected keys: {len(unexpected)}")
            
            print(f"Finished loading from {restore_path}")

    def find_step_positions_in_ref(self, ref_input_ids, start_ids=(2501, 1134), pad_id=128256):
        """
        找到每个step在ref_input_ids中的位置（start token的位置）
        返回: [batch, num_steps] 的位置tensor
        """
        batch_size = ref_input_ids.size(0)
        device = ref_input_ids.device
        
        positions = []
        for b in range(batch_size):
            seq = ref_input_ids[b].tolist()
            step_positions = []
            for i, tok in enumerate(seq):
                if tok in start_ids:
                    step_positions.append(i)
            positions.append(step_positions)
        
        # Pad to same length
        max_steps = max(len(p) for p in positions) if positions else 0
        padded = []
        for p in positions:
            p_padded = p + [-1] * (max_steps - len(p))  # -1表示无效位置
            padded.append(p_padded)
        
        return torch.tensor(padded, dtype=torch.long, device=device)

    def _zero_loss(self):
        """创建一个值为0但有grad_fn的标量"""
        for p in self.codi.parameters():
            if p.requires_grad:
                return (p.flatten()[0] * 0.0)
        return torch.tensor(0.0, device=self.codi.device, requires_grad=True)
    

    def forward(
        self,
        encoder_input_ids: torch.LongTensor = None,
        decoder_input_ids: torch.LongTensor = None,
        ref_input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        ref_answer_position: Optional[torch.LongTensor] = None,
        model_answer_position: Optional[torch.LongTensor] = None,
        ref_attention_mask: Optional[torch.LongTensor] = None,
        ref_labels: torch.LongTensor = None,
        step: int = None,
        step_ratio: float = None,
        debug: bool = False,
    ):
        """
        修复版Forward：正确处理PAD和attention_mask
        - 方案A：维护累积的attention_mask
        - 方案B：动态去除trailing PAD
        """
        
        # ============ 辅助函数：去除trailing PAD ============
        def trim_trailing_pads(token_ids, pad_id):
            """
            去除batch内所有样本共同的trailing PAD
            返回: trimmed_ids, valid_mask
            """
            batch_size, seq_len = token_ids.shape
            
            # 找到每个样本最后一个非PAD位置
            is_not_pad = (token_ids != pad_id)
            if self.tokenizer.pad_token_id is not None and self.tokenizer.pad_token_id != pad_id:
                is_not_pad = is_not_pad & (token_ids != self.tokenizer.pad_token_id)
            
            # 计算每个样本的有效长度
            valid_lens = is_not_pad.long().sum(dim=1)  # [B]
            
            if valid_lens.max().item() == 0:
                # 全是PAD，返回单个PAD
                return token_ids[:, :1], torch.zeros(batch_size, 1, dtype=torch.bool, device=token_ids.device)
            
            # 裁剪到batch内最大有效长度
            max_valid_len = valid_lens.max().item()
            trimmed_ids = token_ids[:, :max_valid_len]
            
            # 创建mask：[B, max_valid_len]
            valid_mask = is_not_pad[:, :max_valid_len]
            
            return trimmed_ids, valid_mask
        
        def get_mode_decision(swir_state, cur_entropy, step_idx):
            """根据baseline_mode决定模式"""
            if self.baseline_mode == "random":
                random_val = torch.rand(batch_size, device=device)
                if step_idx == 0:
                    swir_state.mode = (random_val < self.random_prob).long()
                else:
                    to_normal = (random_val < self.random_prob) & (swir_state.mode == 0)
                    to_soft = (random_val >= self.random_prob) & (swir_state.mode == 1)
                    swir_state.mode = torch.where(to_normal, torch.ones_like(swir_state.mode), swir_state.mode)
                    swir_state.mode = torch.where(to_soft, torch.zeros_like(swir_state.mode), swir_state.mode)
                return swir_state, torch.zeros(batch_size, dtype=torch.bool, device=device), torch.zeros(batch_size, dtype=torch.bool, device=device)
            else:
                return self.swir_controller.update(swir_state, cur_entropy, step=step_idx)
        
        # ============ 初始化 ============
        if not self.fix_attn_mask:
            ref_attention_mask = None
        
        batch_size = encoder_input_ids.shape[0]
        device = encoder_input_ids.device
        
        if 'llama' in self.model_name.lower() or 'qwen' in self.model_name.lower():
            model_pad_id = self.pad_token_id
        else:
            model_pad_id = self.tokenizer.pad_token_id
        
        if debug:
            print("=" * 100)
            print("[DEBUG] FORWARD START - With Cumulative Attention Mask")
            print("=" * 100)
        
        # ============ 解析步骤 ============
        if 'llama' in self.model_name.lower() or 'qwen' in self.model_name.lower():
            steps_list = get_steps(ref_input_ids, self.num_latent + 1)
            steps_pad_list = pad_steps(steps_list, pad_id=model_pad_id)
        elif 'gpt' in self.model_name.lower():
            steps_list = get_steps(
                ref_input_ids, self.num_latent + 1,
                start_ids=(16791, 9959), end_id=4211,
                eos_id=self.tokenizer.eos_token_id, pad_id=self.tokenizer.pad_token_id,
                stop_ids=(self.tokenizer.eos_token_id, self.tokenizer.pad_token_id)
            )
            steps_pad_list = pad_steps(steps_list, pad_id=self.tokenizer.pad_token_id)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        num_steps = len(steps_pad_list[0]) if steps_pad_list else self.num_latent
        
        if 'llama' in self.model_name.lower() or 'qwen' in self.model_name.lower():
            step_positions = self.find_step_positions_in_ref(
                ref_input_ids, start_ids=(2501, 1134), pad_id=model_pad_id
            )
        elif 'gpt' in self.model_name.lower():
            step_positions = self.find_step_positions_in_ref(
                ref_input_ids, start_ids=(16791, 9959), pad_id=model_pad_id
            )
        
        if step_positions.size(1) < num_steps:
            pad_cols = num_steps - step_positions.size(1)
            step_positions = F.pad(step_positions, (0, pad_cols), value=-1)
        
        # ============ Encode问题 + 初始化累积mask ============
        past_key_values = None
        outputs = self.codi(
            input_ids=encoder_input_ids,
            use_cache=True,
            output_hidden_states=True,
            attention_mask=encoder_attention_mask
        )
        past_key_values = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        current_logits = outputs.logits[:, -1, :]
        
        # ★ 初始化累积attention_mask
        if encoder_attention_mask is not None:
            cumulative_attention_mask = encoder_attention_mask.clone()  # [B, Q_len]
        else:
            cumulative_attention_mask = torch.ones(
                batch_size, encoder_input_ids.size(1), 
                dtype=torch.long, device=device
            )
        
        if debug:
            print(f"\n[DEBUG] Initial cumulative_attention_mask shape: {cumulative_attention_mask.shape}")
        
        # ============ 教师模型输出 ============
        with torch.no_grad():
            ref_outputs = self.codi(input_ids=ref_input_ids, output_hidden_states=True, attention_mask=ref_attention_mask)
        ref_outputs_with_grad = self.codi(input_ids=ref_input_ids, output_hidden_states=True, attention_mask=ref_attention_mask)
        
        # ============ 初始化SwiR状态 ============
        swir_state = SwiRModeState.init(batch_size, device)
        cur_entropy = compute_entropy_swir(current_logits)
        swir_state, _, _ = self.swir_controller.update(swir_state, cur_entropy, step=0)
        
        if self.baseline_mode == "random":
            random_init = torch.rand(batch_size, device=device)
            swir_state.mode = (random_init < self.random_prob).long()
        
        latent_embd = last_hidden.unsqueeze(1)
        if self.use_prj:
            latent_embd = self.prj(latent_embd)
        
        # ============ Loss初始化 ============
        _zero = self._zero_loss()
        ce_loss_total = _zero.clone()
        distill_loss_total = _zero.clone()
        explain_loss_total = _zero.clone()
        align_loss_total = _zero.clone()
        
        mode_history = []
        entropy_history = []
        effective_ce_steps = 0
        effective_explain_steps = 0
        ce_token_count = 0
        alignment_computed_at_steps = []
        prev_was_latent = False
        
        # ============ 逐Step推理 ============
        if debug:
            print("\n" + "=" * 100)
            print("[DEBUG] STEP-BY-STEP PROCESSING WITH ATTENTION MASK")
            print("=" * 100)
        
        for step_i in range(num_steps):
            current_mode = swir_state.mode[0].item()
            is_latent_mode = (current_mode == 0)
            mode_str = 'L' if is_latent_mode else 'S'
            mode_history.append(mode_str)
            
            if debug:
                print(f"\n{'='*80}")
                print(f"[STEP {step_i}] MODE: {'LATENT' if is_latent_mode else 'EXPLICIT'}")
                print(f"{'='*80}")
            
            # ===== 收集当前step tokens =====
            current_step_list = []
            for b in range(batch_size):
                current_step_list.append(steps_pad_list[b][step_i])
            current_step_list = dedup_trailing_pads(current_step_list, pad_id=model_pad_id)
            current_step_ids = torch.tensor(current_step_list, dtype=torch.long, device=device)
            
            # 收集下一step（用于alignment/explain）
            next_step_ids = None
            if step_i + 1 < num_steps:
                next_step_list = []
                for b in range(batch_size):
                    next_step_list.append(steps_pad_list[b][step_i + 1])
                next_step_list = dedup_trailing_pads(next_step_list, pad_id=model_pad_id)
                next_step_ids = torch.tensor(next_step_list, dtype=torch.long, device=device)
            
            # ============================================================
            # ★★★ Trim trailing PAD ★★★
            # ============================================================
            current_step_ids_trimmed, current_step_mask = trim_trailing_pads(current_step_ids, model_pad_id)
            
            if debug:
                print(f"\n[PAD TRIMMING]")
                print(f"  Original shape: {current_step_ids.shape}")
                print(f"  Trimmed shape:  {current_step_ids_trimmed.shape}")
                print(f"  Mask shape:     {current_step_mask.shape}")
                print(f"  Sample 0 mask:  {current_step_mask[0].tolist()}")
            
            # ============================================================
            
            if not is_latent_mode:
                # ===== 显式模式 (S) =====
                if debug:
                    print(f"\n[EXPLICIT MODE]")
                
                step_len = current_step_ids_trimmed.size(1)
                
                # ===== 处理step入口的CE loss =====
                if step_len >= 1 and not prev_was_latent:
                    first_token = current_step_ids_trimmed[:, 0].clone()
                    first_token[first_token == model_pad_id] = -100
                    if self.tokenizer.pad_token_id is not None:
                        first_token[first_token == self.tokenizer.pad_token_id] = -100
                    first_token[first_token == self.eot_id] = -100
                    first_token[first_token == self.bot_id] = -100
                    
                    if (first_token != -100).any():
                        valid_entry = (first_token != -100).sum().item()
                        explicit_entry_ce_loss = self.loss_fct_sum(current_logits, first_token)
                        ce_loss_total = ce_loss_total + explicit_entry_ce_loss
                        ce_token_count += valid_entry
                        
                        if debug:
                            print(f"  Explicit Entry CE: {explicit_entry_ce_loss.item():.6f}")
                
                # ===== Forward step tokens =====
                if step_len > 1:
                    step_embds = self.get_embd(self.codi, self.model_name)(current_step_ids_trimmed)
                    
                    # ★★★ 方案A：更新累积mask ★★★
                    cumulative_attention_mask = torch.cat([
                        cumulative_attention_mask, 
                        current_step_mask
                    ], dim=1)
                    
                    if debug:
                        print(f"\n[CUMULATIVE MASK UPDATE]")
                        print(f"  New cumulative_mask shape: {cumulative_attention_mask.shape}")
                        print(f"  Sample 0 mask (last 20): {cumulative_attention_mask[0, -20:].tolist()}")
                    
                    # Forward with attention_mask
                    with autocast(device_type='cuda',dtype=torch.bfloat16):
                        step_out = self.codi(
                            inputs_embeds=step_embds,
                            use_cache=True,
                            output_hidden_states=True,
                            past_key_values=past_key_values,
                            attention_mask=cumulative_attention_mask  # ★ 传入完整mask
                        )
                    past_key_values = step_out.past_key_values
                    
                    # ===== CE Loss =====
                    step_logits = step_out.logits[:, :-1, :]
                    step_targets = current_step_ids_trimmed[:, 1:]
                    
                    step_labels = step_targets.clone()
                    step_labels[~current_step_mask[:, 1:]] = -100
                    step_labels[step_targets == model_pad_id] = -100
                    if self.tokenizer.pad_token_id is not None:
                        step_labels[step_targets == self.tokenizer.pad_token_id] = -100
                    
                    if (step_labels != -100).sum() > 0:
                        valid_step_tokens = (step_labels != -100).sum().item()
                        ce_loss = self.loss_fct_sum(
                            step_logits.reshape(-1, step_logits.size(-1)),
                            step_labels.reshape(-1)
                        )
                        ce_loss_total = ce_loss_total + ce_loss
                        ce_token_count += valid_step_tokens
                        effective_ce_steps += 1
                        
                        if debug:
                            print(f"  Step CE Loss: {ce_loss.item():.6f} (valid_tokens={valid_step_tokens})")
                    
                    last_hidden = step_out.hidden_states[-1][:, -1, :]
                    current_logits = step_out.logits[:, -1, :]
                    latent_embd = last_hidden.unsqueeze(1)
                    if self.use_prj:
                        latent_embd = self.prj(latent_embd)
                
                elif step_len == 1:
                    # 单token step
                    step_embds = self.get_embd(self.codi, self.model_name)(current_step_ids_trimmed)
                    
                    # ★ 更新累积mask
                    cumulative_attention_mask = torch.cat([
                        cumulative_attention_mask, 
                        current_step_mask
                    ], dim=1)
                    
                    with autocast(device_type='cuda',dtype=torch.bfloat16):
                        step_out = self.codi(
                            inputs_embeds=step_embds,
                            use_cache=True,
                            output_hidden_states=True,
                            past_key_values=past_key_values,
                            attention_mask=cumulative_attention_mask  # ★
                        )
                    past_key_values = step_out.past_key_values
                    last_hidden = step_out.hidden_states[-1][:, -1, :]
                    current_logits = step_out.logits[:, -1, :]
                    latent_embd = last_hidden.unsqueeze(1)
                    if self.use_prj:
                        latent_embd = self.prj(latent_embd)
                
                # 更新熵和模式
                cur_entropy = compute_entropy_swir(current_logits)
                entropy_history.append(cur_entropy.mean().item())
                swir_state, _, _ = get_mode_decision(swir_state, cur_entropy, step_i + 1)
                prev_was_latent = False
            
            else:
                # ===== 隐式模式 (L) =====
                if debug:
                    print(f"\n[LATENT MODE]")
                
                # ===== 1. BOT token (只在第一个L时) =====
                if not prev_was_latent:
                    bot_ids = torch.full((batch_size, 1), self.bot_id, dtype=torch.long, device=device)
                    bot_embd = self.get_embd(self.codi, self.model_name)(bot_ids)
                    
                    # ★ BOT是有效token，mask=1
                    bot_mask = torch.ones(batch_size, 1, dtype=torch.long, device=device)
                    cumulative_attention_mask = torch.cat([cumulative_attention_mask, bot_mask], dim=1)
                    
                    if debug:
                        print(f"  BOT added, new mask shape: {cumulative_attention_mask.shape}")
                    
                    with autocast(device_type='cuda',dtype=torch.bfloat16):
                        bot_out = self.codi(
                            inputs_embeds=bot_embd,
                            use_cache=True,
                            output_hidden_states=True,
                            past_key_values=past_key_values,
                            attention_mask=cumulative_attention_mask  # ★
                        )
                    past_key_values = bot_out.past_key_values
                
                # ===== 2. Latent embedding =====
                # ★ Latent也是有效位置，mask=1
                latent_mask = torch.ones(batch_size, 1, dtype=torch.long, device=device)
                cumulative_attention_mask = torch.cat([cumulative_attention_mask, latent_mask], dim=1)
                
                if debug:
                    print(f"  LATENT added, new mask shape: {cumulative_attention_mask.shape}")
                
                with autocast(device_type='cuda',dtype=torch.bfloat16):
                    latent_out = self.codi(
                        inputs_embeds=latent_embd,
                        use_cache=True,
                        output_hidden_states=True,
                        past_key_values=past_key_values,
                        attention_mask=cumulative_attention_mask  # ★
                    )
                past_key_values = latent_out.past_key_values
                last_hidden = latent_out.hidden_states[-1][:, -1, :]
                latent_embd = last_hidden.unsqueeze(1)
                if self.use_prj:
                    latent_embd = self.prj(latent_embd)
                
                # ===== 3. Explain Loss =====
                if self.model_args.use_decoder:
                    current_step_len = current_step_ids_trimmed.size(1)
                    if current_step_len > 0:
                        current_embds = self.get_embd(self.codi, self.model_name)(current_step_ids_trimmed)
                        decoder_input = torch.cat([latent_embd, current_embds], dim=1)
                        
                        prefix_placeholder = torch.full((batch_size, 1), -570, dtype=current_step_ids_trimmed.dtype, device=device)
                        indices_with_prefix = torch.cat([prefix_placeholder, current_step_ids_trimmed], dim=1)
                        
                        explain_labels = indices_with_prefix.clone()
                        explain_labels = explain_labels.masked_fill(
                            (explain_labels == -570) | (explain_labels == model_pad_id),
                            -100
                        )
                        if self.tokenizer.pad_token_id is not None:
                            explain_labels = explain_labels.masked_fill(explain_labels == self.tokenizer.pad_token_id, -100)
                        
                        explain_attention_mask = (indices_with_prefix != model_pad_id) & (indices_with_prefix != -570)
                        if self.tokenizer.pad_token_id is not None:
                            explain_attention_mask = explain_attention_mask & (indices_with_prefix != self.tokenizer.pad_token_id)
                        
                        if hasattr(self, 'pj_in'):
                            decoder_input = self.pj_in(decoder_input)
                        
                        if (explain_labels != -100).sum() > 0:
                            with autocast(device_type='cuda',dtype=torch.bfloat16):
                                dec_out = self.decoder(
                                    inputs_embeds=decoder_input,
                                    attention_mask=explain_attention_mask,
                                    output_hidden_states=True
                                )
                            
                            dec_logits = dec_out.logits
                            if hasattr(self, 'pj_out'):
                                dec_logits = self.pj_out(dec_logits)
                            
                            shift_logits = dec_logits[:, :-1, :].contiguous()
                            shift_labels = explain_labels[:, 1:].contiguous()
                            
                            if (shift_labels != -100).sum() > 0:
                                explain_loss = self.loss_fct(
                                    shift_logits.reshape(-1, shift_logits.size(-1)),
                                    shift_labels.reshape(-1)
                                )
                                explain_loss_total = explain_loss_total + explain_loss
                                effective_explain_steps += 1
                                
                                if debug:
                                    print(f"  Explain Loss: {explain_loss.item():.6f}")
                
                # ===== 4. 判断是否是最后一个L =====
                current_logits = latent_out.logits[:, -1, :]
                cur_entropy = compute_entropy_swir(current_logits)
                entropy_history.append(cur_entropy.mean().item())
                old_mode = swir_state.mode.clone()
                swir_state, _, _ = get_mode_decision(swir_state, cur_entropy, step_i + 1)
                new_mode = swir_state.mode[0].item()
                
                is_last_latent = (new_mode == 1) or (step_i == num_steps - 1)
                
                # ===== 5. EOT token (只在最后一个L时) =====
                if is_last_latent:
                    eot_ids = torch.full((batch_size, 1), self.eot_id, dtype=torch.long, device=device)
                    eot_embd = self.get_embd(self.codi, self.model_name)(eot_ids)
                    
                    # ★ EOT是有效token，mask=1
                    eot_mask = torch.ones(batch_size, 1, dtype=torch.long, device=device)
                    cumulative_attention_mask = torch.cat([cumulative_attention_mask, eot_mask], dim=1)
                    
                    if debug:
                        print(f"  EOT added, new mask shape: {cumulative_attention_mask.shape}")
                    
                    with autocast(device_type='cuda',dtype=torch.bfloat16):
                        eot_out = self.codi(
                            inputs_embeds=eot_embd,
                            use_cache=True,
                            output_hidden_states=True,
                            past_key_values=past_key_values,
                            attention_mask=cumulative_attention_mask  # ★
                        )
                    past_key_values = eot_out.past_key_values
                    current_logits = eot_out.logits[:, -1, :]
                    
                    # Latent退出CE loss
                    eot_logits = eot_out.logits[:, -1, :]
                    if next_step_ids is not None:
                        target_token = next_step_ids[:, 0].clone()
                    else:
                        target_token = decoder_input_ids[:, 0].clone()
                    
                    target_token[target_token == model_pad_id] = -100
                    if self.tokenizer.pad_token_id is not None:
                        target_token[target_token == self.tokenizer.pad_token_id] = -100
                    
                    if (target_token != -100).any():
                        valid_exit = (target_token != -100).sum().item()
                        latent_exit_ce_loss = self.loss_fct_sum(eot_logits, target_token)
                        ce_loss_total = ce_loss_total + latent_exit_ce_loss
                        ce_token_count += valid_exit
                        
                        if debug:
                            print(f"  Latent Exit CE: {latent_exit_ce_loss.item():.6f}")
                    
                    # Alignment Loss
                    if step_i + 1 < num_steps:
                        next_step_ref_pos = step_positions[:, step_i + 1]
                        valid_mask = (next_step_ref_pos >= 0) & (next_step_ref_pos < ref_input_ids.size(1))
                        
                        if valid_mask.any():
                            align_loss = torch.tensor(0.0, device=device)
                            
                            for layer_idx, (eot_h, ref_h) in enumerate(zip(eot_out.hidden_states, ref_outputs.hidden_states)):
                                student_hidden = eot_h[:, -1:, :]
                                
                                ref_pos_expanded = next_step_ref_pos.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, ref_h.size(-1))
                                ref_pos_expanded = ref_pos_expanded.clamp(0, ref_h.size(1) - 1)
                                teacher_hidden = ref_h.gather(1, ref_pos_expanded)
                                
                                if valid_mask.any():
                                    student_valid = student_hidden[valid_mask]
                                    teacher_valid = teacher_hidden[valid_mask]
                                    layer_loss = self.distill_loss_fct(student_valid, teacher_valid.detach())
                                    if self.distill_loss_div_std:
                                        layer_loss = layer_loss / teacher_valid.std().clamp(min=1e-6)
                                    align_loss = align_loss + layer_loss
                            
                            align_loss = align_loss / len(eot_out.hidden_states)
                            align_loss_total = align_loss_total + align_loss
                            alignment_computed_at_steps.append(step_i)
                            
                            if debug:
                                print(f"  Alignment Loss: {align_loss.item():.6f}")
                
                prev_was_latent = True
        
        # ============ 预测答案 ============
        if debug:
            print(f"\n{'='*80}")
            print("[ANSWER PREDICTION]")
            print(f"{'='*80}")
        
        # Answer入口CE loss
        if not prev_was_latent:
            ans_first_token = decoder_input_ids[:, 0].clone()
            ans_first_token[ans_first_token == model_pad_id] = -100
            if self.tokenizer.pad_token_id is not None:
                ans_first_token[ans_first_token == self.tokenizer.pad_token_id] = -100
            
            if (ans_first_token != -100).any():
                valid_ans_entry = (ans_first_token != -100).sum().item()
                ans_entry_ce_loss = self.loss_fct_sum(current_logits, ans_first_token)
                ce_loss_total = ce_loss_total + ans_entry_ce_loss
                ce_token_count += valid_ans_entry
                
                if debug:
                    print(f"  Answer Entry CE: {ans_entry_ce_loss.item():.6f}")
        
        decoder_embds = self.get_embd(self.codi, self.model_name)(decoder_input_ids)
        
        # ★ 更新累积mask（answer tokens）
        answer_mask = (decoder_input_ids != model_pad_id)
        if self.tokenizer.pad_token_id is not None:
            answer_mask = answer_mask & (decoder_input_ids != self.tokenizer.pad_token_id)
        cumulative_attention_mask = torch.cat([cumulative_attention_mask, answer_mask], dim=1)
        
        if debug:
            print(f"  Final cumulative_mask shape: {cumulative_attention_mask.shape}")
        
        with autocast(device_type='cuda',dtype=torch.bfloat16):
            final_out = self.codi(
                inputs_embeds=decoder_embds,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
                attention_mask=cumulative_attention_mask  # ★
            )
        
        # ===== Answer CE Loss =====
        ans_logits = final_out.logits[:, :-1, :]
        ans_labels = labels[:, 1:]
        
        valid_ans_tokens = (ans_labels != -100).sum().item()
        ans_ce_loss = self.loss_fct_sum(
            ans_logits.reshape(-1, ans_logits.size(-1)),
            ans_labels.reshape(-1)
        )
        ce_loss_total = ce_loss_total + ans_ce_loss
        ce_token_count += valid_ans_tokens
        
        if ce_token_count > 0:
            ce_loss_total = ce_loss_total / ce_token_count
        
        ce_loss_total = ce_loss_total * self.ce_loss_factor
        effective_ce_steps += 1
        
        if debug:
            print(f"  Answer CE Loss: {ans_ce_loss.item():.6f}")
        
        # ============ Distillation Loss ============
        if "llama" in self.model_name.lower() or "qwen" in self.model_name.lower():
            model_answer_position = model_answer_position + 1
            ref_answer_position = ref_answer_position + 1
        model_answer_position = model_answer_position - 1
        ref_answer_position = ref_answer_position - 1
        
        for layer_idx, (out_h, ref_h) in enumerate(zip(final_out.hidden_states, ref_outputs.hidden_states)):
            ref_sel = ref_h.gather(1, ref_answer_position.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, ref_h.size(-1)))
            out_sel = out_h[:, -1:, :]
            d_loss = self.distill_loss_fct(out_sel, ref_sel.detach())
            
            if self.distill_loss_div_std:
                teacher_std = ref_sel.std()
                d_loss = d_loss / teacher_std.clamp(min=1e-6)
            
            distill_loss_total = distill_loss_total + d_loss
        
        distill_loss_total = distill_loss_total / len(final_out.hidden_states)
        
        # ============ Reference CE Loss ============
        ref_logits = ref_outputs_with_grad.logits[:, :-1, :]
        ref_labels_shifted = ref_labels[:, 1:]
        
        ref_ce_loss = self.loss_fct(
            ref_logits.reshape(-1, ref_logits.size(-1)),
            ref_labels_shifted.reshape(-1)
        )
        ref_ce_loss = ref_ce_loss * self.ref_loss_factor
        
        # ============ 汇总Loss ============
        distill_loss_total = distill_loss_total * self.distill_loss_factor
        explain_loss_total = explain_loss_total * self.explain_loss_factor
        if effective_explain_steps > 0:
            explain_loss_total = explain_loss_total / effective_explain_steps
        align_loss_total = align_loss_total * self.align_loss_factor
        
        total_loss = ce_loss_total + distill_loss_total + ref_ce_loss + explain_loss_total + align_loss_total
        
        mode_str = ''.join(mode_history)
        if self.print_loss:
            align_info = f"(at steps {alignment_computed_at_steps})" if alignment_computed_at_steps else "(none)"
            print(f"Modes: {mode_str} | Loss={total_loss.item():.4f} | "
                f"CE={ce_loss_total.item():.4f} | Distill={distill_loss_total.item():.4f} | "
                f"RefCE={ref_ce_loss.item():.4f} | Explain={explain_loss_total.item() if isinstance(explain_loss_total, torch.Tensor) else explain_loss_total:.4f} | "
                f"Align={align_loss_total.item():.4f} {align_info}")
        
        if debug:
            print("\n" + "=" * 100)
            print("[DEBUG] FORWARD END")
            print("=" * 100)
        
        return {
            "loss": total_loss,
            "logits": final_out.logits,
            "ce_loss": ce_loss_total.detach(),
            "distill_loss": distill_loss_total.detach(),
            "ref_ce_loss": ref_ce_loss.detach(),
            "explain_loss": explain_loss_total.detach() if isinstance(explain_loss_total, torch.Tensor) else explain_loss_total,
            "align_loss": align_loss_total.detach(),
            "mode_history": mode_history,
            "entropy_history": entropy_history,
            "alignment_computed_at_steps": alignment_computed_at_steps,
        }

