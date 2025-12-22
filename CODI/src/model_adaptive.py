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

        # 在 forward 方法开头添加一个辅助函数:
        def get_mode_decision(swir_state, cur_entropy, step_idx):
            """根据 baseline_mode 决定模式"""
            if self.baseline_mode == "random":
                # random 模式：随机决定模式
                random_val = torch.rand(batch_size, device=device)
                if step_idx == 0:
                    # 初始化时随机选择模式
                    swir_state.mode = (random_val < self.random_prob).long()
                else:
                    # 随机切换
                    to_normal = (random_val < self.random_prob) & (swir_state.mode == 0)
                    to_soft = (random_val >= self.random_prob) & (swir_state.mode == 1)
                    swir_state.mode = torch.where(to_normal, torch.ones_like(swir_state.mode), swir_state.mode)
                    swir_state.mode = torch.where(to_soft, torch.zeros_like(swir_state.mode), swir_state.mode)
                return swir_state, torch.zeros(batch_size, dtype=torch.bool, device=device), torch.zeros(batch_size, dtype=torch.bool, device=device)
            else:
                # adaptive 模式：使用原有的熵判断
                return self.swir_controller.update(swir_state, cur_entropy, step=step_idx)


        if not self.fix_attn_mask:
            ref_attention_mask = None
        
        batch_size = encoder_input_ids.shape[0]
        device = encoder_input_ids.device
        
        # 确定pad_id
        if 'llama' in self.model_name.lower() or 'qwen' in self.model_name.lower():
            model_pad_id = self.pad_token_id
        else:
            model_pad_id = self.tokenizer.pad_token_id
        
        # ========== DEBUG: 打印输入信息 ==========
        if debug:
            print("=" * 100)
            print("[DEBUG] " + "=" * 40 + " FORWARD START " + "=" * 40)
            print("=" * 100)
            print(f"\n[DEBUG] >>> BASIC INFO <<<")
            print(f"  batch_size={batch_size}, device={device}")
            print(f"  model_pad_id={model_pad_id}, bot_id={self.bot_id}, eot_id={self.eot_id}")
            print(f"  encoder_input_ids shape: {encoder_input_ids.shape}")
            print(f"  decoder_input_ids shape: {decoder_input_ids.shape}")
            print(f"  ref_input_ids shape: {ref_input_ids.shape}")
            print(f"  labels shape: {labels.shape}")
            
            print(f"\n[DEBUG] >>> ENCODER INPUT (Question) <<<")
            for b in range(min(batch_size, 2)):
                enc_text = self.tokenizer.decode(encoder_input_ids[b], skip_special_tokens=False)
                print(f"  Sample {b}: {enc_text[:300]}...")
            
            print(f"\n[DEBUG] >>> REF INPUT (Q + full CoT + A) <<<")
            for b in range(min(batch_size, 2)):
                ref_text = self.tokenizer.decode(ref_input_ids[b], skip_special_tokens=False)
                print(f"  Sample {b}: {ref_text[:500]}...")
                print(f"  Last token: {ref_input_ids[b][-1].item()}, EOT={self.eot_id}, EOS={self.tokenizer.eos_token_id}")
                
        
        # ========== 2. 解析步骤 ==========
        if debug:
            print(f"\n[DEBUG] >>> PARSING STEPS <<<")
            
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
        
        # ========== [修改点1] 打印原始steps数量 vs 处理后数量 ==========
        if debug:
            print(f"\n[DEBUG] >>> STEP COUNT CHECK <<<")
            print(f"  num_latent (config): {self.num_latent}")
            print(f"  num_steps (after processing): {num_steps}")
            for b in range(min(batch_size, 2)):
                original_count = len([s for s in steps_list[b] if s != [model_pad_id]])
                print(f"  Sample {b}: original steps parsed = {original_count}, final = {len(steps_pad_list[b])}")
                if original_count > self.num_latent + 1:
                    print(f"    ⚠️  MERGED: Steps exceeded num_latent+1, last steps were merged!")
        
        # 找到每个step的位置
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
        
        # ========== DEBUG: 打印解析出的步骤 ==========
        if debug:
            print(f"  num_steps={num_steps}")
            print(f"  step_positions shape: {step_positions.shape}")
            print(f"  step_positions: {step_positions.tolist()}")
            
            print(f"\n[DEBUG] >>> VERIFY STEP POSITIONS <<<")
            for b in range(min(batch_size, 2)):
                print(f"  Sample {b}:")
                for s_i in range(min(num_steps, step_positions.size(1))):
                    pos = step_positions[b, s_i].item()
                    if pos >= 0 and pos < ref_input_ids.size(1):
                        start = max(0, pos - 2)
                        end = min(ref_input_ids.size(1), pos + 8)
                        context_ids = ref_input_ids[b, start:end].tolist()
                        context_decoded = self.tokenizer.decode(context_ids, skip_special_tokens=False)
                        token_at_pos = ref_input_ids[b, pos].item()
                        token_decoded = self.tokenizer.decode([token_at_pos])
                        print(f"    Step {s_i}: pos={pos}, token_id={token_at_pos}, token='{token_decoded}'")
                        print(f"      Context [...{start}:{end}...]: {context_decoded}")
                    else:
                        print(f"    Step {s_i}: pos={pos} (INVALID)")

            print(f"\n[DEBUG] >>> PARSED STEPS CONTENT <<<")
            for b in range(min(batch_size, 2)):
                print(f"\n  Sample {b}:")
                for s_i in range(num_steps):
                    tokens = steps_pad_list[b][s_i]
                    valid_tokens = [t for t in tokens if t != model_pad_id]
                    if valid_tokens:
                        decoded = self.tokenizer.decode(valid_tokens, skip_special_tokens=False)
                    else:
                        decoded = "[EMPTY/PAD ONLY]"
                    
                    num_eot = tokens.count(self.eot_id)
                    num_bot = tokens.count(self.bot_id)
                    num_pad = tokens.count(model_pad_id)
                    
                    print(f"    Step {s_i}: len={len(tokens)}, valid={len(valid_tokens)}, "
                        f"eot={num_eot}, bot={num_bot}, pad={num_pad}")
                    print(f"      Raw IDs (first 20): {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
                    print(f"      Decoded: {decoded[:120]}{'...' if len(decoded) > 120 else ''}")

        # ========== 3. Encode问题 ==========
        if debug:
            print(f"\n[DEBUG] >>> ENCODER FORWARD <<<")
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
        
        if debug:
            print(f"  past_key_values: {len(past_key_values)} layers")
            print(f"  Each KV shape: K={past_key_values[0][0].shape}, V={past_key_values[0][1].shape}")
            print(f"  last_hidden shape: {last_hidden.shape}")
            print(f"  current_logits shape: {current_logits.shape}")
            
            top_probs, top_ids = torch.softmax(current_logits, dim=-1).topk(5, dim=-1)
            for b in range(min(batch_size, 2)):
                top_tokens = [self.tokenizer.decode([tid]) for tid in top_ids[b].tolist()]
                print(f"  Sample {b} top5 after encoder: {list(zip(top_tokens, [f'{p:.4f}' for p in top_probs[b].tolist()]))}")
        
        # ========== 1. 教师模型输出 ==========
        if debug:
            print(f"\n[DEBUG] >>> TEACHER MODEL FORWARD <<<")
            
        with torch.no_grad():
            ref_outputs = self.codi(input_ids=ref_input_ids, output_hidden_states=True, attention_mask=ref_attention_mask)
        ref_outputs_with_grad = self.codi(input_ids=ref_input_ids, output_hidden_states=True, attention_mask=ref_attention_mask)
        
        if debug:
            print(f"  ref_outputs.logits shape: {ref_outputs.logits.shape}")
            print(f"  ref_outputs.hidden_states: {len(ref_outputs.hidden_states)} layers")
            print(f"  Each hidden state shape: {ref_outputs.hidden_states[0].shape}")

        # ========== 4. 初始化SwiR状态 ==========
        if debug:
            print(f"\n[DEBUG] >>> SWIR INIT <<<")
            
        swir_state = SwiRModeState.init(batch_size, device)
        cur_entropy = compute_entropy_swir(current_logits)
        swir_state, _, _ = self.swir_controller.update(swir_state, cur_entropy, step=0)
        
        # ===== 新增：random 模式下随机初始化第一个 step =====
        if self.baseline_mode == "random":
            random_init = torch.rand(batch_size, device=device)
            swir_state.mode = (random_init < self.random_prob).long()

        if debug:
            print(f"  Initial entropy: {cur_entropy.tolist()}")
            print(f"  Initial mode: {swir_state.mode.tolist()} (0=Latent, 1=Explicit)")
            print(f"  SwiR thresholds - e_to_l: {self.swir_controller.window_e_to_l}, l_to_e: {self.swir_controller.window_l_to_e}")
        
        latent_embd = last_hidden.unsqueeze(1)
        if self.use_prj:
            latent_embd = self.prj(latent_embd)
            
        if debug:
            print(f"  Initial latent_embd shape: {latent_embd.shape}")
            print(f"  Initial latent_embd norm: {latent_embd.norm(dim=-1).mean().item():.4f}")
        
        # ========== 5. Loss累积 ==========
        _zero = self._zero_loss()
        ce_loss_total = _zero.clone()
        distill_loss_total = _zero.clone()
        explain_loss_total = _zero.clone()
        align_loss_total = _zero.clone()

        mode_history = []
        entropy_history = []
        effective_ce_steps = 0
        effective_explain_steps = 0
        ce_token_count = 0  # 总token数
        
        # ========== 用于追踪alignment loss计算 ==========
        alignment_computed_at_steps = []  # 记录在哪些step计算了alignment loss
        
        # ===== 追踪上一步是否是Latent =====
        prev_was_latent = False
        # ========== 6. 逐Step推理 ==========
        if debug:
            print("\n" + "=" * 100)
            print("[DEBUG] " + "=" * 35 + " STEP-BY-STEP PROCESSING " + "=" * 35)
            print("=" * 100)
        
        for step_i in range(num_steps):


            current_mode = swir_state.mode[0].item()
            is_latent_mode = (current_mode == 0)
            mode_str = 'L' if is_latent_mode else 'S'
            mode_history.append(mode_str)
            
            if debug:
                print("\n" + "-" * 80)
                print(f"[DEBUG] STEP {step_i}/{num_steps-1} | MODE: {'LATENT (L)' if is_latent_mode else 'EXPLICIT (S)'}")
                print("-" * 80)
            
            # ===== 收集当前step所有batch的tokens =====
            current_step_list = []
            for b in range(batch_size):
                current_step_list.append(steps_pad_list[b][step_i])
            current_step_list = dedup_trailing_pads(current_step_list, pad_id=model_pad_id)
            current_step_ids = torch.tensor(current_step_list, dtype=torch.long, device=device)
            
            next_step_ids = None
            if step_i + 1 < num_steps:
                next_step_list = []
                for b in range(batch_size):
                    next_step_list.append(steps_pad_list[b][step_i + 1])
                next_step_list = dedup_trailing_pads(next_step_list, pad_id=model_pad_id)
                next_step_ids = torch.tensor(next_step_list, dtype=torch.long, device=device)
            
            if debug:
                print(f"\n[DEBUG] Current Step Tokens:")
                print(f"  current_step_ids shape: {current_step_ids.shape}")
                for b in range(min(batch_size, 2)):
                    step_tokens = current_step_ids[b].tolist()
                    valid_tokens = [t for t in step_tokens if t != model_pad_id]
                    decoded = self.tokenizer.decode(valid_tokens, skip_special_tokens=False) if valid_tokens else "[EMPTY]"
                    print(f"  Sample {b} tokens (first 15): {step_tokens[:15]}...")
                    print(f"  Sample {b} decoded: {decoded[:100]}...")
                
                if next_step_ids is not None:
                    print(f"\n[DEBUG] Next Step Tokens (for alignment/explain):")
                    print(f"  next_step_ids shape: {next_step_ids.shape}")
                    for b in range(min(batch_size, 2)):
                        next_tokens = next_step_ids[b].tolist()
                        valid_next = [t for t in next_tokens if t != model_pad_id]
                        next_decoded = self.tokenizer.decode(valid_next, skip_special_tokens=False) if valid_next else "[EMPTY]"
                        print(f"  Sample {b} tokens (first 15): {next_tokens[:15]}...")
                        print(f"  Sample {b} decoded: {next_decoded[:100]}...")
            
                
            if not is_latent_mode:
                # ===== 显式模式 (S) =====
                if debug:
                    print(f"\n[DEBUG] >>> EXPLICIT MODE PROCESSING <<<")
                
                step_len = current_step_ids.size(1)
                

                # 处理当前句子和下一个步骤第一个token的loss，只有当上一步不是latent时才需要（latent→显式已在latent_exit_ce_loss处理）
                if step_len >= 1 and not prev_was_latent:
                    first_token = current_step_ids[:, 0].clone()
                    # 过滤特殊token和pad
                    first_token[first_token == model_pad_id] = -100
                    if self.tokenizer.pad_token_id is not None:
                        first_token[first_token == self.tokenizer.pad_token_id] = -100
                    first_token[first_token == self.eot_id] = -100
                    first_token[first_token == self.bot_id] = -100
                    
                    # if (first_token != -100).any():
                    #     explicit_entry_ce_loss = self.loss_fct(current_logits, first_token)
                    #     ce_loss_total = ce_loss_total + explicit_entry_ce_loss
                    if (first_token != -100).any():
                        valid_entry = (first_token != -100).sum().item()
                        explicit_entry_ce_loss = self.loss_fct_sum(current_logits, first_token)
                        ce_loss_total = ce_loss_total + explicit_entry_ce_loss
                        ce_token_count += valid_entry

                        if debug:
                            target_decoded = self.tokenizer.decode([first_token[0].item()]) if first_token[0] != -100 else "[masked]"
                            print(f"  ★ Explicit Entry CE Loss: {explicit_entry_ce_loss.item():.6f}, target='{target_decoded}'")


                if step_len > 1:
                    step_embds = self.get_embd(self.codi, self.model_name)(current_step_ids)
                    
                    if debug:
                        print(f"  step_embds shape: {step_embds.shape}")
                    
                    step_attn_mask = (current_step_ids != model_pad_id)
                    if self.tokenizer.pad_token_id is not None and self.tokenizer.pad_token_id != model_pad_id:
                        step_attn_mask = step_attn_mask & (current_step_ids != self.tokenizer.pad_token_id)
                    step_attn_mask = step_attn_mask & (current_step_ids != self.eot_id) & (current_step_ids != self.bot_id)
                    
                    if debug:
                        print(f"  step_attn_mask (sample 0, first 20): {step_attn_mask[0].tolist()[:20]}...")
                        valid_count = step_attn_mask.sum(dim=1).tolist()
                        print(f"  valid tokens per sample: {valid_count}")
                    
                    with autocast(device_type='cuda',dtype=torch.bfloat16):
                        step_out = self.codi(
                            inputs_embeds=step_embds,
                            use_cache=True,
                            output_hidden_states=True,
                            past_key_values=past_key_values,
                            attention_mask=None
                        )
                    past_key_values = step_out.past_key_values
                    
                    if debug:
                        print(f"\n[DEBUG] >>> CE LOSS CALCULATION <<<")
                        
                    step_logits = step_out.logits[:, :-1, :]
                    step_targets = current_step_ids[:, 1:]
                    
                    step_labels = step_targets.clone()
                    step_labels[~step_attn_mask[:, 1:]] = -100
                    step_labels[step_targets == model_pad_id] = -100
                    if self.tokenizer.pad_token_id is not None:
                        step_labels[step_targets == self.tokenizer.pad_token_id] = -100
                    
                    if debug:
                        print(f"  step_logits shape: {step_logits.shape} [batch, seq-1, vocab]")
                        print(f"  step_targets shape: {step_targets.shape} [batch, seq-1]")
                        print(f"  step_labels (sample 0, first 15): {step_labels[0, :15].tolist()}")
                        valid_labels = (step_labels != -100).sum().item()
                        print(f"  total valid labels for CE: {valid_labels}")
                        
                        if valid_labels > 0:
                            pred_ids = step_logits.argmax(dim=-1)
                            for b in range(min(batch_size, 1)):
                                mask_b = step_labels[b] != -100
                                if mask_b.any():
                                    pred_tokens = pred_ids[b][mask_b][:5]
                                    true_tokens = step_labels[b][mask_b][:5]
                                    pred_decoded = [self.tokenizer.decode([t]) for t in pred_tokens.tolist()]
                                    true_decoded = [self.tokenizer.decode([t]) for t in true_tokens.tolist()]
                                    print(f"  Sample {b} pred vs true (first 5):")
                                    print(f"    Pred: {pred_decoded}")
                                    print(f"    True: {true_decoded}")
                    
                    # if (step_labels != -100).sum() > 0:
                    #     ce_loss = self.loss_fct(
                    #         step_logits.reshape(-1, step_logits.size(-1)),
                    #         step_labels.reshape(-1)
                    #     )
                    #     ce_loss_total = ce_loss_total + ce_loss
                    #     effective_ce_steps += 1
                                    
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
                            print(f"\n  *** CE Loss = CrossEntropyLoss(logits[{step_logits.shape}], labels[{step_labels.shape}])")
                            print(f"  *** CE Loss value: {ce_loss.item():.6f} ***")
                    else:
                        if debug:
                            print(f"\n  *** No valid labels, skipping CE loss ***")
                    
                    last_hidden = step_out.hidden_states[-1][:, -1, :]
                    current_logits = step_out.logits[:, -1, :]
                    latent_embd = last_hidden.unsqueeze(1)
                    if self.use_prj:
                        latent_embd = self.prj(latent_embd)
                    
                    if debug:
                        print(f"\n[DEBUG] After explicit step:")
                        print(f"  last_hidden shape: {last_hidden.shape}")
                        print(f"  latent_embd shape: {latent_embd.shape}")
                        top_probs, top_ids = torch.softmax(current_logits, dim=-1).topk(5, dim=-1)
                        for b in range(min(batch_size, 1)):
                            top_tokens = [self.tokenizer.decode([tid]) for tid in top_ids[b].tolist()]
                            print(f"  Sample {b} top5 next: {list(zip(top_tokens, [f'{p:.4f}' for p in top_probs[b].tolist()]))}")
                
                # elif step_len == 1:
                #     if debug:
                #         print(f"  Single token step, just forwarding")
                #     step_embds = self.get_embd(self.codi, self.model_name)(current_step_ids)
                #     with autocast(device_type='cuda',dtype=torch.bfloat16):
                #         step_out = self.codi(
                #             inputs_embeds=step_embds,
                #             use_cache=True,
                #             output_hidden_states=True,
                #             past_key_values=past_key_values
                #         )
                #     past_key_values = step_out.past_key_values
                #     last_hidden = step_out.hidden_states[-1][:, -1, :]
                #     current_logits = step_out.logits[:, -1, :]
                #     latent_embd = last_hidden.unsqueeze(1)
                #     if self.use_prj:
                #         latent_embd = self.prj(latent_embd)
                
                # ========== 显式模式结束：更新熵和模式 ==========
                cur_entropy = compute_entropy_swir(current_logits)
                entropy_history.append(cur_entropy.mean().item())
                old_mode = swir_state.mode.clone()
                swir_state, _, _ = get_mode_decision(swir_state, cur_entropy, step_i + 1)
                
                prev_was_latent = False

                if debug:
                    print(f"\n[DEBUG] === END OF EXPLICIT STEP {step_i} ===")
                    print(f"  Entropy: {cur_entropy.tolist()}")
                    print(f"  Mode changed: {old_mode.tolist()} -> {swir_state.mode.tolist()}")
            
            else:
                # ===== 隐式模式 (L) =====
                if debug:
                    print(f"\n[DEBUG] >>> LATENT MODE PROCESSING <<<")
                    print(f"  prev_was_latent: {prev_was_latent}")
                    print(f"  Pipeline: [BOT if first L] → LATENT → [EOT if last L]")
                
                # ===== 1. 只在Latent序列开始时输入<bot> =====
                if not prev_was_latent:
                    bot_ids = torch.full((batch_size, 1), self.bot_id, dtype=torch.long, device=device)
                    bot_embd = self.get_embd(self.codi, self.model_name)(bot_ids)
                    
                    if debug:
                        print(f"\n[DEBUG] Step 1: BOT Token (first L in sequence)")
                        print(f"  BOT token ID: {self.bot_id}")
                        print(f"  bot_embd shape: {bot_embd.shape}")
                    
                    with autocast(device_type='cuda',dtype=torch.bfloat16):
                        bot_out = self.codi(
                            inputs_embeds=bot_embd,
                            use_cache=True,
                            output_hidden_states=True,
                            past_key_values=past_key_values
                        )
                    past_key_values = bot_out.past_key_values
                    
                    if debug:
                        bot_logits = bot_out.logits[:, -1, :]
                        top_probs, top_ids = torch.softmax(bot_logits, dim=-1).topk(5, dim=-1)
                        for b in range(min(batch_size, 1)):
                            top_tokens = [self.tokenizer.decode([tid]) for tid in top_ids[b].tolist()]
                            print(f"  After BOT, Sample {b} top5: {list(zip(top_tokens, [f'{p:.4f}' for p in top_probs[b].tolist()]))}")
                else:
                    if debug:
                        print(f"\n[DEBUG] Skipping BOT (continuing Latent sequence)")
                
                # ===== 2. 输入latent embedding =====
                if debug:
                    print(f"\n[DEBUG] Step 2: LATENT Embedding")
                    print(f"  latent_embd shape: {latent_embd.shape}")
                    print(f"  latent_embd norm: {latent_embd.norm(dim=-1).mean().item():.4f}")
                
                with autocast(device_type='cuda',dtype=torch.bfloat16):
                    latent_out = self.codi(
                        inputs_embeds=latent_embd,
                        use_cache=True,
                        output_hidden_states=True,
                        past_key_values=past_key_values
                    )
                past_key_values = latent_out.past_key_values
                last_hidden = latent_out.hidden_states[-1][:, -1, :]
                latent_embd = last_hidden.unsqueeze(1)
                if self.use_prj:
                    latent_embd = self.prj(latent_embd)
                
                if debug:
                    latent_logits = latent_out.logits[:, -1, :]
                    top_probs, top_ids = torch.softmax(latent_logits, dim=-1).topk(5, dim=-1)
                    for b in range(min(batch_size, 1)):
                        top_tokens = [self.tokenizer.decode([tid]) for tid in top_ids[b].tolist()]
                        print(f"  After LATENT, Sample {b} top5: {list(zip(top_tokens, [f'{p:.4f}' for p in top_probs[b].tolist()]))}")
                    print(f"  Updated latent_embd shape: {latent_embd.shape}")


                # ===== 5. Explain Loss (改为对齐当前step) =====
                if debug:
                    print(f"\n[DEBUG] >>> EXPLAIN LOSS CALCULATION <<<")
                    
                if self.model_args.use_decoder:
                    # 改动1: 使用 current_step_ids 而不是 next_step_ids
                    current_step_len = current_step_ids.size(1)
                    if current_step_len > 0:
                        if debug:
                            print(f"\n  Input preparation:")
                            print(f"    current_step_ids shape: {current_step_ids.shape}")
                        
                        # 改动2: 用当前step的embedding
                        current_embds = self.get_embd(self.codi, self.model_name)(current_step_ids)
                        
                        # 改动3: decoder输入为 [latent, current_step]，不需要eot分隔符
                        decoder_input = torch.cat([latent_embd, current_embds], dim=1)
                        
                        if debug:
                            print(f"    decoder_input shape: {decoder_input.shape} = [LATENT(1) + current({current_step_len})]")
                        
                        # 改动4: prefix只有1个位置（latent），不是2个
                        prefix_placeholder = torch.full((batch_size, 1), -570, dtype=current_step_ids.dtype, device=device)
                        indices_with_prefix = torch.cat([prefix_placeholder, current_step_ids], dim=1)
                        
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
                                    print(f"  *** Explain Loss value: {explain_loss.item():.6f} ***")
                
                # ===== 3. 基于latent输出计算熵，更新模式 =====
                if debug:
                    print(f"\n[DEBUG] >>> CHECK MODE TRANSITION (based on latent output) <<<")
                
                current_logits = latent_out.logits[:, -1, :]  # 基于latent输出计算熵
                cur_entropy = compute_entropy_swir(current_logits)
                entropy_history.append(cur_entropy.mean().item())
                old_mode = swir_state.mode.clone()
                swir_state, _, _ = get_mode_decision(swir_state, cur_entropy, step_i + 1)
                new_mode = swir_state.mode[0].item()
                
                # 判断是否是Latent序列的最后一个
                is_last_latent = (new_mode == 1) or (step_i == num_steps - 1)
                
                if debug:
                    print(f"  Current entropy: {cur_entropy.tolist()}")
                    print(f"  Old mode: {old_mode.tolist()} (0=L, 1=S)")
                    print(f"  New mode: {swir_state.mode.tolist()} (0=L, 1=S)")
                    print(f"  step_i = {step_i}, num_steps = {num_steps}")
                    print(f"  ★ IS_LAST_LATENT = {is_last_latent}")
                
                # ===== 4. 只在Latent序列结束时输入<eot> =====
                eot_out = None  # 初始化，用于后续alignment loss
                if is_last_latent:
                    eot_ids = torch.full((batch_size, 1), self.eot_id, dtype=torch.long, device=device)
                    eot_embd = self.get_embd(self.codi, self.model_name)(eot_ids)
                    
                    if debug:
                        print(f"\n[DEBUG] Step 3: EOT Token (last L in sequence)")
                        print(f"  EOT token ID: {self.eot_id}")
                        print(f"  eot_embd shape: {eot_embd.shape}")
                    
                    with autocast(device_type='cuda',dtype=torch.bfloat16):
                        eot_out = self.codi(
                            inputs_embeds=eot_embd,
                            use_cache=True,
                            output_hidden_states=True,
                            past_key_values=past_key_values
                        )
                    past_key_values = eot_out.past_key_values
                    current_logits = eot_out.logits[:, -1, :]  # 更新logits



                    if debug:
                        top_probs, top_ids = torch.softmax(current_logits, dim=-1).topk(5, dim=-1)
                        for b in range(min(batch_size, 1)):
                            top_tokens = [self.tokenizer.decode([tid]) for tid in top_ids[b].tolist()]
                            print(f"  After EOT, Sample {b} top5: {list(zip(top_tokens, [f'{p:.4f}' for p in top_probs[b].tolist()]))}")
                    
                    
                    # ==== latent_exit_ce_loss补丁，eot后面需要学会预测"<<"
                    eot_logits = eot_out.logits[:, -1, :]
                    
                    if next_step_ids is not None:
                        # 有下一步：预测 
                        target_token = next_step_ids[:, 0].clone()
                    else:
                        # 最后一步：预测 decoder_input_ids 的第一个 token（比如 "The"）
                        target_token = decoder_input_ids[:, 0].clone()
                    
                    # 过滤 pad
                    target_token[target_token == model_pad_id] = -100
                    if self.tokenizer.pad_token_id is not None:
                        target_token[target_token == self.tokenizer.pad_token_id] = -100
                    
                    # if (target_token != -100).any():
                    #     latent_exit_ce_loss = self.loss_fct(eot_logits, target_token)
                    #     ce_loss_total = ce_loss_total + latent_exit_ce_loss
                    if (target_token != -100).any():
                        valid_exit = (target_token != -100).sum().item()
                        latent_exit_ce_loss = self.loss_fct_sum(eot_logits, target_token)
                        ce_loss_total = ce_loss_total + latent_exit_ce_loss
                        ce_token_count += valid_exit
                        if debug:
                            target_decoded = self.tokenizer.decode([target_token[0].item()]) if target_token[0] != -100 else "[masked]"
                            print(f"Latent_exit_ce_loss: {latent_exit_ce_loss.item():.6f}, target='{target_decoded}'")



                    # ===== Alignment Loss (只在最后一个Latent且有下一步时计算) =====
                    if debug:
                        print(f"\n[DEBUG] >>> ALIGNMENT LOSS CALCULATION <<<")
                    
                    if step_i + 1 < num_steps:
                        next_step_ref_pos = step_positions[:, step_i + 1]
                        valid_mask = (next_step_ref_pos >= 0) & (next_step_ref_pos < ref_input_ids.size(1))
                        
                        if debug:
                            print(f"  Target: Align EOT output hidden state with teacher's hidden state at next step start")
                            print(f"  next_step_ref_pos: {next_step_ref_pos.tolist()}")
                            print(f"  valid_mask: {valid_mask.tolist()}")
                        
                        if valid_mask.any():
                            align_loss = torch.tensor(0.0, device=device)
                            layer_losses = []
                            
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
                                    layer_losses.append(layer_loss.item())
                            
                            align_loss = align_loss / len(eot_out.hidden_states)
                            align_loss_total = align_loss_total + align_loss
                            alignment_computed_at_steps.append(step_i)
                            
                            if debug:
                                print(f"  ★ Alignment Loss value: {align_loss.item():.6f}")
                        else:
                            if debug:
                                print(f"  *** No valid positions for alignment, skipping ***")
                    else:
                        if debug:
                            print(f"  *** Last step, no next step to align to ***")
                else:
                    if debug:
                        print(f"\n[DEBUG] Skipping EOT (Latent sequence continues)")
                

                # ===== 设置标记：当前步是Latent =====
                prev_was_latent = True
                
                if debug:
                    print(f"\n[DEBUG] === END OF LATENT STEP {step_i} ===")
        
        # ========== 7. 预测答案 ==========
        if debug:
            print("\n" + "=" * 100)
            print("[DEBUG] " + "=" * 40 + " ANSWER PREDICTION " + "=" * 40)
            print("=" * 100)
            print(f"\n[DEBUG] >>> ANSWER GENERATION <<<")
            print(f"  decoder_input_ids shape: {decoder_input_ids.shape}")
            for b in range(min(batch_size, 2)):
                dec_text = self.tokenizer.decode(decoder_input_ids[b], skip_special_tokens=False)
                print(f"  Sample {b} decoder_input: {dec_text[:100]}...")
        

        #  新增：答案入口loss 
        # 如果最后一个step是latent，已在latent_exit_ce_loss中处理（next_step_ids为None时）
        # 如果最后一个step是显式，需要额外计算
        if not prev_was_latent:
            ans_first_token = decoder_input_ids[:, 0].clone()
            ans_first_token[ans_first_token == model_pad_id] = -100
            if self.tokenizer.pad_token_id is not None:
                ans_first_token[ans_first_token == self.tokenizer.pad_token_id] = -100
            
            # if (ans_first_token != -100).any():
            #     ans_entry_ce_loss = self.loss_fct(current_logits, ans_first_token)
            #     ce_loss_total = ce_loss_total + ans_entry_ce_loss
            if (ans_first_token != -100).any():
                valid_ans_entry = (ans_first_token != -100).sum().item()
                ans_entry_ce_loss = self.loss_fct_sum(current_logits, ans_first_token)
                ce_loss_total = ce_loss_total + ans_entry_ce_loss
                ce_token_count += valid_ans_entry
                if debug:
                    target_decoded = self.tokenizer.decode([ans_first_token[0].item()]) if ans_first_token[0] != -100 else "[masked]"
                    print(f"  ★ Answer Entry CE Loss: {ans_entry_ce_loss.item():.6f}, target='{target_decoded}'")

        decoder_embds = self.get_embd(self.codi, self.model_name)(decoder_input_ids)
        
        with autocast(device_type='cuda',dtype=torch.bfloat16):
            final_out = self.codi(
                inputs_embeds=decoder_embds,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
            )
        
        if debug:
            print(f"  final_out.logits shape: {final_out.logits.shape}")
        
        # ===== Answer CE Loss =====
        if debug:
            print(f"\n[DEBUG] >>> ANSWER CE LOSS <<<")
            
        ans_logits = final_out.logits[:, :-1, :]
        ans_labels = labels[:, 1:]
        
        if debug:
            print(f"  ans_logits shape: {ans_logits.shape}")
            print(f"  ans_labels shape: {ans_labels.shape}")
            print(f"  ans_labels (sample 0, first 15): {ans_labels[0, :15].tolist()}")
            valid_ans_labels = (ans_labels != -100).sum().item()
            print(f"  valid answer labels: {valid_ans_labels}")
        
        # ans_ce_loss = self.loss_fct(
        #     ans_logits.reshape(-1, ans_logits.size(-1)),
        #     ans_labels.reshape(-1)
        # )
        # ce_loss_total = ce_loss_total + ans_ce_loss
        # ce_loss_total = ce_loss_total * self.ce_loss_factor
        # effective_ce_steps += 1

        valid_ans_tokens = (ans_labels != -100).sum().item()
        ans_ce_loss = self.loss_fct_sum(
            ans_logits.reshape(-1, ans_logits.size(-1)),
            ans_labels.reshape(-1)
        )
        ce_loss_total = ce_loss_total + ans_ce_loss
        ce_token_count += valid_ans_tokens

        # 归一化：除以总token数
        if ce_token_count > 0:
            ce_loss_total = ce_loss_total / ce_token_count

        ce_loss_total = ce_loss_total * self.ce_loss_factor
        effective_ce_steps += 1
        
        if debug:
            print(f"\n  *** Answer CE Loss = CrossEntropyLoss(ans_logits, ans_labels)")
            print(f"  *** Answer CE Loss value: {ans_ce_loss.item():.6f} ***")
            print(f"  Total effective CE steps: {effective_ce_steps}")
            print(f"  Total effective explain steps: {effective_explain_steps}")
        
        # ========== 8. Distillation Loss ==========

        if debug:
            print(f"\\n[DEBUG] >>> DISTILLATION LOSS (Final Answer) <<<")
            
        if "llama" in self.model_name.lower() or "qwen" in self.model_name.lower():
            model_answer_position = model_answer_position + 1
            ref_answer_position = ref_answer_position + 1
        model_answer_position = model_answer_position - 1
        ref_answer_position = ref_answer_position - 1
        
        if debug:
            print(f"  model_answer_position: {model_answer_position.tolist()}")
            print(f"  ref_answer_position: {ref_answer_position.tolist()}")
            print(f"  Target: Align model's final hidden state with teacher's hidden state at answer position")
            
            # ===== 增强Debug: 打印教师对齐点及附近tokens =====
            print(f"\\n  [TEACHER ALIGNMENT POINT TOKENS]:")
            for b in range(min(batch_size, 2)):
                ref_pos = ref_answer_position[b].item()
                if ref_pos >= 0 and ref_pos < ref_input_ids.size(1):
                    # 对齐点token
                    target_token_id = ref_input_ids[b, ref_pos].item()
                    target_token_decoded = self.tokenizer.decode([target_token_id])
                    
                    # 扩大上下文窗口: 前10个token到后5个token
                    ctx_start = max(0, ref_pos - 10)
                    ctx_end = min(ref_input_ids.size(1), ref_pos + 6)
                    ctx_ids = ref_input_ids[b, ctx_start:ctx_end].tolist()
                    ctx_decoded = self.tokenizer.decode(ctx_ids, skip_special_tokens=False)
                    
                    # 逐token打印上下文
                    print(f"    Sample {b}:")
                    print(f"      ★ Alignment Point: pos={ref_pos}, token_id={target_token_id}, token='{target_token_decoded}'")
                    print(f"      Context tokens [{ctx_start}:{ctx_end}]:")
                    for i, tok_id in enumerate(ctx_ids):
                        actual_pos = ctx_start + i
                        tok_str = self.tokenizer.decode([tok_id])
                        marker = ">>>>" if actual_pos == ref_pos else "    "
                        print(f"        {marker} pos={actual_pos}: id={tok_id:6d}, token='{tok_str}'")
                else:
                    print(f"    Sample {b}: ref_pos={ref_pos} (INVALID or out of range)")
            
            # ===== 增强Debug: 打印学生模型对齐点信息 =====
            print(f"\\n  [STUDENT ALIGNMENT POINT INFO]:")
            print(f"    Student uses final hidden state (position -1 in final_out)")
            print(f"    final_out.logits shape: {final_out.logits.shape}")
            print(f"    final_out.hidden_states[-1] shape: {final_out.hidden_states[-1].shape}")
            
            # 学生模型最后位置的预测
            student_last_logits = final_out.logits[:, -1, :]
            student_top_probs, student_top_ids = torch.softmax(student_last_logits, dim=-1).topk(5, dim=-1)
            for b in range(min(batch_size, 2)):
                top_tokens = [self.tokenizer.decode([tid]) for tid in student_top_ids[b].tolist()]
                print(f"    Sample {b} student top5 predictions: {list(zip(top_tokens, [f'{p:.4f}' for p in student_top_probs[b].tolist()]))}")
        
        # ===== 计算Distill Loss =====
        layer_distill_losses = []
        layer_hidden_stats = []  # 记录每层的hidden state统计信息
        
        for layer_idx, (out_h, ref_h) in enumerate(zip(final_out.hidden_states, ref_outputs.hidden_states)):
            ref_sel = ref_h.gather(1, ref_answer_position.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, ref_h.size(-1)))
            out_sel = out_h[:, -1:, :]
            d_loss = self.distill_loss_fct(out_sel, ref_sel.detach())
            
            # 记录统计信息（用于debug）
            if debug:
                layer_hidden_stats.append({
                    'layer': layer_idx,
                    'student_norm': out_sel.norm(dim=-1).mean().item(),
                    'teacher_norm': ref_sel.norm(dim=-1).mean().item(),
                    'cosine_sim': F.cosine_similarity(out_sel.squeeze(1), ref_sel.squeeze(1), dim=-1).mean().item(),
                    'l2_dist': (out_sel - ref_sel).pow(2).sum(dim=-1).sqrt().mean().item(),
                    'loss_before_std': d_loss.item(),
                })
            
            if self.distill_loss_div_std:
                teacher_std = ref_sel.std()
                d_loss = d_loss / teacher_std.clamp(min=1e-6)
                if debug and layer_idx < 5:
                    layer_hidden_stats[-1]['teacher_std'] = teacher_std.item()
                    layer_hidden_stats[-1]['loss_after_std'] = d_loss.item()
            
            distill_loss_total = distill_loss_total + d_loss
            layer_distill_losses.append(d_loss.item())
            
        distill_loss_total = distill_loss_total / len(final_out.hidden_states)
        
        # ===== 增强Debug: 打印详细的层级Loss信息 =====
        if debug:
            print(f"\\n  [LAYER-WISE DISTILL LOSS DETAILS]:")
            print(f"  Total layers: {len(final_out.hidden_states)}")
            print(f"  distill_loss_div_std: {self.distill_loss_div_std}")
            print(f"  distill_loss_type: {self.distill_loss_type}")
            
            print(f"\\n  === First 5 Layers (Embedding + Early Layers) ===")
            for i in range(min(5, len(layer_hidden_stats))):
                stats = layer_hidden_stats[i]
                print(f"    Layer {i}:")
                print(f"      Student norm: {stats['student_norm']:.4f}, Teacher norm: {stats['teacher_norm']:.4f}")
                print(f"      Cosine similarity: {stats['cosine_sim']:.4f}")
                print(f"      L2 distance: {stats['l2_dist']:.4f}")
                print(f"      Loss (before std): {stats['loss_before_std']:.6f}")
                if 'loss_after_std' in stats:
                    print(f"      Teacher std: {stats['teacher_std']:.4f}, Loss (after std): {stats['loss_after_std']:.6f}")
            
            print(f"\\n  === Middle Layers Summary ===")
            mid_start = 5
            mid_end = len(layer_hidden_stats) - 5
            if mid_end > mid_start:
                mid_losses = [layer_hidden_stats[i]['loss_before_std'] for i in range(mid_start, mid_end)]
                mid_cosines = [layer_hidden_stats[i]['cosine_sim'] for i in range(mid_start, mid_end)]
                print(f"    Layers {mid_start} to {mid_end-1}:")
                print(f"      Loss range: [{min(mid_losses):.6f}, {max(mid_losses):.6f}], mean: {sum(mid_losses)/len(mid_losses):.6f}")
                print(f"      Cosine sim range: [{min(mid_cosines):.4f}, {max(mid_cosines):.4f}], mean: {sum(mid_cosines)/len(mid_cosines):.4f}")
            
            print(f"\\n  === Last 5 Layers ===")
            for i in range(max(0, len(layer_hidden_stats) - 5), len(layer_hidden_stats)):
                stats = layer_hidden_stats[i]
                print(f"    Layer {i}:")
                print(f"      Student norm: {stats['student_norm']:.4f}, Teacher norm: {stats['teacher_norm']:.4f}")
                print(f"      Cosine similarity: {stats['cosine_sim']:.4f}")
                print(f"      L2 distance: {stats['l2_dist']:.4f}")
                print(f"      Loss (before std): {stats['loss_before_std']:.6f}")
                if 'loss_after_std' in stats:
                    print(f"      Teacher std: {stats['teacher_std']:.4f}, Loss (after std): {stats['loss_after_std']:.6f}")
            
            print(f"\\n  === Aggregate Statistics ===")
            all_losses = layer_distill_losses
            print(f"    Total distill loss (before factor): {distill_loss_total.item():.6f}")
            print(f"    Layer losses - min: {min(all_losses):.6f}, max: {max(all_losses):.6f}, mean: {sum(all_losses)/len(all_losses):.6f}")
            
            # 检查是否有异常层
            mean_loss = sum(all_losses) / len(all_losses)
            std_loss = (sum((x - mean_loss)**2 for x in all_losses) / len(all_losses)) ** 0.5
            anomaly_layers = [i for i, l in enumerate(all_losses) if abs(l - mean_loss) > 2 * std_loss]
            if anomaly_layers:
                print(f"    ⚠️ Anomaly layers (>2 std from mean): {anomaly_layers}")
                for al in anomaly_layers[:3]:  # 只打印前3个异常层
                    print(f"      Layer {al}: loss={all_losses[al]:.6f} (mean={mean_loss:.6f}, std={std_loss:.6f})")

        
        # ========== 9. Reference CE Loss ==========
        if debug:
            print(f"\n[DEBUG] >>> REFERENCE CE LOSS <<<")
            print(f"  Target: Teacher model's CE loss on full CoT sequence")
            
        ref_logits = ref_outputs_with_grad.logits[:, :-1, :]
        ref_labels_shifted = ref_labels[:, 1:]
        
        if debug:
            print(f"  ref_logits shape: {ref_logits.shape}")
            print(f"  ref_labels_shifted shape: {ref_labels_shifted.shape}")
            valid_ref_labels = (ref_labels_shifted != -100).sum().item()
            print(f"  valid reference labels: {valid_ref_labels}")
        
        ref_ce_loss = self.loss_fct(
            ref_logits.reshape(-1, ref_logits.size(-1)),
            ref_labels_shifted.reshape(-1)
        )
        ref_ce_loss = ref_ce_loss * self.ref_loss_factor
        
        if debug:
            print(f"\n  *** Reference CE Loss = CrossEntropyLoss(ref_logits, ref_labels) * {self.ref_loss_factor}")
            print(f"  *** Reference CE Loss value: {ref_ce_loss.item():.6f} ***")
        
        # ========== 10. 汇总Loss ==========
        if debug:
            print("\n" + "=" * 100)
            print("[DEBUG] " + "=" * 40 + " LOSS SUMMARY " + "=" * 40)
            print("=" * 100)
        
        distill_loss_total = distill_loss_total * self.distill_loss_factor
        explain_loss_total = explain_loss_total * self.explain_loss_factor
        if effective_explain_steps > 0:
            explain_loss_total = explain_loss_total / effective_explain_steps
        align_loss_total = align_loss_total * self.align_loss_factor
        
        total_loss = ce_loss_total + distill_loss_total + ref_ce_loss + explain_loss_total + align_loss_total
        
        if debug:
            print(f"\n[DEBUG] >>> LOSS FACTORS <<<")
            print(f"  distill_loss_factor: {self.distill_loss_factor}")
            print(f"  explain_loss_factor: {self.explain_loss_factor}")
            print(f"  align_loss_factor: {self.align_loss_factor}")
            print(f"  ref_loss_factor: {self.ref_loss_factor}")
            
            print(f"\n[DEBUG] >>> FINAL LOSS VALUES <<<")
            print(f"  CE Loss (total):        {ce_loss_total.item():.6f}")
            print(f"  Distill Loss (scaled):  {distill_loss_total.item():.6f}")
            print(f"  Reference CE Loss:      {ref_ce_loss.item():.6f}")
            print(f"  Explain Loss (scaled):  {explain_loss_total.item() if isinstance(explain_loss_total, torch.Tensor) else explain_loss_total:.6f}")
            print(f"  Align Loss (scaled):    {align_loss_total.item():.6f}")
            print(f"  -" * 40)
            print(f"  TOTAL LOSS:             {total_loss.item():.6f}")
            
            print(f"\n[DEBUG] >>> MODE HISTORY <<<")
            print(f"  Modes: {''.join(mode_history)}")
            print(f"  Entropy history: {[f'{e:.4f}' for e in entropy_history]}")
            
            # ========== [修改点5] 打印alignment loss计算位置 ==========
            print(f"\n[DEBUG] >>> ALIGNMENT LOSS COMPUTATION SUMMARY <<<")
            print(f"  ★ Alignment computed at steps: {alignment_computed_at_steps}")
            if alignment_computed_at_steps:
                print(f"  ★ These correspond to modes: {[mode_history[i] for i in alignment_computed_at_steps]}")
                print(f"  ★ Total alignment losses computed: {len(alignment_computed_at_steps)}")
            else:
                print(f"  ★ No alignment loss computed (no L→S transitions)")
        
        mode_str = ''.join(mode_history)
        if self.print_loss:
            align_info = f"(at steps {alignment_computed_at_steps})" if alignment_computed_at_steps else "(none)"
            print(f"Modes: {mode_str} | Loss={total_loss.item():.4f} | "
                f"CE={ce_loss_total.item():.4f} | Distill={distill_loss_total.item():.4f} | "
                f"RefCE={ref_ce_loss.item():.4f} | Explain={explain_loss_total.item() if isinstance(explain_loss_total, torch.Tensor) else explain_loss_total:.4f} | "
                f"Align={align_loss_total.item():.4f} {align_info}")
        
        if debug:
            print("\n" + "=" * 100)
            print("[DEBUG] " + "=" * 42 + " FORWARD END " + "=" * 42)
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
            "alignment_computed_at_steps": alignment_computed_at_steps,  # [修改点6] 返回alignment计算位置
        }

