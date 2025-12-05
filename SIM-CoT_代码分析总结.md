# SIM-CoT 项目详细分析

## 1. 项目概述

**SIM-CoT (Supervised Implicit Chain-of-Thought)** 是一个针对隐式思维链(Implicit CoT)方法的改进项目。该项目在两个主流的隐式CoT框架上进行了改进：
- **Coconut** (Meta)
- **CODI** (另一种隐式CoT方法)

### 核心问题
原始的隐式CoT方法存在**latent instability（隐状态不稳定性）**问题：当隐式token数量增加时，模型倾向于collapse到同质化的隐状态，丢失运算符语义。

### 核心改进
SIM-CoT引入了**步级监督(Step-Level Supervision)**，通过添加一个**辅助解码器(Auxiliary Decoder)**来稳定优化过程，防止collapse，确保隐式token能捕获有意义的推理步骤。

---

## 2. 核心改进文件和代码详解

### 2.1 Coconut 部分改进

#### 核心文件: `Coconut/coconut.py`

**原始Coconut类 (第14-263行)**: 基础隐式思维链实现，保持不变

**新增SIM-CoT类: `CoconutGPT_Same_Word_Embedding` (第265-856行)**

关键改进点：

```python
class CoconutGPT_Same_Word_Embedding(nn.Module):
    def __init__(
        self,
        base_causallm,         # 主模型（如GPT2/LLaMA）
        expainable_llm,        # 辅助解码器（用于解释隐式token）
        tokenizer,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
        step_start_id,
        c_thought,             # 每个推理步骤的隐式token数量
        configs,
    ):
```

**核心forward逻辑**:

1. **基础前向传播** (第341-470行): 与原始Coconut相同，处理隐式token的多轮前向传播

2. **SIM-CoT改进 - 解释损失计算** (第515-786行):
   - 提取每个latent token对应的hidden state
   - 将hidden state输入到辅助解码器
   - 计算解释损失(explain_loss)，监督latent token生成人类可读的推理步骤

```python
# 关键代码片段 (约第695-712行)
explainable_outputs = self.expainable_llm(
    inputs_embeds=input_explain_input_embeds_batch_tensor,
    attention_mask=input_explain_attention_mask_batch_tensor,
    position_ids=input_explain_position_ids_batch_tensor.to(torch.long),
    output_hidden_states=True,
)

# 计算解释损失
loss_explain = loss_explain_fct(
    shift_explain_logits.view(-1, shift_explain_logits.size(-1)), 
    shift_explain_labels.view(-1)
)

# 最终损失 = 原始损失 + 解释损失
loss += 1.0 * loss_explain_all / c_thought_num
```

### 2.2 CODI 部分改进

#### 核心文件: `CODI/src/model.py`

**CODI类** (第272-766行)

关键改进点：

1. **添加解码器支持** (第305-334行):
```python
if model_args.use_decoder:
    if model_args.decoder_path:
        self.decoder = model_wrapper_class.from_pretrained(model_args.decoder_path, ...)
        self.pj_in = nn.Linear(...)  # 输入投影
        self.pj_out = LowRankProjector(...)  # 输出投影
    else:
        self.decoder = model_wrapper_class.from_pretrained(self.model_name, ...)
```

2. **投影层** (第364-376行):
```python
if training_args.use_prj:
    self.prj = nn.Sequential(
        nn.Dropout(training_args.prj_dropout),
        nn.Linear(self.dim, training_args.prj_dim),
        nn.GELU(),
        nn.Linear(training_args.prj_dim, self.dim),
    )
```

3. **解释损失计算** (第611-667行):
```python
if self.model_args.use_decoder:
    # 将latent embedding与步骤标签拼接
    explain_embds = torch.concat([latent_embd, explain_embds], dim=1)
    
    # 通过辅助解码器
    explain_outputs = self.decoder(
        inputs_embeds=explain_embds,
        attention_mask=explain_attention_mask,
        output_hidden_states=True
    )
    
    # 计算解释损失
    explain_loss = self.loss_fct(shift_explain_logits, shift_explain_labels)
```

4. **最终损失组合** (第744-748行):
```python
loss = ce_loss_total + distill_loss_total + ref_ce_loss
if self.model_args.use_decoder:
    loss += explain_loss_total
```

---

## 3. 运行Baseline的方法

### 3.1 GPT2 模型

#### 训练CODI Baseline:
```bash
cd CODI
python train.py \
    --model_name_or_path gpt2 \
    --data_name icot \
    --num_latent 6 \
    --use_prj True \
    --prj_dim 768 \
    --num_train_epochs 40 \
    --learning_rate 3e-3 \
    --use_lora True \
    --lora_r 128 --lora_alpha 32
```

#### 训练GPT2 + SIM-CoT (CODI框架):
```bash
cd CODI
python train.py \
    --model_name_or_path gpt2 \
    --data_name icot \
    --num_latent 6 \
    --use_prj True \
    --prj_dim 768 \
    --use_decoder True  # 关键参数：启用辅助解码器
```

#### 测试GPT2:
```bash
python test.py \
    --data_name "gsm8k" \
    --model_name_or_path gpt2 \
    --num_latent 6 \
    --use_prj True \
    --prj_dim 768 \
    --ckpt_dir <你的checkpoint路径>
```

### 3.2 LLaMA3 模型

#### 训练CODI Baseline (LLaMA 3B):
```bash
cd CODI
python train.py \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --data_name icot \
    --num_latent 6 \
    --use_prj True \
    --prj_dim 3072 \
    --num_train_epochs 8 \
    --learning_rate 3e-4 \
    --use_lora True \
    --lora_r 128 --lora_alpha 32
```

#### 训练LLaMA3 + SIM-CoT (CODI框架):
```bash
# 添加 --use_decoder True
python train.py \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --data_name icot \
    --num_latent 6 \
    --use_prj True \
    --prj_dim 3072 \
    --use_decoder True  # 关键参数
```

### 3.3 Coconut框架

#### 训练GPT2 Coconut Baseline:
```bash
cd Coconut
torchrun --nnodes 1 --nproc_per_node 8 run.py args/gsm_coconut.yaml
```

#### 训练GPT2 + SIM-CoT (Coconut框架):
```bash
# 先训练Coconut baseline，然后继续训练SIM-CoT
torchrun --nnodes 1 --nproc_per_node 8 run.py args/gsm_simcot.yaml
```

#### 评估:
```bash
torchrun --nnodes 1 --nproc_per_node 8 run.py args/gsm_simcot_eval.yaml
```

---

## 4. 添加新数据集

### 4.1 CODI框架添加数据集

**修改文件**: `CODI/train.py`

在`make_supervised_data_module`函数中添加新数据集支持:

```python
# 在train.py约第396-424行
def make_supervised_data_module(tokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    logging.warning("Downloading Data")
    if "icot" in data_args.data_name:
        dataset = None  # GSM8k-Aug
        train_dataset = SupervisedDataset(...)
    elif "strategy" in data_args.data_name:
        dataset = load_dataset("zen-E/StrategyQA_CoT_GPT4o")["train"]
        train_dataset = SupervisedDataset(...)
    elif "commonsense" in data_args.data_name:
        dataset = load_dataset("zen-E/CommonsenseQA-GPT4omini")["train"]
        train_dataset = SupervisedDataset(...)
    
    # === 添加你的新数据集 ===
    elif "your_dataset_name" in data_args.data_name:
        # 方式1: 从HuggingFace加载
        dataset = load_dataset("your/dataset_path")["train"]
        
        # 方式2: 从本地JSON加载
        with open("path/to/your/data.json") as f:
            dataset = json.load(f)
        
        train_dataset = SupervisedDataset(
            data_name=data_args.data_name, 
            raw_data=dataset, 
            tokenizer=tokenizer, 
            bot=model.bot_id, 
            eot=model.eot_id
        )
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
```

**数据格式要求**（在SupervisedDataset类中约第270-346行）:
```python
# 你的数据需要包含以下字段:
{
    "question": "问题文本",
    "cot": "思维链推理过程",  # 用". "分隔各步骤
    "answer": "最终答案"
}
```

### 4.2 Coconut框架添加数据集

**修改文件**: `Coconut/dataset.py`

添加新的数据集加载函数，参考现有的`get_cot_with_explainable_latent_dataset`:

```python
def get_your_dataset(path, tokenizer, latent_id, start_id, end_id, configs, max_size=1000000000):
    """
    加载你的自定义数据集
    """
    def tokenize_sample(sample):
        question_tokenized = tokenizer.encode(
            sample["question"] + "\n", add_special_tokens=True
        )
        steps_tokenized = [
            tokenizer.encode(s + "\n", add_special_tokens=False)
            for s in sample["steps"]  # 思维链步骤
        ]
        answer_tokenized = tokenizer.encode(
            "### " + sample["answer"], add_special_tokens=False
        ) + [tokenizer.eos_token_id]
        
        return {
            "question_tokenized": question_tokenized,
            "steps_tokenized": steps_tokenized,
            "answer_tokenized": answer_tokenized,
            "idx": sample["idx"],
        }
    
    # 加载数据
    data = json.load(open(path))[:max_size]
    data = [{**d, "idx": idx} for idx, d in enumerate(data)]
    
    # 处理数据集...
    return dataset
```

**然后在`run.py`中调用**:
```python
# 在run.py中修改数据集选择逻辑
if configs.dataset == "your_dataset":
    train_data = get_your_dataset(
        configs.train_path, tokenizer, latent_id, start_id, end_id, configs
    )
```

---

## 5. 在Hidden State添加MLP层估计熵

### 5.1 CODI框架实现

**修改文件**: `CODI/src/model.py`

**步骤1**: 在CODI类的`__init__`中添加熵估计MLP:

```python
class CODI(torch.nn.Module):
    def __init__(self, model_args, training_args, lora_config):
        super().__init__()
        # ... 现有代码 ...
        
        # === 添加熵估计MLP ===
        self.dim = self.codi.config.hidden_size
        self.entropy_mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim // 2),
            nn.GELU(),
            nn.Linear(self.dim // 2, 1),  # 输出一个标量表示熵
            nn.Softplus()  # 确保熵为正值
        )
        
        # 可选：添加熵损失的权重
        self.entropy_loss_factor = training_args.entropy_loss_factor if hasattr(training_args, 'entropy_loss_factor') else 0.1
```

**步骤2**: 在`forward`方法中使用熵估计:

```python
def forward(self, ...):
    # ... 现有代码 ...
    
    # 在latent embedding生成后（约第600行后）
    entropy_loss_total = 0.0
    
    if self.num_latent != 0:
        for i in range(num_latent):
            # 生成latent embedding
            outputs = self.codi(inputs_embeds=latent_embd, ...)
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            
            # === 估计熵 ===
            estimated_entropy = self.entropy_mlp(latent_embd)  # [batch, 1, 1]
            
            # 计算熵损失（可以根据需求设计）
            # 方式1: 最小化熵（鼓励确定性表示）
            entropy_loss = estimated_entropy.mean()
            
            # 方式2: 让熵接近目标值（平衡探索与利用）
            # target_entropy = 1.0
            # entropy_loss = F.mse_loss(estimated_entropy, torch.ones_like(estimated_entropy) * target_entropy)
            
            entropy_loss_total += entropy_loss
            
            if self.use_prj:
                latent_embd = self.prj(latent_embd)
    
    # 在最终损失中加入熵损失
    loss = ce_loss_total + distill_loss_total + ref_ce_loss
    if self.model_args.use_decoder:
        loss += explain_loss_total
    
    # === 添加熵损失 ===
    loss += self.entropy_loss_factor * entropy_loss_total
    
    return {"loss": loss, ..., "entropy_loss": entropy_loss_total.detach()}
```

**步骤3**: 在`TrainingArguments`中添加配置（约第78行）:

```python
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # ... 现有参数 ...
    entropy_loss_factor: float = field(default=0.1, metadata={"help": "熵损失的权重系数"})
    use_entropy_estimation: bool = field(default=False, metadata={"help": "是否启用熵估计"})
```

### 5.2 Coconut框架实现

**修改文件**: `Coconut/coconut.py`

在`CoconutGPT_Same_Word_Embedding`类中添加:

```python
class CoconutGPT_Same_Word_Embedding(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # ... 现有代码 ...
        
        # === 添加熵估计MLP ===
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            hidden_dim = self.base_causallm.config.n_embd
        else:
            hidden_dim = self.base_causallm.config.hidden_size
            
        self.entropy_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )
    
    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):
        # ... 现有代码直到hidden_states获取 ...
        
        entropy_loss_total = 0.0
        
        for pass_idx in range(max_n_latents):
            # ... 现有前向传播代码 ...
            
            hidden_states = outputs.hidden_states[-1]
            
            # === 估计熵 ===
            # 对每个latent token位置的hidden state估计熵
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair
                h = hidden_states[batch_idx, token_idx - 1 - hidden_states_offset, :]
                estimated_entropy = self.entropy_mlp(h.unsqueeze(0))
                entropy_loss_total += estimated_entropy.squeeze()
        
        # 归一化熵损失
        if max_n_latents > 0:
            entropy_loss_total = entropy_loss_total / (max_n_latents * len(filling_indices))
        
        # 添加到总损失
        loss = loss + 0.1 * entropy_loss_total
        
        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)
```

### 5.3 更高级的熵估计方法

如果你想要估计真实的信息熵，可以使用以下方法:

```python
class EntropyEstimator(nn.Module):
    """
    基于hidden state估计token分布的熵
    """
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, hidden_states):
        # hidden_states: [batch, seq_len, hidden_dim]
        logits = self.proj(hidden_states)
        probs = F.softmax(logits, dim=-1)
        
        # 计算信息熵: H = -sum(p * log(p))
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)  # [batch, seq_len]
        
        return entropy
```

---

## 6. 配置文件说明

### Coconut配置 (`args/gsm_simcot.yaml`)

```yaml
# 关键SIM-CoT参数
mode: coconutgpt_same_word_embedding  # 使用SIM-CoT模式
explain_mode: v1_aug                  # 解释模式
training_method: full                 # 完整训练（主模型+辅助解码器）
c_thought: 2                          # 每个推理步骤的隐式token数
max_latent_stage: 5                   # 最大隐式阶段数
```

### CODI配置（通过命令行参数）

```bash
--use_decoder True        # 启用辅助解码器（SIM-CoT核心）
--num_latent 6           # 隐式token数量
--use_prj True           # 使用投影层
--prj_dim 768            # 投影层维度（GPT2用768，LLaMA用3072）
--explain_loss_factor 1.0 # 解释损失权重
```

---

## 7. 总结

| 组件 | 文件 | 作用 |
|------|------|------|
| SIM-CoT核心类(Coconut) | `Coconut/coconut.py` | `CoconutGPT_Same_Word_Embedding`类实现步级监督 |
| SIM-CoT核心类(CODI) | `CODI/src/model.py` | `CODI`类的`use_decoder`参数启用辅助解码器 |
| 训练入口(Coconut) | `Coconut/run.py` | 分布式训练脚本 |
| 训练入口(CODI) | `CODI/train.py` | 使用HuggingFace Trainer |
| 测试脚本 | `CODI/test.py` | 评估模型精度 |
| 数据处理 | `Coconut/dataset.py`, `CODI/train.py` | 数据集加载和预处理 |
| 配置文件 | `Coconut/args/*.yaml` | Coconut框架配置 |

**关键开关参数**:
- `--use_decoder True` (CODI): 启用SIM-CoT
- `mode: coconutgpt_same_word_embedding` (Coconut): 启用SIM-CoT
- `explain_mode: v1_aug` (Coconut): 使用增强版解释模式
