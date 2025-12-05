# Entropy Predictor 集成指南 (Wrapper 形式)

## 概述

采用 **Wrapper 模式**，完全不修改原有的 Coconut/CODI 代码，通过包装类实现功能扩展。

**优点**：
- ✅ 原有代码完全不受影响
- ✅ 可以包装任意社区预训练模型
- ✅ EntropyPredictor 单独保存/加载
- ✅ 灵活组合不同模型

---

## 文件结构

```
SIM-CoT-main/
├── entropy_predictor.py          # 新增：独立模块（包含 Wrapper 类）
├── Coconut/
│   ├── coconut.py               # 不修改
│   └── run.py                   # 少量修改：使用 Wrapper
└── CODI/
    ├── src/model.py             # 不修改
    └── train.py                 # 少量修改：使用 Wrapper
```

---

## 核心类

| 类名 | 说明 |
|-----|------|
| `EntropyPredictor` | 独立的 MLP 熵预测模块 |
| `CoconutWithEntropyPredictor` | Coconut 的 Wrapper |
| `CODIWithEntropyPredictor` | CODI 的 Wrapper |
| `AdaptiveThinkingController` | 推理时的思考模式控制器 |

---

## 使用方式

### 1. 训练阶段

```python
from coconut import Coconut
from entropy_predictor import CoconutWithEntropyPredictor

# 1. 正常创建 Coconut 模型
coconut = Coconut(base_model, latent_id, start_id, end_id, eos_id)

# 2. 用 Wrapper 包装
wrapped = CoconutWithEntropyPredictor(
    coconut_model=coconut,
    entropy_loss_weight=0.1,  # 熵损失权重
    predictor_config={"num_layers": 3}
)

# 3. 训练循环 (与原来一样)
for batch in dataloader:
    outputs = wrapped(**batch)
    loss = outputs.loss  # 已自动包含 entropy_loss
    loss.backward()
    optimizer.step()

# 4. 分别保存
wrapped.save_all(
    base_path="checkpoints/coconut.pt",
    predictor_path="checkpoints/entropy_predictor.pt"
)
```

### 2. 推理阶段 (使用社区模型)

```python
from transformers import AutoModelForCausalLM
from coconut import Coconut
from entropy_predictor import CoconutWithEntropyPredictor, EntropyPredictor

# 1. 下载社区预训练模型
base_model = AutoModelForCausalLM.from_pretrained("community/coconut-gpt2")
coconut = Coconut(base_model, latent_id, start_id, end_id, eos_id)

# 2. 加载自己训练的 entropy predictor
predictor = EntropyPredictor.load("checkpoints/entropy_predictor.pt")

# 3. 组合
wrapped = CoconutWithEntropyPredictor(coconut, entropy_predictor=predictor)

# 4. 自适应生成
outputs, mode, entropy = wrapped.generate_adaptive(
    input_ids,
    attention_mask,
    entropy_threshold=0.5  # 阈值
)
print(f"Used {mode} thinking (entropy={entropy:.3f})")
```

---

## 修改 run.py (Coconut)

只需要在 `run.py` 中做少量修改：

### 修改位置 1: 导入

```python
# 在文件开头添加
from entropy_predictor import CoconutWithEntropyPredictor
```

### 修改位置 2: 模型包装 (约第 183-190 行)

```python
if configs.coconut:
    if configs.mode == 'coconutgpt_same_word_embedding':
        model = CoconutGPT_Same_Word_Embedding(...)
    elif configs.mode == 'coconut_baseline':
        model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)
    
    # 新增：如果启用 entropy predictor，用 Wrapper 包装
    if getattr(configs, 'use_entropy_predictor', False):
        model = CoconutWithEntropyPredictor(
            coconut_model=model,
            entropy_loss_weight=getattr(configs, 'entropy_loss_weight', 0.1),
            predictor_config={"num_layers": getattr(configs, 'entropy_predictor_layers', 3)}
        )
```

### 修改位置 3: 保存逻辑 (约第 425-430 行)

```python
if rank == 0:
    torch.save(states, os.path.join(save_dir, f"checkpoint_{epoch + 1}"))
    
    # 新增：单独保存 entropy predictor
    if hasattr(parallel_model.module, 'entropy_predictor'):
        predictor_path = os.path.join(save_dir, f"entropy_predictor_{epoch + 1}.pt")
        parallel_model.module.entropy_predictor.save(predictor_path)
    
    print("saving model.")
```

### 修改位置 4: 配置文件

在 `args/gsm_coconut.yaml` 中添加：

```yaml
# Entropy Predictor 配置
use_entropy_predictor: true
entropy_loss_weight: 0.1
entropy_predictor_layers: 3
```

---

## 修改 train.py (CODI)

### 修改位置 1: 导入

```python
# 在文件开头添加
import sys
sys.path.append('..')
from entropy_predictor import CODIWithEntropyPredictor
```

### 修改位置 2: 模型包装 (约第 174 行之后)

```python
model = CODI(model_args, training_args, lora_config)

# 新增：如果启用 entropy predictor
if getattr(training_args, 'use_entropy_predictor', False):
    model = CODIWithEntropyPredictor(
        codi_model=model,
        entropy_loss_weight=getattr(training_args, 'entropy_loss_weight', 0.1),
        predictor_config={"num_layers": getattr(training_args, 'entropy_predictor_layers', 3)}
    )
```

### 修改位置 3: 保存逻辑 (约第 447 行)

```python
trainer.save_model(output_dir=training_args.output_dir)

# 新增：单独保存 entropy predictor
if hasattr(model, 'entropy_predictor'):
    predictor_path = os.path.join(training_args.output_dir, "entropy_predictor.pt")
    model.entropy_predictor.save(predictor_path)
```

### 修改位置 4: TrainingArguments

在 `src/model.py` 的 `TrainingArguments` 中添加：

```python
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # ... 原有参数 ...
    
    # 新增
    use_entropy_predictor: bool = field(default=False)
    entropy_loss_weight: float = field(default=0.1)
    entropy_predictor_layers: int = field(default=3)
```

---

## 完整示例

### 示例 1: Coconut 训练

```bash
# 在 yaml 中设置 use_entropy_predictor: true
torchrun --nproc_per_node=4 run.py args/gsm_coconut.yaml
```

### 示例 2: CODI 训练

```bash
python train.py \
    --use_entropy_predictor \
    --entropy_loss_weight 0.1 \
    --entropy_predictor_layers 3 \
    ...
```

### 示例 3: 加载社区模型 + 自己的 predictor

```python
# 社区模型
base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
coconut = Coconut(base, ...)

# 自己的 predictor
predictor = EntropyPredictor.load("my_entropy_predictor.pt")

# 组合
wrapped = CoconutWithEntropyPredictor(coconut, entropy_predictor=predictor)

# 推理
outputs, mode, entropy = wrapped.generate_adaptive(input_ids, entropy_threshold=0.5)
```

---

## 总结

| 改动项 | 文件 | 改动量 |
|-------|------|-------|
| 新增 | `entropy_predictor.py` | 新文件 |
| 修改 | `Coconut/run.py` | ~15行 |
| 修改 | `CODI/train.py` | ~10行 |
| 修改 | `CODI/src/model.py` | ~3行 (TrainingArguments) |
| 修改 | `args/*.yaml` | ~3行配置 |

**原有的 `coconut.py` 和 `model.py` 完全不需要修改！**
