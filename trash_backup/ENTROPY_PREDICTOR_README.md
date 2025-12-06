# Entropy Predictor for Adaptive Thinking (半显半隐式推理)

基于 SIM-CoT/Coconut/CODI 的改进，通过预测 token 熵来决定何时使用显式思考、何时使用隐式思考。

## 核心思路

```
┌─────────────────────────────────────────────────────────────────────┐
│                         整体流程                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  阶段1: 数据收集                                                     │
│  ┌──────────────┐      ┌───────────────┐     ┌────────────────────┐ │
│  │ 普通CoT模型   │ ──→  │ 推理得到      │ ──→ │ (hidden, entropy)  │ │
│  │              │      │ hidden_states │     │ 数据对              │ │
│  │              │      │ + logits      │     │                    │ │
│  └──────────────┘      └───────────────┘     └────────────────────┘ │
│                                                                      │
│  阶段2: 训练 Predictor                                               │
│  ┌────────────────────┐      ┌───────────────────┐                  │
│  │ (hidden, entropy)  │ ──→  │ EntropyPredictor  │                  │
│  │ 数据对              │      │ (MLP)             │                  │
│  └────────────────────┘      └───────────────────┘                  │
│                                                                      │
│  阶段3: 自适应推理                                                   │
│  ┌─────────────────┐   ┌───────────────────┐   ┌──────────────────┐ │
│  │ Coconut/CODI    │ + │ EntropyPredictor  │ = │ 半显半隐推理      │ │
│  │ (已训练好)       │   │ (阶段2训练)       │   │                  │ │
│  └─────────────────┘   └───────────────────┘   └──────────────────┘ │
│                                                                      │
│  推理逻辑:                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ for each token position:                                        ││
│  │   predicted_entropy = EntropyPredictor(hidden_state)            ││
│  │   if entropy > threshold:                                       ││
│  │     → 使用显式思考 (生成可见token)                               ││
│  │   else:                                                         ││
│  │     → 使用隐式思考 (使用latent token)                            ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

## 文件结构

```
SIM-CoT-main/
├── entropy_predictor.py       # 核心模块 (EntropyPredictor, Wrappers)
├── collect_entropy_data.py    # 阶段1: 数据收集脚本
├── train_entropy_predictor.py # 阶段2: 训练脚本
├── adaptive_inference.py      # 阶段3: 半显半隐推理脚本
├── Coconut/                   # 原有Coconut代码 (不修改)
└── CODI/                      # 原有CODI代码 (不修改)
```

## 使用流程

### 阶段1: 收集数据

用普通 CoT 模型推理，收集 `(hidden_states, entropy)` 数据对：

```bash
python step1_collect_entropy_data.py \
    --model_path ./Coconut/pretrained/gpt2 \
    --data_name icot \
    --output_path data/entropy_data.pt \
    --collect_positions cot \
    --batch_size 8
```

收集策略改进：

"all": 收集所有位置
"cot": 只收集CoT推理部分（更有针对性）
"answer": 只收集答案部分


新增参数：

--data_name: 指定数据集名称
--batch_size: 支持批量处理
--max_token_num: 过滤超长样本
--include_last_cot: 是否包含最后一个CoT步骤

参数说明：
- `--model_path`: 预训练的 CoT 模型路径
- `--data_path`: CoT 数据文件 (txt/jsonl/json)
- `--output_path`: 输出的熵数据文件
- `--collect_positions`: 收集哪些位置的数据
  - `all`: 所有位置
  - `last`: 只收集最后一个位置
  - `cot`: 只收集 CoT 推理部分

### 阶段2: 训练 EntropyPredictor

用收集的数据训练 MLP predictor：

```bash
python step2_train_entropy_predictor.py \
    --data_path data/entropy_data.pt \
    --output_path checkpoints/entropy_predictor.pt \
    --hidden_dim 768 \
    --num_layers 3 \
    --epochs 10 \
    --batch_size 256 \
    --lr 1e-3
```

参数说明：
- `--hidden_dim`: 隐藏维度 (应与 base model 一致，脚本会自动从数据推断)
- `--num_layers`: MLP 层数
- `--epochs`: 训练轮数
- `--batch_size`: 批大小

训练完成后会输出：
1. `entropy_predictor.pt` - 模型权重
2. `entropy_predictor_results.png` - 预测结果可视化
3. `entropy_predictor_training_curve.png` - 训练曲线
4. 阈值建议 (基于熵分布的百分位数)

### 阶段3: 半显半隐推理

在已训练好的 Coconut/CODI 上使用 EntropyPredictor 进行自适应推理：

```bash
# 序列级别自适应
python step3_adaptive_inference.py \
    --model_type coconut \
    --model_path checkpoints/coconut.pt \
    --tokenizer_path meta-llama/Llama-2-7b-hf \
    --predictor_path checkpoints/entropy_predictor.pt \
    --mode sequence \
    --entropy_threshold 0.5 \
    --input "Question: What is 15 + 27?"

# Token级别自适应
python adaptive_inference.py \
    --model_type coconut \
    --model_path checkpoints/coconut.pt \
    --predictor_path checkpoints/entropy_predictor.pt \
    --mode token \
    --high_threshold 0.7 \
    --low_threshold 0.3 \
    --input "Question: What is 15 + 27?"

# 分析模式 (只预测熵，不生成)
python adaptive_inference.py \
    --model_type base \
    --model_path gpt2 \
    --predictor_path checkpoints/entropy_predictor.pt \
    --mode analyze \
    --input "Question: What is 15 + 27?"

# 交互模式
python adaptive_inference.py \
    --model_type base \
    --model_path gpt2 \
    --predictor_path checkpoints/entropy_predictor.pt \
    --mode sequence
```

## 推理模式详解

### 1. 序列级别 (sequence)

根据输入的**整体熵**决定整个生成使用显式还是隐式思考：

```
输入: "What is 2+3?"
       ↓
预测整体熵 = 0.65
       ↓
entropy > threshold (0.5)?
       ↓
是 → 使用显式思考 (生成可见的推理步骤)
否 → 使用隐式思考 (使用 latent tokens)
```

### 2. Token级别 (token)

**逐 token** 决定使用哪种模式（更细粒度）：

```
输入: "What is 2+3?"
       ↓
Token 1: entropy=0.3 → latent (隐式)
Token 2: entropy=0.8 → explicit (显式)
Token 3: entropy=0.5 → mixed
Token 4: entropy=0.2 → latent (隐式)
...
       ↓
输出: 部分 token 可见，部分 token 隐式
```

### 3. 分析模式 (analyze)

只预测熵，不生成，用于分析输入文本：

```
输入: "What is 2+3?"
输出:
  Position 0: "What"    entropy=0.45  mode=mixed
  Position 1: "is"      entropy=0.32  mode=latent
  Position 2: "2"       entropy=0.78  mode=explicit
  Position 3: "+"       entropy=0.85  mode=explicit
  Position 4: "3"       entropy=0.72  mode=explicit
  Position 5: "?"       entropy=0.28  mode=latent
```

## 代码示例

### Python API 使用

```python
from entropy_predictor import EntropyPredictor, AdaptiveThinkingController

# 1. 加载训练好的 predictor
predictor = EntropyPredictor.load("checkpoints/entropy_predictor.pt")

# 2. 创建控制器
controller = AdaptiveThinkingController(
    predictor,
    high_threshold=0.7,  # 高于此值 → 显式
    low_threshold=0.3,   # 低于此值 → 隐式
)

# 3. 推理时使用
with torch.no_grad():
    outputs = base_model(input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]
    
    # 决定整体模式
    mode, entropy = controller.decide_mode(hidden_states)
    print(f"Mode: {mode}, Entropy: {entropy:.4f}")
    
    # 获取 token 级别 mask
    explicit_mask = controller.get_explicit_mask(hidden_states)
    # explicit_mask[i] = True 表示第 i 个 token 应该使用显式思考
```

### 与 Coconut 集成

```python
from coconut import Coconut
from entropy_predictor import CoconutWithEntropyPredictor, EntropyPredictor

# 1. 加载 Coconut 模型
coconut = Coconut(base_model, latent_id, start_id, end_id, eos_id)
coconut.load_state_dict(torch.load("coconut.pt"))

# 2. 加载 EntropyPredictor
predictor = EntropyPredictor.load("entropy_predictor.pt")

# 3. 包装
wrapped = CoconutWithEntropyPredictor(coconut, entropy_predictor=predictor)

# 4. 自适应生成
outputs, mode, entropy = wrapped.generate_adaptive(
    input_ids,
    attention_mask,
    max_new_tokens=100,
    entropy_threshold=0.5,
)
print(f"Used {mode} thinking (entropy={entropy:.3f})")
```

## 阈值选择建议

训练脚本会根据熵分布自动给出建议阈值：

```
Suggested thresholds:
  Low threshold (latent thinking):   0.35
  High threshold (explicit thinking): 0.65
```

**一般准则**：
- 数学计算等需要精确推理的：使用较低阈值，更多显式思考
- 常识问答等简单任务：使用较高阈值，更多隐式思考
- 混合任务：使用中等阈值，让模型自适应

## 原理说明

### 为什么熵可以指导思考模式？

- **高熵** = 模型不确定，需要更多"思考"
  → 使用显式推理，生成可见的推理步骤
  
- **低熵** = 模型有信心，可以快速处理
  → 使用隐式推理，使用 latent tokens 压缩推理

### 与原有 Coconut/CODI 的关系

1. **训练阶段**：Coconut/CODI 按原有方式训练（不修改）
2. **EntropyPredictor**：独立训练，用普通 CoT 数据
3. **推理阶段**：两者组合，predictor 指导 Coconut/CODI 的推理模式

这种设计的好处：
- ✅ 原有代码完全不修改
- ✅ 可以使用任意预训练的 Coconut/CODI 模型
- ✅ EntropyPredictor 可以独立训练/更新
- ✅ 灵活组合不同模型

## 常见问题

### Q: hidden_dim 不匹配怎么办？

A: 训练脚本会自动从数据推断 hidden_dim。如果你要用预训练的 predictor 在不同 hidden_dim 的模型上，需要重新训练。

### Q: 如何选择 collect_positions？

A: 
- `all`: 最全面，数据量大，但可能包含噪声
- `last`: 只关注预测位置，数据量小
- `cot`: 只关注推理部分，最相关但需要数据格式支持

### Q: 训练需要多少数据？

A: 建议至少 10k-100k 个 token 位置的数据。数据越多，predictor 越准确。

---

## License

MIT License
