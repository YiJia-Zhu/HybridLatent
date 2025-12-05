# SwiReasoning 风格的自适应推理控制

## 概述

本项目将 [SwiReasoning](https://github.com/sdc17/SwiReasoning) 的熵计算和模式切换策略适配到 CODI/Coconut 框架。

### SwiReasoning vs CODI/Coconut

| 特性 | SwiReasoning | CODI/Coconut |
|------|--------------|--------------|
| 隐式思考方式 | Soft embedding (概率加权) | Latent token |
| 训练需求 | Training-free | 需要训练 |
| 熵计算 | 实时从 logits 计算 | 可通过 Predictor 预测 |

### 核心设计思想

SwiReasoning 的模式切换基于**熵的相对变化**：

```python
# 熵下降 -> 模型更确定 -> 使用显式输出
to_normal = (mode == soft) & (cur_entropy < ref_entropy)

# 熵上升 -> 模型不确定 -> 继续隐式思考
to_soft = (mode == normal) & (cur_entropy > ref_entropy)
```

## 文件说明

### 1. `step1_collect_entropy_data_swir.py`

收集训练数据，采用 SwiReasoning 风格的熵计算：

```bash
python step1_collect_entropy_data_swir.py \
    --model_path <pretrained_model> \
    --data_name icot \
    --output_path data/entropy_data_swir.pt \
    --max_samples 10000 \
    --compute_dynamics  # 计算熵动态特征
```

**新增特性**：
- `compute_entropy_swir()`: 非归一化熵计算
- `compute_entropy_dynamics()`: 计算熵变化趋势
- 存储扩展特征：raw_entropy, entropy_delta, entropy_trend

### 2. `step2_train_entropy_predictor.py`

训练 EntropyPredictor（与原版相同）：

```bash
python step2_train_entropy_predictor.py \
    --data_path data/entropy_data_swir.pt \
    --output_path checkpoints/entropy_predictor.pt \
    --hidden_dim 768 \
    --num_layers 3 \
    --epochs 10
```

### 3. `step3_adaptive_inference_swir.py`

SwiReasoning 风格的自适应推理：

```bash
# 使用预测熵 + latent token 模式
python step3_adaptive_inference_swir.py \
    --model_type coconut \
    --model_path <model_path> \
    --predictor_path checkpoints/entropy_predictor.pt \
    --window_size 10 \
    --use_predicted_entropy \
    --input "Question: What is 2+3?"

# 使用 soft embedding 模式（更接近 SwiReasoning 原版）
python step3_adaptive_inference_swir.py \
    --model_type base \
    --model_path <model_path> \
    --predictor_path checkpoints/entropy_predictor.pt \
    --use_soft_embedding \
    --input "Question: What is 2+3?"
```

### 4. `entropy_predictor.py`

核心模块，包含：
- `EntropyPredictor`: MLP 熵预测器
- `EntropyDataset`: 数据集类
- 熵计算工具函数

## 关键参数

### SwiReasoning 风格参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `window_size` | 512 | 切换前最少保持步数 |
| `max_switch_count` | None | 最大切换次数（防止 overthinking） |
| `alpha_0` | 1.0 | Soft embedding 混合参数 |
| `beta_0` | 0.7 | Normal embedding 混合参数 |

### 模式选择

```python
# 模式切换策略
if cur_entropy < ref_entropy:
    mode = "normal"  # 显式：使用普通 token
elif cur_entropy > ref_entropy and steps >= window_size:
    mode = "soft"    # 隐式：使用 latent token 或 soft embedding
```

## 使用示例

### 完整流程

```bash
# Step 1: 收集数据
python step1_collect_entropy_data_swir.py \
    --model_path Qwen/Qwen2-1.5B \
    --data_name icot \
    --output_path data/entropy_data.pt \
    --compute_dynamics

# Step 2: 训练预测器
python step2_train_entropy_predictor.py \
    --data_path data/entropy_data.pt \
    --output_path checkpoints/entropy_predictor.pt

# Step 3: 自适应推理
python step3_adaptive_inference_swir.py \
    --model_type coconut \
    --model_path <coconut_checkpoint> \
    --predictor_path checkpoints/entropy_predictor.pt \
    --window_size 10 \
    --use_predicted_entropy \
    --input "What is 15 + 27?"
```

### 交互模式

```bash
python step3_adaptive_inference_swir.py \
    --model_type base \
    --model_path Qwen/Qwen2-1.5B \
    --predictor_path checkpoints/entropy_predictor.pt \
    --verbose

# 命令：
# > What is 2 + 3?
# [analyze]> What is 2 + 3?   # 分析每个 token 的熵
```

## 与原版 SwiReasoning 的对比

### 相同点
1. 熵计算公式相同
2. 模式切换逻辑相同（基于熵的相对变化）
3. window_size 防止频繁切换

### 不同点
1. **熵来源**：
   - SwiReasoning: 实时从 logits 计算
   - 本项目: 可通过 EntropyPredictor 从 hidden states 预测

2. **隐式思考实现**：
   - SwiReasoning: Soft embedding（概率加权）
   - 本项目: 支持 soft embedding 或 latent token

3. **训练**：
   - SwiReasoning: Training-free
   - 本项目: 需要训练 EntropyPredictor（如果使用预测熵）

## 代码结构

```
.
├── entropy_predictor.py          # 核心模块
├── step1_collect_entropy_data_swir.py  # 数据收集
├── step2_train_entropy_predictor.py    # 训练预测器
├── step3_adaptive_inference_swir.py    # 自适应推理
└── README_SWIR.md                      # 本文档
```

## 参考

- [SwiReasoning Paper](https://arxiv.org/abs/2510.05069)
- [SwiReasoning GitHub](https://github.com/sdc17/SwiReasoning)
- [CODI Paper](https://arxiv.org/abs/2404.03989)
- [Coconut Paper](https://arxiv.org/abs/2408.08118)
