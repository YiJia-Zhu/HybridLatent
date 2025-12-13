# Phase 1: 半显半隐推理训练数据生成 (SwiReasoning风格)

## 概述

本模块实现了半显半隐推理训练数据的自动生成流程，**直接使用step4_adaptive_eval.py中的SwiRController切换逻辑**：

1. **熵分析**: 对每个token位置计算熵值
2. **SwiR模拟**: 使用与推理时相同的切换逻辑决定每个位置的模式
3. **多样化数据生成**: 生成三种pattern的训练样本

## 核心思想

### SwiReasoning切换逻辑 (与step4一致)

```
初始模式: normal (显式)

每一步:
  计算当前token的熵值
  
  if 熵上升 && 当前是normal && 等待步数>=window_e_to_l:
      → 切换到latent (隐式)
  
  if 熵下降 && 当前是latent && 等待步数>=window_l_to_e:
      → 切换到normal (显式)
  
  if 连续latent步数 >= max_latent_steps:
      → 强制切换到normal
  
  if 进入答案阶段:
      → 锁定normal模式
```

### 训练数据类型

| 类型 | 比例 | 目的 |
|------|------|------|
| 熵引导划分 | ~50% | 使用SwiR逻辑，学会在"合适"的地方用隐式 |
| 随机划分 | ~30% | 鲁棒性，任意位置都能切换 |
| 纯显式CoT | ~20% | 保持显式推理能力不退化 |

## 文件结构

```
phase1_data_generation.py    # 主数据生成脚本 (使用SwiRController)
phase1_data_converter.py     # CODI格式转换器 (可选)
run_phase1.sh               # 运行脚本
```

## 使用方法

### 生成训练数据

```bash
python phase1_data_generation.py \
    --model_path ./CODI/pretrained/Llama-3.2-1B-Instruct \
    --data_name icot \
    --output_dir ./data/hybrid_training \
    --window_e_to_l 5 \
    --window_l_to_e 0 \
    --max_switch_count 5 \
    --max_latent_steps 6 \
    --entropy_guided_ratio 0.5 \
    --random_ratio 0.3 \
    --explicit_ratio 0.2 \
    --bf16
```

### 一键运行

```bash
bash run_phase1.sh
```

## 输出数据格式

### hybrid_training_data.json

```json
{
  "question": "Tom has 5 apples...",
  "full_cot": "Tom starts with 5 apples. He buys 3 more. 5+3=8.",
  "answer": "8",
  "hybrid_sequence": "Tom starts with 5 apples.[L][L][L] 5+3=8.",
  "token_modes": [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
  "entropies": [2.3, 1.8, 2.1, 3.5, 4.2, 3.8, 2.1, 1.5, 1.2, 0.8, 0.5, 0.3],
  "switch_events": [
    {"step": 5, "event": "normal->latent", "entropy": 4.2},
    {"step": 8, "event": "latent->normal", "entropy": 1.5}
  ],
  "num_latent_tokens": 3,
  "num_explicit_tokens": 9,
  "pattern_type": "entropy_guided"
}
```

**关键字段说明**:
- `token_modes`: 每个token的模式列表 (0=latent, 1=normal)
- `entropies`: 每个token位置的熵值
- `switch_events`: 模式切换事件记录
- `hybrid_sequence`: 用[L]标记latent位置的混合文本

## 参数说明

### SwiReasoning 参数 (与step4_adaptive_eval.py一致)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--window_e_to_l` | 5 | Explicit→Latent 切换需等待的步数 |
| `--window_l_to_e` | 0 | Latent→Explicit 切换需等待的步数 |
| `--max_switch_count` | 5 | 最大切换次数 |
| `--max_latent_steps` | 6 | 最大连续latent步数 |

### 数据生成参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--entropy_guided_ratio` | 0.5 | 使用SwiR逻辑的样本比例 |
| `--random_ratio` | 0.3 | 随机划分样本比例 |
| `--explicit_ratio` | 0.2 | 纯显式样本比例 |

## 数据生成示例

### 原始数据
```
Q: Tom has 5 apples, buys 3 more. How many?
CoT: Tom starts with 5 apples. He buys 3 more. 5 + 3 = 8.
Answer: 8
```

### 熵引导划分 (假设"5 + 3 = 8"是低熵步骤)
```
Q: Tom has 5 apples, buys 3 more. How many?
Hybrid: Tom starts with 5 apples. He buys 3 more. [L][L]
Answer: The answer is: 8
```

### 随机划分
```
Q: Tom has 5 apples, buys 3 more. How many?
Hybrid: Tom starts with 5 apples. [L][L] 5 + 3 = 8.
Answer: The answer is: 8
```

### 纯显式
```
Q: Tom has 5 apples, buys 3 more. How many?
Hybrid: Tom starts with 5 apples. He buys 3 more. 5 + 3 = 8.
Answer: The answer is: 8
```

## 后续Phase 2训练建议

### Loss设计

```python
def compute_hybrid_loss(
    model_outputs,
    labels,
    implicit_mask,
    explicit_mask,
    ref_hidden_states,  # teacher模型的hidden states
    alpha_ce=1.0,       # CE loss权重
    alpha_distill=0.5,  # Distillation loss权重
):
    # 1. 显式部分: Cross-Entropy Loss
    ce_loss = cross_entropy(
        model_outputs.logits[explicit_mask],
        labels[explicit_mask]
    )
    
    # 2. 隐式部分: Distillation Loss
    distill_loss = mse_loss(
        model_outputs.hidden_states[implicit_mask],
        ref_hidden_states[implicit_mask]  # 对齐到teacher对应位置
    )
    
    return alpha_ce * ce_loss + alpha_distill * distill_loss
```

### 训练策略

1. **Warm-up**: 先用较多显式样本预热
2. **渐进式**: 逐步增加隐式比例
3. **Curriculum**: 从短序列到长序列

## 注意事项

1. **SFT-CoT模型选择**: 确保用于熵分析的模型已经有良好的显式推理能力
2. **熵阈值调整**: 根据实际数据分布调整`entropy_threshold_percentile`
3. **[L]数量**: `num_latent_per_implicit`应与目标模型的隐式循环能力匹配
4. **数据平衡**: 确保三种pattern的比例合理，避免模型偏向某一种模式

## 常见问题

### Q: 熵分析结果显示所有步骤熵都很高怎么办？
A: 可能SFT-CoT模型还不够好，或者数据集本身就比较复杂。可以尝试降低`entropy_threshold_percentile`。

### Q: 生成的数据中隐式步骤太少怎么办？
A: 增加`entropy_threshold_percentile`，或者增加`random_ratio`以获得更多带隐式步骤的样本。

### Q: 如何验证生成的数据质量？
A: 检查`generation_stats.json`中的统计信息，并查看`visualizations/`目录下的可视化图表。
