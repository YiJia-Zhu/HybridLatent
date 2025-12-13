# Phase 2: 半显半隐推理训练

## 概述

本模块实现了半显半隐推理的训练流程，使CODI模型学会：
1. **显式生成**：正常输出文本token
2. **隐式压缩**：将若干显式步骤压缩到隐层循环
3. **无缝切换**：从隐层状态能顺畅接续显式生成

## 与原始CODI的兼容性

**关键设计**：本训练代码使用原始的CODI模型结构（`model_hybrid.py`），这意味着：

✅ 训练好的模型可以**直接使用** `step4_adaptive_eval.py` 进行推理  
✅ 保持与原始CODI相同的特殊token IDs  
✅ 保持相同的projection层结构  
✅ 保持相同的forward接口  

## 文件结构

```
phase2_model_hybrid.py       # 半显半隐模型定义（可选，用于实验）
phase2_hybrid_train.py       # 训练脚本（独立版本）
phase2_train_compatible.py   # 训练脚本（与原始CODI兼容，推荐）
run_full_pipeline.sh         # 完整流程运行脚本
```

## 训练方法

### 方法1：使用兼容版本（推荐）

```bash
python phase2_train_compatible.py \
    --model_name_or_path ./CODI/pretrained/Llama-3.2-1B-Instruct \
    --hybrid_data_path ./data/hybrid_training/hybrid_training_data.json \
    --output_dir ./checkpoints/hybrid_codi \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-5 \
    --num_latent 6 \
    --prj_dim 2048 \
    --hybrid_cot_only_ratio 0.2 \
    --bf16
```

### 方法2：从现有CODI checkpoint继续训练

```bash
python phase2_train_compatible.py \
    --model_name_or_path ./CODI/pretrained/Llama-3.2-1B-Instruct \
    --restore_from ./CODI/pretrained/CODI-llama3.2-1b-Instruct \
    --hybrid_data_path ./data/hybrid_training/hybrid_training_data.json \
    --output_dir ./checkpoints/hybrid_codi_v2 \
    --num_train_epochs 3 \
    --bf16
```

## 训练后推理

训练完成后，可以直接使用原始的 `step4_adaptive_eval.py` 进行推理：

```bash
python step4_adaptive_eval.py \
    --model_type codi \
    --base_model_path ./CODI/pretrained/Llama-3.2-1B-Instruct \
    --ckpt_dir ./checkpoints/hybrid_codi \
    --prj_dim 2048 \
    --data_name gsm8k \
    --bf16 \
    --baseline_mode adaptive \
    --max_switch_count 5 \
    --window_e_to_l 5 \
    --window_l_to_e 0
```

## 参数说明

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_train_epochs` | 5 | 训练轮数 |
| `--per_device_train_batch_size` | 4 | 每设备batch大小 |
| `--learning_rate` | 2e-5 | 学习率 |
| `--warmup_ratio` | 0.1 | 预热比例 |

### CODI参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_latent` | 6 | 隐式循环步数 |
| `--prj_dim` | 2048 | Projection层维度 |
| `--use_prj` | True | 是否使用Projection层 |

### Loss参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--distill_loss_factor` | 1.0 | 蒸馏loss权重 |
| `--ref_loss_factor` | 1.0 | 参考loss权重 |
| `--hybrid_cot_only_ratio` | 0.2 | 纯CoT样本比例（保持显式推理能力） |

## 训练策略

### 混合训练

训练数据包含三种类型：
1. **熵引导样本** (~50%): 低熵步骤隐式化，高熵步骤保持显式
2. **随机样本** (~30%): 随机选择隐式化位置，提高鲁棒性
3. **纯显式样本** (~20%): 完整显式CoT，保持显式推理能力不退化

### Loss设计

```
Total Loss = CE_Loss + Distill_Loss * factor + Ref_CE_Loss * factor

其中：
- CE_Loss: 对decoder输出计算cross-entropy
- Distill_Loss: 隐层状态对齐到teacher对应位置
- Ref_CE_Loss: 对完整显式序列计算CE（保持显式能力）
```

### `hybrid_cot_only_ratio` 参数

这个参数控制训练时随机跳过隐式推理部分的比例：
- 设为0.0: 所有样本都进行半显半隐训练
- 设为0.2: 20%的样本只做显式CoT训练
- 设为1.0: 所有样本只做显式CoT训练（退化为纯SFT-CoT）

**建议值**: 0.2-0.3，既学习隐式推理又保持显式能力

## 完整流程示例

```bash
# 1. Phase 1: 生成数据
python phase1_data_generation.py \
    --model_path ./CODI/pretrained/Llama-3.2-1B-Instruct \
    --data_name icot \
    --output_dir ./data/hybrid_training \
    --bf16

# 2. Phase 2: 训练
python phase2_train_compatible.py \
    --model_name_or_path ./CODI/pretrained/Llama-3.2-1B-Instruct \
    --hybrid_data_path ./data/hybrid_training/hybrid_training_data.json \
    --output_dir ./checkpoints/hybrid_codi \
    --num_train_epochs 5 \
    --bf16

# 3. 评估
python step4_adaptive_eval.py \
    --model_type codi \
    --base_model_path ./CODI/pretrained/Llama-3.2-1B-Instruct \
    --ckpt_dir ./checkpoints/hybrid_codi \
    --data_name gsm8k \
    --baseline_mode adaptive \
    --bf16
```

或者直接运行：
```bash
bash run_full_pipeline.sh
```

## 预期效果

训练后的模型应该能够：

1. **自适应切换**: 根据熵值在显式/隐式模式间切换
2. **保持显式能力**: 在需要时能够完整输出推理过程
3. **提升效率**: 对简单计算步骤使用隐式推理
4. **保持准确性**: 在GSM8K等数据集上保持或提升准确率

## 常见问题

### Q: 训练后准确率下降怎么办？
A: 尝试以下方法：
- 增加 `hybrid_cot_only_ratio` 到 0.3-0.4
- 增加 `ref_loss_factor` 到 1.5
- 增加训练轮数

### Q: 如何验证模型学会了半显半隐？
A: 使用 `step4_adaptive_eval.py` 的 `--verbose` 模式，观察输出中的模式切换情况

### Q: 可以用于Coconut模型吗？
A: 目前主要针对CODI设计，Coconut需要相应调整数据格式

## 参考

- 原始CODI代码: `CODI/src/model_hybrid.py`
- 原始训练代码: `CODI/train.py`
- 推理代码: `step4_adaptive_eval.py`
