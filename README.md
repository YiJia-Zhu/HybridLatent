<!-- <p align="center" width="100%">
<img src="./docs/static/images/logo_resize.png"  width="80%">
</p> -->

<div align="center">
    <h1 align="center"> SIM-CoT: Supervised Implicit Chain-of-Thought
    </h1>
</div>

<p align="center">
  <img src="assets/coconut_teaser.png">
</p>


- **Authors**: [Xilin Wei](https://github.com/Wiselnn570), [Xiaoran Liu](https://scholar.google.de/citations?user=Qe6F4J4AAAAJ&hl=en), [Yuhang Zang](https://yuhangzang.github.io), [Xiaoyi Dong](https://lightdxy.github.io), [Yuhang Cao](https://scholar.google.com/citations?user=sJkqsqkAAAAJ&hl=en), [Jiaqi Wang](https://myownskyw7.github.io/), [Xipeng Qiu](https://xpqiu.github.io/en.html), [Dahua Lin](http://dahua.site/)
- **Institutes**: Fudan University; Shanghai AI Laboratory; The Chinese University of Hong Kong; Shanghai Innovation Institute; 
- **Resources**: [📖[Paper](https://arxiv.org/pdf/2509.20317)] [[🏠Project Page]()] [[🤗Huggingface](https://huggingface.co/collections/Wiselnn/sim-cot-supervised-implicit-chain-of-thought-68d895b00576f6166c19ab4f)]
## 💡 Highlights

- 🔥 **Latent Instability in Implicit CoT:** We systematically analyze the limitations of implicit Chain-of-Thought methods and reveal a **latent instability issue**—as the number of implicit tokens increases, models tend to collapse into homogeneous latent states that lose operator semantics.  

- 🔥 **Step-Level Supervision with SIM-CoT:** We propose **S**upervised **IM**plicit-CoT (**SIM-CoT**), a plug-and-play module that introduces **step-level supervision** via an auxiliary decoder. This stabilizes optimization, prevents collapse, and ensures that latent tokens capture meaningful reasoning steps.

- 🔥 **Strong and Consistent Performance:** SIM-CoT consistently outperforms both explicit and implicit baselines. On GPT-2, it exceeds supervised CoT by **+2.1%**, Coconut by **+8.2%**, and CODI by **+4.3%**. Across larger LLaMA models (1B/3B/8B), it delivers **+1.5% to +9.0%** gains, and remains stable even with **8–16 implicit tokens**, where prior methods collapse.  

- 🔥 **Efficiency and Interpretability:** SIM-CoT adds **no extra inference cost** since the auxiliary decoder is discarded after training. It also provides **interpretability**, allowing each latent token to be decoded into a human-readable reasoning step.  

## 📜 News

**[2025/9/24]** [Code]() and [Paper](https://arxiv.org/pdf/2509.20317) are released!

## 👨‍💻 Todo

- [x] Code Release
- [x] Checkpoint Release
- [x] Usage Instructions Release


## 🛠️ Usage

### 1. Clone the repository
```bash
git clone https://github.com/InternLM/SIM-CoT.git
cd SIM-CoT
```

### 2. Install dependencies
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---
export CUDA_VISIBLE_DEVICES="4,5,6,7"
### 3. Training with Coconut + SIM-CoT

#### Step 1: Train the Coconut baseline
```bash
cd Coconut
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut.yaml
```

#### Step 2: Continue training with SIM-CoT
Select a checkpoint that has been expanded to predefined implicit tokens, then continue training with SIM-CoT:
```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_simcot.yaml
```

---

### 4. Evaluation with Coconut + SIM-CoT
```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_simcot_eval.yaml
```

---

### 5. Training with CODI + SIM-CoT
```bash
cd CODI
bash scripts/train_llama3b_gsm8k-aug-decoder-2.sh
```

---

### 6. Evaluation with CODI + SIM-CoT
```bash
bash CODI/scripts/test_llama3b-copy.sh
```

根据我从 CODI 项目 GitHub 和论文中获取的信息，让我为你详细解释这个训练脚本中每个参数的含义：

## 📁 基础路径与环境设置

| 参数 | 值 | 含义 |
|------|-----|------|
| `SAVE_DIR` | `./codi_llama1b_full` | 模型保存目录 |
| `CUDA_VISIBLE_DEVICES` | `4,5,6,7` | 使用的 GPU 编号 |

## 🔧 Hugging Face Trainer 标准参数

| 参数 | 值 | 含义 |
|------|-----|------|
| `--output_dir` | `$SAVE_DIR` | 输出目录 |
| `--expt_name` | `gsm8k_llama1b_latent_baseline` | 实验名称 |
| `--logging_dir` | `$SAVE_DIR/logs` | TensorBoard 日志目录 |
| `--logging_steps` | `10` | 每 10 步记录一次日志 |
| `--model_name_or_path` | `./pretrained/Llama-3.2-1B-Instruct` | 预训练模型路径 |
| `--data_name` | `icot` | 数据集名称 (implicit CoT) |
| `--seed` | `11` | 随机种子 |
| `--model_max_length` | `512` | 最大序列长度 |
| `--per_device_train_batch_size` | `16` | 每个 GPU 的 batch size |
| `--gradient_accumulation_steps` | `4` | 梯度累积步数 (有效 batch = 16×4×4GPU = 256) |
| `--bf16` | - | 使用 BFloat16 混合精度训练 |
| `--num_train_epochs` | `10` | 训练轮数 |
| `--learning_rate` | `8e-4` | 学习率 |
| `--max_grad_norm` | `2.0` | 梯度裁剪阈值 |
| `--save_strategy` | `no` | 不保存中间 checkpoint |
| `--save_total_limit` | `1` | 最多保留 1 个 checkpoint |
| `--save_safetensors` | `False` | 不使用 safetensors 格式 |
| `--weight_decay` | `0.1` | 权重衰减 |
| `--warmup_ratio` | `0.03` | 预热比例 (3% 的训练步数) |
| `--lr_scheduler_type` | `cosine` | 余弦学习率调度 |
| `--do_train` | - | 执行训练 |
| `--report_to` | `tensorboard` | 日志报告到 TensorBoard |
| `--logging_strategy` | `steps` | 按步数记录日志 |

## 🎯 LoRA 相关参数

| 参数 | 值 | 含义 |
|------|-----|------|
| `--use_lora` | `True` | 启用 LoRA 微调 |
| `--lora_r` | `128` | LoRA 的秩 (rank)，越大表达能力越强 |
| `--lora_alpha` | `32` | LoRA 缩放因子 |
| `--lora_init` | - | 使用特殊的 LoRA 初始化 |

## 🧠 CODI 核心参数

| 参数 | 值 | 含义 |
|------|-----|------|
| `--num_latent` | `6` | 训练时使用的隐式思维 token 数量 |
| `--use_prj` | `True` | 是否对最后一层 hidden state 使用投影层 |
| `--prj_dim` | `2048` | 投影层的隐藏维度 |
| `--prj_dropout` | `0.0` | 投影层的 dropout 率 |
| `--distill_loss_div_std` | `True` | 是否用 teacher hidden state 的标准差来归一化蒸馏损失 |
| `--distill_loss_factor` | `20` | 蒸馏损失的权重系数 |
| `--max_token_num` | `200` | 丢弃超过此 token 长度的训练样本 |
| `--remove_eos` | `True` | 移除 EOS token |
| `--print_ref_model_stats` | `True` | 打印参考模型的统计信息 |

## 🔬 实验模式参数

| 参数 | 值 | 含义 |
|------|-----|------|
| `--exp_mode` | `False` | 是否为实验模式 (用于快速调试) |
| `--exp_data_num` | `200` | 实验模式下使用的数据量 |

## 📖 CODI 框架简介

CODI 是一个自蒸馏框架，包含 teacher 任务和 student 任务。Teacher 任务学习显式 CoT 推理，student 任务学习隐式 CoT 推理。知识蒸馏通过对齐关键 token 的 hidden activation 来实现。

核心思想是：
- **Teacher**：使用标准的 Chain-of-Thought（显式推理步骤）
- **Student**：使用连续空间中的隐式思维 token（`num_latent=6` 个）
- **蒸馏**：通过对齐两者在生成答案位置的 hidden state 来传递知识

## ✒️ Citation

If you find our work helpful for your research, please consider giving a star ⭐ and citation 📝

```bibtex
@article{wei2025simcot,
  title={{SIM-COT}: Supervised Implicit Chain-of-Thought},
  author={Wei, Xilin and Liu, Xiaoran and Zang, Yuhang and Dong, Xiaoyi and Cao, Yuhang and Wang, Jiaqi and Qiu, Xipeng and Lin, Dahua},
  journal={arXiv preprint arXiv:2509.20317},
  year={2025}
}
```

## ❤️ Acknowledgments

- [Coconut](https://github.com/facebookresearch/coconut): The codebase we built upon. Thanks for their wonderful work.
- [CODI](https://github.com/zhenyi4/codi): Our work is based on this codebase; we are grateful for their valuable contribution.
- [LLaMA series](https://huggingface.co/meta-llama/collections): The amazing open-sourced large language model!
- [GPT2](https://huggingface.co/openai-community/gpt2): An impressive open-source large language model!


