"""
阶段2: 训练 EntropyPredictor (SwiReasoning 风格)

支持:
1. 标准训练: 预测归一化熵
2. 多任务学习: 同时预测熵和熵趋势
3. 针对 SwiReasoning 模式切换的评估指标

使用方法:
    # 标准训练
    python step2_train_entropy_predictor.py \
        --data_path data/entropy_data.pt \
        --output_path checkpoints/entropy_predictor.pt \
        --hidden_dim 768 \
        --epochs 10

    # 多任务学习（同时预测熵和趋势）
    python step2_train_entropy_predictor.py \
        --data_path data/entropy_data_swir.pt \
        --output_path checkpoints/entropy_predictor_mt.pt \
        --multi_task \
        --trend_weight 0.3
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, List, Tuple

# 导入本地模块
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from entropy_predictor import EntropyPredictor


# ============================================================================
# 扩展的 EntropyPredictor（支持多任务）
# ============================================================================

class EntropyPredictorMultiTask(nn.Module):
    """
    多任务 EntropyPredictor
    
    同时预测:
    1. 归一化熵 (0-1)
    2. 熵趋势 (-1, 0, 1) - 用于 SwiReasoning 模式切换
    
    Architecture:
        hidden_states -> Shared MLP -> [Entropy Head, Trend Head]
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 3,
        intermediate_dim: Optional[int] = None,
        dropout: float = 0.1,
        predict_trend: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.intermediate_dim = intermediate_dim or hidden_dim * 2
        self.predict_trend = predict_trend
        
        # 共享的 backbone
        backbone_layers = []
        current_dim = hidden_dim
        
        for i in range(num_layers - 1):
            backbone_layers.append(nn.Linear(current_dim, self.intermediate_dim))
            backbone_layers.append(nn.ReLU())
            backbone_layers.append(nn.Dropout(dropout))
            current_dim = self.intermediate_dim
        
        self.backbone = nn.Sequential(*backbone_layers)
        
        # 熵预测头
        self.entropy_head = nn.Sequential(
            nn.Linear(current_dim, 1),
            nn.Sigmoid(),
        )
        
        # 趋势预测头 (3分类: -1, 0, 1)
        if predict_trend:
            self.trend_head = nn.Sequential(
                nn.Linear(current_dim, 3),
            )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        return_trend: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            hidden_states: (batch, hidden_dim) 或 (batch, seq_len, hidden_dim)
            return_trend: 是否返回趋势预测
        
        Returns:
            entropy: (batch, 1) 预测的归一化熵
            trend: (batch, 3) 趋势 logits (if return_trend and predict_trend)
        """
        original_shape = hidden_states.shape
        
        if hidden_states.dim() == 3:
            batch_size, seq_len, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)
        
        # Backbone
        features = self.backbone(hidden_states)
        
        # Entropy prediction
        entropy = self.entropy_head(features)
        
        # Trend prediction
        trend = None
        if return_trend and self.predict_trend:
            trend = self.trend_head(features)
            if len(original_shape) == 3:
                trend = trend.view(batch_size, seq_len, 3)
        
        if len(original_shape) == 3:
            entropy = entropy.view(batch_size, seq_len, 1)
        
        return entropy, trend
    
    def predict_mode_switch(
        self,
        hidden_states: torch.Tensor,
        ref_entropy: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        预测模式切换（SwiReasoning 风格）
        
        Args:
            hidden_states: (batch, hidden_dim)
            ref_entropy: (batch,) 参考熵
        
        Returns:
            cur_entropy: (batch,) 当前熵
            should_go_normal: (batch,) 是否应该切换到 normal
            should_go_soft: (batch,) 是否应该切换到 soft
        """
        entropy, trend = self.forward(hidden_states, return_trend=True)
        cur_entropy = entropy.squeeze(-1)
        
        # 基于相对熵的判断
        should_go_normal = cur_entropy < ref_entropy  # 熵下降
        should_go_soft = cur_entropy > ref_entropy    # 熵上升
        
        # 如果有趋势预测，可以作为辅助参考
        if trend is not None:
            trend_pred = torch.argmax(trend, dim=-1)  # 0=下降, 1=稳定, 2=上升
            # 可以结合趋势预测进行更精细的控制
        
        return cur_entropy, should_go_normal, should_go_soft
    
    def save(self, path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'intermediate_dim': self.intermediate_dim,
                'predict_trend': self.predict_trend,
            }
        }, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: torch.device = None) -> 'EntropyPredictorMultiTask':
        """加载模型"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        model = cls(
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            intermediate_dim=config.get('intermediate_dim'),
            predict_trend=config.get('predict_trend', False),
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print(f"Model loaded from {path}")
        return model


# ============================================================================
# 扩展的数据集类
# ============================================================================

class EntropyDatasetExtended(Dataset):
    """
    扩展的熵数据集，支持:
    - normalized_entropy: 归一化熵 (训练目标)
    - raw_entropy: 原始熵 (可选)
    - entropy_delta: 熵变化量 (可选)
    - entropy_trend: 熵趋势 (可选，用于多任务学习)
    """
    
    def __init__(self, data_path: str = None):
        self.hidden_states: List[torch.Tensor] = []
        self.entropies: List[torch.Tensor] = []
        self.raw_entropies: List[torch.Tensor] = []
        self.entropy_deltas: List[torch.Tensor] = []
        self.entropy_trends: List[torch.Tensor] = []
        
        self.has_raw = False
        self.has_delta = False
        self.has_trend = False
        
        if data_path is not None:
            self.load(data_path)
    
    def __len__(self):
        return len(self.hidden_states)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = {
            "hidden_state": self.hidden_states[idx],
            "entropy": self.entropies[idx],
        }
        
        if self.has_raw and idx < len(self.raw_entropies):
            item["raw_entropy"] = self.raw_entropies[idx]
        
        if self.has_delta and idx < len(self.entropy_deltas):
            item["entropy_delta"] = self.entropy_deltas[idx]
        
        if self.has_trend and idx < len(self.entropy_trends):
            # 将趋势转换为分类标签 (-1->0, 0->1, 1->2)
            trend = self.entropy_trends[idx]
            if trend.dim() == 0:
                trend_label = int(trend.item()) + 1  # -1,0,1 -> 0,1,2
            else:
                trend_label = int(trend[0].item()) + 1
            item["trend_label"] = torch.tensor(trend_label, dtype=torch.long)
        
        return item
    
    def load(self, path: str):
        """加载数据集"""
        data = torch.load(path)
        
        if data.get('hidden_states') is not None:
            self.hidden_states = list(data['hidden_states'])
        
        if data.get('entropies') is not None:
            self.entropies = list(data['entropies'])
        
        if data.get('raw_entropies') is not None:
            self.raw_entropies = list(data['raw_entropies'])
            self.has_raw = True
        
        if data.get('entropy_deltas') is not None:
            self.entropy_deltas = list(data['entropy_deltas'])
            self.has_delta = True
        
        if data.get('entropy_trends') is not None:
            self.entropy_trends = list(data['entropy_trends'])
            self.has_trend = True
        
        print(f"Loaded {len(self)} samples from {path}")
        print(f"  Has raw entropy: {self.has_raw}")
        print(f"  Has entropy delta: {self.has_delta}")
        print(f"  Has entropy trend: {self.has_trend}")


# ============================================================================
# 训练函数
# ============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    scheduler=None,
    multi_task: bool = False,
    trend_weight: float = 0.3,
):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    total_entropy_loss = 0
    total_trend_loss = 0
    num_batches = 0
    
    criterion_entropy = nn.MSELoss()
    criterion_trend = nn.CrossEntropyLoss()
    
    for batch in dataloader:
        hidden_states = batch["hidden_state"].to(device).float()
        target_entropy = batch["entropy"].to(device).float()
        
        if target_entropy.dim() == 1:
            target_entropy = target_entropy.unsqueeze(-1)
        
        optimizer.zero_grad()
        
        if multi_task and "trend_label" in batch:
            # 多任务学习
            trend_labels = batch["trend_label"].to(device)
            
            if isinstance(model, EntropyPredictorMultiTask):
                pred_entropy, pred_trend = model(hidden_states, return_trend=True)
            else:
                pred_entropy = model(hidden_states)
                pred_trend = None
            
            # 熵损失
            entropy_loss = criterion_entropy(pred_entropy, target_entropy)
            
            # 趋势损失
            if pred_trend is not None:
                trend_loss = criterion_trend(pred_trend.squeeze(1), trend_labels)
                loss = (1 - trend_weight) * entropy_loss + trend_weight * trend_loss
                total_trend_loss += trend_loss.item()
            else:
                loss = entropy_loss
        else:
            # 单任务学习
            if isinstance(model, EntropyPredictorMultiTask):
                pred_entropy, _ = model(hidden_states, return_trend=False)
            else:
                pred_entropy = model(hidden_states)
            
            entropy_loss = criterion_entropy(pred_entropy, target_entropy)
            loss = entropy_loss
        
        loss.backward()
        optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        total_loss += loss.item()
        total_entropy_loss += entropy_loss.item()
        num_batches += 1
    
    return {
        "total_loss": total_loss / num_batches,
        "entropy_loss": total_entropy_loss / num_batches,
        "trend_loss": total_trend_loss / num_batches if multi_task else 0,
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    multi_task: bool = False,
) -> Dict:
    """评估模型"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_trend_preds = []
    all_trend_labels = []
    
    total_mse = 0
    total_mae = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            hidden_states = batch["hidden_state"].to(device).float()
            target_entropy = batch["entropy"].to(device).float()
            
            if target_entropy.dim() == 1:
                target_entropy = target_entropy.unsqueeze(-1)
            
            if isinstance(model, EntropyPredictorMultiTask):
                pred_entropy, pred_trend = model(hidden_states, return_trend=multi_task)
            else:
                pred_entropy = model(hidden_states)
                pred_trend = None
            
            mse = nn.MSELoss()(pred_entropy, target_entropy)
            mae = nn.L1Loss()(pred_entropy, target_entropy)
            
            total_mse += mse.item()
            total_mae += mae.item()
            num_batches += 1
            
            all_predictions.append(pred_entropy.cpu())
            all_targets.append(target_entropy.cpu())
            
            if multi_task and pred_trend is not None and "trend_label" in batch:
                all_trend_preds.append(torch.argmax(pred_trend, dim=-1).cpu())
                all_trend_labels.append(batch["trend_label"])
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    results = {
        "mse": total_mse / num_batches,
        "mae": total_mae / num_batches,
        "predictions": all_predictions,
        "targets": all_targets,
    }
    
    # 趋势预测准确率
    if all_trend_preds:
        all_trend_preds = torch.cat(all_trend_preds, dim=0)
        all_trend_labels = torch.cat(all_trend_labels, dim=0)
        trend_acc = (all_trend_preds.squeeze() == all_trend_labels).float().mean().item()
        results["trend_accuracy"] = trend_acc
        results["trend_predictions"] = all_trend_preds
        results["trend_labels"] = all_trend_labels
    
    return results


def evaluate_mode_switch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict:
    """
    评估模式切换预测准确率（SwiReasoning 风格）
    
    模拟 SwiReasoning 的切换逻辑，评估:
    - 熵下降时是否正确预测切换到 normal
    - 熵上升时是否正确预测切换到 soft
    """
    model.eval()
    
    correct_normal = 0  # 正确预测切换到 normal
    correct_soft = 0    # 正确预测切换到 soft
    total_normal = 0    # 实际应该切换到 normal 的次数
    total_soft = 0      # 实际应该切换到 soft 的次数
    
    prev_entropy = None
    
    with torch.no_grad():
        for batch in dataloader:
            hidden_states = batch["hidden_state"].to(device).float()
            target_entropy = batch["entropy"].to(device).float()
            
            if target_entropy.dim() == 1:
                target_entropy = target_entropy.unsqueeze(-1)
            
            if isinstance(model, EntropyPredictorMultiTask):
                pred_entropy, _ = model(hidden_states, return_trend=False)
            else:
                pred_entropy = model(hidden_states)
            
            # 对于每个样本，模拟模式切换
            for i in range(pred_entropy.shape[0]):
                cur_pred = pred_entropy[i, 0].item()
                cur_true = target_entropy[i, 0].item()
                
                if prev_entropy is not None:
                    # 判断真实的切换方向
                    true_delta = cur_true - prev_entropy
                    pred_delta = cur_pred - prev_entropy
                    
                    if true_delta < -0.01:  # 真实熵下降，应该切换到 normal
                        total_normal += 1
                        if pred_delta < 0:  # 预测也认为熵下降
                            correct_normal += 1
                    
                    elif true_delta > 0.01:  # 真实熵上升，应该切换到 soft
                        total_soft += 1
                        if pred_delta > 0:  # 预测也认为熵上升
                            correct_soft += 1
                
                prev_entropy = cur_true
    
    return {
        "normal_accuracy": correct_normal / total_normal if total_normal > 0 else 0,
        "soft_accuracy": correct_soft / total_soft if total_soft > 0 else 0,
        "total_normal": total_normal,
        "total_soft": total_soft,
    }


# ============================================================================
# 可视化
# ============================================================================

def plot_results(eval_results: Dict, save_path: str, multi_task: bool = False):
    """绘制预测结果"""
    predictions = eval_results["predictions"].numpy().flatten()
    targets = eval_results["targets"].numpy().flatten()
    
    n_plots = 4 if multi_task and "trend_predictions" in eval_results else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    
    # 1. 散点图
    axes[0].scatter(targets, predictions, alpha=0.3, s=1)
    axes[0].plot([0, 1], [0, 1], 'r--', label='Perfect prediction')
    axes[0].set_xlabel('True Entropy')
    axes[0].set_ylabel('Predicted Entropy')
    axes[0].set_title('Prediction vs True')
    axes[0].legend()
    
    # 2. 误差分布
    errors = predictions - targets
    axes[1].hist(errors, bins=50, edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--')
    axes[1].set_xlabel('Prediction Error')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Error Distribution (MAE={eval_results["mae"]:.4f})')
    
    # 3. 熵分布对比
    axes[2].hist(targets, bins=50, alpha=0.5, label='True', edgecolor='black')
    axes[2].hist(predictions, bins=50, alpha=0.5, label='Predicted', edgecolor='black')
    axes[2].set_xlabel('Entropy')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Entropy Distribution')
    axes[2].legend()
    
    # 4. 趋势预测混淆矩阵（如果有）
    if multi_task and "trend_predictions" in eval_results:
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        trend_preds = eval_results["trend_predictions"].numpy().flatten()
        trend_labels = eval_results["trend_labels"].numpy().flatten()
        
        cm = confusion_matrix(trend_labels, trend_preds, labels=[0, 1, 2])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[3],
                   xticklabels=['Down', 'Stable', 'Up'],
                   yticklabels=['Down', 'Stable', 'Up'])
        axes[3].set_xlabel('Predicted Trend')
        axes[3].set_ylabel('True Trend')
        axes[3].set_title(f'Trend Confusion Matrix (Acc={eval_results["trend_accuracy"]:.2%})')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Results plot saved to {save_path}")


def plot_training_curve(
    train_losses: List[Dict],
    val_losses: List[float],
    save_path: str,
    multi_task: bool = False,
):
    """绘制训练曲线"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(1, 2 if multi_task else 1, figsize=(12 if multi_task else 8, 5))
    
    if not multi_task:
        axes = [axes]
    
    # 主损失曲线
    train_total = [l["total_loss"] for l in train_losses]
    axes[0].plot(epochs, train_total, label='Train Loss', marker='o')
    axes[0].plot(epochs, val_losses, label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Training Curve')
    axes[0].legend()
    axes[0].grid(True)
    
    # 多任务损失分解
    if multi_task:
        train_entropy = [l["entropy_loss"] for l in train_losses]
        train_trend = [l["trend_loss"] for l in train_losses]
        
        axes[1].plot(epochs, train_entropy, label='Entropy Loss', marker='o')
        axes[1].plot(epochs, train_trend, label='Trend Loss', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Multi-Task Loss Components')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curve saved to {save_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train EntropyPredictor (SwiReasoning style)")
    
    # 数据参数
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to entropy data")
    parser.add_argument("--output_path", type=str, default="checkpoints/entropy_predictor_swir.pt",
                        help="Output path for trained predictor")
    
    # 模型参数
    parser.add_argument("--hidden_dim", type=int, default=768,
                        help="Hidden dimension (should match base model)")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Number of MLP layers")
    parser.add_argument("--intermediate_dim", type=int, default=None,
                        help="Intermediate dimension")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    
    # 多任务参数
    parser.add_argument("--multi_task", action="store_true",
                        help="Enable multi-task learning (entropy + trend)")
    parser.add_argument("--trend_weight", type=float, default=0.3,
                        help="Weight for trend loss in multi-task learning")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Validation set ratio")
    
    # 其他参数
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据
    print(f"Loading data from {args.data_path}...")
    dataset = EntropyDatasetExtended(args.data_path)
    print(f"Total samples: {len(dataset)}")
    
    # 检查是否支持多任务
    if args.multi_task and not dataset.has_trend:
        print("Warning: Multi-task enabled but dataset has no trend labels.")
        print("  Consider running step1 with --compute_dynamics")
        args.multi_task = False
    
    # 从数据推断 hidden_dim
    sample = dataset[0]
    inferred_hidden_dim = sample["hidden_state"].shape[0]
    if args.hidden_dim != inferred_hidden_dim:
        print(f"Warning: specified hidden_dim ({args.hidden_dim}) != data hidden_dim ({inferred_hidden_dim})")
        print(f"Using inferred hidden_dim: {inferred_hidden_dim}")
        args.hidden_dim = inferred_hidden_dim
    
    # 划分训练/验证集
    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # 创建模型
    if args.multi_task:
        model = EntropyPredictorMultiTask(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            intermediate_dim=args.intermediate_dim,
            dropout=args.dropout,
            predict_trend=True,
        ).to(device)
    else:
        model = EntropyPredictor(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            intermediate_dim=args.intermediate_dim,
            dropout=args.dropout,
        ).to(device)
    
    print(f"\nModel architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Multi-task learning: {args.multi_task}")
    
    # 优化器和调度器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    total_steps = len(train_loader) * args.epochs
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=args.warmup_ratio,
    )
    
    # 训练循环
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        # 训练
        train_result = train_epoch(
            model, train_loader, optimizer, device, scheduler,
            multi_task=args.multi_task,
            trend_weight=args.trend_weight,
        )
        train_losses.append(train_result)
        
        # 验证
        eval_results = evaluate(model, val_loader, device, multi_task=args.multi_task)
        val_loss = eval_results["mse"]
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch + 1}/{args.epochs}:")
        print(f"  Train Loss: {train_result['total_loss']:.6f} "
              f"(entropy: {train_result['entropy_loss']:.6f}, trend: {train_result['trend_loss']:.6f})")
        print(f"  Val Loss:   {val_loss:.6f} (MAE: {eval_results['mae']:.6f})")
        
        if args.multi_task and "trend_accuracy" in eval_results:
            print(f"  Trend Acc:  {eval_results['trend_accuracy']:.2%}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
            model.save(args.output_path)
            print(f"  -> New best model saved!")
    
    # 最终评估
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    
    # 加载最佳模型
    if args.multi_task:
        model = EntropyPredictorMultiTask.load(args.output_path, device)
    else:
        model = EntropyPredictor.load(args.output_path, device)
    
    eval_results = evaluate(model, val_loader, device, multi_task=args.multi_task)
    
    print(f"Best Val MSE: {eval_results['mse']:.6f}")
    print(f"Best Val MAE: {eval_results['mae']:.6f}")
    
    if args.multi_task and "trend_accuracy" in eval_results:
        print(f"Trend Accuracy: {eval_results['trend_accuracy']:.2%}")
    
    # 模式切换评估
    print("\n" + "-" * 40)
    print("Mode Switch Prediction (SwiReasoning style)")
    print("-" * 40)
    
    switch_results = evaluate_mode_switch(model, val_loader, device)
    print(f"Normal switch accuracy: {switch_results['normal_accuracy']:.2%} "
          f"({switch_results['total_normal']} cases)")
    print(f"Soft switch accuracy: {switch_results['soft_accuracy']:.2%} "
          f"({switch_results['total_soft']} cases)")
    
    # 绘制结果
    plot_path = args.output_path.replace('.pt', '_results.png')
    plot_results(eval_results, plot_path, multi_task=args.multi_task)
    
    # 绘制训练曲线
    curve_path = args.output_path.replace('.pt', '_training_curve.png')
    plot_training_curve(train_losses, val_losses, curve_path, multi_task=args.multi_task)
    
    # 阈值建议（SwiReasoning 风格）
    print("\n" + "=" * 60)
    print("Recommendations for SwiReasoning-style Control")
    print("=" * 60)
    
    targets = eval_results["targets"].numpy().flatten()
    predictions = eval_results["predictions"].numpy().flatten()
    
    print("\nEntropy Distribution Statistics:")
    print(f"  True - Mean: {targets.mean():.4f}, Std: {targets.std():.4f}")
    print(f"  Pred - Mean: {predictions.mean():.4f}, Std: {predictions.std():.4f}")
    
    print("\nSwiReasoning uses RELATIVE entropy comparison:")
    print("  - cur_entropy < ref_entropy -> Switch to NORMAL (explicit)")
    print("  - cur_entropy > ref_entropy -> Switch to SOFT (latent)")
    
    # 计算预测误差统计
    errors = predictions - targets
    print(f"\nPrediction Error Statistics:")
    print(f"  Mean error: {errors.mean():.4f}")
    print(f"  Std error:  {errors.std():.4f}")
    print(f"  This affects relative comparison accuracy")
    
    # 建议的 window_size
    # 基于熵变化的自相关性来建议
    print("\nSuggested Parameters:")
    print(f"  window_size: 5-20 (start with 10)")
    print(f"  max_switch_count: 2-5 (to prevent overthinking)")
    
    # 也提供绝对阈值参考（作为备选）
    print("\nAbsolute Threshold Reference (if needed):")
    percentiles = [25, 50, 75]
    for p in percentiles:
        val = np.percentile(targets, p)
        print(f"  {p}th percentile: {val:.4f}")


if __name__ == "__main__":
    main()