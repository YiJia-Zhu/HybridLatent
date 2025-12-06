"""
训练 EntropyPredictor: hidden_state -> raw_entropy

使用方法:
    python step2_train_entropy_predictor.py \
        --data_path data/entropy_data.pt \
        --output_path checkpoints/entropy_predictor.pt \
        --epochs 10
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional


class RawEntropyPredictor(nn.Module):
    """预测 raw entropy 的 MLP"""
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 3,
        intermediate_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.intermediate_dim = intermediate_dim or hidden_dim * 2
        
        layers = []
        cur_dim = hidden_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(cur_dim, self.intermediate_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            cur_dim = self.intermediate_dim
        layers.append(nn.Linear(cur_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, hidden_dim)
        Returns:
            entropy: (batch, 1)
        """
        return self.mlp(hidden_states)
    
    def predict(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """预测 raw entropy，返回 (batch,)"""
        return self.forward(hidden_states).squeeze(-1)
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'intermediate_dim': self.intermediate_dim,
            }
        }, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location=device)
        cfg = ckpt['config']
        model = cls(
            hidden_dim=cfg['hidden_dim'],
            num_layers=cfg['num_layers'],
            intermediate_dim=cfg.get('intermediate_dim'),
        )
        model.load_state_dict(ckpt['state_dict'])
        model.to(device).eval()
        print(f"Model loaded from {path}")
        return model


class EntropyDataset(Dataset):
    """加载 hidden_states 和 raw_entropies"""
    
    def __init__(self, data_path: str):
        data = torch.load(data_path)
        
        self.hidden_states = data['hidden_states']
        
        # 优先使用 raw_entropies
        # if 'raw_entropies' in data and data['raw_entropies'] is not None:
        self.entropies = data['raw_entropies']
        print("Using raw_entropies")
        # else:
        #     self.entropies = data['entropies']
        #     print("Warning: raw_entropies not found, using entropies")
        
        print(f"Loaded {len(self)} samples")
        
        # 打印统计信息
        ent = torch.stack(list(self.entropies)).float()
        print(f"Entropy stats: mean={ent.mean():.4f}, std={ent.std():.4f}, "
              f"min={ent.min():.4f}, max={ent.max():.4f}")
    
    def __len__(self):
        return len(self.hidden_states)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return {
            "hidden_state": self.hidden_states[idx],
            "entropy": self.entropies[idx],
        }


def train_epoch(model, loader, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0
    n = 0
    
    for batch in loader:
        h = batch["hidden_state"].to(device).float()
        y = batch["entropy"].to(device).float().unsqueeze(-1)
        
        optimizer.zero_grad()
        pred = model(h)
        loss = nn.MSELoss()(pred, y)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        total_loss += loss.item()
        n += 1
    
    return total_loss / n


def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    
    with torch.no_grad():
        for batch in loader:
            h = batch["hidden_state"].to(device).float()
            y = batch["entropy"].to(device).float()
            
            pred = model(h).squeeze(-1)
            preds.append(pred.cpu())
            targets.append(y.cpu())
    
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    
    mse = ((preds - targets) ** 2).mean().item()
    mae = (preds - targets).abs().mean().item()
    
    return {"mse": mse, "mae": mae, "preds": preds, "targets": targets}


def plot_results(results, save_path):
    preds = results["preds"].numpy()
    targets = results["targets"].numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 散点图
    axes[0].scatter(targets, preds, alpha=0.3, s=1)
    lims = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
    axes[0].plot(lims, lims, 'r--')
    axes[0].set_xlabel('True Entropy')
    axes[0].set_ylabel('Predicted Entropy')
    axes[0].set_title('Prediction vs True')
    
    # 误差分布
    errors = preds - targets
    axes[1].hist(errors, bins=50, edgecolor='black')
    axes[1].axvline(0, color='r', linestyle='--')
    axes[1].set_xlabel('Error')
    axes[1].set_title(f'Error Distribution (MAE={results["mae"]:.4f})')
    
    # 分布对比
    axes[2].hist(targets, bins=50, alpha=0.5, label='True')
    axes[2].hist(preds, bins=50, alpha=0.5, label='Pred')
    axes[2].set_xlabel('Entropy')
    axes[2].legend()
    axes[2].set_title('Distribution')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="checkpoints/entropy_predictor.pt")
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 数据
    dataset = EntropyDataset(args.data_path)
    
    # 推断 hidden_dim
    h_dim = dataset[0]["hidden_state"].shape[0]
    if args.hidden_dim != h_dim:
        print(f"Adjusting hidden_dim: {args.hidden_dim} -> {h_dim}")
        args.hidden_dim = h_dim
    
    # 划分
    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(args.seed))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # 模型
    model = RawEntropyPredictor(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} params")
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps)
    
    # 训练
    best_val = float('inf')
    train_losses, val_losses = [], []
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, scheduler)
        val_res = evaluate(model, val_loader, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_res["mse"])
        
        print(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.6f}, "
              f"val_mse={val_res['mse']:.6f}, val_mae={val_res['mae']:.6f}")
        
        if val_res["mse"] < best_val:
            best_val = val_res["mse"]
            model.save(args.output_path)
            print("  -> Best model saved")
    
    # 最终评估
    print("\n" + "="*50)
    model = RawEntropyPredictor.load(args.output_path, device)
    results = evaluate(model, val_loader, device)
    print(f"Final: MSE={results['mse']:.6f}, MAE={results['mae']:.6f}")
    
    # 绘图
    plot_path = args.output_path.replace('.pt', '_results.png')
    plot_results(results, plot_path)
    
    # 训练曲线
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    curve_path = args.output_path.replace('.pt', '_curve.png')
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"Curve saved to {curve_path}")


if __name__ == "__main__":
    main()