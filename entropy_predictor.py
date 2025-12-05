"""
EntropyPredictor 模块

从 hidden states 预测归一化熵值，用于自适应推理控制
支持 SwiReasoning 风格的模式切换

包含:
- EntropyPredictor: MLP 熵预测器
- EntropyDataset: 熵数据集类
- AdaptiveThinkingController: 自适应思考控制器 (legacy)
- CoconutWithEntropyPredictor: 集成了熵预测的 Coconut 包装器 (legacy)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
from typing import Optional, Dict, Tuple, List


# ============================================================================
# EntropyPredictor
# ============================================================================

class EntropyPredictor(nn.Module):
    """
    MLP 熵预测器
    
    从 hidden states 预测归一化熵值 (0-1 范围)
    用于在推理时避免重复计算 logits
    
    Architecture:
        hidden_states -> Linear -> ReLU -> Linear -> ... -> Linear -> Sigmoid
    """
    
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
        
        # 构建 MLP 层
        layers = []
        current_dim = hidden_dim
        
        for i in range(num_layers):
            if i == num_layers - 1:
                # 最后一层输出维度为 1
                layers.append(nn.Linear(current_dim, 1))
            else:
                layers.append(nn.Linear(current_dim, self.intermediate_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                current_dim = self.intermediate_dim
        
        self.mlp = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        return_all_positions: bool = False,
    ) -> torch.Tensor:
        """
        预测熵值
        
        Args:
            hidden_states: (batch, hidden_dim) 或 (batch, seq_len, hidden_dim)
            return_all_positions: 如果输入是 3D，是否返回所有位置的预测
        
        Returns:
            predicted_entropy: (batch, 1) 或 (batch, seq_len, 1)
        """
        original_shape = hidden_states.shape
        
        if hidden_states.dim() == 3:
            batch_size, seq_len, hidden_dim = hidden_states.shape
            # 展平为 (batch * seq_len, hidden_dim)
            hidden_states = hidden_states.view(-1, hidden_dim)
        
        # MLP forward
        output = self.mlp(hidden_states)
        # Sigmoid 确保输出在 0-1 范围
        output = self.sigmoid(output)
        
        if len(original_shape) == 3 and return_all_positions:
            # 恢复形状
            output = output.view(batch_size, seq_len, 1)
        
        return output
    
    def save(self, path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'intermediate_dim': self.intermediate_dim,
            }
        }, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: torch.device = None) -> 'EntropyPredictor':
        """加载模型"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        model = cls(
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            intermediate_dim=config.get('intermediate_dim'),
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print(f"Model loaded from {path}")
        return model


# ============================================================================
# EntropyDataset
# ============================================================================

class EntropyDataset(Dataset):
    """
    熵数据集类
    
    存储 (hidden_state, entropy) 数据对
    用于训练 EntropyPredictor
    """
    
    def __init__(self, data_path: str = None):
        self.hidden_states: List[torch.Tensor] = []
        self.entropies: List[torch.Tensor] = []
        
        if data_path is not None:
            self.load(data_path)
    
    def add_sample(self, hidden_state: torch.Tensor, entropy: torch.Tensor):
        """添加一个样本"""
        self.hidden_states.append(hidden_state)
        self.entropies.append(entropy)
    
    def __len__(self):
        return len(self.hidden_states)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return {
            "hidden_state": self.hidden_states[idx],
            "entropy": self.entropies[idx],
        }
    
    def save(self, path: str):
        """保存数据集"""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'hidden_states': torch.stack(self.hidden_states) if self.hidden_states else None,
            'entropies': torch.stack(self.entropies) if self.entropies else None,
        }, path)
        print(f"Saved {len(self)} samples to {path}")
    
    def load(self, path: str):
        """加载数据集"""
        data = torch.load(path)
        if data.get('hidden_states') is not None:
            self.hidden_states = list(data['hidden_states'])
        if data.get('entropies') is not None:
            self.entropies = list(data['entropies'])
        print(f"Loaded {len(self)} samples from {path}")


# ============================================================================
# Legacy: AdaptiveThinkingController (保留兼容性)
# ============================================================================

class AdaptiveThinkingController:
    """
    自适应思考控制器 (Legacy)
    
    基于阈值的简单控制器，用于决定使用显式还是隐式思考
    
    注意: 推荐使用 step3_adaptive_inference_swir.py 中的 SwiRAdaptiveController
    """
    
    def __init__(
        self,
        entropy_predictor: EntropyPredictor,
        high_threshold: float = 0.7,
        low_threshold: float = 0.3,
    ):
        self.predictor = entropy_predictor
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
    
    def predict_entropy(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """预测熵"""
        return self.predictor(hidden_states)
    
    def decide_mode(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[str, torch.Tensor]:
        """
        决定思考模式
        
        Args:
            hidden_states: (batch, seq_len, hidden_dim) 或 (batch, hidden_dim)
        
        Returns:
            mode: "explicit", "latent", 或 "mixed"
            entropy: 预测的熵值
        """
        with torch.no_grad():
            if hidden_states.dim() == 3:
                # 使用最后一个位置
                entropy = self.predictor(hidden_states[:, -1, :])
            else:
                entropy = self.predictor(hidden_states)
        
        avg_entropy = entropy.mean().item()
        
        if avg_entropy > self.high_threshold:
            mode = "explicit"
        elif avg_entropy < self.low_threshold:
            mode = "latent"
        else:
            mode = "mixed"
        
        return mode, entropy
    
    def get_token_modes(
        self,
        hidden_states: torch.Tensor,
    ) -> List[str]:
        """
        获取每个 token 的模式
        
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
        
        Returns:
            modes: 每个位置的模式列表
        """
        with torch.no_grad():
            entropy = self.predictor(hidden_states, return_all_positions=True)
        
        entropy = entropy[:, :, 0]  # (batch, seq_len)
        modes = []
        
        for e in entropy[0]:  # 假设 batch_size = 1
            e_val = e.item()
            if e_val > self.high_threshold:
                modes.append("explicit")
            elif e_val < self.low_threshold:
                modes.append("latent")
            else:
                modes.append("mixed")
        
        return modes


# ============================================================================
# Legacy: CoconutWithEntropyPredictor (保留兼容性)
# ============================================================================

class CoconutWithEntropyPredictor(nn.Module):
    """
    集成了 EntropyPredictor 的 Coconut 包装器 (Legacy)
    
    注意: 推荐使用 step3_adaptive_inference_swir.py 中的 AdaptiveGeneratorSwiR
    """
    
    def __init__(
        self,
        coconut_model,
        entropy_predictor: EntropyPredictor,
        high_threshold: float = 0.7,
        low_threshold: float = 0.3,
    ):
        super().__init__()
        self.coconut = coconut_model
        self.predictor = entropy_predictor
        self.controller = AdaptiveThinkingController(
            entropy_predictor,
            high_threshold,
            low_threshold,
        )
    
    @property
    def base_causallm(self):
        return self.coconut.base_causallm
    
    def forward(self, *args, **kwargs):
        return self.coconut(*args, **kwargs)
    
    def generate_adaptive(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 100,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        自适应生成
        
        根据熵预测动态选择显式或隐式思考
        """
        device = input_ids.device
        
        # 获取初始 hidden states
        with torch.no_grad():
            outputs = self.base_causallm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-1]
        
        # 决定模式
        mode, entropy = self.controller.decide_mode(hidden_states)
        
        if verbose:
            print(f"Predicted entropy: {entropy.mean().item():.4f}, Mode: {mode}")
        
        # 使用 Coconut 生成
        return self.coconut.generate(
            input_ids,
            attention_mask,
            max_new_tokens,
        )


# ============================================================================
# SwiReasoning 风格熵计算工具
# ============================================================================

def compute_entropy_swir(logits: torch.Tensor) -> torch.Tensor:
    """
    SwiReasoning 风格的熵计算 (非归一化)
    
    参考 SwiReasoning/generation_utils.py 第252行:
    cur_entropy = -(probs_original * (probs_original.clamp(min=1e-12).log())).sum(dim=-1)
    """
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * probs.clamp(min=1e-12).log()).sum(dim=-1)
    return entropy


def compute_normalized_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    归一化熵计算 (0-1 范围)
    """
    vocab_size = logits.size(-1)
    max_entropy = torch.log(torch.tensor(vocab_size, dtype=logits.dtype, device=logits.device))
    entropy = compute_entropy_swir(logits)
    return entropy / max_entropy