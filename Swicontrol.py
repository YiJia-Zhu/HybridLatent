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
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


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
    
    def predict(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """预测 entropy，返回 (batch,)"""
        return self.forward(hidden_states).squeeze(-1)
    
    def save(self, path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'state_dict': self.state_dict(),  # 兼容 RawEntropyPredictor 格式
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
        
        # 兼容两种保存格式: 'model_state_dict' (EntropyPredictor) 和 'state_dict' (RawEntropyPredictor)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            raise KeyError(f"Checkpoint missing state dict. Available keys: {list(checkpoint.keys())}")
        
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


# ============================================================================
# 时间统计器
# ============================================================================

@dataclass
class TimeStats:
    """时间统计"""
    normal_time: float = 0.0
    soft_time: float = 0.0
    normal_steps: int = 0
    soft_steps: int = 0
    
    def add_normal(self, duration: float):
        self.normal_time += duration
        self.normal_steps += 1
    
    def add_soft(self, duration: float):
        self.soft_time += duration
        self.soft_steps += 1
    
    def get_avg_normal(self) -> float:
        return self.normal_time / self.normal_steps if self.normal_steps > 0 else 0
    
    def get_avg_soft(self) -> float:
        return self.soft_time / self.soft_steps if self.soft_steps > 0 else 0
    
    def merge(self, other: 'TimeStats'):
        self.normal_time += other.normal_time
        self.soft_time += other.soft_time
        self.normal_steps += other.normal_steps
        self.soft_steps += other.soft_steps


# ============================================================================
# SwiReasoning 风格的模式状态和控制器
# ============================================================================

@dataclass
class SwiRModeState:
    """SwiReasoning 风格的模式状态"""
    mode: torch.Tensor
    mode_stay_steps: torch.Tensor
    ref_entropy: torch.Tensor
    locked_normal: torch.Tensor
    switch_count: torch.Tensor
    answer_locked: torch.Tensor
    
    @classmethod
    def init(cls, batch_size: int, device: torch.device):
        return cls(
            mode=torch.ones(batch_size, dtype=torch.long, device=device),
            mode_stay_steps=torch.zeros(batch_size, dtype=torch.long, device=device),
            ref_entropy=torch.zeros(batch_size, dtype=torch.float, device=device),
            locked_normal=torch.zeros(batch_size, dtype=torch.bool, device=device),
            switch_count=torch.zeros(batch_size, dtype=torch.long, device=device),
            answer_locked=torch.zeros(batch_size, dtype=torch.bool, device=device),
        )


class SwiRController:
    """SwiReasoning 风格的模式切换控制器"""
    
    def __init__(
        self,         
        window_e_to_l: int = 5,
        window_l_to_e: int = 0,
        max_switch_count: Optional[int] = None
        ):
        self.window_e_to_l = window_e_to_l
        self.window_l_to_e = window_l_to_e
        self.max_switch_count = max_switch_count
    
    def update(
        self,
        state: SwiRModeState,
        cur_entropy: torch.Tensor,
        step: int,
        end_token_mask: Optional[torch.Tensor] = None,
        at_step_boundary: Optional[torch.Tensor] = None,  # 新增：是否在 step 边界
    ) -> Tuple[SwiRModeState, torch.Tensor, torch.Tensor]:
        """更新模式状态"""
        device = cur_entropy.device
        batch_size = cur_entropy.shape[0]
        
        if end_token_mask is not None:
            state.locked_normal = state.locked_normal | end_token_mask
        
        if state.answer_locked.any():
            state.mode = torch.where(state.answer_locked, torch.ones_like(state.mode), state.mode)
            to_normal = torch.zeros(batch_size, dtype=torch.bool, device=device)
            to_soft = torch.zeros(batch_size, dtype=torch.bool, device=device)
            return state, to_normal, to_soft

        if step == 0:
            state.ref_entropy = cur_entropy.clone()
            to_normal = torch.zeros(batch_size, dtype=torch.bool, device=device)
            to_soft = torch.zeros(batch_size, dtype=torch.bool, device=device)
        else:
            state.mode_stay_steps += 1
            allow_l_to_e = (state.mode_stay_steps >= self.window_l_to_e)
            allow_e_to_l = (state.mode_stay_steps >= self.window_e_to_l)
            
            to_normal = (state.mode == 0) & (cur_entropy < state.ref_entropy) & allow_l_to_e
            to_soft = (state.mode == 1) & (cur_entropy > state.ref_entropy) & allow_e_to_l & (~state.locked_normal)
            
            # ===== 新增：只在 step 边界才允许从 explicit 切换到 latent =====
            if at_step_boundary is not None:
                to_soft = to_soft & at_step_boundary
            # =============================================================

            state.mode = torch.where(to_normal, torch.ones_like(state.mode), state.mode)
            state.mode = torch.where(to_soft, torch.zeros_like(state.mode), state.mode)
            switched = to_normal | to_soft
            state.mode_stay_steps = torch.where(switched, torch.zeros_like(state.mode_stay_steps), state.mode_stay_steps)
            state.ref_entropy = torch.where(switched, cur_entropy, state.ref_entropy)

            if self.max_switch_count is not None:
                state.switch_count = state.switch_count + switched.long()
                limit_reached = (state.switch_count >= self.max_switch_count)
                state.mode = torch.where(limit_reached, torch.ones_like(state.mode), state.mode)
                state.locked_normal = state.locked_normal | limit_reached
        
        return state, to_normal, to_soft
        

class AdaptiveController:
    """结合 EntropyPredictor 和 SwiRController 的自适应控制器"""
    
    def __init__(
        self,
        entropy_predictor: Optional[EntropyPredictor],
        window_e_to_l: int = 5,
        window_l_to_e: int = 0,
        max_switch_count: Optional[int] = None,
        use_predicted_entropy: bool = True,
        baseline_mode: str = "adaptive",
        random_prob: float = 0.5,
    ):
        self.predictor = entropy_predictor
        self.controller = SwiRController(window_e_to_l=window_e_to_l, window_l_to_e=window_l_to_e, max_switch_count=max_switch_count)
        self.use_predicted_entropy = use_predicted_entropy
        self.state = None
        
        self.baseline_mode = baseline_mode
        self.random_prob = random_prob
        self.is_initialized = False
        
    def init_state(self, batch_size: int, device: torch.device):
        """初始化模式状态"""
        self.state = SwiRModeState.init(batch_size, device)
        self.is_initialized = True
        return self.state
    
    def get_entropy(self, hidden_states: torch.Tensor, logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_predicted_entropy and self.predictor is not None:
            with torch.no_grad():
                predicted = self.predictor(hidden_states)
            return predicted.squeeze(-1)
        else:
            if logits is None:
                raise ValueError("logits required when use_predicted_entropy=False")
            return compute_entropy_swir(logits)
    
    def step(
        self,
        hidden_states: torch.Tensor,
        step: int,
        logits: Optional[torch.Tensor] = None,
        end_token_mask: Optional[torch.Tensor] = None,
        at_step_boundary: Optional[torch.Tensor] = None,  # 新增
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """更新模式状态并返回当前模式、切换事件和熵"""
        if self.state is None:
            self.init_state(hidden_states.shape[0], hidden_states.device)
            
        device = hidden_states.device
        batch_size = hidden_states.shape[0]
        
        if self.baseline_mode == "random":
            if step == 0 and self.is_initialized:
                random_values = torch.rand(batch_size, device=device)
                self.state.mode = (random_values < self.random_prob).long()
                
            random_switch = torch.rand(batch_size, device=device)
            
            to_normal_new = (random_switch < self.random_prob) & (self.state.mode == 0)
            to_soft_new = (random_switch >= self.random_prob) & (self.state.mode == 1) & (~self.state.locked_normal)
            
            # ===== 新增：random 模式也只在 step 边界切换到 soft =====
            if at_step_boundary is not None:
                to_soft_new = to_soft_new & at_step_boundary
            # ======================================================
            
            mode = self.state.mode.clone()
            mode = torch.where(to_normal_new, torch.ones_like(mode), mode)
            mode = torch.where(to_soft_new, torch.zeros_like(mode), mode)
            
            self.state.mode = mode
            
            to_normal = to_normal_new
            to_soft = to_soft_new
            cur_entropy = torch.zeros(batch_size, dtype=torch.float, device=device) 
            
            return mode, to_normal, to_soft, cur_entropy
        
        elif self.baseline_mode == "adaptive":
            cur_entropy = self.get_entropy(hidden_states, logits)
            # ===== 修改：传递 at_step_boundary =====
            self.state, to_normal, to_soft = self.controller.update(
                self.state, cur_entropy, step, end_token_mask, at_step_boundary
            )
            # ========================================
            
            return self.state.mode.clone(), to_normal, to_soft, cur_entropy
        
        else:
            raise ValueError(f"Unknown baseline_mode: {self.baseline_mode}")

