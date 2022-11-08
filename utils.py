import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from typing import Optional, Callable, Tuple, Union, Dict
import os
import numpy as np
import random
import gym
import ot

def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)
    
    
class ReplayBuffer:
    def __init__(self, buffer_size: int,
                 state_dim: Union[Tuple[int], int],
                 action_dim: Union[Tuple[int], int],
                 device: str = "cuda") -> None:
        
        self._buffer_size = buffer_size
        obs_dtype = np.float32 if len(state_dim) == 1 else np.uint8
        
        self._states = np.zeros(
            (buffer_size, *state_dim), dtype=obs_dtype, device=device
        )
        self._next_states = np.zeros(
            (buffer_size, *state_dim), dtype=obs_dtype, device=device
        )
        self._actions = np.zeros(
            (buffer_size, *action_dim), dtype=np.float32, device=device
        )
        self._rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self._gw_rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self._dones = torch.zeros(
            (buffer_size, 1), dtype=torch.float32, device=device
        )
        
        self.device = device
        
    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)
    
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")
    
    
def compute_initialization(batch_trajectories_expert, batch_trajectories_agent):
    
    