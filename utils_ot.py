import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from typing import Optional, Callable, Tuple, Union, Dict, List
import os
import numpy as np
import random
import gym
from sklearn.metrics import pairwise_distances
import ot

TensorBatch = List[torch.Tensor]

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
                 device: torch.device) -> None:
        
        self._buffer_size = buffer_size
        self._device = device
        obs_dtype = torch.float32 if len(state_dim) == 1 else torch.uint8
        
        state_dim = state_dim[0]
        action_dim = action_dim[0]
        
        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=obs_dtype, device=device
        )
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=obs_dtype, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32)
        self._gw_rewards = torch.zeros((buffer_size, 1), dtype=torch.float32)
        self._dones = torch.zeros(
            (buffer_size, 1), dtype=torch.float32, device=device
        )
        
        self.device = device
        
    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)
    
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        n_transitions = data["observations"].shape[0]
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])


        print(f"Dataset size: {n_transitions}")
        
    
def compute_initialization(mode, expert_ds, agent_ds, expert_metric,
                           agent_metric, entropic, sinkhorn_reg):
    distances_expert = pairwise_distances(expert_ds, expert_ds, metric=expert_metric) # (s_i, s_{i+1})
    distances_agent = pairwise_distances(agent_ds, agent_ds, metric=agent_metric)
    
    p = np.zeros(len(expert_ds)) + 1. / len(expert_ds)
    q = np.zeros(len(agent_ds)) + 1. / len(agent_ds)
    M = np.outer(p, q)
    
    distances_expert /= distances_expert.max()
    distances_agent /= distances_agent.max()
    if mode == "gw":
        if entropic:
            T = ot.gromov.entropic_gromov_wasserstein(
                    distances_expert, distances_agent, 
                    p, q,
                    'square_loss', epsilon=sinkhorn_reg, max_iter=1000, tol=1e-7)
            
    # else mode == "infoot"
    
    return torch.from_numpy(T, dtype=torch.double)


def align_pairs(T_init, agent_trajectories):
    return torch.matmul(T_init, agent_trajectories)
    
    
    
    