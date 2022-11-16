import pyrallis
import numpy as np
import ot
#import gymnasium as gym
import d4rl
import gym
from dataclasses import field, dataclass, asdict
import typing as tp
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import uuid
from typing import Optional, Callable
from functorch import vmap


TensorBatch = tp.List[torch.Tensor]

#Utils import
from utils_ot import set_seed, ReplayBuffer, compute_initialization, align_pairs
from video import VideoRecorder
from tqdm import trange


def compute_mean_std(states: np.ndarray, eps: float) -> tp.Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: tp.Union[np.ndarray, float] = 0.0,
    state_std: tp.Union[np.ndarray, float] = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    env = gym.wrappers.TransformObservation(env, normalize_state)
    return env


@dataclass
class CILOT_cfg:
    device: str = "cuda"
    expert: str = "hopper-medium-expert-v2"
    agent: str = "walker2d-random-v2"
    logger: str = field(default="wandb")
    seed: int = field(default=42)
    save_video: bool = field(default=True)
    work_dir: str = "expr/"
    delta: int = 1 # window size to check for next states
    num_train_steps: int = 1e6
    eval_steps: int = 5000
    batch_size: int = 100 # trajectory length
        
    #Wandb logger params
    project: str = "CILOT"
    group: str = "CILOT"
    name: str = "CILOT_VICreg"
    exper_name: str = "pendulum"
    
    #replay buffer
    capacity: int = 1e6
    #GW params
    gw_include_actions_expert: bool = field(default=False)
    metric_expert: str = "euclidean"
    metric_agent: str = "euclidean"
    norm_agent_with_expert: bool = field(default=True)
    entropic: bool = field(default=True)
    sinkhorn_reg: float = field(default=5e-3)
    gw_size_comp: int = field(default=2000)
    
    def __post_init__(self):
        self.group = self.name + "expert_" + self.exper_name

        
def load_wandb(cfg: dict) -> None:
    wandb.init(
        config=cfg,
        project=cfg["project"],
        group=cfg["group"],
        name=cfg["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()
    
    
@pyrallis.wrap()
def entry(cfg: CILOT_cfg):
    
    #Process logger
    if cfg.logger == "wandb":
        pass
        #load_wandb(asdict(cfg))
        
    env_agent = gym.make(cfg.agent, render_mode="rgb_array_list")
    env_expert = gym.make(cfg.expert)
    
    
    set_seed(cfg.seed, env_agent)
    device = torch.device(cfg.device)
    
    dataset_expert = d4rl.qlearning_dataset(env_expert) # (999999, 17)
    
    #normalize rewards of agent ? 
    if cfg.norm_agent_with_expert:
        state_mean_expert, state_std_expert = compute_mean_std(dataset_expert["observations"], eps=1e-3)
        env_agent = wrap_env(env_agent, state_mean=state_mean_expert, state_std=state_std_expert)
    
    dataset_agent = d4rl.qlearning_dataset(env_agent) 
    #dataset_expert = d4rl.qlearning_dataset(env_expert) # (999999, 17)
    
    
    video_recorder = VideoRecorder(
            cfg.work_dir if cfg.save_video else None)
    
    if cfg.gw_include_actions_expert:
            dataset_expert = np.concatenate((dataset_expert,
                                             dataset_expert['actions']), axis=1)
    
    
    #maybe wrap env (reward normalization?)    
    replay_buffer_agent = ReplayBuffer(dataset_agent['observations'].shape[0],
                                env_agent.observation_space.shape,
                                env_agent.action_space.shape,
                                device)
    
    replay_buffer_expert = ReplayBuffer(
                                dataset_expert['observations'].shape[0],
                                env_expert.observation_space.shape,
                                env_expert.action_space.shape,
                                device)
    replay_buffer_agent.load_d4rl_dataset(dataset_agent)
    replay_buffer_expert.load_d4rl_dataset(dataset_expert)
    
    D_agent = torch.concat([replay_buffer_agent._states, replay_buffer_agent._next_states], dim=1)
    D_expert = torch.concat([replay_buffer_expert._states, replay_buffer_expert._next_states], dim=1)
    
    division_states = replay_buffer_expert._actions.shape[0] + 1

    # Computing 1M for GW is bad
    # until first done
    # 
    for step in trange(1): #max(D_agent.shape[0], D_expert.shape[0]) // cfg.gw_size_comp, ncols=200):
        if step == 0:
            stride = 0
        else:
            stride = 500
            
        batch_trajectories_agent = D_agent[stride : cfg.gw_size_comp, :]
        batch_trajectories_expert = D_expert[stride : cfg.gw_size_comp, :]
        
        batch_trajectories_agent = vmap(lambda x: x.to("cpu"), in_dims=0)(batch_trajectories_agent) #[b.to(cfg.device) for b in batch_trajectories_agent]
        batch_trajectories_expert = vmap(lambda x: x.to("cpu"), in_dims=0)(batch_trajectories_expert) #[b.to(cfg.device) for b in batch_trajectories_expert]
        
        T_init = compute_initialization(batch_trajectories_expert,
                                        batch_trajectories_agent,
                                        cfg.metric_expert,
                                        cfg.metric_agent,
                                        cfg.entropic,
                                        cfg.sinkhorn_reg) # GW
        
        #Write function to align best pairs based on T_init
        aligned_states = align_pairs(T_init, \
                                        batch_trajectories_agent)
        loss = batch_trajectories_expert - aligned_states
        
        '''if step != 0:
            merged_states = merged_states[:, :division_states]
            merged_states = torch.concat([merged_states, aligned_states], dim=1)
        else:
            #tuple
            merged_states = torch.concat([batch_trajectories_expert, aligned_states], dim=1)'''
        
        # compute embeddings of initial datasets, then concat based on 
        # T * u справа умножить на матрицу от эксперта (барицентрическая проекция) потом сравниваем с экспертом
        
        stride = cfg.gw_size_comp
        cfg.gw_size_comp += 2000
        
        
        
        
        
        
        
        
        
        
    









if __name__ == "__main__":
    entry()

