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
from utils import set_seed, ReplayBuffer, compute_initialization
from video import VideoRecorder
from tqdm import trange

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
    #normalize rewards of agent ? 
    
    dataset_agent = d4rl.qlearning_dataset(env_agent) 
    dataset_expert = d4rl.qlearning_dataset(env_expert) # (999999, 17)
    
    video_recorder = VideoRecorder(
            cfg.work_dir if cfg.save_video else None)
    
    if cfg.gw_include_actions_expert:
            dataset_expert = np.concatenate((dataset_expert, dataset_expert['actions']), axis=1)
    
    
    #maybe wrap env (reward normalization?)    
    replay_buffer_agent = ReplayBuffer(env_agent.observation_space.shape,
                                 env_agent.action_space.shape,
                                 int(cfg.capacity),
                                 device)
    
    replay_buffer_expert = ReplayBuffer(env_expert.observation_space.shape,
                                 env_expert.action_space.shape,
                                 int(cfg.capacity),
                                 device)
    replay_buffer_agent.load_d4rl_dataset(dataset_agent)
    replay_buffer_expert.load_d4rl_dataset(dataset_expert)
    
    D_agent = torch.concat([replay_buffer_agent._states, replay_buffer_agent._next_states])
    D_expert = torch.concat([replay_buffer_expert._states, replay_buffer_expert._next_states])
    
    # Computing 1M for GW is bad
    for step in trange(cfg.num_train_steps, ncols=200):
        batch_trajectories_agent = D_agent[step:cfg.batch_size, :] #????
        batch_trajectories_expert = D_expert[step:cfg.batch_size, :]
        
        batch_trajectories_agent = vmap(lambda x: x.to(device), in_dims=0)(batch_trajectories_agent) #[b.to(cfg.device) for b in batch_trajectories_agent]
        batch_trajectories_expert = vmap(lambda x: x.to(device), in_dims=0)(batch_trajectories_expert) #[b.to(cfg.device) for b in batch_trajectories_expert]
        
        T_init = compute_initialization(batch_trajectories_expert,
                                        batch_trajectories_agent)
        
        
        
        
        
        
        
        
        
    









if __name__ == "__main__":
    entry()

