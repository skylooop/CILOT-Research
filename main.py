import pyrallis
import numpy as np
import ot
#import gymnasium as gym
import d4rl
import gym
from dataclasses import field, dataclass, asdict
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import uuid
from typing import Optional, Callable

#Utils import
from utils import set_seed
from video import VideoRecorder


@dataclass
class CILOT_cfg:
    device: str = "cuda"
    expert: str = "hopper-medium-expert-v2"
    agent: str = "walker2d-random-v2"
    logger: str = field(default="wandb")
    seed: int = field(default=42)
    save_video: bool = field(default=True)
    work_dir: str = "expr/"
    
    #Wandb logger params
    project: str = "CILOT"
    group: str = "CILOT"
    name: str = "CILOT_VICreg"
    exper_name: str = "pendulum"
    
    
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
    
    dataset_agent = d4rl.qlearning_dataset(env_agent) # (999999, 17)
    dataset_expert = d4rl.qlearning_dataset(env_expert) # ()
    
    video_recorder = VideoRecorder(
            cfg.work_dir if cfg.save_video else None)
    
    if cfg.gw_include_actions_expert:
            traj_expert = np.concatenate((traj_expert, dataset_expert['actions']), axis=1)
    
    #maybe wrape env (reward normalization?)
    
    agent_obs_dim = env_agent.observation_space.shape[0]
    agent_action_dim = env_agent.action_space.shape[0]
    
    replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                 self.env.action_space.shape,
                                 int(cfg.replay_buffer_capacity),
                                 self.device, cfg)
    









if __name__ == "__main__":
    entry()

