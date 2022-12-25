import pyrallis
import numpy as np
import ot

# import gymnasium as gym
import d4rl
import gym
from dataclasses import field, dataclass, asdict
import typing as tp
import pickle as pkl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
import uuid
from typing import Optional, Callable
from functorch import vmap
from test_encoder import Encoder
from utils_ot import euclidean_distance


device = torch.device("cuda")
TensorBatch = tp.List[torch.Tensor]

from utils_ot import set_seed, ReplayBuffer, compute_initialization, align_pairs
from video import VideoRecorder
from tqdm import trange


def compute_mean_std(
    states: np.ndarray, eps: float
) -> tp.Tuple[np.ndarray, np.ndarray]:
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
    agent: str = "walker2d-random-v2"  # Try to imitate jumps
    logger: str = field(default="wandb")
    seed: int = field(default=42)
    save_video: bool = field(default=True)
    work_dir: str = "expr/"
    delta: int = 1  # window size to check for next states
    num_train_steps: int = 1e6
    eval_steps: int = 5000
    batch_size: int = 100  # trajectory length
    optimizer: optim.Optimizer = field(default="Adam")

    # Wandb logger params
    project: str = "CILOT"
    group: str = "CILOT"
    name: str = "CILOT_VICreg"
    exper_name: str = "pendulum"

    # replay buffer
    capacity: int = 1e6

    # GW params
    gw_include_actions_expert: bool = field(default=False)
    metric_expert: str = "euclidean"
    metric_agent: str = "euclidean"
    norm_agent_with_expert: bool = field(default=True)
    entropic: bool = field(default=True)
    sinkhorn_reg: float = field(default=5e-3)
    gw_size_comp: int = field(default=1000)
    mode: str = "gw"  # gw/infoot

    def __post_init__(self):
        self.group = self.name + "expert_" + self.exper_name
        self.video_recorder = VideoRecorder(self.work_dir if self.save_video else None)
        print(f"Optimal transport T")


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

    # Process logger
    if cfg.logger == "wandb":
        pass
        # load_wandb(asdict(cfg))

    env_agent = gym.make(cfg.agent, render_mode="rgb_array_list")
    env_expert = gym.make(cfg.expert)

    # set_seed(cfg.seed, env_agent)
    device = torch.device(cfg.device)

    dataset_expert = d4rl.qlearning_dataset(
        env_expert
    )  # (999999, 17), 1000 steps each trajectory

    # normalize rewards of agent ?
    if cfg.norm_agent_with_expert:
        state_mean_expert, state_std_expert = compute_mean_std(
            dataset_expert["observations"], eps=1e-3
        )
        env_agent = wrap_env(
            env_agent, state_mean=state_mean_expert, state_std=state_std_expert
        )

    dataset_agent = d4rl.qlearning_dataset(env_agent)

    video_recorder = VideoRecorder(cfg.work_dir if cfg.save_video else None)

    if cfg.gw_include_actions_expert:
        dataset_expert = np.concatenate(
            (dataset_expert, dataset_expert["actions"]), axis=1
        )

    replay_buffer_agent = ReplayBuffer(
        dataset_agent["observations"].shape[0],
        env_agent.observation_space.shape,
        env_agent.action_space.shape,
        device,
    )

    replay_buffer_expert = ReplayBuffer(
        dataset_expert["observations"].shape[0],
        env_expert.observation_space.shape,
        env_expert.action_space.shape,
        device,
    )

    replay_buffer_agent.load_d4rl_dataset(dataset_agent)
    replay_buffer_expert.load_d4rl_dataset(dataset_expert)

    # FIX: Take trajectories only up to terminal point
    D_agent = torch.concat(
        [replay_buffer_agent._states, replay_buffer_agent._next_states], dim=1
    )
    D_expert = torch.concat(
        [replay_buffer_expert._states, replay_buffer_expert._next_states], dim=1
    )

    if cfg.optimizer == "Adam":
        optimizer = optim.Adam(cfg.encoder.parameters(), lr=3e-4)

    encoder = Encoder(D_agent.shape[1])

    for step in trange(1):
        print("Computing T_init")

        batch_trajectories_expert = D_expert[
            step : cfg.gw_size_comp, :
        ]  # take first trajectory of expert

        batch_trajectories_expert = vmap(lambda x: x.to("cpu"), in_dims=0)(
            batch_trajectories_expert
        )
        trajectory_slice = 0

        for t_bar in range(1, 4):  # take 3 trajectories from
            batch_trajectories_agent = D_agent[
                trajectory_slice : cfg.gw_size_comp * t_bar, :
            ]
            trajectory_slice = cfg.gw_size_comp

            batch_trajectories_agent = vmap(lambda x: x.to("cpu"), in_dims=0)(
                batch_trajectories_expert
            )
            if t_bar == 1:
                merged_trajectories_agent = batch_trajectories_agent
            else:
                merged_trajectories_agent = torch.vstack(
                    [merged_trajectories_agent, batch_trajectories_agent]
                )

        agent_traj_embeds = encoder(
            merged_trajectories_agent
        )  # embedding of merged_traj
        M = torch.nn.pairwise_distances(batch_trajectories_expert, agent_traj_embeds)

        # compute T init between first trajectory of agent and several traj of agent
        T_init = compute_initialization(
            cfg.mode,
            batch_trajectories_expert,
            merged_trajectories_agent,
            cfg.metric_expert,
            cfg.metric_agent,
            cfg.entropic,
            cfg.sinkhorn_reg,
            device=device,
        )  # GW

        loss = torch.sum(T_init * M)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    entry()
