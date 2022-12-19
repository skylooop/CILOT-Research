import collections
from abc import ABC, abstractmethod
from typing import Optional
import os
import d4rl
import gym
import numpy as np
from tqdm import tqdm
import ot


Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])



class RewardsExpert(ABC):

    def compute_rewards(self, observations: np.ndarray, next_observations: np.ndarray, dones_float: np.ndarray) -> np.ndarray:
        assert dones_float[-1] > 0.5
        i0 = 0
        rewards = []
        for i1 in tqdm(np.where(dones_float > 0.5)[0].tolist()):
            rewards.append(self.compute_rewards_one_episode(observations[i0:i1+1], next_observations[i0:i1+1]))
            i0 = i1+1

        return np.concatenate(rewards)

    @abstractmethod
    def compute_rewards_one_episode(self, observations: np.ndarray, next_observations: np.ndarray) -> np.ndarray:
        pass


class RewardsScaler(ABC):
    @abstractmethod
    def init(self, rewards: np.ndarray) -> None:
       pass

    @abstractmethod
    def scale(self, rewards: np.ndarray) -> np.ndarray:
        pass


class Dataset(object):
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 size: int):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx])


class D4RLDataset(Dataset):
    def __init__(self,
                 env: gym.Env,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5):
        if os.path.isfile(f"./tmp/dataset_{env.spec.id}.npz"):
            dataset = dict(np.load(f"./tmp/dataset_{env.spec.id}.npz"))
        else:
            dataset = d4rl.qlearning_dataset(env)
            np.savez(f"./tmp/dataset_{env.spec.id}.npz", **dataset)
            print("saved")

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset['observations'][i + 1] -
                              dataset['next_observations'][i]
                              ) > 1e-6 or dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        super().__init__(dataset['observations'].astype(np.float32),
                         actions=dataset['actions'].astype(np.float32),
                         rewards=dataset['rewards'].astype(np.float32),
                         masks=1.0 - dataset['terminals'].astype(np.float32),
                         dones_float=dones_float.astype(np.float32),
                         next_observations=dataset['next_observations'].astype(
                             np.float32),
                         size=len(dataset['observations']))


class ReplayBufferWithDynamicRewards(Dataset):

    def __init__(self, observation_space: gym.spaces.Box, action_dim: int, capacity: int,
                 rewards_scaler: RewardsScaler,
                 rewards_expert: RewardsExpert):

        observations = np.empty((capacity, *observation_space.shape),
                                dtype=observation_space.dtype)
        actions = np.empty((capacity, action_dim), dtype=np.float32)
        rewards = np.empty((capacity,), dtype=np.float32)
        masks = np.empty((capacity,), dtype=np.float32)
        dones_float = np.empty((capacity,), dtype=np.float32)
        next_observations = np.empty((capacity, *observation_space.shape),
                                     dtype=observation_space.dtype)
        super().__init__(observations=observations,
                         actions=actions,
                         rewards=rewards,
                         masks=masks,
                         dones_float=dones_float,
                         next_observations=next_observations,
                         size=0)

        self.size = 0

        self.insert_index = 0
        self.capacity = capacity


        self.observations_cur = []
        self.actions_cur = []
        self.masks_cur = []
        self.dones_float_cur = []
        self.next_observations_cur = []

        self.scaler = rewards_scaler
        self.expert = rewards_expert

    def initialize_with_dataset(self, dataset: D4RLDataset, num_samples: int):

        assert self.capacity > num_samples

        for i in range(num_samples - 1, len(dataset.observations)):
            if dataset.dones_float[i-1] == 1.0:
                self.observations[0:i] = dataset.observations[0:i]
                self.actions[0:i] = dataset.actions[0:i]
                self.masks[0:i] = dataset.masks[0:i]
                self.dones_float[0:i] = dataset.dones_float[0:i]
                self.next_observations[0:i] = dataset.next_observations[0:i]

                self.rewards[0:i] = self.expert.compute_rewards(
                    self.observations[0:i],
                    self.next_observations[0:i],
                    self.dones_float[0:i]
                )

                self.insert_index = i
                self.size = i
                break

        self.scaler.init(self.rewards[:self.size])
        self.rewards[:self.size] = self.scaler.scale(self.rewards[:self.size])
        print("rewards:", np.min(self.rewards[:self.size]), np.mean(self.rewards[:self.size]), np.max(self.rewards[:self.size]))

    def insert(self, observation: np.ndarray, action: np.ndarray,
               mask: float, done_float: float, next_observation: np.ndarray):

        self.observations_cur.append(observation)
        self.actions_cur.append(action)
        self.masks_cur.append(mask)
        self.dones_float_cur.append(done_float)
        self.next_observations_cur.append(next_observation)

        if done_float == 1.0:

            i0 = self.insert_index
            i1 = min(i0 + len(self.observations_cur), self.capacity)
            obs = np.stack(self.observations_cur)
            next_obs = np.stack(self.next_observations_cur)

            self.observations[i0:i1] = obs[:i1-i0]
            self.actions[i0:i1] = np.stack(self.actions_cur)[:i1-i0]
            self.masks[i0:i1] = np.stack(self.masks_cur)[:i1-i0]
            self.dones_float[i0:i1] = np.stack(self.dones_float_cur)[:i1-i0]
            self.next_observations[i0:i1] = next_obs[:i1-i0]

            self.rewards[i0:i1] = self.scaler.scale(
                self.expert.compute_rewards_one_episode(obs, next_obs)
            )[:i1-i0]

            self.insert_index = i1 % self.capacity
            self.size = min(self.size + (i1 - i0), self.capacity)

            self.observations_cur = []
            self.actions_cur = []
            self.masks_cur = []
            self.dones_float_cur = []
            self.next_observations_cur = []