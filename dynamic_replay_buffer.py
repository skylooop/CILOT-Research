import collections
import random
from typing import Optional
import os
from absl import flags
FLAGS = flags.FLAGS
import gym
import numpy as np


from agent.iql.dataset_utils import D4RLDataset
from compute_rewards import RewardsScaler, RewardsExpert

Batch = collections.namedtuple(
    "Batch", ["observations", "actions", "rewards", "masks", "next_observations"]
)


class Dataset(object):
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        masks: np.ndarray,
        dones_float: np.ndarray,
        next_observations: np.ndarray,
        size: int,
    ):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(
            observations=self.observations[indx],
            actions=self.actions[indx],
            rewards=self.rewards[indx],
            masks=self.masks[indx],
            next_observations=self.next_observations[indx],
        )

    def sample_episode(self):

        sep = [-1] + np.where(self.dones_float > 0.5)[0].tolist()
        index = random.randint(0, len(sep) - 1)
        i0 = sep[index] + 1
        i1 = sep[index + 1] + 1

        assert i1 > i0

        return Batch(
            observations=self.observations[i0:i1],
            actions=self.actions[i0:i1],
            rewards=self.rewards[i0:i1],
            masks=self.masks[i0:i1],
            next_observations=self.next_observations[i0:i1]
        )


class ReplayBuffer(Dataset):
    def __init__(
        self, observation_space: gym.spaces.Box, action_dim: int, capacity: int
    ):

        observations = np.empty(
            (capacity, *observation_space.shape), dtype=observation_space.dtype
        )
        actions = np.empty((capacity, action_dim), dtype=np.float32)
        rewards = np.empty((capacity,), dtype=np.float32)
        masks = np.empty((capacity,), dtype=np.float32)
        dones_float = np.empty((capacity,), dtype=np.float32)
        next_observations = np.empty(
            (capacity, *observation_space.shape), dtype=observation_space.dtype
        )

        super().__init__(
            observations=observations,
            actions=actions,
            rewards=rewards,
            masks=masks,
            dones_float=dones_float,
            next_observations=next_observations,
            size=0,
        )

        self.size = 0

        self.insert_index = 0
        self.capacity = capacity

    def insert(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        mask: float,
        done_float: float,
        next_observation: np.ndarray,
    ):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = mask
        self.dones_float[self.insert_index] = done_float
        self.next_observations[self.insert_index] = next_observation

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
        
class D4RLDatasetWithOTRewards:
    @staticmethod
    def save(dataset: D4RLDataset, rewards_expert: RewardsExpert, num_samples: int):

        sep = np.where(dataset.dones_float > 0.5)[0].tolist()
        index = sep[0] + 1
        for i in sep:
            index = i + 1
            if index > num_samples:
                break
                
        # self.size = index
        data = {}
            
        data["observations"] = dataset.observations[0:index]
        data["next_observations"] = dataset.next_observations[0:index]
        data["actions"] = dataset.actions[0:index]
        data["masks"] = dataset.masks[0:index]
        data["dones_float"] = dataset.dones_float[0:index]

        data["rewards"] = rewards_expert.compute_rewards(
            data["observations"],
            data["next_observations"],
            data["dones_float"],
        )
        assert data["rewards"].shape[0] == data["observations"].shape[0]

        print(f"Saving dataset with {index} from agent dataset")
        print(f"dataset_{FLAGS.env_name}_{FLAGS.expert_env_name}_ot_rewards.npz")

        np.savez(
            os.path.join(FLAGS.path_to_save_env, f"dataset_{FLAGS.env_name}_{FLAGS.expert_env_name}_ot_rewards.npz"),
            **data,
        )

    @staticmethod
    def load():
        dataset = dict(
            np.load(
                os.path.join(FLAGS.path_to_save_env, f"dataset_{FLAGS.env_name}_{FLAGS.expert_env_name}_ot_rewards.npz"),
            )
        )

        return Dataset(
            observations=dataset["observations"].astype(np.float32),
            actions=dataset["actions"].astype(np.float32),
            rewards=dataset["rewards"].astype(np.float32),
            masks=dataset["masks"].astype(np.float32),
            dones_float=dataset["dones_float"].astype(np.float32),
            next_observations=dataset["next_observations"].astype(np.float32),
            size=len(dataset["observations"])
        )


class ReplayBufferWithDynamicRewards(ReplayBuffer):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_dim: int,
        capacity: int,
        rewards_scaler: RewardsScaler,
        rewards_expert: RewardsExpert,
    ):
        """
        Args:
            observation_space (gym.spaces.Box): Agent env observation space dim
            action_dim (int): Agent action space dim
            capacity (int): size of buffer
        """
        super().__init__(observation_space, action_dim, capacity)

        self.observations_cur = []
        self.actions_cur = []
        self.masks_cur = []
        self.dones_float_cur = []
        self.next_observations_cur = []

        self.scaler = rewards_scaler
        self.expert = rewards_expert

    def initialize_with_dataset(self, dataset: Dataset, num_samples: int):

        assert self.capacity > num_samples
        assert dataset.size >= num_samples

        i = num_samples

        self.observations[0:i] = dataset.observations[0:i]
        self.actions[0:i] = dataset.actions[0:i]
        self.masks[0:i] = dataset.masks[0:i]
        self.dones_float[0:i] = dataset.dones_float[0:i]
        self.next_observations[0:i] = dataset.next_observations[0:i]
        self.rewards[0:i] = dataset.rewards[0:i]
        self.insert_index = i
        self.size = i

        self.scaler.init(self.rewards[: self.size])
        self.rewards[: self.size] = self.scaler.scale(self.rewards[: self.size])
        print(
            "rewards:",
            np.min(self.rewards[: self.size]),
            np.mean(self.rewards[: self.size]),
            np.max(self.rewards[: self.size]),
        )

    def insert(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        mask: float,
        done_float: float,
        next_observation: np.ndarray,
    ):

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

            self.observations[i0:i1] = obs[: i1 - i0]
            self.actions[i0:i1] = np.stack(self.actions_cur)[: i1 - i0]
            self.masks[i0:i1] = np.stack(self.masks_cur)[: i1 - i0]
            self.dones_float[i0:i1] = np.stack(self.dones_float_cur)[: i1 - i0]
            self.next_observations[i0:i1] = next_obs[: i1 - i0]
            self.rewards[i0:i1] = self.scaler.scale(
                self.expert.compute_rewards_one_episode(obs, next_obs)
            )[ :i1 - i0]

            self.insert_index = i1 % self.capacity
            self.size = min(self.size + (i1 - i0), self.capacity)

            self.observations_cur = []
            self.actions_cur = []
            self.masks_cur = []
            self.dones_float_cur = []
            self.next_observations_cur = []
