import collections
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from jax import numpy as jnp
from ott.geometry import pointcloud, costs
from ott.geometry.costs import CostFn
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from ott.solvers.linear.sinkhorn_lr import LRSinkhorn
from sklearn import preprocessing
from tqdm import tqdm
import torch

import typing as tp

import torch.nn as nn


from absl import flags

FLAGS = flags.FLAGS

from agent.iql.dataset_utils import D4RLDataset

ExpertData = collections.namedtuple("ExpertData", ["observations", "next_observations"])
AgentData = collections.namedtuple("AgentData", ["observations_shape"])


class ListBuffer:
    def __init__(self, n: int) -> None:
        self.data = []
        self.n = n

    def append(self, triple: tp.Tuple[np.ndarray, np.ndarray, np.ndarray]):
        self.data.append(triple)

        if len(self.data) > self.n:
            self.data = self.data[1:]

    def sample(self):
        return self.data[np.random.randint(0, len(self.data))]


class RewardsScaler(ABC):
    @abstractmethod
    def init(self, rewards: np.ndarray) -> None:
        pass

    @abstractmethod
    def scale(self, rewards: np.ndarray) -> np.ndarray:
        pass


class ExpRewardsScaler(RewardsScaler):
    def init(self, rewards: np.ndarray):
        self.min = np.quantile(np.abs(rewards).reshape(-1), 0.01)
        self.max = np.quantile(np.abs(rewards).reshape(-1), 0.99)

    def scale(self, rewards: np.ndarray):
        # From paper
        return 5 * np.exp(rewards / self.max) - 2.5


class RewardsExpert(ABC):
    def compute_rewards(
        self,
        observations: np.ndarray,
        next_observations: np.ndarray,
        dones_float: np.ndarray,
    ) -> np.ndarray:

        assert dones_float[-1] > 0.5
        i0 = 0
        rewards = []

        # Each 1k step - terminate
        for i1 in tqdm(np.where(dones_float > 0.5)[0].tolist()):  #
            rewards.append(
                self.compute_rewards_one_episode(
                    observations[i0 : i1 + 1], next_observations[i0 : i1 + 1]
                )
            )
            i0 = i1 + 1

        return np.concatenate(rewards)

    @abstractmethod
    def warmup(self) -> None:
        pass

    @abstractmethod
    def compute_rewards_one_episode(
        self, observations: np.ndarray, next_observations: np.ndarray
    ) -> np.ndarray:
        pass


class Preprocessor:
    def __init__(self, partial_updates=True, update_preprocessor_every_episode=1):
        self.preprocessor = preprocessing.StandardScaler()
        self.update_preprocessor_every_episode = update_preprocessor_every_episode
        self.partial_updates = partial_updates
        self.e = 0
        self.enabled = True

    def fit(self, observations):
        if self.e % self.update_preprocessor_every_episode == 0 and self.enabled:
            self.e += 1
            if self.partial_updates:
                self.preprocessor.partial_fit(observations)
            else:
                self.preprocessor.fit(observations)

    def transform(self, observations):
        if self.enabled:
            return self.preprocessor.transform(observations)
        else:
            return observations


class OTRewardsExpert(RewardsExpert):
    def __init__(
        self, expert_data: ExpertData, cost_fn: CostFn = costs.Euclidean(), epsilon=0.01
    ):
        self.expert_states_pair = np.concatenate(
            [expert_data.observations, expert_data.next_observations], axis=1
        )
        self.cost_fn = cost_fn
        self.epsilon = epsilon
        self.expert_data = expert_data

        self.preproc = Preprocessor()

    def compute_rewards_one_episode(
        self, observations: np.ndarray, next_observations: np.ndarray
    ) -> np.ndarray:

        states_pair = np.concatenate([observations, next_observations], axis=1)

        self.preproc.fit(states_pair)
        x = jnp.asarray(self.preproc.transform(states_pair))
        y = jnp.asarray(self.preproc.transform(self.expert_states_pair))

        a = jnp.ones((x.shape[0],)) / x.shape[0]
        b = jnp.ones((y.shape[0],)) / y.shape[0]

        geom = pointcloud.PointCloud(x, y, epsilon=self.epsilon, cost_fn=self.cost_fn)
        ot_prob = linear_problem.LinearProblem(geom, a, b)
        solver = sinkhorn.Sinkhorn()

        ot_sink = solver(ot_prob)
        transp_cost = jnp.sum(ot_sink.matrix * geom.cost_matrix, axis=1)
        rewards = -transp_cost

        return np.asarray(rewards)

    def warmup(self) -> None:
        pass


class OTRewardsExpertCrossDomain(RewardsExpert):
    def __init__(
        self,
        params,
        expert_data: ExpertData,
        embed_model: nn.Module,
        opt_fn: tp.Callable[[torch.Tensor], None],
        cost_fn: CostFn = costs.Euclidean(),
        epsilon=0.01,
    ):
        self.expert_states_pair = np.concatenate(
            [expert_data.observations, expert_data.next_observations], axis=1
        )
        self.cost_fn = cost_fn
        self.opt_fn = opt_fn
        self.epsilon = epsilon

        self.preproc = Preprocessor()
        self.states_pair_buffer = ListBuffer(n=50)

        # Init model
        self.model = embed_model
        self.params = params
        
    def cost_matrix_fn(self, states_pair):
        dist = jnp.sum(jnp.sqrt(jnp.power(
            
                states_pair[:, None]
                - self.expert_states_pair[
                    None,
                ]
            ,jnp.array(2))), axis=-1)
        return dist
    
    def loss_fn(self, torch_t_matrix, torch_cost):
        loss = (torch_t_matrix * torch_cost).sum()
        
        return loss
    
    def optim_embed(self) -> None:

        (
            observations,
            next_observations,
            transport_matrix,
        ) = self.states_pair_buffer.sample()
        
        '''
        embed_obs = self.model(
            torch.from_numpy(observations).to(torch.device("cuda:1"))
        )
        next_embed_obs = self.model(
            torch.from_numpy(next_observations).to(torch.device("cuda:1"))
        )
        states_pair_torch = torch.cat([embed_obs, next_embed_obs], 1)

        torch_t_matrix = torch.from_numpy(transport_matrix).to(torch.device("cuda:1"))
        torch_cost = self.torch_cost_matrix(states_pair_torch)
        loss = (torch_t_matrix * torch_cost).sum() 
        '''
        
        embed_obs, obs_updated_params = self.model.apply(self.params, observations, mutable=['batch_stats'])
        next_embed_obs, next_updated_params = self.model.apply(self.params, next_observations, mutable=['batch_stats'])
        states_pair = jnp.concatenate([embed_obs, next_embed_obs], axis=1)
        cost_matrix = self.cost_matrix_fn(states_pair)
        
        loss = self.loss_fn(transport_matrix, cost_matrix)
        self.opt_fn(loss_fn=loss, model=self.model, obs=embed_obs)

    def warmup(self) -> None:
        self.optim_embed()

    def compute_rewards_one_episode(
        self, observations: np.ndarray, next_observations: np.ndarray
    ) -> np.ndarray:
        embeded_observations = self.model.apply(self.params, observations, mutable=['batch_stats'])
        embeded_next_observations = self.model.apply(self.params, next_observations, mutable=['batch_stats'])

        states_pair = np.concatenate(
            [embeded_observations[0], embeded_next_observations[0]], axis=1
        )

        self.preproc.fit(states_pair)

        x = jnp.asarray(self.preproc.transform(states_pair))
        y = jnp.asarray(self.preproc.transform(self.expert_states_pair))

        a = jnp.ones((x.shape[0],)) / x.shape[0]
        b = jnp.ones((y.shape[0],)) / y.shape[0]

        geom = pointcloud.PointCloud(x, y, epsilon=self.epsilon, cost_fn=self.cost_fn)
        ot_prob = linear_problem.LinearProblem(geom, a, b)
        solver = sinkhorn.Sinkhorn()

        ot_sink = solver(ot_prob)
        transp_cost = jnp.sum(ot_sink.matrix * geom.cost_matrix, axis=1)
        rewards = -transp_cost

        # ((999, obs_shape), (999, obs_shape), (999, 999))
        self.states_pair_buffer.append(
            (observations, next_observations, np.asarray(ot_sink.matrix))
        )

        return np.asarray(rewards)


def split_into_trajectories(
    observations, actions, rewards, masks, dones_float, next_observations
):
    trajs = [[]]

    print("Splitting into trajectories based on terminal states")
    for i in tqdm(range(len(observations))):
        trajs[-1].append(
            (
                observations[i],
                actions[i],
                rewards[i],
                masks[i],
                dones_float[i],
                next_observations[i],
            )
        )
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


class OTRewardsExpertFactory:
    def apply(self, dataset: D4RLDataset) -> OTRewardsExpert:
        trajs = split_into_trajectories(
            dataset.observations,
            dataset.actions,
            dataset.rewards,
            dataset.masks,
            dataset.dones_float,
            dataset.next_observations,
        )

        def compute_returns(traj):
            episode_return = 0
            for _, _, rew, _, _, _ in traj:
                episode_return += rew

            return episode_return

        # Choose best trajectories
        trajs.sort(key=compute_returns)
        best_traj = trajs[-1]

        # Concat observations
        best_traj_states = np.stack([el[0] for el in best_traj])
        best_traj_next_states = np.stack([el[-1] for el in best_traj])

        return OTRewardsExpert(
            ExpertData(
                observations=best_traj_states, next_observations=best_traj_next_states
            )
        )


class OTRewardsExpertFactoryCrossDomain(OTRewardsExpertFactory):
    def apply(
        self,
        params,
        dataset: D4RLDataset,
        embed_model: torch.nn.Module,
        opt_fn: tp.Callable[[torch.Tensor], None],
    ) -> OTRewardsExpert:

        expert = super().apply(dataset)
        return OTRewardsExpertCrossDomain(
            params=params, expert_data=expert.expert_data, embed_model=embed_model, opt_fn=opt_fn
        )
