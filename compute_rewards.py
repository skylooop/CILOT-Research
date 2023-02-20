import collections
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from flax.training import train_state
from jax import numpy as jnp
from ott.geometry import pointcloud, costs
from ott.geometry.costs import CostFn
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from ott.solvers.linear.sinkhorn_lr import LRSinkhorn
from sklearn import preprocessing
from tqdm import tqdm

from optimization import uptade_encoder, embed
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
        for i1 in tqdm(np.where(dones_float > 0.5)[0].tolist()):
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
        ot_prob = linear_problem.LinearProblem(geom, a, b, tau_a=0.8, tau_b=0.8)
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
        expert_data: ExpertData,
        encoder_class: train_state.TrainState,
        cost_fn: CostFn = costs.Euclidean(),
        epsilon=0.01,
    ):
        self.expert_states_pair = np.concatenate(
            [expert_data.observations, expert_data.next_observations], axis=1
        )
        self.cost_fn = cost_fn
        self.epsilon = epsilon

        self.preproc = Preprocessor()
        self.states_pair_buffer = ListBuffer(n=50)
        
        self.encoder_class = encoder_class
        
    def optim_embed(self) -> None:

        (
            observations,
            next_observations,
            transport_matrix,
        ) = self.states_pair_buffer.sample()

        self.encoder_class, loss = uptade_encoder(self.encoder_class, observations, next_observations, self.expert_states_pair, transport_matrix)
        
    def warmup(self) -> None:
        self.optim_embed()

    def compute_rewards_one_episode(
        self, observations: np.ndarray, next_observations: np.ndarray
    ) -> np.ndarray:
        embeded_observations, embeded_next_observations = embed(self.encoder_class, observations, next_observations)

        states_pair = np.concatenate(
            [embeded_observations, embeded_next_observations], axis=1
        )

        self.preproc.fit(states_pair)

        x = jnp.asarray(self.preproc.transform(states_pair))
        y = jnp.asarray(self.preproc.transform(self.expert_states_pair))

        a = jnp.ones((x.shape[0],)) / x.shape[0]
        b = jnp.ones((y.shape[0],)) / y.shape[0]

        geom = pointcloud.PointCloud(x, y, epsilon=self.epsilon, cost_fn=self.cost_fn)
        ot_prob = linear_problem.LinearProblem(geom, a, b, tau_a=0.95, tau_b=0.95)
        solver = sinkhorn.Sinkhorn()

        ot_sink = solver(ot_prob)
        transp_cost = jnp.sum(ot_sink.matrix * geom.cost_matrix, axis=1)
        rewards = -transp_cost
        
        self.states_pair_buffer.append(
            (observations, next_observations, np.asarray(ot_sink.matrix))
        )
        
        self.warmup()
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
    def apply(self, dataset: D4RLDataset, type: str, encoder_class: train_state.TrainState) -> OTRewardsExpert:
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

        
        if type == "CrossDomain":
            return OTRewardsExpertCrossDomain(
                ExpertData(
                    observations=best_traj_states, next_observations=best_traj_next_states
            ), encoder_class=encoder_class)
        else: 
            return OTRewardsExpert(
                ExpertData(
                    observations=best_traj_states, next_observations=best_traj_next_states
                )
            )


class OTRewardsExpertFactoryCrossDomain(OTRewardsExpertFactory): #OTRewardsExpertCrossDomain
    def apply(
        self,
        dataset: D4RLDataset,
        encoder_class,
        type="CrossDomain"
    ) -> OTRewardsExpert:

        expert = super().apply(dataset, type, encoder_class)
        
        return expert
        #return OTRewardsExpertCrossDomain(
        #    expert_data=expert.expert_data, encoder_class=encoder_class
        #)
