import collections
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from jax import numpy as jnp
from ott.geometry import pointcloud, costs
from ott.geometry.costs import CostFn
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from tqdm import tqdm

from dynamic_replay_buffer import D4RLDataset, RewardsScaler, RewardsExpert

ExpertData = collections.namedtuple('ExpertData', ['observations', 'next_observations'])


class ExpRewardsScaler(RewardsScaler):
    def init(self, rewards: np.ndarray):
        self.min = np.quantile(np.abs(rewards).reshape(-1), 0.01)
        self.max = np.quantile(np.abs(rewards).reshape(-1), 0.99)

    def scale(self, rewards: np.ndarray):
        return 5 * np.exp((rewards - self.min) / self.max) - 2



class OTRewardsExpert(RewardsExpert):

    def __init__(self, expert_data: ExpertData, cost_fn: CostFn = costs.Euclidean(), epsilon=0.01):
        self.expert_states_pair = np.concatenate([expert_data.observations, expert_data.next_observations], axis=1)
        self.cost_fn = cost_fn
        self.epsilon = epsilon

    def compute_rewards_one_episode(self, observations: np.ndarray, next_observations: np.ndarray) -> np.ndarray:
        states_pair = np.concatenate([observations, next_observations], axis=1)

        x = jnp.asarray(states_pair)
        y = jnp.asarray(self.expert_states_pair)
        a = jnp.ones((x.shape[0],)) / x.shape[0]
        b = jnp.ones((y.shape[0],)) / y.shape[0]

        geom = pointcloud.PointCloud(x, y, epsilon=self.epsilon, cost_fn=self.cost_fn)
        ot_prob = linear_problem.LinearProblem(geom, a, b)
        solver = sinkhorn.Sinkhorn()
        ot_sink = solver(ot_prob)

        transp_cost = jnp.sum(ot_sink.matrix * geom.cost_matrix, axis=1)
        rewards = -transp_cost

        return np.asarray(rewards)


def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                          dones_float[i], next_observations[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


class OTRewardsExpertFactory:

    def apply(self, dataset: D4RLDataset) -> OTRewardsExpert:
        trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                        dataset.rewards, dataset.masks,
                                        dataset.dones_float,
                                        dataset.next_observations)

        def compute_returns(traj):
            episode_return = 0
            for _, _, rew, _, _, _ in traj:
                episode_return += rew

            return episode_return

        trajs.sort(key=compute_returns)

        best_traj = trajs[-1]
        best_traj_states = np.stack([el[0] for el in best_traj])
        best_traj_next_states = np.stack([el[-1] for el in best_traj])

        return OTRewardsExpert(
            ExpertData(observations=best_traj_states, next_observations=best_traj_next_states))





