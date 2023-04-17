import collections
from abc import ABC, abstractmethod
from typing import List, Tuple, Any
import os
from absl import flags

from optimization import create_encoder, embed, update_encoder

FLAGS = flags.FLAGS
import jax
import numpy as np
import typing as tp
from jax import numpy as jnp
from ott.geometry import pointcloud, costs
from ott.geometry.costs import CostFn
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from ott.solvers.linear.sinkhorn_lr import LRSinkhorn
from sklearn import preprocessing
from tqdm import tqdm
from agent.iql.common import MLP, Model, Params, InfoDict
from dataset_utils import D4RLDataset, Dataset

ExpertData = collections.namedtuple('ExpertData', ['observations', 'next_observations'])


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
        return 5 * np.exp(rewards / self.max) - 2.5


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

    def warmup(self):
        pass


class Preprocessor:
    def __init__(self, partial_updates=True,
                 update_preprocessor_every_episode=1):
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

    def __init__(self, expert_data: ExpertData, cost_fn: CostFn = costs.Euclidean(), epsilon=0.01):
        self.expert_states_pair = jnp.asarray(np.concatenate([expert_data.observations, expert_data.next_observations], axis=1))
        self.cost_fn = cost_fn
        self.epsilon = epsilon
        self.expert_data = expert_data

        self.preproc = Preprocessor()

        dim = self.expert_states_pair.shape[1] // 2
        self.encoder = create_encoder(17, dim, lr=2e-4)

        self.ott = jax.jit(
            self._ott_computation
        )

    def _ott_computation(self, expert_traj, expert_weights, agent_traj, agent_weights):

        geom = pointcloud.PointCloud(agent_traj, expert_traj, epsilon=self.epsilon, cost_fn=self.cost_fn)
        ot_prob = linear_problem.LinearProblem(geom, b=expert_weights, a=agent_weights)
        solver = sinkhorn.Sinkhorn()

        ot_sink = solver(ot_prob)
        transp_cost = jnp.sum(ot_sink.matrix * geom.cost_matrix, axis=0)
        pseudo_rewards = -transp_cost

        return pseudo_rewards, ot_sink.matrix

    def warmup(self, observations, next_observations):
        embeded_agent_observations, embeded_agent_next_observations = embed(self.encoder, observations,
                                                                            next_observations)
        x = jnp.stack(
            jnp.concatenate((embeded_agent_observations, embeded_agent_next_observations), axis=-1))

        y = self.expert_states_pair
        a = jnp.ones((x.shape[0],)) / x.shape[0]
        b = jnp.ones((y.shape[0],)) / y.shape[0]

        _, matrix = self.ott(y, b, x, a)

        assert matrix.shape[0] == x.shape[0]

        self.encoder, loss = update_encoder(self.encoder, observations, next_observations,
                                                  y, matrix, cost_fn=self.cost_fn)

        return {"loss": np.asarray(loss)}


    def compute_rewards_one_episode(self, observations: np.ndarray, next_observations: np.ndarray) -> np.ndarray:
        # states_pair = np.concatenate([observations, next_observations], axis=1)
        embeded_agent_observations, embeded_agent_next_observations = embed(self.encoder, observations, next_observations)
        x = jnp.stack(
            jnp.concatenate((embeded_agent_observations, embeded_agent_next_observations), axis=-1))
        y = self.expert_states_pair

        # self.preproc.fit(states_pair)
        # x = jnp.asarray(self.preproc.transform(states_pair))
        # y = jnp.asarray(self.preproc.transform(self.expert_states_pair))
        # x = jnp.asarray(states_pair)
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


@jax.jit
def optim_embed(embed: Model,
                observations: jnp.array,
                next_observations: jnp.array,
                transport_matrix: jnp.array,
                expert_states_pair: jnp.array,
                epsilon: float,
                cost_fn: CostFn) -> tuple[Any, Model]:

    def embed_loss_fn(params: Params) -> tp.Tuple[jnp.ndarray, InfoDict]:
        embed_obs = embed.apply({'params':  params}, observations)
        next_embed_obs = embed.apply({'params':  params}, next_observations)
        states_pair_jax = jnp.concatenate([embed_obs, next_embed_obs], 1)
        x = states_pair_jax
        y = expert_states_pair

        geom = pointcloud.PointCloud(x, y, epsilon=epsilon, cost_fn=cost_fn)
        loss = jnp.sum(transport_matrix * geom.cost_matrix)

        return loss, {'loss': loss}

    return embed.apply_gradient(embed_loss_fn)


class D4RLDatasetWithOTRewards:
    @staticmethod
    def save(dataset: D4RLDataset, rewards_expert: RewardsExpert, num_samples: int):

        sep = np.where(dataset.dones_float > 0.5)[0].tolist()
        index = sep[0] + 1
        for i in sep:
            index = i + 1
            if index > num_samples:
                break

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
                os.path.join(FLAGS.path_to_save_env,
                             f"dataset_{FLAGS.env_name}_{FLAGS.expert_env_name}_ot_rewards.npz"),
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


class OTRewardsExpertCrossDomain(RewardsExpert):
    def __init__(
            self,
            expert_data: ExpertData,
            embed_model: Model,
            cost_fn: CostFn = costs.Euclidean(),
            epsilon=0.01,
    ):
        self.expert_states_pair = np.concatenate(
            [expert_data.observations, expert_data.next_observations], axis=1
        )
        self.cost_fn = cost_fn
        self.epsilon = epsilon

        self.preproc = Preprocessor()
        self.states_pair_buffer = ListBuffer(n=100)

        # Init model
        self.model = embed_model

    # def torch_cost_matrix(self, states_pair_torch: torch.Tensor):
    #     expert_states_pair = torch.from_numpy(self.expert_states_pair)
    #     dist = (states_pair_torch[:, None] - expert_states_pair[None, ]).pow(2).sum(-1).sqrt()
    #
    #     return dist

    def warmup(self) -> None:
        (
            observations,
            next_observations,
            transport_matrix,
        ) = self.states_pair_buffer.sample()

        new_model, info = optim_embed(self.model,
                    jnp.asarray(observations),
                    jnp.asarray(next_observations),
                    jnp.asarray(transport_matrix),
                    jnp.asarray(self.expert_states_pair),
                    self.epsilon,
                    self.cost_fn)
        # print(info)
        self.model = new_model

    def compute_rewards_one_episode(
            self, observations: np.ndarray, next_observations: np.ndarray
    ) -> np.ndarray:

        embeded_observations = np.asarray(self.model(observations))
        embeded_next_observations = np.asarray(self.model(next_observations))

        states_pair = np.concatenate(
            [embeded_observations, embeded_next_observations], axis=1
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

        self.warmup()

        return np.asarray(rewards)


class OTRewardsExpertFactoryCrossDomain(OTRewardsExpertFactory):
    def apply(
        self,
        dataset: D4RLDataset,
        embed_model: Model,
    ) -> OTRewardsExpertCrossDomain:

        expert = super().apply(dataset)
        return OTRewardsExpertCrossDomain(
            expert_data=expert.expert_data, embed_model=embed_model
        )



