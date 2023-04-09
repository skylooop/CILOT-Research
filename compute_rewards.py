import collections
from abc import ABC, abstractmethod

import numpy as np
from flax.training import train_state
from jax import numpy as jnp

import ott
import jax
from ott.geometry import pointcloud, costs
from ott.geometry.costs import CostFn
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from sklearn import preprocessing
from tqdm import tqdm

from optimization import update_encoder, embed
import typing as tp
import torch.nn as nn

from absl import flags

FLAGS = flags.FLAGS

from agent.iql.dataset_utils import D4RLDataset

ExpertData = collections.namedtuple("ExpertData", ["packed_trajectories", "expert_pairs_weights"])
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
        i0 = 0
        rewards = []

        # number of terminal states in agent dataset
        for i1 in tqdm(np.where(dones_float > 0.5)[0].tolist()):
            rewards.append(
                self.compute_rewards_one_episode(observations[i0 : i1 + 1], next_observations[i0 : i1 + 1])
            )
            i0 = i1 + 1
        
        return np.concatenate(jax.device_get(rewards))
    
    @abstractmethod
    def warmup(self):
        ...

    @abstractmethod
    def compute_rewards_one_episode(
        self, observations_agent: np.ndarray, next_observations_agent: np.ndarray
    ) -> np.ndarray:
        ...


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

    def _pad(self, x, max_sequence_length: int = 1000):
        paddings = [(0, max_sequence_length - x.shape[0])]
        paddings.extend([(0, 0) for _ in range(x.ndim - 1)])
        return np.pad(x, paddings, mode='constant', constant_values=0.)
    
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
        cost_fn: CostFn = ott.geometry.costs.Cosine(), # same as in PWIL and OTR, no explanation why. Reviewers dont understand this decision
        epsilon: float = 1e-2,
    ):
    
        self.episode_length_expert = 1000                                                        # topk                  1000
        self.batched_trajectories_pairs = expert_data.packed_trajectories # shape: num of [num_best_trajectories=topk x traj_episode_length x (expert.obs_shape * 2)]
        self.expert_trajectory_weights = expert_data.expert_pairs_weights # [num_best_trajectories x episode_length]
        
        self.cost_fn = cost_fn
        self.epsilon = epsilon

        self.preproc = Preprocessor()
        self.states_pair_buffer = ListBuffer(n=10)
        
        self.encoder_class = encoder_class
        
        # OTT
        self.vectorized_ot_rewards = jax.jit(
            jax.vmap(self._ott_computation, in_axes=(0, 0, None, None))
        )
        
    def optim_embed(self) -> None:
        sampled_agent_observations, sampled_agent_next_observations, transport_matrix = self.states_pair_buffer.sample()
        
        # since we have topk batch expert trajectories - take only one
        best_expert_traj_pairs = self.batched_trajectories_pairs[-1]
        self.encoder_class, loss = update_encoder(self.encoder_class, sampled_agent_observations, sampled_agent_next_observations,
                                                  best_expert_traj_pairs, transport_matrix, cost_fn=self.cost_fn)
        
    def warmup(self) -> None:
        self.optim_embed()
    
    def _ott_computation(self, batched_expert_trajs, expert_trajs_weights, agent_trajectory, agent_traj_weights):
        '''
        Main OTT library computations here
        '''
        # TODO: Add preprocessing function for trajectories and observations
        geom = pointcloud.PointCloud(batched_expert_trajs, agent_trajectory, epsilon=self.epsilon, cost_fn=self.cost_fn)
        ot_prob = linear_problem.LinearProblem(geom, a=expert_trajs_weights, b=agent_traj_weights, tau_a=1., tau_b=0.95)
        solver = sinkhorn.Sinkhorn()

        ot_sink = solver(ot_prob)
        transp_cost = jnp.sum(ot_sink.matrix * geom.cost_matrix, axis=0)

        pseudo_rewards = -transp_cost
        return pseudo_rewards, ot_sink.matrix
        
    def compute_cross_domain_OT(self, batched_expert_trajs, expert_trajs_weights,
                                      agent_trajectory, agent_traj_weights):
        
        transport_costs, ot_sink_matrix = self.vectorized_ot_rewards(batched_expert_trajs, expert_trajs_weights,
                                               agent_trajectory, agent_traj_weights)
        
        return transport_costs, ot_sink_matrix
    
    def aggregate_top_k(self, rewards, k=1):
        """Aggregate rewards from batched (top) expert demonstrations by mean of top-K demos."""
        scores = jnp.sum(rewards, axis=-1)
        _, indices = jax.lax.top_k(scores, k=k)
        return jnp.mean(rewards[indices], axis=0)
    
    def _pad(self, x, max_sequence_length: int):
        paddings = [(0, max_sequence_length - x.shape[0])]
        paddings.extend([(0, 0) for _ in range(x.ndim - 1)])
        return np.pad(x, paddings, mode='constant', constant_values=0.)
    
    def compute_rewards_one_episode(
        self, observations_agent: np.ndarray, next_observations_agent: np.ndarray, train: bool = True
    ) -> np.ndarray:
        
        #one trajectory
        embeded_agent_observations, embeded_agent_next_observations = embed(self.encoder_class, observations_agent, next_observations_agent)
        agent_traj_pairs = jnp.stack(jnp.concatenate((embeded_agent_observations, embeded_agent_next_observations), axis=-1))
        
        agent_traj_weights = jnp.ones((agent_traj_pairs.shape[0], )) / agent_traj_pairs.shape[0]

        # Experiment without Preprocessing
        #self.preproc.fit(agent_obs_pairs)
        
        #x = jnp.asarray(self.preproc.transform(agent_obs_pairs))
        #y = jnp.asarray(self.preproc.transform(self.expert_states_pair))
        
        rewards, ot_sink_matrix = self.compute_cross_domain_OT(self.batched_trajectories_pairs, self.expert_trajectory_weights, 
                                               agent_traj_pairs, agent_traj_weights)
        rewards = self.aggregate_top_k(rewards, k=1)
        ####
        self.states_pair_buffer.append(
            (observations_agent, next_observations_agent, ot_sink_matrix)
        )
        return rewards


def split_into_trajectories(
    observations, actions, rewards, dones_float, next_observations
):
    trajs = [[]]

    print("Splitting dataset into trajectories based on terminal condition")
    for i in tqdm(range(len(observations))):
        trajs[-1].append(
            (
                observations[i],
                actions[i],
                rewards[i],
                #masks[i],
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
            #dataset.masks,
            dataset.dones_float,
            dataset.next_observations,
        )
        
        episodic_returns = [sum([info[2] for info in traj]) for traj in trajs] # reward on each step is 2nd element in array
        idx = np.argpartition(episodic_returns, -FLAGS.topk)[-FLAGS.topk:] # choose top k best trajectories from expert
        expert_best_returns = [episodic_returns[i] for i in idx] 
        print(f"Best return examples: {expert_best_returns}")
        best_traj = [trajs[i] for i in idx] # best trajectories that gave maximal return
        
        expert_pairs_states = []
        expert_pairs_weights = []
        
        for cur_traj in best_traj:
            best_traj_paired_states = [
                np.concatenate([transition[0], transition[-1]], axis=-1) for transition in cur_traj
            ]
            # TODO: is it possible for trajectory of expert to be less than 1k? Maybe, bec 1k only for Gym, and can make <1k steps
            # Add padding to 1k
            # same like in OTR, make 0 probability for padded with 0 states
            atoms = np.stack(best_traj_paired_states, axis=0) # equaivalent to pairs of (cur_state, next_state) in one trajectory
            atoms = self._pad(atoms, max_sequence_length=999)
            expert_pairs_states.append(atoms)
            
            num_weights = len(cur_traj)
            expert_pairs_weights_cur_traj = np.ones((num_weights, )) / 999
            expert_pairs_weights_cur_traj = self._pad(expert_pairs_weights_cur_traj, max_sequence_length=999)
            expert_pairs_weights.append(expert_pairs_weights_cur_traj) # for gym
            
        batched_expert_pairs_states = np.stack(expert_pairs_states) # shape: num of [num_best_trajectories x episode_length x (expert.obs_shape * 2)]
        expert_pairs_weights = np.stack(expert_pairs_weights) # [num_best_trajectories x episode_length]
        
        if type == "CrossDomain":
            return OTRewardsExpertCrossDomain(
                ExpertData(packed_trajectories=batched_expert_pairs_states,
                           expert_pairs_weights=expert_pairs_weights), 
                encoder_class=encoder_class)
        else: 
            return OTRewardsExpert(
                ExpertData(
                    packed_trajectories=batched_expert_pairs_states
                )
            )
            
    def _pad(self, x, max_sequence_length: int):
        if x.shape[0] >= 999:
            return x
        paddings = [(0, max_sequence_length - x.shape[0])]
        paddings.extend([(0, 0) for _ in range(x.ndim - 1)])
        return np.pad(x, paddings, mode='constant', constant_values=0.)

class OTRewardsExpertFactoryCrossDomain(OTRewardsExpertFactory):
    
    def apply(self, dataset: D4RLDataset, encoder_class, type="CrossDomain") -> OTRewardsExpert:
        expert = super().apply(dataset, type, encoder_class)
        return expert

