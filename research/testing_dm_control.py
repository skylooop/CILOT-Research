import dmc2gym
import numpy as np
from collections import defaultdict
from tqdm import tqdm
"""
 A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.

"""
env = dmc2gym.make(domain_name="cartpole", task_name="balance")
"""expert_trajs = dict(np.load("/home/m_bobrin/CILOT-Research/dataAgg/expert_trajectory_cartpole_swingup.npz"))
#expert_trajs = np.load("/home/m_bobrin/CILOT-Research/tmp_data/dataset_halfcheetah-medium-v2.npz")
expert_trajs['actions'] = np.zeros_like(expert_trajs['observations'].mean(-1))
print(expert_trajs.keys())"""

qlearning = defaultdict()

for i in tqdm(range(2)):
  done = False
  obs = env.reset()
  actions = []
  rewards = []
  terminals = []
  observations = []
  
  while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    actions.append(action)
    rewards.append(reward)
    terminals.append(done)
    observations.append(obs)
    
  qlearning['actions'] = np.stack(np.array(actions), axis=0)
  qlearning['rewards'] = np.stack(np.array(rewards), axis=0)
  qlearning['terminals'] = np.stack(np.array(terminals), axis=0)
  qlearning['next_observations'] = np.stack(np.roll(np.array(observations), 1), axis=0)
  qlearning['observations'] = np.stack(np.array(observations), axis=0)
  
np.savez("research/agent_cartpole_balance", **qlearning)

  
  
    