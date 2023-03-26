import dmc2gym
import numpy as np
env = dmc2gym.make(domain_name="cartpole", task_name="balance")
expert_trajs = dict(np.load("/home/m_bobrin/CILOT-Research/dataAgg/expert_trajectory_cartpole_swingup.npz"))
#expert_trajs = np.load("/home/m_bobrin/CILOT-Research/tmp_data/dataset_halfcheetah-medium-v2.npz")
expert_trajs['actions'] = np.zeros_like(expert_trajs['observations'].mean(-1))
print(expert_trajs.keys())

'''done = False
obs = env.reset()
while not done:
  action = env.action_space.sample()
  print(action)
  obs, reward, done, info = env.step(action)'''