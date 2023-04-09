import pickle

with open("/home/m_bobrin/CILOT-Research/dmc_rollouts/pendulum/expert_trajectory_pendulum_swingup.pickle",'rb') as handle:
    dict_demonstration = pickle.load(handle)
print(dict_demonstration.keys())