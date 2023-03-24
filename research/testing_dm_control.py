import dmc2gym
env = dmc2gym.make(domain_name="cartpole", task_name="swingup")

done = False
obs = env.reset()
while not done:
  action = env.action_space.sample()
  print(action)
  obs, reward, done, info = env.step(action)