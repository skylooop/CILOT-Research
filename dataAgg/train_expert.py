from absl import app, flags
import os
import gymnasium as gym
from gymnasium.utils.save_video import save_video
import sys
sys.modules["gym"] = gym

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["MUJOCO_GL"] = "egl"
os.environ["MUJOCO_EGL_DEVICES_ID"] = "4"

FLAGS = flags.FLAGS

flags.DEFINE_string(name="expert_env", default="InvertedPendulum-v4", help="Provide name of env to train on.")

from stable_baselines3 import A2C, PPO

def main(_):
    env = gym.make(FLAGS.expert_env, render_mode='rgb_array')
    
    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000)
    
    vec_env = model.get_env()
    obs = vec_env.reset()
    
    step_starting_index = 0
    episode_index = 0
    frames = []
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action) 
        frames.append(vec_env.render())
        if done:
            save_video(
                frames,
                "/home/m_bobrin/CILOT-Research/dataAgg",
                fps=env.metadata["render_fps"]
            )
            frames = []


if __name__ == "__main__":
    app.run(main)