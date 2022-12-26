import os
from typing import Tuple, Dict, Union
import gym
import numpy as np
import tqdm
import flax.linen as nn
from absl import app, flags

import wandb
from tensorboardX import SummaryWriter

from agent.iql.dataset_utils import D4RLDataset
from compute_rewards import OTRewardsExpertFactory, OTRewardsExpert, ExpRewardsScaler
from agent.iql.learner import Learner
from agent.iql.wrappers.episode_monitor import EpisodeMonitor
from agent.iql.wrappers.single_precision import SinglePrecision
from dynamic_replay_buffer import ReplayBufferWithDynamicRewards
from video import VideoRecorder

# Loggers builder
from loggers.loggers_wrapper import InitTensorboard, InitWandb

# Environmental variables
os.environ["CUDA_VISIBLE_DEVICES"] = "4" #opengl on dgx works only here 
os.environ["MUJOCO_GL"] = "egl"
os.environ["MUJOCO_EGL_DEVICES_ID"] = "4"
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

# Arguments
FLAGS = flags.FLAGS

# Choose agent/expert datasets
flags.DEFINE_string("env_name", "halfcheetah-medium-v2", "Environment name.")
flags.DEFINE_string("expert_env_name", "halfcheetah-expert-v2", "Environment name.")

# Define Loggers (Wandb/Tensorboard)
flags.DEFINE_enum(
    "logger", "Wandb", ["Wandb", "Tensorboard"], help="define loggers"
)

# Wandb params
flags.DEFINE_string("wandb_project_name", "CILOT", help="Current run name")
flags.DEFINE_string("wandb_entity", "cilot", help="Team name.")
flags.DEFINE_string("wandb_job_type", "training", help="Set job type.")

flags.DEFINE_string("save_dir", "CILOT-Research/assets", "Logger logging dir.")


flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 30, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 10000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(2e6), "Number of training steps.")
flags.DEFINE_integer("num_pretraining_steps", int(1e5), "Number of pretraining steps.")
flags.DEFINE_integer(
    "replay_buffer_size", 200000, "Replay buffer size (=max_steps if unspecified)."
)
flags.DEFINE_integer(
    "init_dataset_size", 100000, "Offline data size (uses all data if unspecified)."
)
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")


def make_env_and_dataset(env_name: str, seed: int) -> Tuple[gym.Env, D4RLDataset]:
    env = gym.make(env_name)

    env = EpisodeMonitor(env) #action wrapper
    env = SinglePrecision(env) #observation wrapper
    
    env.seed(seed)

    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    dataset = D4RLDataset(env)

    return env, dataset


def make_expert() -> OTRewardsExpert:
    expert_env = gym.make(FLAGS.expert_env_name)
    expert_env = SinglePrecision(expert_env)
    expert_dataset = D4RLDataset(expert_env)
    return OTRewardsExpertFactory().apply(expert_dataset)


def update_buffer(
    observation: np.ndarray,
    action: np.ndarray,
    replay_buffer: ReplayBufferWithDynamicRewards,
    env: gym.Env,
    summary_writer: SummaryWriter,
):

    next_observation, reward, done, info = env.step(action)
    mask = float(not done or "TimeLimit.truncated" in info)

    replay_buffer.insert(observation, action, mask, float(done), next_observation)

    if done:
        next_observation = env.reset()
        for k, v in info["episode"].items():
            summary_writer.add_scalar(f"training/{k}", v, info["total"]["timesteps"])

    return next_observation


def evaluate(
    step: int,
    agent: Learner,
    env: gym.Env,
    num_episodes: int,
    summary_writer: Union[SummaryWriter, None],
):
    stats = {"return": [], "length": []}

    video = VideoRecorder(FLAGS.save_dir, fps=20)
    env.reset()
    video.init(enabled=True)

    for en in range(num_episodes):
        observation, done = env.reset(), False
        video.record(env)

        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, _, done, info = env.step(action)
            if en < 2:
                video.record(env)

        for k in stats.keys():
            stats[k].append(info["episode"][k])

    for k, v in stats.items():
        stats[k] = np.mean(v)

    video.save(f"eval_{FLAGS.env_name}_{FLAGS.seed}_{step}.mp4")
    wandb.log({"video": wandb.Video(f"{FLAGS.save_dir}/video/eval_{FLAGS.env_name}_{FLAGS.seed}_{step}.mp4", fps=4, format='gif')})
    
    if summary_writer is not None:
        for k, v in stats.items():
            summary_writer.add_scalar(f"evaluation/average_{k}s", v, step)
        summary_writer.flush()
    else:
        for k, v in stats.items():
            wandb.log({f"Evaluation/average_{k}s": v.item()}, step=step)

def main(_):
    
    """
    if FLAGS.logger == "Tensorboard":
        summary_writer = SummaryWriter(
            os.path.join(FLAGS.save_dir, "tb", str(FLAGS.seed)), write_to_disk=True
        )
    """
    # os.makedirs(FLAGS.save_dir, exist_ok=True)

    if FLAGS.logger == "Tensorboard":
        summary_writer = InitTensorboard().init(
            save_dir=FLAGS.save_dir, seed=FLAGS.seed
        )
        logger = "Tensorboard"

    if FLAGS.logger == "Wandb":
        wandb_logger = InitWandb().init(
            config=FLAGS,
            save_dir=FLAGS.save_dir, seed=FLAGS.seed, wandb_project_name=FLAGS.wandb_project_name,
            wandb_entity=FLAGS.wandb_entity, wandb_job_type=FLAGS.wandb_job_type
        )
        logger = "Wandb"
        summary_writer = None

    # Making agent
    env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed)
    # Making expert
    expert = make_expert()

    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBufferWithDynamicRewards(
        env.observation_space,
        action_dim,
        FLAGS.replay_buffer_size,
        ExpRewardsScaler(),
        expert,
    )
    replay_buffer.initialize_with_dataset(dataset, FLAGS.init_dataset_size)

    agent = Learner(
        FLAGS.seed,
        env.observation_space.sample()[np.newaxis],
        env.action_space.sample()[np.newaxis],
        max_steps=FLAGS.max_steps,
        temperature=3.0,
        expectile=0.7,
    )

    observation = env.reset()

    for i in tqdm.tqdm(
        range(FLAGS.num_pretraining_steps + FLAGS.max_steps),
        smoothing=0.1,
        disable=not FLAGS.tqdm,
    ):
        if i >= FLAGS.num_pretraining_steps:
            action = agent.sample_actions(
                observation,
            )
            action = np.clip(action, -1, 1)
            observation = update_buffer(
                observation, action, replay_buffer, env, summary_writer
            )

        batch = replay_buffer.sample(FLAGS.batch_size)
        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            # k - name of loss (e.g Actor loss)
            # V - loss value
            
            for k, v in update_info.items():
                if v.ndim == 0:
                    if logger == "Tensorboard":
                        summary_writer.add_scalar(f"training/{k}", v, i)
                    else:
                        wandb.log({f"Training {k}": v.item()}, step=i)
                else:
                    if logger == "Tensorboard":
                        summary_writer.add_histogram(f"training/{k}", v, i)
                        
            if logger == "Tensorboard":
                summary_writer.flush()
                
        if i % FLAGS.eval_interval == 0:
            evaluate(i, agent, env, FLAGS.eval_episodes, summary_writer)

    if logger == "Wandb":
        wandb.finish()
        

if __name__ == "__main__":
    app.run(main)
