import os
import random
from typing import Tuple, Union
import gym
import gymnasium as gym
import numpy as np
#import dmc2gym
from tqdm.auto import tqdm
from absl import app, flags
import wandb
from flax.training import checkpoints
#from tensorboardX import SummaryWriter
from agent.iql.dataset_utils import D4RLDataset
from compute_rewards import (
    OTRewardsExpertFactoryCrossDomain, RewardsExpert, OTRewardsExpertCrossDomain,
)
from environments.utils import get_dataset, crossembodiment_dataset
from agent.iql.learner import Learner
from agent.iql.wrappers.episode_monitor import EpisodeMonitor
from agent.iql.wrappers.single_precision import SinglePrecision
from dynamic_replay_buffer import D4RLDatasetWithOTRewards
from video import VideoRecorder

from environments.utils import make_env
from environments.gc_dataset import GCDataset
from optimization import create_encoder
from loggers.loggers_wrapper import InitTensorboard, InitWandb

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Environmental variables
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["MUJOCO_GL"] = "egl"
os.environ["MUJOCO_EGL_DEVICES_ID"] = "4"
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

# Arguments
FLAGS = flags.FLAGS

# Choose agent/expert datasets
flags.DEFINE_bool("dmc_env", default=False, help="Whether DMC env is used.")


flags.DEFINE_bool("xmagical", default=True,
                  help="Whether to use cross-domain x-magical dataset.")
flags.DEFINE_string("modality", default="gripper", 
                    help="Which modality to use in xmagical dataset.")
flags.DEFINE_string("dataset", default="/home/m_bobrin/CILOT-Research/datasets/x_magical/xmagical_replay_icvf",
                    help="Path to dataset.")

flags.DEFINE_string("env_name", "halfcheetah-random-v2",
                    help="Environment agent name.")
flags.DEFINE_string("expert_env_name", "halfcheetah-medium-replay-v2",
                    help="Environment expert name.")

flags.DEFINE_string(
    "save_dir", "assets/", "Logger logging dir."
)

flags.DEFINE_string(
    "path_to_save_env",
    "tmp_data",
    help="Path where .npz numpy file with environment will be saved.",
)

flags.DEFINE_integer("seed", np.random.choice(100000), "Random seed.")
flags.DEFINE_integer("max_steps", int(3e4), "Number of training steps.")
flags.DEFINE_integer("log_interval", 10, "Log interval.")
flags.DEFINE_integer("topk", default=15,
                     help="Number of trajectories to use from")


def make_env_and_dataset(env_name: str, seed: int) -> Tuple[gym.Env, D4RLDataset]:
    """
    Makes d4rl dataset from offline agent dataset and return its environment

    Args:
        env_name (str): Name of the agent environment
        seed (int): Seed

    Returns:
        Tuple[gym.Env, D4RLDataset]
    """
    if FLAGS.dmc_env:
        domain_name = FLAGS.env_name.split("_")[0]
        task_name = '_'.join(FLAGS.env_name.split('_')[1:])
        env = dmc2gym.make(domain_name=domain_name,
                           task_name=task_name,
                           visualize_reward=False)
    elif FLAGS.xmagical:
        env = make_env(FLAGS.modality)
        #currently only expert, just for debug
        env = SinglePrecision(env)
        dataset = get_dataset(FLAGS.modality)
        return env, dataset
    else:
        env = gym.make(env_name)

    env = EpisodeMonitor(env)
    env = SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    dataset = D4RLDataset(env)

    return env, dataset


def make_expert(agent_state_shape: int) -> OTRewardsExpertCrossDomain:
    """
    Building expert trajectories dataset
    """
    if FLAGS.dmc_env:
        domain_name = FLAGS.expert_env_name.split("_")[0]
        task_name = '_'.join(FLAGS.expert_env_name.split('_')[1:])

        expert_env = dmc2gym.make(domain_name=domain_name,
                                  task_name=task_name,
                                  visualize_reward=False)
    elif FLAGS.xmagical:
        # ['gripper', 'shortstick', 'mediumstick', 'longstick']
        if FLAGS.video_type == "same":
            video_dataset = get_dataset(FLAGS.modality, FLAGS.dataset)
        elif FLAGS.video_type == "cross":
            # Collect datasets from other embodiments
            video_dataset = crossembodiment_dataset(FLAGS.modality, FLAGS.dataset)
        expert_dataset = GCSDataset(video_dataset)
        example_batch = gc_dataset.sample(1) # (1 sample, img_size, img_size, 3)
        encoder_class = create_encoder(
            agent_state_shape, 3, # for x-magical only 3 actions available
            lr=5e-5)
    else:
        expert_env = gym.make(FLAGS.expert_env_name)

    if not FLAGS.xmagical:
        expert_env = SinglePrecision(expert_env)
        expert_dataset = D4RLDataset(expert_env)

        encoder_class = create_encoder(
            agent_state_shape, expert_env.observation_space.shape[0], lr=5e-5)
        # encoder_class = checkpoints.restore_checkpoint(os.path.join(FLAGS.save_dir, 'checkpoints'),
        #                                                target=encoder_class,
        #                                                step=30000,
        #                                                prefix='encoder')

        return OTRewardsExpertFactoryCrossDomain().apply(
            expert_dataset,
            encoder_class,
            type="CrossDomain"
        )
    return 


def sample_episodes(dataset: D4RLDataset, n: int, max_len=128):
    block_lens = []
    sep = [-1] + np.where(dataset.dones_float > 0.5)[0].tolist()
    index = random.randint(0, len(sep) - 1 - n)
    i0 = sep[index] + 1
    i1 = sep[index + 1] + 1
    block_lens.append(i1 - i0)
    k = 2
    while i1 - i0 < max_len and k <= n:
        i1 = sep[index + k] + 1
        block_lens.append(i1 - sep[index + k - 1] - 1)
        k += 1

    assert i1 > i0

    return dataset.observations[i0:i1], dataset.next_observations[i0:i1], block_lens


def main(_):
    print(f"Gym Version: {gym.__version__}")

    summary_writer = InitTensorboard().init(
        save_dir=FLAGS.save_dir, seed=FLAGS.seed
    )
    print("Path to Tensorboard logs: ",FLAGS.save_dir)

    env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed)
    expert = make_expert(agent_state_shape=env.observation_space.shape[0])

    for i in tqdm(range(FLAGS.max_steps), smoothing=0.1):

        obs, next_obs, block_lens = sample_episodes(
            dataset, random.randint(1, 3))
        update_info = expert.warmup(obs, next_obs)

        if i % FLAGS.log_interval == 0:
            print(i, update_info)
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.add_scalar(f"training/{k}", v, i)
                else:
                    summary_writer.add_histogram(f"training/{k}", v, i)

    D4RLDatasetWithOTRewards.save(dataset, expert, 300_000)

    checkpoints.save_checkpoint(ckpt_dir=os.path.join(FLAGS.save_dir, 'checkpoints'),
                                target=expert.encoder_class,
                                step=FLAGS.max_steps,
                                prefix='encoder',
                                overwrite=True)


if __name__ == "__main__":
    app.run(main)
