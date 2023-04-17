import os
import random
from typing import Tuple

import gym
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
from tensorboardX import SummaryWriter

from dataset_utils import D4RLDataset
from agent.iql.learner import Learner
from ot_replay import ReplayBufferWithDynamicRewards
from wrappers.episode_monitor import EpisodeMonitor
from wrappers.single_precision import SinglePrecision
from compute_rewards import OTRewardsExpert, OTRewardsExpertFactory, ExpRewardsScaler, D4RLDatasetWithOTRewards
from video import VideoRecorder

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'walker2d-random-v2', 'Environment name.')
flags.DEFINE_string('expert_env_name', 'halfcheetah-expert-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_string('path_to_save_env', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 79, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 30,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 10, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 50000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e3), 'Number of training steps.')
flags.DEFINE_integer('num_pretraining_steps', 0,
                     'Number of pretraining steps.')
flags.DEFINE_integer('replay_buffer_size', 300000,
                     'Replay buffer size (=max_steps if unspecified).')
flags.DEFINE_integer('init_dataset_size', 100000,
                     'Offline data size (uses all data if unspecified).')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')


def make_env_and_dataset(env_name: str, seed: int) -> Tuple[gym.Env, D4RLDataset]:
    env = gym.make(env_name)

    env = EpisodeMonitor(env)
    env = SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    dataset = D4RLDataset(env)

    return env, dataset


def make_expert() -> OTRewardsExpert:
    expert_env = gym.make(FLAGS.expert_env_name)
    expert_env = SinglePrecision(expert_env)
    expert_dataset = D4RLDataset(expert_env)

    return OTRewardsExpertFactory().apply(expert_dataset, )


def sample_episode(dataset: D4RLDataset):

    sep = [-1] + np.where(dataset.dones_float > 0.5)[0].tolist()
    index = random.randint(0, len(sep) - 2)
    i0 = sep[index] + 1
    i1 = sep[index + 1] + 1

    assert i1 > i0

    return dataset.observations[i0:i1], dataset.next_observations[i0:i1]


def main(_):
    print(f"Gym Version: {gym.__version__}")
    print("tensorboard", FLAGS.save_dir)

    summary_writer = SummaryWriter(os.path.join(FLAGS.save_dir, 'tb', str(FLAGS.seed)), write_to_disk=True)

    env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed)
    expert = make_expert()

    # encoder = checkpoints.restore_checkpoint(
    #     os.path.join(FLAGS.save_dir, 'checkpoints'),
    #     target=expert.encoder,
    #     step=10000,
    #     prefix=f'encoder_{FLAGS.env_name}_{FLAGS.expert_env_name}')
    #
    # expert.encoder = encoder

    for i in tqdm.tqdm(range(FLAGS.max_steps), smoothing=0.1):

        obs, next_obs = sample_episode(dataset)
        update_info = expert.warmup(obs, next_obs)

        if i % FLAGS.log_interval == 0:
            print(i, update_info)
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.add_scalar(f"training/{k}", v, i)
                else:
                    summary_writer.add_histogram(f"training/{k}", v, i)

    D4RLDatasetWithOTRewards.save(dataset, expert, 10_000)

    checkpoints.save_checkpoint(ckpt_dir=os.path.join(FLAGS.save_dir, 'checkpoints'),
                                target=expert.encoder,
                                step=FLAGS.max_steps,
                                prefix=f'encoder_{FLAGS.env_name}_{FLAGS.expert_env_name}',
                                overwrite=True)


if __name__ == "__main__":
    app.run(main)
