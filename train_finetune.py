import os
from typing import Tuple

import gym
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
from tensorboardX import SummaryWriter

from dataset_utils import D4RLDataset, Dataset
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
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 50000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 128, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(3e6), 'Number of training steps.')
flags.DEFINE_integer('num_pretraining_steps', 0,
                     'Number of pretraining steps.')
flags.DEFINE_integer('replay_buffer_size', 50000,
                     'Replay buffer size (=max_steps if unspecified).')
flags.DEFINE_integer('init_dataset_size', 10000,
                     'Offline data size (uses all data if unspecified).')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')


def make_env_and_dataset(env_name: str, seed: int) -> Tuple[gym.Env, Dataset]:
    env = gym.make(env_name)

    env = EpisodeMonitor(env)
    env = SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    # dataset = D4RLDataset(env)
    dataset = D4RLDatasetWithOTRewards.load()

    return env, dataset


def make_expert() -> OTRewardsExpert:
    expert_env = gym.make(FLAGS.expert_env_name)
    expert_env = SinglePrecision(expert_env)
    expert_dataset = D4RLDataset(expert_env)

    expert = OTRewardsExpertFactory().apply(expert_dataset, )

    # encoder = checkpoints.restore_checkpoint(
    #     os.path.join(FLAGS.save_dir, 'checkpoints'),
    #     target=expert.encoder,
    #     step=3000,
    #     prefix=f'encoder_{FLAGS.env_name}_{FLAGS.expert_env_name}')
    # expert.encoder = encoder

    return expert


def evaluate(step: int, agent: Learner, env: gym.Env, num_episodes: int, summary_writer: SummaryWriter):
    stats = {'return': [], 'length': []}

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
            stats[k].append(info['episode'][k])

    for k, v in stats.items():
        stats[k] = np.mean(v)

    video.save(f'eval_{FLAGS.env_name}_{FLAGS.seed}_{step}.mp4')
    print(f'eval_{FLAGS.env_name}_{FLAGS.seed}_{step}.mp4')

    for k, v in stats.items():
        summary_writer.add_scalar(f'evaluation/average_{k}s', v, step)
    summary_writer.flush()


def main(_):
    summary_writer = SummaryWriter(os.path.join(FLAGS.save_dir, 'tb',
                                                str(FLAGS.seed)),
                                   write_to_disk=True)
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed)
    expert = make_expert()

    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBufferWithDynamicRewards(env.observation_space, action_dim, FLAGS.replay_buffer_size,
                                                   ExpRewardsScaler(), expert)
    replay_buffer.initialize_with_dataset(dataset, FLAGS.init_dataset_size)

    agent = Learner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis],
                    max_steps=FLAGS.max_steps,
                    temperature=3.0,
                    expectile=0.9)

    eval_returns = []
    observation, done = env.reset(), False

    # Use negative indices for pretraining steps.
    for i in tqdm.tqdm(range(1 - FLAGS.num_pretraining_steps,
                             FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):

        if i == 500000:
            agent.expectile = 0.8
            expert.preproc.enabled = False

        if i >= 10000 and i % 5 == 0:
            action = agent.sample_actions(observation, )
            action = np.clip(action, -1, 1)
            next_observation, reward, done, info = env.step(action)

            if not done or 'TimeLimit.truncated' in info:
                mask = 1.0
            else:
                mask = 0.0

            replay_buffer.insert(observation, action, reward, mask,
                                 float(done), next_observation)
            observation = next_observation

            if done:
                observation, done = env.reset(), False
                for k, v in info['episode'].items():
                    summary_writer.add_scalar(f'training/{k}', v,
                                              info['total']['timesteps'])
        else:
            info = {}
            info['total'] = {'timesteps': i}

        batch = replay_buffer.sample(FLAGS.batch_size)
        update_info = agent.update(batch)
        if i % 500 == 0:
            obs1, next_obs1 = replay_buffer.sample_episode()
            expert.warmup(obs1, next_obs1)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.add_scalar(f'training/{k}', v, i)
                else:
                    summary_writer.add_histogram(f'training/{k}', v, i)
            summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            evaluate(i, agent, env, FLAGS.eval_episodes, summary_writer)


if __name__ == '__main__':
    app.run(main)
