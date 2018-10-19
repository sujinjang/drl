# Deep Q-Network (DQN)

import sys

sys.path.append("../")
import argparse
import datetime
import inspect
import os
import random
import time
import itertools
from collections import namedtuple
import numpy as np
import gym
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logz

from deepq.value_function_model import ValueEstimator, StateProcessor
from gym.wrappers.monitoring.video_recorder import VideoRecorder

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.

    Returns:
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation, epsilon, sess=None):
        sess = sess or tf.get_default_session()
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(np.expand_dims(observation, 0), sess)[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def copy_model_parameters(estimator1, estimator2, sess=None):
    """
    Copies the model parameters of one estimator to another.

    Args:
      sess: Tensorflow session instance
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """

    sess = sess or tf.get_default_session()
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)


def deepq(env, max_episode_steps, n_experiments, n_total_steps, seed, gamma, learning_rate, conv_sizes, fc_sizes,
          n_init_buffer_size, n_buffer_size, batch_size, epsilon_start, epsilon_end, exploration_fraction,
          update_target_freq,
          logging_dir="log", isRenderding=True, isRecordingVideo=True, recordingVideo_dir="video", rec_per_episodes=100,
          chckp_dir="checkpoint", checkpt_save_freq=100,
          test_name="test", device="CPU"):
    # Get environment name
    env_name = env.spec.id
    if max_episode_steps > 0:
        env._max_episode_steps = max_episode_steps
    print("Env max_step_per_episode:{}".format(env._max_episode_steps))

    # Identify states and action dimensions
    isDiscrete = isinstance(env.action_space, gym.spaces.Discrete)
    n_actions = env.action_space.n if isDiscrete else env.action_space.shape[0]

    # State processor
    state_shape = env.observation_space.shape
    state_size = [84, 84, 4]  # list(state_shape)
    state_processor = StateProcessor(input_shape=state_shape, output_shape=state_size[:-1])

    if device in {"gpu", "GPU"}:
        tf_device = '/device:GPU:0'
    else:
        tf_device = '/device:CPU:0'

    with tf.device(tf_device):
        value_model = ValueEstimator(scope="q_func", state_size=state_size, action_size=n_actions,
                                     conv_sizes=conv_sizes, fc_sizes=fc_sizes, learning_rate=learning_rate,
                                     isDiscrete=isDiscrete)

        target_value_model = ValueEstimator(scope="t_q_func", state_size=state_size, action_size=n_actions,
                                            conv_sizes=conv_sizes, fc_sizes=fc_sizes,
                                            learning_rate=learning_rate, isDiscrete=isDiscrete)

    init_time = time.strftime("%d-%m-%Y_%H-%M-%S")
    for exp in range(n_experiments):

        # Set random seed
        rand_seed = seed + 10 * exp
        tf.set_random_seed(rand_seed)
        np.random.seed(rand_seed)

        # Global step
        global_step = tf.Variable(0, name="global_step", trainable=False)

        # Init TF session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # TF saver
            saver = tf.train.Saver()
            chckp_dir = os.path.join(chckp_dir, env_name, test_name, init_time, str(exp))
            if not os.path.exists(chckp_dir):
                os.makedirs(chckp_dir)
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=chckp_dir)
            if latest_checkpoint:
                print("Loading model checkpoint {}...".format(latest_checkpoint))
                saver.restore(sess, latest_checkpoint)

            # Configure output directory for logging
            # Data logging paths
            if isRecordingVideo:
                recordingVideo_dir = os.path.join(recordingVideo_dir, env_name, test_name, init_time, str(exp))
                if not os.path.exists(recordingVideo_dir):
                    os.makedirs(recordingVideo_dir)

            logging_dir = os.path.join(logging_dir, env_name, test_name, init_time)
            if not os.path.exists(logging_dir):
                os.makedirs(logging_dir)
            logz.configure_output_dir(os.path.join(logging_dir, str(exp)))

            # Log experimental parameters
            args = inspect.getargspec(deepq)[0]
            locals_ = locals()
            params = {k: locals_[k] if k in locals_ and isinstance(locals_[k], (int, str, float)) else None for k in
                      args}
            logz.save_params(params)

            print("Parameter Lists")
            for param in params:
                if params[param]:
                    print(param + ": {}".format(params[param]))

            # Global step
            total_step = tf.train.global_step(sess, global_step)

            # Epsilon decaying schedule
            epsilons = np.linspace(epsilon_start, epsilon_end, int(exploration_fraction*n_total_steps))

            # The policy we're following
            policy = make_epsilon_greedy_policy(value_model, env.action_space.n)

            # Create a replay buffer
            replay_memory = []
            print("Collecting initial replay buffer")
            state = env.reset()
            state = state_processor.process(state, sess)  # TODO: DO NOT PROCESS IMAGE TO GRAYSCALE
            state = np.stack([state] * 4, axis=2)  # Sequential images (4 frames)
            for idx in range(n_init_buffer_size):
                action_probs = policy(state, epsilons[0], sess)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, _ = env.step(action)
                next_state = state_processor.process(next_state, sess)

                # Append next_state
                next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, axis=2), axis=2)
                replay_memory.append(Transition(state, action, reward, next_state, done))

                if done:
                    state = env.reset()
                    state = state_processor.process(state, sess)
                    state = np.stack([state] * 4, axis=2)
                else:
                    state = next_state

            print("==========================================")
            print("Exp: ", exp)
            print("==========================================")

            # Stat variables
            episode_reward_sum = 0
            episode_length = 0
            loss_sum = 0
            loss_steps = 0
            ep = 0  # Episode

            # Reset env variables
            state = env.reset()
            state = state_processor.process(state, sess)  # TODO: DO NOT PROCESS IMAGE TO GRAYSCALE
            state = np.stack([state] * 4, axis=2)  # Sequential images (4 frames)

            video_recorder = None
            if isRenderding and isRecordingVideo and (ep == 0 or ep % rec_per_episodes == 0):
                video_recorder = VideoRecorder(env,
                                               os.path.join(
                                                   recordingVideo_dir,
                                                   "vid_{}_{}_{}_{}.mp4".format(env_name, exp, test_name, ep)),
                                               enabled=True)
                print("Recording a video of this episode {} in experiment {}".format(ep, exp))
            # Iterate total n steps of simulation across numerous episodes
            for total_step in range(n_total_steps):

                # Epsilon for this time step
                epsilon = epsilons[min(total_step, int(exploration_fraction*n_total_steps) - 1)]

                # Update target Q-function with online Q-function
                if total_step % update_target_freq == 0:
                    copy_model_parameters(value_model, target_value_model, sess)
                    print("Copied model parameters to target network.")

                # Take a step
                action_probs = policy(state, epsilon, sess)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, _ = env.step(action)
                next_state = state_processor.process(next_state, sess)
                next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, axis=2), axis=2)

                if video_recorder and isRenderding:
                    env.render()
                    video_recorder.capture_frame()

                # Check whether replay buffer is full
                if len(replay_memory) == n_buffer_size:
                    replay_memory.pop(0)

                # Save transition to replay buffer
                replay_memory.append(Transition(state, action, reward, next_state, done))

                # Update online Q-function
                # Sample randomized minibatch from replay buffer
                samples = random.sample(replay_memory, batch_size)
                states_batch, actions_batch, reward_batch, next_state_batch, done_batch = map(np.array, zip(*samples))

                # Calculate action-values from double Q-functions
                q_values_next = value_model.predict(next_state_batch, sess)  # Q values per each possible actions
                selected_actions = np.argmax(q_values_next, axis=1)  # Use Q-function (not target) to get the max action

                # Get max action-value using max action from online Q-values
                target_q_values_next = target_value_model.predict(next_state_batch, sess)

                selected_target_values = gamma * target_q_values_next[np.arange(batch_size), selected_actions]
                targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * selected_target_values

                # Update Q(action-value) function
                states_batch = np.array(states_batch)
                loss = value_model.update(states_batch, actions_batch, targets_batch, sess=sess)
                loss_sum += loss
                loss_steps += 1

                if done:
                    # Close video recorder
                    if video_recorder:
                        video_recorder.close()
                        video_recorder = None

                    print("===================== End of Episode:{} @ step:{} =====================".format(ep, total_step))

                    # Log progress
                    logz.log_tabular("Episode", ep)
                    logz.log_tabular("Episode length", episode_length)
                    logz.log_tabular("Total steps", total_step)
                    logz.log_tabular("Mean rewards", episode_reward_sum / episode_length)
                    logz.dump_tabular()
                    logz.pickle_tf_vars()

                    # Reset env and stat variables
                    state = env.reset()
                    state = state_processor.process(state, sess)  # TODO: DO NOT PROCESS IMAGE TO GRAYSCALE
                    state = np.stack([state] * 4, axis=2)  # Sequential images (4 frames)

                    episode_reward_sum = 0
                    episode_length = 0
                    loss_sum = 0
                    loss_steps = 0
                    ep += 1

                    # Save model per episode
                    if ep % checkpt_save_freq == 0 or ep == 0:
                        saver.save(tf.get_default_session(), chckp_dir, global_step=total_step)

                    # Recording videos
                    if video_recorder:
                        video_recorder.close()
                    if isRenderding and isRecordingVideo and (ep == 0 or ep % rec_per_episodes == 0):
                        video_recorder = VideoRecorder(env,
                                                       os.path.join(
                                                           recordingVideo_dir,
                                                           "vid_{}_{}_{}_{}.mp4".format(env_name, exp, test_name,
                                                                                        ep)),
                                                       enabled=True)
                        print("Recording a video of this episode {} in experiment {}".format(ep, exp))

                else:
                    # Update episode stats
                    episode_reward_sum += reward
                    episode_length += 1
                    state = next_state

            print("===================== End of Last Episode:{} @ step:{} =====================".format(ep, total_step))

            # Log progress
            logz.log_tabular("Episode", ep)
            logz.log_tabular("Episode length", episode_length)
            logz.log_tabular("Total steps", total_step)
            logz.log_tabular("Mean rewards", episode_reward_sum / episode_length)
            logz.dump_tabular()
            logz.pickle_tf_vars()

            # Save session
            saver.save(tf.get_default_session(), chckp_dir, global_step=total_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default="Breakout-v0") # "PongNoFrameskip-v4"
    parser.add_argument('--max_episode_steps', type=int, default=0)
    parser.add_argument('--n_experiments', type=int, default=1)
    parser.add_argument('--n_total_steps', type=int, default=int(1e7))
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--n_buffer_size', type=int, default=50000)
    parser.add_argument('--n_init_buffer_size', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.01)
    parser.add_argument('--exploration_fraction', type=float, default=0.1)
    parser.add_argument('--update_target_freq', type=int, default=1000)
    parser.add_argument('--isRendering', type=bool, default=False)
    parser.add_argument('--isRecordingVideo', type=bool, default=False)
    parser.add_argument('--rec_per_ep', type=int, default=100)
    parser.add_argument('--recordingVideo_dir', type=str, default="video")
    parser.add_argument('--logging_dir', type=str, default="log")
    parser.add_argument('--test_name', type=str, default="test")
    parser.add_argument('--device', type=str, default="CPU")
    args = parser.parse_args()

    # Build gym environment
    env = gym.make(args.env_name)

    # Run DQN
    conv_sizes = [[32, 8, 4], [64, 4, 2], [64, 3, 1]]
    fc_sizes = [256]
    deepq(env=env,
          max_episode_steps=args.max_episode_steps,
          n_experiments=args.n_experiments,
          n_total_steps=args.n_total_steps,
          seed=args.seed,
          gamma=args.gamma,
          learning_rate=args.learning_rate,
          conv_sizes=conv_sizes,
          fc_sizes=fc_sizes,
          n_init_buffer_size=args.n_init_buffer_size,
          n_buffer_size=args.n_buffer_size,
          batch_size=args.batch_size,
          epsilon_start=args.epsilon_start,
          epsilon_end=args.epsilon_end,
          exploration_fraction=args.exploration_fraction,
          update_target_freq=args.update_target_freq,
          logging_dir=args.logging_dir,
          isRenderding=args.isRendering,
          isRecordingVideo=args.isRecordingVideo,
          recordingVideo_dir=args.recordingVideo_dir,
          rec_per_episodes=args.rec_per_ep,
          test_name=args.test_name,
          device=args.device)

    # # Identify states and action dimensions
    # isDiscrete = isinstance(env.action_space, gym.spaces.Discrete)
    # n_actions = env.action_space.n if isDiscrete else env.action_space.shape[0]

    # print("Argument Lists")
    # for arg in vars(args):
    #     print(arg + ": ", getattr(args, arg))

    # # State processor
    # state_shape = env.observation_space.shape
    # state_processor = StateProcessor(input_shape=state_shape, output_shape=[80, 80])
    #
    # # Value estimation model
    # args.layer_sizes = [[32, 8, 4], [64, 4, 2], [64, 3, 1]]
    # value_model = ValueEstimator(scope="q_func", state_size=list(state_shape), action_size=n_actions, isDiscrete=isDiscrete,
    #                              conv_sizes=args.layer_sizes, fc_sizes=[256], learning_rate=args.learning_rate)
    #
    # target_value_model = ValueEstimator(scope="t_q_func", state_size=list(state_shape), action_size=n_actions,
    #                                     isDiscrete=isDiscrete, conv_sizes=args.layer_sizes, fc_sizes=[256],
    #                                     learning_rate=args.learning_rate)
    #
    #
    #
    # global_step = tf.Variable(0, name="global_step", trainable=False)
    #
    # for exp in range(args.n_experiments):
    #     # Set random
    #     seed = args.seed + 10 * exp
    #     tf.set_random_seed(seed)
    #     np.random.seed(seed)
    #
    #     print("==========================================")
    #     print("Exp: ", exp)
    #     print("==========================================")
    #     with tf.Session() as sess:
    #         sess.run(tf.global_variables_initializer())
    #
    #         # Run DQN algorithm
    #         deepq(sess=sess, exp=exp, value_model=value_model, target_value_model=target_value_model,
    #               state_processor=state_processor, global_step=global_step, env=env, train_freq=args.train_freq,
    #               gamma=args.gamma, n_episodes=args.n_episodes, n_buffer_size=args.n_buffer_size,
    #               n_init_buffer_size=args.n_init_buffer_size, batch_size=args.batch_size,
    #               epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
    #               epsilon_decay_steps=args.epsilon_decay_steps, update_target_estimator_freq=args.target_update_freq,
    #               isRenderding=args.isRendering,
    #               isRecordingVideo=args.isRecordingVideo, recordingVideo_dir=recordingVideo_dir,
    #               rec_per_ep=args.rec_per_ep,
    #               test_name=args.test_name, logging_dir=logging_dir, seed=seed)
