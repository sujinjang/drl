# Policy Gradient with (1) value function as a baseline and (2) reward-to-go as a reward summation
#

import argparse
import datetime
import inspect
import os
import time
import numpy as np
import gym
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from policy_estimator_model import PolicyEstimator
from value_estimator_model import ValueEstimator
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import logz


def reinforce(sess, exp, pg_model, value_model, env, gamma, isRTG=True, n_iterations=100, n_batch=100,
              isRenderding=True, isRecordingVideo=True, recordingVideo_dir="video",
              isNNBaseLine=True, isNormalizeAdvantage=True, isAdaptiveStd=False,
              test_name="test", logging_dir="log", seed=0):
    # Get environment name
    env_name = env.spec.id

    # Configure output directory for logging
    logz.configure_output_dir(os.path.join(logging_dir, '%d' % exp))
    recordingVideo_dir = os.path.join(recordingVideo_dir, '%d' % exp)
    if not os.path.exists(recordingVideo_dir):
        os.makedirs(recordingVideo_dir)

    # Log experimental parameters
    args = inspect.getargspec(reinforce)[0]
    locals_ = locals()
    params = {k: locals_[k] if k in locals_ and isinstance(locals_[k], (int, str, float)) else None for k in args}
    logz.save_params(params)

    print("Policy Gradient for {} Environment".format(env_name))
    for iter in range(n_iterations):
        print("==========================================")
        print("Iteration: ", iter)

        steps_in_batch = 0
        trajectories = []
        tic = time.clock()
        episode = 1

        video_recorder = None

        # Outer loop for collecting a trajectory batch
        while True:
            episode_states, episode_actions, episode_rewards, episode_returns, episode_advantages = [], [], [], [], []
            episode_steps = 0
            state = env.reset()

            if isRecordingVideo and episode == 1 and (iter % 10 == 0 or iter == n_iterations - 1 or iter == 0):
                video_recorder = VideoRecorder(env,
                                               os.path.join(
                                                   recordingVideo_dir,
                                                   "vid_{}_{}_{}_{}.mp4".format(env_name, exp, test_name, iter)),
                                               enabled=True)
                print("Recording a video of this episode {} in iteration {}".format(episode, iter))

            # Roll-out trajectory to collect a batch
            while True:
                if isRenderding:
                    env.render()

                    if video_recorder:
                        video_recorder.capture_frame()

                # Choose an action based on observation
                action = pg_model.predict(state, sess=sess)
                action = action[0]

                # Simulate one time step from action
                nex_state, reward, done, info = env.step(action=action)

                # Collect data for a trajectory
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                state = nex_state

                episode_steps += 1

                if done:
                    break

            # Compute returns (Reward-To-Go or Full trajectory-centric)
            if isRTG:
                episode_returns = get_discounted_rewards_to_go(episode_rewards, gamma=gamma)
            else:
                episode_returns = [get_sum_of_reward(episode_rewards, gamma=gamma)] * len(episode_rewards)

            # Compute Value function per trajectory
            if isNNBaseLine:
                episode_baseline = value_model.predict(state=episode_states, sess=sess)

                # Normalize baseline estimation w.r.t returns
                # episode_baseline = normalize(episode_baseline, np.mean(episode_returns), np.std(episode_returns))

                # Get advantage
                episode_advantages = np.squeeze(episode_returns) - np.squeeze(episode_baseline)
            else:
                episode_advantages = episode_returns.copy()

            # Normalize advantage
            if isNormalizeAdvantage:
                # episode_advantages = normalize(episode_advantages)
                episode_advantages = (episode_advantages - np.mean(episode_advantages)) \
                                     / (np.std(episode_advantages) + 1e-8)

            # # Normalize Target (Q)
            # episode_returns = normalize(episode_returns)

            # Append to trajectory batch
            trajectory = {"state": np.array(episode_states),
                          "action": np.array(episode_actions),
                          "reward": np.array(episode_rewards),
                          "return": np.array(episode_returns),
                          "advantage": np.array(episode_advantages)}
            trajectories.append(trajectory)

            # Increase episode step
            steps_in_batch += len(trajectory["reward"])
            episode += 1

            # Close video recording
            if video_recorder:
                video_recorder.close()
                video_recorder = None

            # Break loop when enough episode batch is collected
            if episode > n_batch:  # steps_in_batch > min_steps_in_batch:
                break

        # Batching sample trajectories
        # Generate 'ready-to-use' batch arrays for state, action, and reward

        # pg_model.sample_trajectories(trajectories)
        batch_states = np.concatenate([traj["state"] for traj in trajectories])
        batch_actions = np.concatenate([traj["action"] for traj in trajectories])
        batch_returns = np.concatenate([traj["return"] for traj in trajectories])
        batch_advantages = np.concatenate([traj["advantage"] for traj in trajectories])

        # # Compute trajectory-centric reward sum
        # if isRTG:
        #     batch_rewards = np.concatenate([
        #         get_discounted_rewards_to_go(traj["reward"], gamma) for traj in trajectories])
        # else:
        #     batch_rewards = np.concatenate([
        #         [get_sum_of_reward(traj["reward"], gamma=gamma)] * len(traj["reward"])
        #         for traj in trajectories
        #     ])

        # Compute estimated V(s) and A(s) (= Sum(rewards) - V(s))
        # if isNNBaseLine:
        #     # Compute NN baseline estimation
        #     value_estimates = value_model.predict(state=batch_states)
        #     # value_estimates = normalize(value_estimates, np.mean(value_estimates), np.std(value_estimates))
        #     # value_estimates = value_estimates * np.std(value_estimates, axis=0) + np.mean(value_estimates, axis=0)
        #
        #     # Compute advantages and normalize it per trajectory
        #     advantages = np.squeeze(batch_rewards) - np.squeeze(value_estimates)
        #     # advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        # else:
        #     advantages = batch_rewards.copy()

        # if isNormalizeAdvantage:
        #     # advantages = normalize(advantages)
        #     advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # if isNNBaseLine:
        #     # Normalize rewards (targets) and update value estimator
        #     # batch_rewards = (batch_rewards - np.mean(batch_rewards)) / (np.std(batch_rewards) + 1e-8)
        #     batch_rewards = normalize(batch_rewards)
        #
        #     # Update value estimator
        #     value_model.update(states=batch_states, targets=batch_rewards)

        # Update value estimator
        if isNNBaseLine:
            value_model.update(states=batch_states, targets=batch_returns, sess=sess)

        # Update policy estimator
        pg_model.update(states=batch_states, actions=batch_actions, advantages=batch_advantages, sess=sess)

        toc = time.clock()
        elapsed_sec = toc - tic
        rewards = [traj["reward"].sum() for traj in trajectories]
        advantages = [traj["advantage"].sum() for traj in trajectories]
        episode_lengths = [len(traj["reward"]) for traj in trajectories]

        # # Print progress
        # print("------------Return--------------")
        # print("Average_Return", np.mean(rewards))
        # print("Std_Return", np.std(rewards))
        # print("Max_Return", np.max(rewards))
        # print("Min_Return", np.min(rewards))
        # print("------------Advs----------------")
        # print("Average_Advs", np.mean(advantages))
        # print("Std_Advs", np.std(advantages))
        # print("Max_Advs", np.max(advantages))
        # print("Min_Advs", np.min(advantages))
        # print("------------Ep------------------")
        # print("Num_Total_Ep", len(episode_lengths))
        # print("Mean_Ep_Len", np.mean(episode_lengths))
        # print("Std_Ep_Len", np.std(episode_lengths))
        # print("Sec_per_interaction: ", elapsed_sec)

        # Log progress
        logz.log_tabular("Time", elapsed_sec)
        logz.log_tabular("Iteration", iter)
        logz.log_tabular("Average_Return", np.mean(rewards))
        logz.log_tabular("Std_Return", np.std(rewards))
        logz.log_tabular("Max_Return", np.max(rewards))
        logz.log_tabular("Min_Return", np.min(rewards))
        logz.log_tabular("Average_Advs", np.mean(advantages))
        logz.log_tabular("Std_Advs", np.std(advantages))
        logz.log_tabular("Max_Advs", np.max(advantages))
        logz.log_tabular("Min_Advs", np.min(advantages))
        logz.log_tabular("Num_Total_Ep", len(episode_lengths))
        logz.log_tabular("Mean_Ep_Len", np.mean(episode_lengths))
        logz.log_tabular("Std_Ep_Len", np.std(episode_lengths))
        logz.log_tabular("Sec_per_iteration: ", elapsed_sec)
        logz.dump_tabular()
        logz.pickle_tf_vars()


def normalize(data, mean=0.0, std=1.0):
    n_data = (data - np.mean(data)) / (np.std(data) + 1e-8)
    return n_data * (std + 1e-8) + mean


def get_sum_of_reward(rewards, gamma):
    return sum((gamma ** i) * rewards[i] for i in range(len(rewards)))


def get_discounted_rewards_to_go(rewards, gamma):
    """ state/action-centric policy gradients; reward-to-go=True.
    """
    rtgs = []
    future_reward = 0
    # start at time step t and use future_reward to calculate current reward
    for r in reversed(rewards):
        future_reward = future_reward * gamma + r
        rtgs.append(future_reward)
    rtgs.reverse()
    return rtgs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default="HalfCheetah-v2")
    parser.add_argument('--max_episode_steps', type=int, default=0)
    parser.add_argument('--n_episode_per_batch', type=int, default=100)
    parser.add_argument('--n_iterations', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', type=int, default=3)
    parser.add_argument('--isRenderding', type=bool, default=True)
    parser.add_argument('--isRecordingVideo', type=bool, default=True)
    parser.add_argument('--recordingVideo_dir', type=str, default="video")
    parser.add_argument('--logging_dir', type=str, default="log")
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--layer_sizes', nargs='+', type=int, default=[32, 32, 32])
    parser.add_argument('--isAdaptiveStd', '-std', action='store_true')
    parser.add_argument('--std_layer_sizes', nargs='+', type=int, default=[32, 32, 32])
    parser.add_argument('--value_layer_sizes', nargs='+', type=int, default=[32, 32, 32])
    parser.add_argument('--value_learning_rate', type=float, default=0.01)
    parser.add_argument('--activation', type=str, default="None")
    parser.add_argument('--value_activation', type=str, default="None")
    parser.add_argument('--isNNBaseLine', '-bl', action='store_true')
    parser.add_argument('--isNormalizeAdv', '-norm', action='store_true')
    parser.add_argument('--isRTG', '-rtg', action='store_true')
    parser.add_argument('--test_name', type=str, default="test")
    args = parser.parse_args()

    # args.isNNBaseLine = True
    # args.isNormalizeAdv = True
    # args.isRTG = True
    # args.isAdaptiveStd = True

    # Environment variables
    env_name = args.env_name
    max_episode_steps = args.max_episode_steps

    # Data logging paths
    recordingVideo_dir = os.path.join(args.recordingVideo_dir, args.env_name, args.test_name,
                                      time.strftime("%d-%m-%Y_%H-%M-%S"))
    if not os.path.exists(recordingVideo_dir):
        os.makedirs(recordingVideo_dir)

    logging_dir = os.path.join(args.logging_dir, args.env_name, args.test_name, time.strftime("%d-%m-%Y_%H-%M-%S"))
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    # Build gym environment
    env = gym.make(env_name)
    if max_episode_steps > 0:
        env._max_episode_steps = max_episode_steps
    print("Max. episode steps: {}".format(env._max_episode_steps))

    # Identify states and action dimensions
    isDiscrete = isinstance(env.action_space, gym.spaces.Discrete)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n if isDiscrete else env.action_space.shape[0]

    # # Learning model config
    # gamma = args.gamma  # reward decaying factor [0.0, 1.0]
    # learning_rate = args.learning_rate
    # layer_sizes = args.layer_sizes  # will add the last layer based on the action space size

    print("Argument Lists")
    for arg in vars(args):
        print(arg + ": ", getattr(args, arg))

    # Policy estimation model
    pg_model = PolicyEstimator(n_state=n_states, n_action=n_actions, isDiscrete=isDiscrete,
                               layer_sizes=args.layer_sizes,
                               activation_fn=args.activation, learning_rate=args.learning_rate,
                               isAdaptiveStd=args.isAdaptiveStd, std_layer_sizes=args.std_layer_sizes)

    # Value estimation model
    value_model = ValueEstimator(n_state=n_states, isDiscrete=isDiscrete,
                                 layer_sizes=args.value_layer_sizes, activation_fn=args.value_activation,
                                 learning_rate=args.value_learning_rate)

    global_step = tf.Variable(0, name="global_step", trainable=False)

    for exp in range(args.n_experiments):
        # Set random
        seed = args.seed + 10 * exp
        tf.set_random_seed(seed)
        np.random.seed(seed)

        print("==========================================")
        print("Exp: ", exp)
        print("==========================================")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Run REINFORCE algorithm
            reinforce(sess=sess, exp=exp, pg_model=pg_model, value_model=value_model, env=env, gamma=args.gamma,
                      n_iterations=args.n_iterations, n_batch=args.n_episode_per_batch,
                      isRenderding=args.isRenderding, isRecordingVideo=args.isRecordingVideo,
                      isNNBaseLine=args.isNNBaseLine, isNormalizeAdvantage=args.isNormalizeAdv, isRTG=args.isRTG,
                      isAdaptiveStd=args.isAdaptiveStd,
                      test_name=args.test_name, recordingVideo_dir=recordingVideo_dir, logging_dir=logging_dir,
                      seed=seed)
