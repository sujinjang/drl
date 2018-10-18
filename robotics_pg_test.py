import argparse
import os
import time

import numpy as np
import gym
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from policy_estimator_model import PolicyEstimator
from value_estimator_model import ValueEstimator
from gym.wrappers.monitoring.video_recorder import VideoRecorder


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


def reinforce(pg_model, value_model, env, gamma, isRTG=True,
              n_iterations=100, min_steps_in_batch=300,
              isRenderding=True,
              isRecordingVideo=True, recordingVideo_dir="recording"):
    env_name = env.spec.id
    print("Policy Gradient for {} Environment".format(env_name))
    for iter in range(n_iterations):
        print("==========================================")
        print("Iteration: ", iter)

        steps_in_batch = 0
        trajectories = []
        tic = time.clock()
        episode = 1

        while True:  # Outer loop for collecting trajectory batches
            episode_states, episode_actions, episode_rewards = [], [], []
            episode_steps = 0
            state = env.reset()

            if isRecordingVideo and episode == 1 and (iter % 10 == 0 or iter == n_iterations - 1 or iter == 0):
                video_recorder = VideoRecorder(env,
                                               os.path.join(
                                                   recordingVideo_dir,
                                                   "recording_{}_{}.mp4".format(env_name, iter)),
                                               enabled=True)
                print("Recording a video of this episode {} in iteration {}".format(episode, iter))

            while True:  # Inner loop for collecting single trajectory
                if isRenderding:
                    env.render()

                    if video_recorder:
                        video_recorder.capture_frame()

                # Choose an action based on observation
                # action = pg_model.get_next_action(state['observation'])
                # action = np.reshape(action, newshape=[n_actions])

                action = pg_model.predict(state['observation'])
                action = action[0]

                # 2. Simulate one time step from action
                state, reward, done, info = env.step(action=action)

                # 3. Collect data for a trajectory
                episode_states.append(state['observation'])
                episode_actions.append(action)
                episode_rewards.append(reward)

                episode_steps += 1

                if done:
                    break

            trajectory = {"state": np.array(episode_states),
                          "action": np.array(episode_actions),
                          "reward": np.array(episode_rewards)}
            trajectories.append(trajectory)
            steps_in_batch += len(trajectory["reward"])
            episode += 1

            if steps_in_batch > min_steps_in_batch:
                break

            if video_recorder:
                video_recorder.close()
                video_recorder = None

        # 3. Batching sample trajectories
        # Generate 'ready-to-use batch arrays for state, action, and reward
        # pg_model.sample_trajectories(trajectories)
        batch_states = np.concatenate([traj["state"] for traj in trajectories])
        batch_actions = np.concatenate([traj["action"] for traj in trajectories])

        # Compute trajectory-centric reward sum
        if isRTG:
            batch_rewards = np.concatenate([
                get_discounted_rewards_to_go(traj["reward"], gamma) for traj in trajectories])
        else:
            batch_rewards = np.concatenate([
                [get_sum_of_reward(traj["reward"], gamma=gamma)] * len(traj["reward"])
                for traj in trajectories
            ])

        # Compute NN baseline estimation
        value_estimates = value_model.predict(state=batch_states)
        value_estimates = value_estimates * np.std(batch_rewards, axis=0) + np.mean(batch_rewards, axis=0)
        advantages = np.squeeze(batch_rewards) - np.squeeze(value_estimates)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # Normalize rewards (targets) and update value estimator
        batch_rewards = (batch_rewards - np.mean(batch_rewards)) / (np.std(batch_rewards) + 1e-8)

        # Update value estimator
        value_model.update(batch_states, batch_rewards)

        # Update policy estimator
        pg_model.update(batch_states, batch_actions, advantages)

        toc = time.clock()
        elapsed_sec = toc - tic
        returns = [traj["reward"].sum() for traj in trajectories]
        episode_lengths = [len(traj["reward"]) for traj in trajectories]

        # Log progress
        print("Seconds: ", elapsed_sec)
        print("AverageReturn", np.mean(returns))
        print("StdReturn", np.std(returns))
        print("MaxReturn", np.max(returns))
        print("MinReturn", np.min(returns))
        print("EpLenMean", np.mean(episode_lengths))
        print("EpLenStd", np.std(episode_lengths))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default="FetchReach-v1")
    parser.add_argument('--max_episode_steps', type=int, default=0)
    parser.add_argument('--min_steps_in_batch', type=int, default=1000)
    parser.add_argument('--n_iterations', type=int, default=1000)
    parser.add_argument('--isRenderding', type=bool, default=True)
    parser.add_argument('--isRecordingVideo', type=bool, default=True)
    parser.add_argument('--recordingVideo_dir', type=str, default="recording")
    parser.add_argument('--logging_dir', type=str, default="log")
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--layer_sizes', nargs='+', type=int, default=[32, 32, 32])
    args = parser.parse_args()

    # Environment variables
    env_name = args.env_name  # "Hopper-v2"  # "CartPole-v0"
    max_episode_steps = args.max_episode_steps
    min_steps_in_batch = args.min_steps_in_batch
    n_iterations = args.n_iterations
    isRenderding = args.isRenderding
    isRecordingVideo = args.isRecordingVideo
    recordingVideo_dir = args.recordingVideo_dir
    if not os.path.exists(recordingVideo_dir):
        os.mkdir(recordingVideo_dir)

    logging_dir = args.logging_dir
    if not os.path.exists(logging_dir):
        os.mkdir(logging_dir)

    # Build gym environment
    env = gym.make(env_name)
    env.render()
    if max_episode_steps > 0:
        env._max_episode_steps = max_episode_steps
    print("Max. episode steps: {}".format(env._max_episode_steps))

    # Build policy network model
    isDiscrete = isinstance(env.action_space, gym.spaces.Discrete)

    n_states = env.observation_space.spaces['observation'].shape[0]
    n_actions = env.action_space.shape[0]

    # Learning model config
    gamma = args.gamma  # reward decaying factor [0.0, 1.0]
    learning_rate = args.learning_rate
    layer_sizes = args.layer_sizes  # will add the last layer based on the action space size

    print("Argument Lists")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    # Policy estimation model
    pg_model = PolicyEstimator(n_state=n_states, n_action=n_actions, isDiscrete=isDiscrete, layer_sizes=layer_sizes,
                               activation_fn="relu", learning_rate=learning_rate)

    # Value estimation model
    value_model = ValueEstimator(n_state=n_states, n_action=n_actions, isDiscrete=isDiscrete,
                                 layer_sizes=layer_sizes,
                                 activation_fn="relu", learning_rate=learning_rate)

    global_step = tf.Variable(0, name="global_step", trainable=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        reinforce(pg_model=pg_model, value_model=value_model, env=env, gamma=gamma, isRTG=True,
                  n_iterations=n_iterations, min_steps_in_batch=min_steps_in_batch,
                  isRenderding=isRenderding, isRecordingVideo=isRecordingVideo,
                  recordingVideo_dir=recordingVideo_dir)

    # for iter in range(n_iterations):
    #     state = env.reset()
    #     steps_in_batch = 0
    #     trajectories = []
    #     while True:  # Outer loop for collecting trajectory batches
    #         episode_states, episode_actions, episode_rewards = [], [], []
    #         episode_steps = 0
    #
    #         while True: # Innter loop for collecting single trajectory
    #
    #             if isRenderding:
    #                 env.render()
    #
    #             # 1. Choose an action based on observation
    #             action = pg_model.get_next_action(state['observation'])
    #             action = np.reshape(action, newshape=[n_actions])
    #
    #             # 2. Simulate one time step from action
    #             rnd_action = env.action_space.sample()
    #             state, reward, done, info = env.step(action=action)
    #
    #             # 3. Collect data for a trajectory
    #             episode_states.append(state['observation'])
    #             episode_actions.append(action)
    #             episode_rewards.append(reward)
    #
    #             episode_steps += 1
    #
    #             if done:
    #                 break
    #
    #         trajectory = {"state": np.array(episode_states),
    #                       "action": np.array(episode_actions),
    #                       "reward": np.array(episode_rewards)}
    #         trajectories.append(trajectory)
    #
    #         steps_in_batch += len(trajectory["reward"])
    #
    #         if steps_in_batch > min_steps_in_batch:
    #             break
    #
    #     # 3. Batching sample trajectories
    #     pg_model.sample_trajectories(trajectories)
    #
    #     # 4. Update policy model
    #     pg_model.update_policy()


if __name__ == "__main__":
    main()