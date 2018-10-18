#!/usr/bin/env bash
source ~/ml_env/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sujinj/.mujoco/mjpro150/bin
#FetchPickAndPlace-v0, FetchReach-v1
#python robotics_pg_test.py --env_name "FetchPickAndPlace-v1" --max_episode_steps 0 --min_steps_in_batch 5000 --n_iterations 1000 --gamma 0.9 --learning_rate 0.01 --layer_size 128 128 128
#python gym_pg_test.py --env_name "HalfCheetah-v2" --max_episode_steps 150 --min_steps_in_batch 15000 --n_iterations 100 --gamma 0.9 --learning_rate 0.001 --layer_size 128 128 128 --activation None -rtg -bl -norm
#python gym_pg_test.py --env_name "CartPole-v0" --max_episode_steps 0 --min_steps_in_batch 1000 --n_iterations 100 --gamma 0.9 --learning_rate 0.01 --layer_size 64 64 64 --activation tanh --test_name pg_rtg_norm -rtg -norm -bl
#python gym_pg_test.py --env_name "CartPole-v1" --max_episode_steps 0 --min_steps_in_batch 100 --n_iterations 100 --gamma 0.99 --learning_rate 0.1 --layer_size 32 32 32 --activation None --test_name test -rtg -norm -bl
python gym_pg_test.py --env_name "CartPole-v1" --test_name rtg_norm_bl -rtg -norm -bl --max_episode_steps 0 --n_episode_per_batch 100 --n_iterations 100 --gamma 0.99 --learning_rate 0.01 --layer_size 32 32 32 --activation tanh
