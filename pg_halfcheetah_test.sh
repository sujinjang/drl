#!/usr/bin/env bash
source ~/ml_env/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sujinj/.mujoco/mjpro150/bin
python gym_pg_test.py --env_name "HalfCheetah-v2" --test_name rtg_norm_bl_100 -rtg -norm -bl -std --max_episode_steps 150 --n_episode_per_batch 50 --n_iterations 100 --n_experiments 3 --gamma 0.90 --learning_rate 0.01 --layer_size 64 64 64 --value_learning_rate 0.01 --value_layer_size 64 64 --std_layer_size 64 64 64 --activation tanh --value_activation tanh
