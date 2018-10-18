#!/usr/bin/env bash
source ~/ml_env/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sujinj/.mujoco/mjpro150/bin
#python gym_pg_test.py --env_name "InvertedPendulum-v2" --test_name rtg_norm_bl -rtg -norm -bl --max_episode_steps 200 --n_episode_per_batch 100 --n_iterations 100 --n_experiments 5 --gamma 0.99 --learning_rate 0.01 --layer_size 32 32 32 --activation tanh
python gym_pg_test.py --env_name "InvertedPendulum-v2" --test_name rtg_norm_bl_50 -rtg -norm -bl --max_episode_steps 50 --n_episode_per_batch 100 --n_iterations 100 --n_experiments 5 --gamma 0.99 --learning_rate 0.01 --layer_size 32 32 32  --value_learning_rate 0.001 --value_layer_size 32 32 32 --activation tanh --value_--activation tanh