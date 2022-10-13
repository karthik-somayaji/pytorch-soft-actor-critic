import argparse

import gym
import dnc2s_rl.agents
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces
from gym import spaces
from gym.utils import seeding

import torch
import seaborn as sns

def plot_figures(env_id, trial_num):
    num_smooth =91
    num_pts = 10

    agent_id = 'Torch_DDPG-v0'
    reward_QR_DDPG = np.zeros((trial_num, num_smooth))

    for i in range(trial_num):
        load_path = 'saved_rewards/' + env_id + '_' + agent_id + '_' + str(i) + '.npy'
        x = np.load(load_path)
        x = np.convolve(x, np.ones(num_pts), 'valid') / num_pts
        reward_QR_DDPG[i,:] = x

    print(reward_QR_DDPG.shape)
    mean_reward_QR_DDPG = reward_QR_DDPG.mean(0)
    std_reward_QR_DDPG = reward_QR_DDPG.std(0)

    agent_id = 'Torch_TD3-v2'
    reward_TD3 = np.zeros((trial_num, num_smooth))

    for i in range(trial_num):
        load_path = 'saved_rewards/' + env_id + '_' + agent_id + '_' + str(i) + '.npy'
        x = np.load(load_path)[:100]
        x = np.convolve(x, np.ones(num_pts), 'valid') / num_pts
        reward_TD3[i,:] = x
    print(reward_TD3.shape)
    mean_reward_TD3 = reward_TD3.mean(0)
    std_reward_TD3 = reward_TD3.std(0)

    agent_id = 'TorchQR_TD3_UP-v2'
    reward_QR_TD3_UP_v1 = np.zeros((trial_num, num_smooth))

    for i in range(trial_num):
        print(i)
        load_path = 'saved_rewards/' + env_id + '_' + agent_id + '_' + str(i) + '.npy'
        x = np.load(load_path)
        x = np.convolve(x, np.ones(num_pts), 'valid') / num_pts
        print(x.shape)
        reward_QR_TD3_UP_v1[i,:] = x
    print(reward_QR_TD3_UP_v1.shape)
    mean_reward_QR_TD3_UP_v1 = reward_QR_TD3_UP_v1.mean(0)
    std_reward_QR_TD3_UP_v1 = reward_QR_TD3_UP_v1.std(0)

    agent_id = 'TorchQR_DDPG_UP-v2'
    reward_QR_DDPG_UP_v1 = np.zeros((trial_num, num_smooth))

    for i in range(trial_num):
        print(i)
        load_path = 'saved_rewards/' + env_id + '_' + agent_id + '_' + str(i) + '.npy'
        x = np.load(load_path)
        x = np.convolve(x, np.ones(num_pts), 'valid') / num_pts
        print(x.shape)
        reward_QR_DDPG_UP_v1[i,:] = x
    print(reward_QR_DDPG_UP_v1.shape)
    mean_reward_QR_DDPG_UP_v1 = reward_QR_DDPG_UP_v1.mean(0)
    std_reward_QR_DDPG_UP_v1 = reward_QR_DDPG_UP_v1.std(0)


    sns.set()
    #plt.figure()
    x = np.arange(91)
    plt.plot(x, mean_reward_QR_DDPG, label='DDPG')
    plt.fill_between(x, mean_reward_QR_DDPG - std_reward_QR_DDPG, mean_reward_QR_DDPG + std_reward_QR_DDPG, alpha=0.2)
    
    plt.plot(x, mean_reward_TD3, label='TD3')
    plt.fill_between(x, mean_reward_TD3 - std_reward_TD3, mean_reward_TD3 + std_reward_TD3, alpha=0.2)
    
    plt.plot(x, mean_reward_QR_TD3_UP_v1, label = 'QR-TD3-Soft-Actor')
    #plt.plot(x, reward_QR_TD3_UP_v1[0,:], label = 'QR-TD3-Soft-Actor1')
    #plt.plot(x, reward_QR_TD3_UP_v1[1,:], label = 'QR-TD3-Soft-Actor2')
    #plt.plot(x, reward_QR_TD3_UP_v1[2,:], label = 'QR-TD3-Soft-Actor3')
    plt.fill_between(x, mean_reward_QR_TD3_UP_v1 - std_reward_QR_TD3_UP_v1, mean_reward_QR_TD3_UP_v1 + std_reward_QR_TD3_UP_v1, alpha=0.2)
    
    plt.plot(x, mean_reward_QR_DDPG_UP_v1, label = 'QR-DDPG-Soft-Actor')
    plt.fill_between(x, mean_reward_QR_DDPG_UP_v1 - std_reward_QR_DDPG_UP_v1, mean_reward_QR_DDPG_UP_v1 + std_reward_QR_DDPG_UP_v1, alpha=0.2)

    plt.xlabel('x5e+3 timesteps')
    plt.ylabel('Average Reward')
    plt.legend()
    y =  env_id + '.png'
    plt.savefig(y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--agent", help="Agent used for RL", type=str, default='TorchQR_DDPG-v0')
    #parser.add_argument("--env", help="Environment used for RL", type=str, default='Pendulum-v1')
    #parser.add_argument("--agent", help="Agent used for RL", type=str, default='TorchQR_DDPG_ensemble-v1')
    #parser.add_argument("--agent", help="Agent used for RL", type=str, default='TorchQR_DDPG_UP-v1')
    #parser.add_argument("--env", help="Environment used for RL", type=str, default='car_1D-v0')
    #parser.add_argument("--env", help="Environment used for RL", type=str, default='HalfCheetah-v3')
    #parser.add_argument("--env", help="Environment used for RL", type=str, default='Ant-v3')
    #parser.add_argument("--env", help="Environment used for RL", type=str, default='Walker2d-v3')
    #parser.add_argument("--env", help="Environment used for RL", type=str, default='Swimmer-v3')
    parser.add_argument("--env", help="Environment used for RL", type=str, default='Hopper-v3')
    #parser.add_argument("--env", help="Environment used for RL", type=str, default='MountainCarContinuous-v0')
    parser.add_argument("--trial_num", help="Trial num of algo run", type=int, default=3)

    # Get input arguments
    args = parser.parse_args()
    env_id = args.env
    trial = args.trial_num

    # Print input settings
    plot_figures(env_id, 3)