# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time
from random import randrange
from collections import deque
from datetime import datetime
import numpy as np

import gym

sys.dont_write_bytecode = True
import q_lambda_agent2 as q_agent

def normalize(env, state):
    return (state - env.observation_space.low) / (env.observation_space.high - env.observation_space.low)


if __name__ == '__main__' :
    # parameters
    n_episodes = 5000
    n_maxframecount = 200 # 
    epsilon_decaying_frames = 100000 # original 1000000
    min_epsilon = 0.1
    iteration_per_save = 10000


    # setup env
    env = gym.make('MountainCar-v0')

    action_num = env.action_space.n
    observation_dims = env.observation_space.shape[0]

    agent = q_agent.QAgent(observation_dims,action_num)

    iteration_count = 0
    for e in xrange(n_episodes):
        observation_t = env.reset()
        observation_t = normalize(env, observation_t)
        done = False

        episode_reward = 0

        epsilon  = min_epsilon
        while not done:
            if iteration_count < epsilon_decaying_frames :
                epsilon = 1.0 - ((1.0 - min_epsilon) / epsilon_decaying_frames) * iteration_count
            
            action_t,q_t = agent.select_action(observation_t, epsilon)
            observation_t1, reward_t, done, info = env.step(action_t)
            observation_t1 = normalize(env, observation_t1)
            
            agent.experience(observation_t,action_t,reward_t,observation_t1,done)

            episode_reward += reward_t
            observation_t = observation_t1

            iteration_count += 1
            
        format_str = ('%s: %d, reward = %d, max_q = (%2.f), itr = %d')
        print (format_str % (datetime.now(), e, episode_reward, q_t, iteration_count))
        #print(dqn.D[0][0])
        if (e+1) % 1000 == 0 :
            agent.display_policy()




