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

import tensorflow as tf
import gym

sys.dont_write_bytecode = True
import q_lambda_agent
import imagetraining_agent


def normalize(env, state):
    return (state - env.observation_space.low) / (env.observation_space.high - env.observation_space.low)


if __name__ == '__main__' :
    if len(sys.argv) < 2 :
        print("Usage : %s image/normal")
        sys.exit()
    
    n_episodes = 3000
    save_per_episodes = 200
    n_maxframecount = 200 # 
    epsilon_decaying_frames = 100000 # original 1000000
    min_epsilon = 0.1
    
    con_reward = deque(maxlen=100)
    con_reward_rec = np.zeros(n_episodes)

    # setup env
    env = gym.make('MountainCar-v0')

    num_act = env.action_space.n
    num_obs = env.observation_space.shape[0]

    if sys.argv[1] == "image" :
        agent = imagetraining_agent.ImageTrainingAgent(num_obs,num_act)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        agent.set_tfsess(sess)

    else :    
        agent = q_lambda_agent.QAgent(num_obs,num_act)

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
            
            action_t = agent.select_action(observation_t, epsilon)
            observation_t1, reward_t, done, info = env.step(action_t)
            observation_t1 = normalize(env, observation_t1)
            
            agent.experience(observation_t,action_t,reward_t,observation_t1,done)

            episode_reward += reward_t
            observation_t = observation_t1

            iteration_count += 1
        
        con_reward.append(episode_reward)
        con_reward_rec[e] = np.mean(con_reward)
        format_str = ('%s: %d, reward = %d, itr = %d')
        print (format_str % (datetime.now(), e, episode_reward, iteration_count))
        np.save(sys.argv[1] + "_rec.npy", con_reward_rec)

    np.save(sys.argv[1] + "_trainrec.npy", con_reward_rec)

