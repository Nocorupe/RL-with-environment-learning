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

sys.dont_write_bytecode = True
import simnet
import q_lambda_agent

class ImageTrainingAgent :
    
    def __init__(self,num_obs,num_act):
        #with tf.Graph().as_default() as tfgraph:
        self.sim = simnet.SimulationNet(num_obs,num_act,max_timestep=200)
        self.sim.build_model()
        #self.sess = tf.Session()
        #self.sess.run(tf.global_variables_initializer())
        #self.sim.set_sess(self.sess)
        #self.saver = tf.train.Saver()

        self.q_agent = q_lambda_agent.QAgent(num_obs,num_act)
        self.S = deque()

        self.sim_start_db = np.zeros((128,8,num_obs + num_act))
        self.sim_start_db_count = 0

        self.sim_iteration = 0
        self.imtr_count = 0
        self.imtr_crash = 0
        self.imtr_success = 0

        self.epsilon = 0.1

    def set_tfsess(self,sess) :
        self.sim.set_sess(sess)
        
    def act_one_hot(self, action) :
        act = np.zeros(3)
        act[action] = 1.0
        return act
        

    def experience(self, state_t, action_t, reward_t, state_t1, done) :
        state_act = np.concatenate((state_t,self.act_one_hot(action_t)))
        self.S.append(state_act)
        if len(self.S) == 8 :
            self.sim_start_db[ self.sim_start_db_count % 128 ] = np.array(self.S)
            self.sim_start_db_count += 1 

        if done :
            self.sim.append_episode(self.S)
            self.S.clear()
            format_str = ('IMTR : imtr = %d(s=%d,c=%d), sim_itr = %d, sim_loss = %f')
            print (format_str % (self.imtr_count,self.imtr_success,self.imtr_crash, self.sim_iteration,self.sim.loss_value))

        if done :
            if self.sim_start_db_count > 50 :
                n = 4
            else :
                n = 8
            for i in range(n) :
                self.sim.episode_replay()
                self.sim_iteration += 1
                    
        if done :
            if self.sim_start_db_count > 50 and self.sim.loss_value < 0.1 :
                self.image_training()
        
        self.q_agent.experience(state_t,action_t,reward_t,state_t1,done)



    def image_training(self) :
        Ssim = deque()
        sim_done = False
        start = np.random.randint(low=0,high=min(self.sim_start_db_count,128))
        
        for i in range(8) :
            Ssim.append(self.sim_start_db[start][i])
        
        observation_t = self.sim.run_inference(Ssim)


        while not sim_done :
            action_t = self.q_agent.select_action(observation_t,self.epsilon) 

            Ssim.append(np.concatenate((observation_t, self.act_one_hot(action_t))))
            observation_t1 = self.sim.run_inference(Ssim)
            
            # MountainCar-v0 code
            done = False
            if observation_t1[0] < 0.0 :
                #simulation failure
                reward_t = -1
                sim_done = True
            elif observation_t1[0] > 2.0 :
                #simulation failure
                reward_t = -1
                sim_done = True
            elif observation_t1[0] > 0.90 : # position = 0.5 -> 0.94
                reward_t = 0
                done = True
                sim_done = True
            else :
                reward_t = -1
                done = False
                
            if len(Ssim) == 200 : 
                done = True
                sim_done = True

            self.q_agent.experience(observation_t,action_t,reward_t,observation_t1,done)
            observation_t = observation_t1
        
        self.imtr_count += 1
        if not done :
            self.imtr_crash += 1
        elif len(Ssim) < 200 :
            self.imtr_success += 1
        
     


    def select_action(self, state, epsilon):
        self.epsilon = epsilon
        return self.q_agent.select_action(state,epsilon)
    
    """
    def save(self, target, episode_count) :
        self.sim.save_episodes(target + ".ckpt-%d.npz" % (episode_count))
        self.saver.save(self.sess, target +".ckpt", global_step=episode_count)
        self.q_agent.save(target + ".ckpt-%d.q"%(episode_count))
    """
