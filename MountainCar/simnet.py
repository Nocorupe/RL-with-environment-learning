# -*- coding: utf-8 -*-
from collections import deque
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


class SimulationNet:
    # Hyper-parameters
    minibatch_size = 32
    N = (10**4) * 5
    learning_rate = 0.01


    def __init__(self,num_obs,num_act,max_timestep):
        self.num_obs = num_obs
        self.num_act = num_act
        self.num_input = self.num_obs + self.num_act
        self.max_timestep = max_timestep
        # replay memory
        self.Episodes = [   np.zeros((self.N, self.max_timestep, self.num_input), dtype=np.float ),
                            np.zeros((self.N) ,dtype=np.int32) ]
        #self.Episodes = deque(maxlen=self.N)
        self.loss_value = 0.0
        self.mem_count = 0

    def set_sess(self,sess) :
        self.sess = sess
       

    def build_model(self) :
        def fully_connected(x,n_in,n_out) :
            w = tf.Variable( tf.truncated_normal([n_in,n_out],stddev=0.01) )
            b = tf.Variable( tf.truncated_normal([n_out],stddev=0.01) )
            return (tf.matmul(x,w) + b)

        self.x = tf.placeholder(tf.float32,[None,self.max_timestep,self.num_input])
        batch_size = tf.shape(self.x)[0]

        self.cell = rnn.BasicLSTMCell(128, forget_bias=0.8)
        self.sequence_length = tf.placeholder(tf.int32, [None])
        initial_state = self.cell.zero_state( batch_size, tf.float32 )
        self.seq_output, states_op = tf.nn.dynamic_rnn(self.cell, self.x ,sequence_length=self.sequence_length ,initial_state=initial_state)
        
        index = tf.range(0, batch_size) * self.max_timestep + (self.sequence_length - 1)
        self.cell_out = tf.gather(tf.reshape(self.seq_output, [-1, 128]), index)

        self.inference = fully_connected(self.cell_out, 128, self.num_obs)
        
        self.y = tf.placeholder("float", shape=(None,self.num_obs))
        self.loss = tf.reduce_mean(tf.nn.l2_loss(self.y  - self.inference )) # average v = 0.01
        
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.training = self.optimizer.minimize(self.loss)

       
    def run_inference(self, sequence):
        slen = min(len(sequence),self.max_timestep)
        x = np.zeros((self.max_timestep,self.num_input))
        x[:slen, 0:self.num_input] = np.array(sequence)[:slen, 0:self.num_input]
        slen_minibatch = [slen]
        return self.sess.run(self.inference, feed_dict={self.x : [x], self.sequence_length: slen_minibatch})[0]

    def append_episode(self, sequence):
        size = min(len(sequence),self.max_timestep)
        data_index = (self.mem_count) % self.N

        self.Episodes[0][data_index] = np.zeros((self.max_timestep,self.num_input))
        self.Episodes[0][data_index][:size, 0:self.num_input] = np.array(sequence)[:size, 0:self.num_input]
        self.Episodes[1][data_index] = size

        self.mem_count = (self.mem_count + 1)
        if self.mem_count > 2**16 :
            self.mem_count = self.N + (self.mem_count%self.N)
        
        
    def episode_replay(self,depth=-1):
        if self.mem_count < 1 :
            return 
        # sample random minibatch
        data_index = ((self.mem_count) % self.N) -1
        minibatch_indexes = np.zeros(self.minibatch_size/4,dtype=int)
        if self.minibatch_size/4 < self.mem_count :
            for i in range(self.minibatch_size/4) :
                minibatch_indexes[i] = data_index -i
        
        mem_max_index = min(self.mem_count, self.N)
        minibatch_indexes = np.concatenate((minibatch_indexes,np.random.randint(low=0, high=mem_max_index, size=(self.minibatch_size/4)*3)))

        sequence_minibatch = []
        y_minibatch = []
        slen_minibatch = np.zeros(len(minibatch_indexes),dtype=np.int32)
        gather_minibatch = []
        for i in range(len(minibatch_indexes)):
            j = minibatch_indexes[i]
            sequence_j = self.Episodes[0][j]
            if depth > 0 :
                ts = min(self.Episodes[1][j] -1,depth)
            else :
                ts = np.random.randint(low=1, high=self.Episodes[1][j])
            y_j = self.Episodes[0][j][ts][0:self.num_obs]
            gather_j = np.array([ts, i])

            sequence_minibatch.append(sequence_j)
            y_minibatch.append(y_j)
            slen_minibatch[i] = ts
            gather_minibatch.append(gather_j)

        # training
        self.sess.run(self.training, feed_dict={self.x: sequence_minibatch, self.y: y_minibatch, self.sequence_length : slen_minibatch })#, self.gather_indices: gather_minibatch})
        self.loss_value = self.sess.run(self.loss, feed_dict={self.x: sequence_minibatch, self.y: y_minibatch, self.sequence_length : slen_minibatch})#, self.gather_indices : gather_minibatch})
    
    """
    def save_episodes(self,name) :
        np.savez(name,ep0=self.Episodes[0],ep1=self.Episodes[1], data_len=self.mem_count)
    
    def load_episodes(self,name) :
        n = np.load(name)
        self.Episodes[0] = n['ep0']
        self.Episodes[1] = n['ep1']
        self.mem_count = n['data_len']
    """