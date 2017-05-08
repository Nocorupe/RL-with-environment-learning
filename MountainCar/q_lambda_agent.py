# -*- coding: utf-8 -*-
# 
# Q(lambda) で学習を行うエージェント
# breekoの実装を参考
# https://gym.openai.com/evaluations/eval_BfzwiYkhQsmV3W8SLUQ0SA
# 
import numpy as np
from itertools import product
import matplotlib.pyplot as plt


def get_action_values(activations, theta):
    """ Returns the value of each action at some state"""
    return np.dot(theta.T, activations)

def get_action_value(activations, action, theta):
    """ Returns the value of an action at some state"""
    return np.dot(theta[:, action], activations)


class QAgent :
    num_state_split = 4
    alpha = 0.01
    gamma = 0.5#0.99
    lambda_ = 0.99#0.5
    

    def __init__(self,num_obs,num_act):
        self.num_obs = num_obs
        self.num_act = num_act

        self.discretization_map = np.array(list(product(np.linspace(0,1,self.num_state_split),repeat=num_obs)))
        self.theta = np.random.random([self.discretization_map.shape[0], num_act]) - .5
        #self.theta = np.zeros([self.discretization_map.shape[0], num_act])
        self.eligibility = np.zeros_like(self.theta) 
        self.a = -1
        self.epsilon = 1.0
 
        

    def activation(self,state) :
        return np.exp([-np.linalg.norm(state - center)**2 / .05 for center in self.discretization_map])

    def get_epsilon_greedy(self, vals, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.randint(self.num_act)
        return vals.argmax()


    def experience(self, state_t, action_t, reward_t, state_t1, done) :
        activations = self.activation(state_t)
        new_activations = self.activation(state_t1)
        new_vals = get_action_values(activations,self.theta)
        new_action = self.get_epsilon_greedy(new_vals, epsilon=self.epsilon)
        q = get_action_value(activations,action_t,self.theta)
        new_q = get_action_value(new_activations,new_action,self.theta)
        
        if done:
            target = reward_t - q
        else:
            target = reward_t + self.lambda_ * (new_q - q)
            
        self.eligibility[:,action_t] = activations
        self.theta += self.alpha * target * self.eligibility
        self.eligibility *= self.lambda_ * self.gamma

        self.a = new_action
        

    def select_action(self, state, epsilon):
        self.epsilon = epsilon
        if self.a == -1 :
            activations = self.activation(state)
            return self.get_epsilon_greedy(get_action_values(activations,self.theta))
        else :
            return self.a
        

