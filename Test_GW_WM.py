#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 17:11:01 2024

@author: josephinepazem
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
from GW_FEPS_functions_parallel import Environment_GW, Agent, Plots, AgentWrapper, Normal_Agent
from GW_FEPS_functions_parallel import rand_choice_nb
#from GW_FEPS_functions import Optimal_agent
import numba
from numba import njit
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed, parallel_backend
import dill as pickle
from collections import Counter

############## Initialize the variables   ############## 
fake_zero = 10 ** (-8)
save_figures = False

# Where to save the results
tests = ["wandering_1", "no_wandering_-3"]
tests_labels = ['wander, ', "task, "]
# tests = ["wandering_1", "wandering_0"]
# tests_labels = ['wander 1', "uniform"]


############## Variables for training and hyperparameters
# Training parameters
N_agents = 30
N_episodes = 40000 #10000
T_episodes = 80 # number of steps per episode

N_tests = 1000
T_test = 80

# PS parameters
causal_model = False

gamma_A = 0.001  # 0.001
scale_reward_posterior = 3. # 3.  
eta_A = 1.
h_0 = 0.1
eta_g = 1/2


# Emotional thinking parameters
wandering_phase = False
preferences_marginal = False
curiosity_wander = -1
curiosity_task = 3 # The "minus" is set in the function to make the policy
emotional_thinking = False
alpha_posterior = 5
emotion_actions = False
alpha_policy = 0.2#0.5 #0.5 #0.5 # 1/5 #1/2 #0.8#/2

# Parameters task phase
prediction_horizon = 3
proba_preferred_obs = 0.99
discount = 0.8
target = 3

# Environement's configuration
dim = 3
observations = np.array([0,1,2,3]) ### numba
directions = {0:np.array([0,1]),
              1:np.array([0,-1]),
              2:np.array([1,0]),
              3:np.array([-1,0])}
symmetric=False
periodic_boundaries = False

# Agent's parameters
actions = np.array([0,1,2,3])   # [Wait, Go]    ### numba
N_actions = len(actions)
N_observations = len(observations)
N_clones = 3
N_states = N_observations * N_clones


# Make the environment for all agents
env = Environment_GW(dim, observations, directions)
env.make_elementary_grid(symmetric=symmetric)

uniform_policy = np.ones((N_states, N_actions)) / N_actions
# gather the results in a list

prediction_lengths = []
prediction_lengths_superposition = []

for wander_task in range(len(tests)):
   prediction_lengths.append([])
   prediction_lengths_superposition.append([])
   
   for actor in range(N_agents):
      # print(actor)
      
      # Load the world model of the agent
      folder = "/Users/josephinepazem/Documents/FEPS_data/19DEC24_GW_" + tests[wander_task] + "/"
      agent = joblib.load(folder + "testing_agent_" + str(actor)+ ".pkl")
      agent.policy = uniform_policy

      #### Test with the bare belief estimation
      np.random.seed(13485248)
      prediction_length_actor = 0
      for episode in range(N_tests):
         t_steps = 0
         
         # Initialize the environment
         env.observation = env.initial_conditions()   # Drop the agent somewhere in the grid
         agent.select_first_state(env.observation)
         agent.previous_action = agent.select_first_action()
         
         for t_steps in range(T_test+1):
            #### For a number of tests, have it predict observations for max 80 steps
            agent.belief_state = agent.deliberate_next_state(False, 10, False)
            agent.obs_agent = agent.deliberate_observation()
            # Apply action in env
            env.agent_position = env.move_the_agent(agent.previous_action, periodic_boundaries=periodic_boundaries)
            env.observation = env.give_observation()
            #### Compare prediction and observation
            # if agent.obs_agent == env.observation:
            #    t_steps += 1
            if t_steps == T_test or agent.obs_agent != env.observation:
               prediction_length_actor += t_steps / N_tests
               break
            # else: 
            #    prediction_length_actor += t_steps / N_tests
            #    break
               
            # Choose the next action and prepare next round
            agent.previous_action = agent.deliberate_next_action(5)
            agent.previous_belief_state = agent.belief_state + 0
            
      prediction_lengths[wander_task].append(prediction_length_actor)   
      
      #### Test in superposition
      prediction_length_superposition_actor = 0
      
      for episode in range(N_tests):
         
          step = 0
         
          # Initialize the environment
          env.observation = env.initial_conditions()   # Drop the agent somewhere in the grid
         
          agent.plausible_previous_belief_states = list(range(env.observation * agent.N_clones, 
                                                           env.observation * agent.N_clones + agent.N_clones))
         
          agent.plausible_previous_actions = [agent.deliberate_next_action() 
                                               for agent.belief_state in agent.plausible_previous_belief_states]                                         
         
          agent.previous_action = Counter(agent.plausible_previous_actions).most_common(1)[0][0]
         
          for step in range(1, T_test):

            # Apply action in env
            env.env_state = env.move_the_agent(agent.previous_action, periodic_boundaries = periodic_boundaries)
            env.observation = env.give_observation()
               
            # Based on the observation, deliberate:
            agent.plausible_belief_states = [agent.deliberate_next_state(emotional_thinking=False, reflection_time=0) 
                                               for agent.previous_belief_state in agent.plausible_previous_belief_states]
            
            agent.plausible_observations = [agent.deliberate_observation() for agent.belief_state 
                                               in agent.plausible_belief_states]
            
            agent.plausible_previous_belief_states = [agent.plausible_belief_states[i] 
                                                        for i in range(len(agent.plausible_belief_states))
                                                        if agent.plausible_observations[i]==env.observation]
            
            if len(agent.plausible_previous_belief_states) == 0:
               prediction_length_superposition_actor += step / N_tests
               break
            
               # agent.plausible_previous_belief_states = list(range(env.observation * agent.N_clones,
               #                                                    env.observation * agent.N_clones + agent.N_clones))
            
            agent.plausible_previous_actions = [agent.deliberate_next_action() 
                                                  for agent.belief_state in agent.plausible_previous_belief_states] 
            
            agent.previous_action = Counter(agent.plausible_previous_actions).most_common(1)[0][0]
               
            if step == T_test - 1:
               prediction_length_superposition_actor += step / N_tests
              
      prediction_lengths_superposition[wander_task].append(prediction_length_superposition_actor)   
            
            
data = {tests_labels[0]+"bare":prediction_lengths[0], 
        tests_labels[0]+"sup.":prediction_lengths_superposition[0],
        tests_labels[1]+"bare":prediction_lengths[1], 
        tests_labels[1]+"sup.":prediction_lengths_superposition[1]} 

data = {"Length of trajectories": prediction_lengths[0] + prediction_lengths_superposition[0] + prediction_lengths[1] + prediction_lengths_superposition[1],
        "Scenarios": [tests_labels[0] + "bare"] * N_agents + [tests_labels[0]+"sup."] * N_agents + [tests_labels[1] + "bare"] * N_agents + [tests_labels[1]+"sup."] * N_agents
       }

palette = [(0.8622837370242215, 0.42952710495963087, 0.34271434063821604),
           (0.9686274509803922, 0.7176470588235294, 0.6),
           (0.142483660130719, 0.4173010380622838, 0.6833525567089581),
           (0.32349096501345653, 0.6149173394848136, 0.7854671280276817)]
sns.set_theme(context="paper", style="whitegrid", font_scale=2)
fig, ax = plt.subplots(1, dpi=300, figsize = (8,6))
sns.violinplot(data=data, x = "Scenarios", y="Length of trajectories", 
                density_norm="count", inner="box", 
                linewidth = 0.1, ax=ax,
                inner_kws=dict(box_width=10, whis_width=3, color="0.1"),
                palette=palette, fill=True) #, height=10, aspect=1.1)

# sns.boxplot(data, palette = palette, saturation=1, linewidth=1.5, fill=False)
sns.catplot(data=data, x = "Scenarios", y="Length of trajectories",
            kind="box",  palette=palette, 
            ax=ax, height=6, aspect=1.4, 
            linewidth = 3, fill=False)
# sns.stripplot(data=data, jitter=True, palette=palette, size=10, edgecolor="0.2", linewidth=0.5)

# sns.boxplot(data=data, x = "Scenarios", y="Length of trajectories",
#             palette=palette, ax=ax, whis_width = 2)
ax.set_ylabel("length of trajectories")
#ax.grid()
plt.show()   
            


      
      

   
# Collect the test outcomes
   
# Save the test outcomes to a file

# Make a figure



