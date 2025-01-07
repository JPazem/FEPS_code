#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:30:45 2024

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

############## Initialize the variables   ############## 
fake_zero = 10 ** (-8)
save_figures = False

# Where to save the results
folder = "/Users/josephinepazem/Documents/FEPS_data/07Jan25_GW_CleanUpTests/"

############## Variables for training and hyperparameters
# Training parameters
N_agents = 1 #30
N_episodes = 4000# 0 #10000
T_episodes = 80 # number of steps per episode

# PS parameters
causal_model = False

gamma_A = 0.001  # 0.001
scale_reward_posterior = 3. # 3.  
eta_A = 1.
h_0 = 0.1
eta_g = 1/2


# Emotional thinking parameters
wandering_phase = True
preferences_marginal = True
curiosity_wander = 2
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

Free_energies = np.zeros((N_agents, N_episodes))
Expected_Free_energies = np.zeros((N_agents, N_episodes))
Rewards_posterior = np.zeros((N_agents, N_episodes))
Rewards_policy = np.zeros((N_agents, N_episodes))
Correct_predictions = np.zeros((N_agents, N_episodes))
Length_trajectories = np.zeros((N_agents, N_episodes))
agents = []

# Initialize the likelihood to make it compatible with jitclass 
ECM_likelihood = np.full((N_states, N_observations), fake_zero)
for observation, duplicate in itertools.product(observations, range(N_clones)):
   ECM_likelihood[observation * N_clones + duplicate, observation] = 1

# train the agents (in parallel):
print("training")

#if __name__ == "__main__":
import multiprocessing as mp
from joblib import parallel_config
mp.set_start_method(method="fork")
if mp.get_start_method() != "spawn":
   def train_agent_SkinnerBox(actor, N_episodes, T_episodes, env,
                                    N_observations, N_clones, N_actions,
                                    observations, actions,
                                    gamma_A, eta_A,scale_reward_posterior,
                                    target, proba_preferred_obs, prediction_horizon,
                                    curiosity_wander, curiosity_task,
                                    emotional_thinking, emotion_actions, 
                                    causal_model,
                                    fake_zero,
                                    folder):
      
      # np.random.seed(seed = 9342 * actor + 139)
      initial_preferences = np.zeros((N_states * N_observations))

      
      # Create the initial conditions for the agent
      agent = Agent(initial_preferences, ECM_likelihood,
                   N_clones, N_observations, N_actions, 
                   observations, actions, T_episodes,
                   fake_zero) # emotional_thinking, 
      agent.make_initial_ECMs()
      agent.make_initial_distributions()
      #agent.boredom_actions = np.zeros((N_states, N_actions), dtype=np.float64)
      
      # Store information on the evolution of the training
      agent.Free_energies = np.zeros((N_episodes))
      agent.Expected_Free_energies = np.zeros((N_episodes))
      agent.Length_trajectories = np.zeros((N_episodes))
      visits = np.zeros((N_states * N_actions, N_states))
      
      for episode in tqdm(range(N_episodes)):
      
         # Initialize the agent in the environment
         env.observation = env.initial_conditions()   # Drop the agent somewhere in the grid
      
         # Initial deliberation
         agent.ECM_g_posterior = np.zeros(agent.ECM_g_posterior.shape)
         agent.previous_belief_state = agent.select_first_state(env.observation)
         agent.previous_action = agent.select_first_action()
         
         max_trajectory=0
         traj_counter=0
         counter_rewards_posterior = 0
         
         for step in range(T_episodes):
            
            # Agent tries to predict its observations
            emotion_boolean = bool(emotional_thinking * (counter_rewards_posterior>0))
            agent.belief_state = agent.deliberate_next_state(#emotion_boolean, 10, False
                                                             )
            visits[agent.input_percept, agent.belief_state] += 1
            agent.obs_agent = agent.deliberate_observation()

            # Apply action in env
            env.agent_position = env.move_the_agent(agent.previous_action, periodic_boundaries=periodic_boundaries)
            env.observation = env.give_observation()

            agent.input_percept = agent.calculate_input_percept()
            
            # Record the length of the trajectories
            if env.observation == agent.obs_agent:
               traj_counter += 1
               max_trajectory = np.maximum(traj_counter, max_trajectory)
            else:
               traj_counter = 0
               counter_rewards_posterior += 1
      
            # Calculate the progress of the agent
            agent.FE_specific, agent.FE = agent.calculate_free_energy(env.observation)
            
            # Update the g-matrices to record trajectories in states AND actions
            agent.ECM_g_posterior = agent.update_g_matrix(env.observation, 
                                                          False, eta_g)   # 0.9**traj_counter
            
            #### Update the world model
            agent.ECM_posterior, agent.posterior = agent.update_posterior_cumulative_rewards(env.observation,
                                                               gamma_A, eta_A, h_0,
                                                               # emotional_thinking, 
                                                               alpha_posterior * (1-episode/N_episodes),
                                                               # emotion_actions,
                                                               scale_reward_posterior,
                                                               bool(step == T_episodes-1),
                                                               False, 1)
      
            
            # # Signal the transition when erroneous (on both states and actions)
            # agent.ECM_emotions_posterior = agent.update_emotions(env.observation,
            #                                                      False)   # applies_to_actions
               
            #### Update the policy 
            target_ON = (episode >= N_episodes and step==T_episodes-1)
            

            if not wandering_phase:
               target_ON = True
               agent.preferences = agent.calculate_preferences(agent.belief_state, agent.previous_action, 
                                                             target_ON, discount, 
                                                             True, 1.,
                                                             target, proba_preferred_obs, 
                                                             prediction_horizon, 0.99,
                                                             False)
            
            agent.ECM_policy = agent.EFE_filter(target_ON, preferences_marginal,
                                                target, proba_preferred_obs, prediction_horizon, discount,
                                                True, True, curiosity_wander)  #,

            agent.EFE_array = agent.EFE_contributions[agent.belief_state,:].copy()

            agent.EFE = np.min(agent.EFE_contributions[agent.belief_state, :].copy())
            
            # Calculate the policy based on the baseline and the EFE
            # agent.policy = agent.ECM_to_probas(agent.ECM_policy, 
            #                                    True, 
            #                                    curiosity_wander)
                     
            
            agent.previous_action = agent.deliberate_next_action() #5, emotion_actions)
            agent.previous_belief_state = agent.belief_state + 0
            
            agent.Free_energies[episode] += agent.FE / T_episodes
            agent.Expected_Free_energies[episode] += agent.EFE_array[agent.previous_action]/ T_episodes
            
            # Correct_predictions[actor, episode] += 1 * (env.observation == np.random.choice(observations, 
            #                                                                                 p=agent.likelihood[agent.belief_state,:]))
         
            
         # Update the prior of the agents at the end of the episode
         agent.ECM_prior = agent.ECM_posterior.copy()
         agent.prior = agent.posterior.copy()
         agent.Length_trajectories[episode] = max_trajectory
         

         agent.policy = agent.ECM_to_probas(agent.EFE_weights, 
                                            True,
                                            curiosity_wander,
                                            np.array([N_states, N_actions]))
         if np.any(np.isnan(np.log(agent.policy))):
            break
         
         # agent.update_boredom_policy(emotion_actions)
         # if emotion_actions:
         #    distrib_boredom = 1/N_actions * np.ones((N_states, N_actions), dtype=np.float64)
         #    for i in range(N_states):
         #       if np.sum(agent.boredom_actions[i,:]) > 0:
         #          distrib_boredom[i,:] = agent.boredom_actions[i,:].copy() / np.sum(agent.boredom_actions[i,:].copy())
         #    agent.emotional_policy = (1-alpha_policy) * agent.policy + alpha_policy * distrib_boredom
         #    for i in range(N_states):
         #       agent.emotional_policy[i,:] /= np.sum(agent.emotional_policy[i,:])
               
         #    if (episode == 1 or episode == N_episodes/2 or episode == N_episodes-1):
         #        print("episode: " + str(episode))
         #        print("policy", agent.policy)
         #        print("boredom_distr", distrib_boredom)
         #        print("emotional_policy", agent.emotional_policy)
                
         # if (episode == 1 or episode == N_episodes/2 or episode == N_episodes-1):       
         #    print("visits", visits)

         
         
      #
      # plt.plot(range(N_episodes), figure_maker.moving_average(agent.Length_trajectories))
      # plt.show()
      # FE_figure = figure_maker.plot_energies_avg_evol(agent.Free_energies.reshape((1, N_episodes)), 
      #                                                 agent.Expected_Free_energies.reshape((1, N_episodes)),
      #                                                 transparence = 0.1/2, size=(7,10),
      #                                                 std=False)
      # plt.show()
      
      # fig_agent = figure_maker.plot_models_square_PGW(agent.posterior, 
      #                                           agent.policy,
      #                                           agent.full_preferences,
      #                                           title="agent " + str(actor))
      # plt.show()   
      # fig_agent = figure_maker.plot_models_square_PGW(visits, 
      #                                           agent.policy,
      #                                           agent.full_preferences,vmax=np.max(visits),
      #                                           title="agent " + str(actor))
      # plt.show()  
      
      
      agent_dealer = AgentWrapper(agent)
      agent_dealer.save_agent(folder + "training_agent_" + str(actor) + ".npy", False, True)
      #return agent

   def train_one_agent(actor):
      return train_agent_SkinnerBox(actor, 
                     N_episodes,T_episodes, 
                     env, 
                     N_observations, N_clones, N_actions, 
                     observations, actions, 
                     gamma_A, eta_A, scale_reward_posterior, 
                     target, proba_preferred_obs, prediction_horizon, 
                     curiosity_wander, curiosity_task, 
                     emotional_thinking, emotion_actions,
                     causal_model, 
                     fake_zero,folder)

   with parallel_config('multiprocessing'):
       agents = Parallel(n_jobs=-1)(delayed(train_one_agent)(actor) for actor in range(N_agents))      

### TASK: Train and test the policy to reach the target
print("testing")

# To initialize the agents
initial_preferences = np.zeros(N_states*N_observations, dtype=np.float64)
uniform_policy = np.ones((N_states, N_actions)) / N_actions
emotion_actions = False
emotional_thinking = False


def test_agent_GW_superposition(agent_task, N_trials, trial_length):
   
   agent_task.time_to_target= np.zeros(dim**2, dtype=np.float64)
   
   env = Environment_GW(dim, observations, directions)
   env.make_elementary_grid(symmetric=symmetric)
   
   for initial_state in env.grid_indices:
      for trial in range(N_trials):
          #print(trial)
          # Initialize the agent in the environment
          env.agent_position = initial_state + 0
          env.observation = env.give_observation()
      
          agent_task.plausible_previous_belief_states = np.zeros(N_states, dtype=np.int64)
          agent_task.plausible_previous_belief_states[env.observation * N_clones : env.observation * N_clones + N_clones] = 1
         
          # Deliberate on the action to take
          agent_task.plausible_previous_actions = np.zeros((N_clones), dtype=np.int64)
          #agent_task.ECM_emotions_policy = np.zeros((N_states, N_actions), dtype=np.int64)
          frequencies_plausible_actions = np.zeros((N_actions), dtype=np.float64)
          for i in range(N_states):
            agent_task.belief_state = i # agent_task.plausible_previous_belief_states[i]
            act = np.int64(agent_task.deliberate_next_action())#5))
            frequencies_plausible_actions[act] += 1. / N_states
          agent_task.previous_action = rand_choice_nb(actions, frequencies_plausible_actions)
         
          for step in range(1, trial_length):
      
            # Apply action in env
            env.agent_position = env.move_the_agent(agent_task.previous_action, 
                                               periodic_boundaries=periodic_boundaries)
            env.observation = env.give_observation()
            #print("env " + str(env.env_state))
               
            # Based on the observation, deliberate on the next plausible belief state:
            agent_task.plausible_belief_states = np.zeros(N_states, dtype=np.int64)
            for state in range(N_states):
                if agent_task.plausible_previous_belief_states[state] > 0:
                  agent_task.previous_belief_state = state    # agent_task.plausible_previous_belief_states[state]
                  # agent_task.input_percept = agent_task.previous_belief_state * N_actions + agent_
                  agent_task.belief_state = agent_task.deliberate_next_state(# False, 0, False
                                                                             )
                  agent_task.obs_agent = agent_task.deliberate_observation()
                  if agent_task.obs_agent == env.observation:
                      agent_task.plausible_belief_states[agent_task.belief_state] = 1
                     
            if np.sum(agent_task.plausible_belief_states) == 0:
                agent_task.plausible_belief_states = np.zeros(N_states, dtype=np.int64)
                agent_task.plausible_belief_states[env.observation * N_clones : env.observation * N_clones + N_clones] = 1
            
            # Update the belief states when they confirmed the observation
            agent_task.plausible_previous_belief_states = agent_task.plausible_belief_states.copy()
            
            # Deliberate on the action to take
            agent_task.plausible_previous_actions = np.zeros((N_clones), dtype=np.int64)
            #agent_task.ECM_emotions_policy = np.zeros((N_states, N_actions), dtype=np.int64)
            frequencies_plausible_actions = np.zeros((N_actions), dtype=np.float64)
            for i in range(N_states):
                agent_task.belief_state = agent_task.plausible_previous_belief_states[i]
                act = int(agent_task.deliberate_next_action())#5))
                frequencies_plausible_actions[act] += 1. / N_states
            agent_task.previous_action = rand_choice_nb(actions, frequencies_plausible_actions)
            
            
            if env.observation != target and step == trial_length-1:
               agent_task.time_to_target[initial_state] += step / N_trials
               
            if env.observation == target:
               agent_task.time_to_target[initial_state] += step / N_trials
               break
            
            


def train_test_GridWorld(actor, 
                        N_iterations_pref, target_ON,
                        proba_preferred_obs, prediction_horizon,
                        N_trials, trial_length, targets,
                        causal_model,
                        folder):
   
   np.random.seed(seed = 9342 * actor + 139)
   # Load the agent
   agent_task = Normal_Agent(initial_preferences, ECM_likelihood,
                 N_clones, N_observations, N_actions, 
                 observations, actions, T_episodes,
                 fake_zero)
   
   # Assign the trained distributions to the object
   file_load = folder + "training_agent_" + str(actor) + ".npy"
   agent_dealer=AgentWrapper(agent_task)
   agent_dealer.load_agent(file_load, False, True)   # testing = False, allow_pickle=False
   
   # agent_task.ECM_emotions_posterior = np.zeros((N_states * N_actions, N_states), dtype=np.int64)
   # agent_task.ECM_emotions_policy = np.zeros((N_states, N_actions), dtype=np.int64)
   
   agent_task.performances=[]
   agent_task.targets = targets
   for target_idx in range(len(targets)):
      agent_task.performances.append([])
      agent_task.policy = uniform_policy
      
      target = targets[target_idx]
      # Train the agent to complete the task
      for i in range(N_iterations_pref):
         agent_task.preferences = agent_task.calculate_preferences(0,0, 
                                                       True, discount, True, 1.,
                                                       target, proba_preferred_obs, 
                                                       prediction_horizon, 0.99,
                                                       False)
         
         agent_task.EFE_contributions = agent_task.EFE_filter(target_ON, target,
                                       proba_preferred_obs, prediction_horizon, 0.8,
                                       True, False, -10)  
         
         
         agent_task.policy = agent_task.ECM_to_probas(agent_task.EFE_weights.copy(),
                                                      True,
                                                      -(np.abs(curiosity_wander) 
                                                        + (i+1)/N_iterations_pref 
                                                        + (i>=N_iterations_pref-1) * (curiosity_task-1-np.abs(curiosity_wander))))
                  
      #test_agent_DR(agent_task, N_trials, trial_length)  
      # test_agent_GW_superposition(agent_task, N_trials, trial_length)
      
      environment = Environment_GW(dim, observations, directions)
      environment.make_elementary_grid(symmetric=symmetric)

      agent_task.time_to_target = agent_task.test_PGW_superposition(environment, N_trials = N_trials, trial_length = trial_length, 
                                        target_obs=target, target_coordinates =(2,2),
                                        causal_model=False,
                                        emotional_thinking=False,
                                        periodic_boundaries = periodic_boundaries)
      agent_task.performances[target_idx] = agent_task.time_to_target
   
   joblib.dump(agent_task, folder + "testing_agent_" + str(actor) + ".pkl")


def iterable_test(actor):
   return train_test_GridWorld(actor, 
                           N_iterations_pref, target_ON,
                           proba_preferred_obs, prediction_horizon,
                           N_trials, trial_length, targets,
                           causal_model,
                           folder)

target_ON = True
N_iterations_pref = 1#000 # To recalculate preferences for a given stand of policy
trial_length = 20
N_trials=1000
#target_obs = 3
prediction_horizon = 3
discount = 0.8
targets = [3]
target_coordinates = (2,2)

# Learn the task and test performances in parallel
Parallel(n_jobs=-1, backend='loky')(delayed(iterable_test)(actor) for actor in range(N_agents))

# test a random agent for each target 
def train_test_GridWorld_random( 
                        N_iterations_pref, target_ON,
                        proba_preferred_obs, prediction_horizon,
                        N_trials, trial_length, targets,
                        causal_model,
                        folder):
   
   # Load the agent
   agent_task = Normal_Agent(initial_preferences, ECM_likelihood,
                 N_clones, N_observations, N_actions, 
                 observations, actions, T_episodes,
                 fake_zero)
      
   # agent_task.ECM_emotions_posterior = np.zeros((N_states * N_actions, N_states), dtype=np.int64)
   # agent_task.ECM_emotions_policy = np.zeros((N_states, N_actions), dtype=np.int64)
   
   agent_task.posterior = np.ones((N_states * N_actions, N_states)) / N_states
   agent_task.likelihood = agent_task.ECM_to_probas(ECM_likelihood)
   agent_task.policy = uniform_policy
   
   agent_task.performances=[]
   agent_task.targets = targets
   for target_idx in range(len(targets)):
      agent_task.performances.append([])
      agent_task.policy = uniform_policy
      
      target = targets[target_idx]
      # Interact with the environment      
      environment = Environment_GW(dim, observations, directions)
      environment.make_elementary_grid(symmetric=symmetric)

      agent_task.time_to_target = agent_task.test_PGW_superposition(environment, N_trials = N_trials, trial_length = trial_length, 
                                        target_obs=target, target_coordinates =(2,2),
                                        causal_model=False,
                                        emotional_thinking=False,
                                        periodic_boundaries = periodic_boundaries)
      agent_task.performances[target_idx] = agent_task.time_to_target
   
   joblib.dump(agent_task, folder + "testing_agent_" + "random" + ".pkl")
   
agent_random = train_test_GridWorld_random( 
                        N_iterations_pref, target_ON,
                        proba_preferred_obs, prediction_horizon,
                        N_trials, trial_length, targets,
                        causal_model,
                        folder)

# gather the results in a list
agents_task = []
for actor in range(N_agents):
   agents_task.append(joblib.load(folder + "testing_agent_" + str(actor)+ ".pkl"))
   
agents_task.append(joblib.load(folder + "testing_agent_" + "random"+ ".pkl"))
   
   
 
########################## Plot the results ##########################
fontsize_titles=16
fontsize_labels = 16
fontsize_annot = 10
window_size = 300
agent_idx = 0

figure_maker = Plots(N_episodes, N_agents, N_clones, N_observations, N_actions,
                     actions, observations, dim,
                     fontsize_titles, fontsize_labels, fontsize_annot, window_size)

# Plot the evolution of the training on average
Free_energies = np.zeros((N_agents, N_episodes))
Expected_Free_energies = np.zeros((N_agents, N_episodes))
Length_trajectories = np.zeros((N_agents, N_episodes))
for actor in range(N_agents):
   Free_energies[actor,:] = agents_task[actor].Free_energies.copy()
   Expected_Free_energies[actor,:] = agents_task[actor].Expected_Free_energies.copy()
   Length_trajectories[actor,:] = agents_task[actor].Length_trajectories.copy()

FE_figure = figure_maker.plot_energies_avg_evol(Free_energies, Expected_Free_energies,
                                                transparence = 0.1/2, size=(7,10),
                                                std=False)
plt.show()

# Lenths of the trajectories
fig_lengths = figure_maker.plot_length_trajectories(Length_trajectories)
plt.show()


# Plot the distributions for one agent
# for actor in range(N_agents+1*(len(agents)>N_agents)):
actor = 0
title = "agent " + str(actor)
# for actor in range(N_agents):
fig_agent = figure_maker.plot_models_square_PGW(agents_task[actor].posterior, 
                                         agents_task[actor].policy,
                                         agents_task[actor].full_preferences,
                                         title="agent " + str(actor))
#if actor == 0:
plt.show()   

   

fig_testing = figure_maker.plot_testing_PGW(agents_task, targets = targets, size=(15,4), 
                                            avg_only=True, min_max=True, 
                                            subtract_random=False,
                                            ylim=21)
plt.show()



   