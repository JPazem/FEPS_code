#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:15:17 2024

@author: josephinepazem
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
from GW_FEPS_functions_parallel import Environment_delayed_rewards, Agent, Plots, AgentWrapper, Normal_Agent
from GW_FEPS_functions_parallel import rand_choice_nb
#from GW_FEPS_functions import Optimal_agent
import numba
from numba import njit
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed, parallel_backend
import dill as pickle
import pandas as pd
############## Initialize the variables   ############## 
fake_zero = 10 ** (-8)
save_figures = False

# Where to save the results
# Grid world
# folder = "/Users/josephinepazem/Documents/FEPS_data/24Sept24_GW_" #"_no_wandering_-1/"
############## Variables for training and hyperparameters
# Training parameters
# N_agents = 10
# N_episodes = 30000 #10000
# T_episodes = 80 # number of steps per episode

env_GW = True
#### Grid world
if env_GW:
   folder = "/Users/josephinepazem/Documents/FEPS_data/19DEC24_GW_" #"_no_wandering_-1/"
   ############## Variables for training and hyperparameters
   # Training parameters
   N_agents = 30   # 30 or 100
   N_episodes = 40000 #40000 or 4000
   T_episodes = 80 # number of steps per episode
#### Skinner box
else:
   folder = "/Users/josephinepazem/Documents/FEPS_data/19DEC24_DR_100Agents_"
   ############## Variables for training and hyperparameters
   # Training parameters
   N_agents = 100   # 30 or 100
   N_episodes = 4000 #40000 or 4000
   T_episodes = 80 # number of steps per episode




pretraining = ["no_wandering_" , "wandering_"]
pretraining_strings = ["task" , "wander"]
if env_GW:
   curiosity_params = [[-3], [1]]
else:
   curiosity_params = [[-1], [0]]
# Select the two types of training to compare
curiosity_1 = 0   # select the curiosity-th entry in curiosity_params
pt_1 = 0
curiosity_2 = 0   # select the curiosity-th entry in curiosity_params
pt_2 = 1
# Make the results more smooth
window_size = 300 * (not env_GW) + 1500 * env_GW #1500 # 300
N_example_actors = 10 - 2    # 2 for the min-max actors

      
#### Make the averages      
def moving_average(data, window_size):
    # Define the kernel for the moving window
    kernel = np.ones(window_size) / window_size
    
    # Convolve the data array with the kernel
    moving_avg = np.convolve(data, kernel, mode='valid')
    first_points = np.zeros((window_size-1))
    for t in range(window_size-1):
        first_points[t] = np.sum(data[:t+1])/(t+1)
        
    return np.concatenate((first_points, moving_avg))    
   
#### Load the agents   
Free_Energies = []   
Expected_Free_Energies = []
Length_trajectories = []
for pt in range(len(pretraining)):
   Free_Energies_pt = []
   Expected_Free_Energies_pt = []
   Length_trajectories_pt = []
   for curiosity in range(len(curiosity_params[pt])):
      Free_Energies_c = []
      Expected_Free_Energies_c = []
      Length_trajectories_c = []
      foldername = folder + pretraining[pt] + str(curiosity_params[pt][curiosity]) #+ "_eta0.5"
      for actor in range(N_agents):
         agent = joblib.load(foldername + "/testing_agent_" + str(actor)+ ".pkl")
         Free_Energies_c.append(moving_average(agent.Free_energies, window_size))
         Expected_Free_Energies_c.append(moving_average(agent.Expected_Free_energies, window_size))
         Length_trajectories_c.append(moving_average(agent.Length_trajectories, window_size))
   
      Free_Energies_c = np.array(Free_Energies_c)
      Expected_Free_Energies_c = np.array(Expected_Free_Energies_c)
      Length_trajectories_c = np.array(Length_trajectories_c)
  
      Free_Energies_pt.append(Free_Energies_c)
      Expected_Free_Energies_pt.append(Expected_Free_Energies_c)
      Length_trajectories_pt.append( Length_trajectories_c)

   Free_Energies.append(Free_Energies_pt)
   Expected_Free_Energies.append(Expected_Free_Energies_pt)
   Length_trajectories.append(Length_trajectories_pt)
      

linestyles = ['-', ':']
transparence = 0.02 #99AD61 #90C184

colors_FE = sns.color_palette("blend:#90C184,#201365", as_cmap=True)# sns.color_palette("crest", as_cmap = True) # sns.color_palette("blend:#88C273,#4A628A", as_cmap=True) #  
colors_EFE = sns.color_palette("flare", as_cmap = True)
colors_lengths = sns.color_palette("blend:#fff400,#ac2020", as_cmap = True)


#### Find the best and worst agent
min_episodes_1 = [np.where(row < 1.5)[0][0] if np.sum(row < 1.5) > 0 else N_episodes for row in Free_Energies[pt_1][curiosity_1]]
min_actor_1 = np.argmin(min_episodes_1)
max_actor_1 = 0
max_FE = 0
for i in range(N_agents):
   if min_episodes_1[i] == max(min_episodes_1) and Free_Energies[pt_1][curiosity_1][i,-1] > max_FE:
      max_actor_1 = i
      max_FE = Free_Energies[pt_1][curiosity_1][i,-1]
# max_actor_1 = np.argmax(min_episodes_1)


#### Repeat with a wandering phase with uniform distribution (curiosity_param = 0)
min_episodes_2 = [np.where(row < 2)[0][0] if np.sum(row < 2) > 0 else N_episodes for row in Free_Energies[pt_2][curiosity_2]]
min_actor_2 = np.argmin(min_episodes_2)
max_actor_2 = np.argmax(min_episodes_2)

#### Sample 8 random agents
seed = np.random.seed(1932)
actors_list = np.arange(N_agents)
actors_list = np.delete(actors_list, np.where(actors_list == min_actor_1))
actors_list = np.delete(actors_list, np.where(actors_list == max_actor_1))
actors_list = np.delete(actors_list, np.where(actors_list == min_actor_2))
actors_list = np.delete(actors_list, np.where(actors_list == max_actor_2))
example_actors = np.random.choice(actors_list, size=N_example_actors, replace=False)


#### Asymptotic behavior of the EFE for wandering agents in the grid world

if env_GW:
   zeta = 1
   pseudo_policy_asymptote_wandering = np.zeros((12, 4))
   pseudo_policy_asymptote_wandering[(1,3,4,5,7,8), :] = 1/4
   numerator = 2**(1/2) + 2
   pseudo_policy_asymptote_wandering[(0,0,2,2,6,6,9,9,10,10,11,11), (1,3,0,3,1,2,0,2,0,2,0,2)] =  2 * 2**(-1/2) / numerator
   pseudo_policy_asymptote_wandering[(0,0,2,2,6,6,9,9,10,10,11,11), (0,2,1,2,0,3,1,3,1,3,1,3)] = 1 / numerator
   
   EFE_asymptote = -np.log(pseudo_policy_asymptote_wandering)
   EFE_asymptote = np.mean(EFE_asymptote)
else:
   zeta = 0
   pseudo_policy_asymptote_wandering = np.ones((6,2))/2 # uniform for zeta = 0
   
   EFE_asymptote = -np.log(pseudo_policy_asymptote_wandering)
   EFE_asymptote[(2,3,4,5),:] /= 2    # the E_0 states have degeneracy 2
   EFE_asymptote = np.mean(EFE_asymptote)
   
   pref_task = np.ones((6,6)) * 10**(-4)
   pref_task[(0,1,2,3,4,5), (4,4,4,4,5,3)] = 1 - 5 * 10**(-4)
   pref_task[5,3] /= 1/2
   EFE_asymptote_task = -np.log(pref_task)
   EFE_asymptote_task[5,3] *= 1/2
   EFE_asymptote_task = np.mean(EFE_asymptote_task)


#### Make the figure
plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels

fig_FE, ax_FE = plt.subplots(nrows=3, sharex=True,
                             dpi=300, figsize=(10,10),gridspec_kw={'height_ratios': [5, 1,1]})

fig_FE.subplots_adjust(hspace=0.1)
## Plot the example agents first
if not env_GW:
   N_example_actors = 4 #3
   example_actors = [52,74,72,61] #[61,14,79]
else:
   N_example_actors = 5 #4 #3
   example_actors = [6,23,21,27,18] #[52,74,72,61]#[61,14,79]

for actor_idx in range(N_example_actors):
   actor = example_actors[actor_idx]
   # task-oriented agents
   ax_FE[0].plot(np.arange(N_episodes), Free_Energies[pt_1][curiosity_1][actor],
                 alpha = 0.15, color=colors_FE(200 + 5 * actor_idx), linewidth=2)
   # wandering agents
   ax_FE[0].plot(np.arange(N_episodes), Free_Energies[pt_2][curiosity_2][actor],
                 alpha = 0.2, color=colors_FE(5 * actor_idx), linewidth=2, linestyle="--")
   
   ax_FE[1].plot(np.arange(N_episodes), Expected_Free_Energies[pt_1][curiosity_1][actor],
                  alpha = 0.2, color=colors_EFE(50 + 5 * actor_idx), linewidth=2)
    # wandering agents
   ax_FE[2].plot(np.arange(N_episodes), Expected_Free_Energies[pt_2][curiosity_2][actor],
                  alpha = 0.2, color=colors_EFE(5 * actor_idx), linewidth=2)




## Min-max agents 
# wandering agents
ax_FE[0].plot(np.arange(N_episodes), np.mean(Free_Energies[pt_2][curiosity_2], axis=0),
              alpha = 1, linestyle = "-", color = colors_FE(20), linewidth = 3,
              label="average agent, wander")

ax_FE[0].plot(np.arange(N_episodes), Free_Energies[pt_2][curiosity_2][min_actor_2],
              alpha = 1, linestyle = (0, (5,3)), color = colors_FE(0), linewidth=2.5,
              label="best agent, wander")

ax_FE[0].plot(np.arange(N_episodes), Free_Energies[pt_2][curiosity_2][max_actor_2],
              alpha = 1, linestyle = (0, (1, 4)), color = colors_FE(0), linewidth=2.5, 
              label="worst agent, wander")

ax_FE[2].plot(np.arange(N_episodes), np.mean(Expected_Free_Energies[pt_2][curiosity_2], axis=0),
              alpha = 1, linestyle = "-", color = colors_EFE(10), linewidth = 3,
              label="average agent, wander")

# ax_FE[1].plot(np.arange(N_episodes), Expected_Free_Energies[pt_2][curiosity_2][min_actor_2],
#               alpha = 0.8, linestyle = "dashdot", color = colors_EFE(10), linewidth=2,
#               label="best agent, wander")

# ax_FE[1].plot(np.arange(N_episodes), Expected_Free_Energies[pt_2][curiosity_2][max_actor_2],
#               alpha = 0.8, linestyle = (0, (5, 5)), color = colors_EFE(10), linewidth=2, 
#               label="worst agent, wander")

ax_FE[2].plot(np.arange(N_episodes), EFE_asymptote * np.ones(N_episodes),
              label="asymptotic EFE, wander", color = colors_EFE(75), linewidth = 3, linestyle=":")

  
# task-oriented agents
ax_FE[0].plot(np.arange(N_episodes), np.mean(Free_Energies[pt_1][curiosity_1], axis=0),
              alpha = 1, linestyle = "-", color = colors_FE(200 + 100), linewidth = 3,
              label="average agent, task")

ax_FE[0].plot(np.arange(N_episodes), Free_Energies[pt_1][curiosity_1][min_actor_1],
              alpha = 0.5, linestyle = (0, (5,3)), color = colors_FE(200 + 100), linewidth=2.5,
              label="best agent, task")

ax_FE[0].plot(np.arange(N_episodes), Free_Energies[pt_1][curiosity_1][max_actor_1],
              alpha = 0.5, linestyle = (0, (1, 4)), color = colors_FE(200 + 100), linewidth=2.5, 
              label="worst agent, task")


ax_FE[1].plot(np.arange(N_episodes), np.mean(Expected_Free_Energies[pt_1][curiosity_1], axis=0),
              alpha = 1, linestyle = "-", color = colors_EFE(50 + 100), linewidth = 3,
              label="average agent, task")

# ax_FE[1].plot(np.arange(N_episodes), Expected_Free_Energies[pt_1][curiosity_1][min_actor_1],
#               alpha = 0.6, linestyle = "dashdot", color = colors_EFE(50 + 100), linewidth=2,
#               label="best agent, task")

# ax_FE[1].plot(np.arange(N_episodes), Expected_Free_Energies[pt_1][curiosity_1][max_actor_1],
#               alpha = 0.6, linestyle = (0, (5, 5)), color = colors_EFE(50 + 100), linewidth=2, 
#               label="worst agent, task")




if env_GW:
   ax_FE[1].set_ylim([9, 15])
   ax_FE[2].set_ylim([0,1.5])
else:
   ax_FE[1].set_ylim([5, 14])
   ax_FE[2].set_ylim([0,1])

# Hide the spines between the axes
ax_FE[1].spines['bottom'].set_visible(False)
ax_FE[2].spines['top'].set_visible(False)
# Add "cut" lines
d = 0.01  # Length of cut line segments
kwargs = dict(transform=ax_FE[1].transAxes, color='k', clip_on=False)
ax_FE[1].plot((-d, +d), (-d, +d), **kwargs)        # Top-left diagonal
ax_FE[1].plot((1 - d, 1 + d), (-d, +d), **kwargs)  # Top-right diagonal

kwargs.update(transform=ax_FE[2].transAxes)  # Switch to the bottom axis
ax_FE[2].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # Bottom-left diagonal
ax_FE[2].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # Bottom-right diagonal

#ax_FE[0].set_xlabel("episode")
ax_FE[0].set_ylabel("variational free energy")
ax_FE[2].set_xlabel("episode")
# ax_FE[2].set_ylabel("expected free energy")
# Center the y-label on the interrupted axis
fig_FE.text(0.055, 0.23, 'expected free energy', va='center', rotation='vertical', fontsize=12)

if env_GW:
   position_EFE = (1, 0.)
   position_EFE_task = (1, 0.6)
   position_EFE_wandering = (1, 1.05)
else:
   position_EFE_task = (1, 0.6)
   position_EFE_wandering = (1, 1.05)
ax_FE[0].legend(ncols=2, fontsize=11, loc="upper right", bbox_to_anchor=(1, 1))
ax_FE[1].legend(ncols=1, fontsize=11, loc="lower right", bbox_to_anchor=position_EFE_task)
ax_FE[2].legend(ncols=1, fontsize=11, loc="upper right", bbox_to_anchor=position_EFE_wandering)

ax_FE[0].set_yscale("log")
ax_FE[0].set_ylim([0, 40])
ax_FE[1].set_yscale("linear")
# ax_FE[0].set_xticklabels(fontsize=12)
# ax_FE[0].set_yticklabels(fontsize=12)


plt.show()

# for index in range(len(example_actors)):
#    fig_FE_EFE, ax_FE_EFE = plt.subplots(1, figsize=(10,5),dpi=200)
#    actor = example_actors[index]
#    ax_FE_EFE.plot(np.arange(N_episodes), Free_Energies[pt_2][curiosity_2][actor],
#                  alpha = 0.2, color=colors_FE(200), linewidth=2,
#                  label="VFE")
#    ax_FE_EFE.plot(np.arange(N_episodes), Expected_Free_Energies[pt_2][curiosity_2][actor],
#                  alpha = 0.2, color=colors_EFE(5 * actor_idx), linewidth=2,
#                  label="EFE")
#    ax_FE_EFE.legend()
#    plt.show()




