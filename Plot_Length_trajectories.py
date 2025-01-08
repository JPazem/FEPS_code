#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 13:47:16 2024

@author: josephinepazem
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import pandas as pd

############## Initialize the variables   ############## 
fake_zero = 10 ** (-8)
save_figures = False

env_GW = True
pretraining = ["no_wandering_" , "wandering_"]
pretraining_strings = ["task" , "wander"]

#### Timed Response
if not env_GW:
   folder = "/Users/josephinepazem/Documents/FEPS_data/19DEC24_DR_100agents_" #"_no_wandering_-1/"
   # Variables for training and hyperparameters
   N_agents = 100
   N_episodes = 4000#10000
   T_episodes = 80 # number of steps per episode
   curiosity_params = [[-1],[0]]
   window_size = 100
   
#### Navigation
elif env_GW:
   folder = "/Users/josephinepazem/Documents/FEPS_data/19DEC24_GW_" #"_no_wandering_-1/"
   # Variables for training and hyperparameters
   N_agents = 30
   N_episodes = 40000#10000
   T_episodes = 80 # number of steps per episode
   curiosity_params = [[-3, -1, 1,3],[0, 1, 2, 3]]
   window_size = 300

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

#### Load the data of the agents
Length_trajectories = []
for pt in range(len(pretraining)):
   Length_trajectories_pt = []
   
   for curiosity in range(len(curiosity_params[pt])):
      Length_trajectories_c = []
      foldername = folder + pretraining[pt] + str(curiosity_params[pt][curiosity]) #+ "_eta0.5"
      for actor in range(N_agents):
         agent = joblib.load(foldername + "/testing_agent_" + str(actor)+ ".pkl")
         Length_trajectories_c.append(moving_average(agent.Length_trajectories, window_size))
   
      Length_trajectories_c = np.array(Length_trajectories_c)
  
      Length_trajectories_pt.append(Length_trajectories_c)

   Length_trajectories.append(Length_trajectories_pt)
   

#### Make the plots
N_curiosity = len(curiosity_params[0]) + len(curiosity_params[1])
palette = sns.color_palette('RdBu_r', N_curiosity + 1)
colors_lengths = palette[:len(curiosity_params[0])] + palette[len(curiosity_params[0])+1:] 

#### Plot the results
sns.set_theme(context="paper", style="whitegrid", font_scale = 1.4)
fig_traj, ax_traj = plt.subplots(1,1,dpi=300)

for pt in range(len(pretraining)):
   for curiosity in range(len(curiosity_params[pt])):
      
      Length_mean = np.mean(Length_trajectories[pt][curiosity], axis=0)
      Length_std = np.std(Length_trajectories[pt][curiosity], axis=0)
      
      # Set lines wider for wandering independent of preferences
      if curiosity_params[pt][curiosity] == 0:
         linewidth= 2
      else:
         linewidth=1*2
         
      # Make lines more transparent for very large curiosity
      if np.abs(curiosity_params[pt][curiosity]) == 10:
         linestyle="-"
         alpha = 0.2
      else:
         if pt == 1:
            linestyle = "-"
         else:
            linestyle = "-"
         alpha = 0.7 #0.7
      
      color_idx = (pt==1) * len(curiosity_params[pt-1]) + curiosity
      
      ax_traj.plot(list(range(N_episodes)), Length_mean,
                   color=colors_lengths[color_idx],
                   linestyle=linestyle, linewidth = linewidth, alpha=1,
                   label = str(curiosity_params[pt][curiosity]) + ', ' + pretraining_strings[pt])
      
ax_traj.legend(fontsize=13, ncols=2, loc="lower right")
ax_traj.set_ylabel("Length of trajectories")
ax_traj.set_xlabel("episodes")
plt.show()
   


