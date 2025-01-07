#!/usr/bin/environment python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:27:02 2024

@author: josephinepazem
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import itertools
from collections import Counter
from markovchain import MarkovChain
from mycolorpy import colorlist as mcp
from numba.experimental import jitclass
from numba import boolean, int64, float64, jit
import pandas as pd


NOPYTHON = True


# Adapt some of the functions to the needs of numba

# no np.random.choice with tailored p
@jit(nopython = NOPYTHON)
def rand_choice_nb(arr : np.array, # 1D numpy array of values to sample from.
                   prob : np.array # 1D numpy array of probabilities for the given samples.
                  ): # Random sample from the given array with a given probability.    
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]
 
# no boolean indexing
@jit(nopython = NOPYTHON)
def boolean_idx(arr_reference, size0, size1, criterion, larger):
   """
   Creates an array to select only indices in an array that satisfy some ordering
   with respect to some criterion

   Parameters
   ----------
   arr_reference : TYPE np.ndarray, 2D
      DESCRIPTION. Reference array to compare with the criterion
   size0 : TYPE int
      DESCRIPTION. size of axis0 of arr_references
   size1 : TYPE int
      DESCRIPTION. size of axis1 of arr_references
   criterion : TYPE float
      DESCRIPTION. value to compare arr_reference against
   larger : TYPE Boolean
      DESCRIPTION. whether the ordering is "larger than" (True) or "smaller than" (False)

   Returns
   -------
   sat_array : TYPE np.ndarray, 2D
      DESCRIPTION. array containing the truths values as 0 and 1.

   """
   sat_array = np.zeros(((size0, size1)), dtype=np.float64)
   for i in range(size0):
      for j in range(size1):
         if larger == True and arr_reference[i, j] > criterion: # select the entry if entry larger than criterion
            sat_array[i,j] = 1.
         elif larger == False and arr_reference[i,j] <= criterion:
            sat_array[i,j] = 1.
   return sat_array
            



   
class Environment_GW:
   def __init__(self, dim, observations, directions):
      self.dim = dim
      self.observations = observations
      self.directions = directions
      self.grid_indices = []
      self.grid_coordinates = []
      self.grid_xy_to_idx = {}
      self.obs_in_coordinates = []


   def make_elementary_grid(self, symmetric=False):
        """ 
        Function that makes the unit cell for the infinite grid and allocates observations to cells
        """
        
        self.grid_coordinates = [np.array(cell) for cell in itertools.product(list(range(self.dim)), repeat=2)] 
        self.grid_indices = list(range(len(self.grid_coordinates)))
        if not symmetric:
           x_max = self.dim
           y_max = self.dim
           
        else:
           x_max = self.dim // 2 + 1
           y_max = self.dim //2 + 1

        for location in self.grid_indices:
            cell = self.grid_coordinates[location]
            self.grid_xy_to_idx[tuple(self.grid_coordinates[location])]=location
            
            x=cell[0]
            y=cell[1]
            
            distance_to_max = np.abs(x_max-1-x) + np.abs(y_max-1-y)
            if distance_to_max >= np.max(self.observations):
                obs = np.min(self.observations)
            else:
                obs = self.observations[-(distance_to_max+1)]
                
            self.obs_in_coordinates.append(obs)
     
   def initial_conditions(self):
      """ 
      Initialize the agent in the grid: chooses its position and initial action.
      """
      self.agent_position = np.random.choice(self.grid_indices)
      self.agent_coordinates = self.grid_coordinates[self.agent_position]
      
      self.observation = self.give_observation()
      
      return self.observation
   
   def move_the_agent(self, action, periodic_boundaries = False):
      """
      Moves the agent in the grid, ie calculates the next position and corresponding
      position coordinates
      

      Parameters
      ----------
      action : int
         DESCRIPTION. index of the direction the agent has selected
         
      agent_position : int
         DESCRIPTION. index of the position of the agent in the grid

      Returns
      -------
      agent_coordinates: tuple
         DESCRIPTION. next coordinates in the grid
      
      agent_position: int.
         DESCRIPTION. index of the position in the grid

      """
      
      self.agent_coordinates = self.grid_coordinates[self.agent_position].copy()
      displacement = self.directions[action].copy()
      
      if periodic_boundaries:
         self.agent_coordinates += displacement
         self.agent_coordinates %= self.dim   # Periodic boundaries
         
      else:
         coordinates = self.agent_coordinates + displacement
         if np.all(np.array(coordinates) < self.dim) and np.all(np.array(coordinates) >= 0):
            self.agent_coordinates = coordinates.copy()
      
      self.agent_position = self.grid_xy_to_idx[tuple(self.agent_coordinates)] + 0
      
      return self.agent_position
   
   def give_observation(self):
      """
      Outputs the observation corresponding to the position of the agent in the grid
      
      Parameters
      ----------
      agent_position : int
         DESCRIPTION. index of the direction the agent has selected

      Returns
      -------
      observation: int
         DESCRIPTION. index of the observation in the observation list
      """
      
      self.observation = self.obs_in_coordinates[self.agent_position] + 0
      return self.observation
   
   
   
   
class Environment_delayed_rewards:
   def __init__(self, proba_ON):
      self.proba_ON = proba_ON
      self.observations = np.array([0,1,2])
      self.observations_labels = ["OFF, empty", "OFF, full", "ON, empty"]
      self.environment_states = [0,1,2] # 0: light off, 1: light on, no food", 2: light on, food
      self.previous_environment_state = 0
      self.waiting_time = 1
      
   def initial_conditions(self):
      self.observation = 0
      self.environment_state = 0
      return self.observation
   
   def state_transition(self, action):
      self.previous_environment_state = self.environment_state + 0
      
      if self.environment_state == 0:
         self.environment_state = np.random.choice(np.array([3,1]), p=np.array([1-self.proba_ON, self.proba_ON]))
         #self.flag_stay = (self.environment_state == 0)
         self.counter = 0
      
      elif self.environment_state == 1:
         if action == 0:   # "Wait"
            if self.counter == self.waiting_time-1:
               self.environment_state = 2

            elif self.counter < self.waiting_time-1:
               self.environment_state = 1
               
            self.counter += 1 
            
         if action == 1:    # "Press the lever"
            self.environment_state = 0
            self.counter = 0
      
      elif self.environment_state == 3: # if light did not turn on
         self.environment_state = 2
         
      elif self.environment_state == 2:
            self.environment_state = 0
            self.counter = 0
            
      return self.environment_state
   
   def give_observation(self, action):
      if self.environment_state == 0:
         if self.previous_environment_state == 2 and action == 1:
            self.observation = 1  # Off, full
         else:
            self.observation = 0   # Off, empty
      if self.environment_state >= 1:
         self.observation = 2   # On, empty
         
      return self.observation
         

# Save and load the data of the trained agents
class AgentWrapper:
    def __init__(self, agent):
        self.agent = agent

    def save_agent(self, filepath, testing, allow_pickle):
       """
       Saves the data of the agent (jitclass does not do this so it is reusable...)
       testing: Boolean, data come from the testing, not the training
       """
       if not testing:
           # Save agent state (e.g., weights) to a file
           data={}
           data["Free_energies"] = self.agent.Free_energies
           data["Expected_Free_energies"] = self.agent.Expected_Free_energies
           data["Length_trajectories"] = self.agent.Length_trajectories
           data["ECM_likelihood"] = self.agent.ECM_likelihood
           data["likelihood"] = self.agent.likelihood
           data["ECM_posterior"] = self.agent.ECM_posterior
           data["posterior"] = self.agent.posterior
           data["preferences"] = self.agent.preferences
           data["policy"] = self.agent.policy
           # data["boredom_actions"] = self.agent.boredom_actions
           np.save(filepath,data, allow_pickle=allow_pickle)

       if testing:
           data = {}
           data["Free_energies"] = self.agent.Free_energies
           data["Expected_Free_energies"] = self.agent.Expected_Free_energies
           data["Length_trajectories"] = self.agent.Length_trajectories
           data["ECM_likelihood"] = self.agent.ECM_likelihood
           data["likelihood"] = self.agent.likelihood
           data["ECM_posterior"] = self.agent.ECM_posterior
           data["posterior"] = self.agent.posterior
           data["preferences"] = self.agent.preferences
           data["full_preferences"] = self.agent.full_preferences
           data["full_preferences_obs"] = self.agent.full_preferences_obs
           data["preferences"] = self.agent.preferences
           data["success_rate"] = self.agent.success_rate
           data["success_distance"] = self.agent.success_distance
           np.savez(filepath, data, allow_pickle=allow_pickle)


    def load_agent(self, filepath, load_from_testing, allow_pickle):
        # Load agent state from a file
        data_obj = np.load(filepath, allow_pickle=allow_pickle)
        data = dict(enumerate(data_obj.flatten(), 1))
        data = data[1]
        if not load_from_testing: # The data does not come from the testing
            self.agent.Free_energies = data["Free_energies"]
            self.agent.Expected_Free_energies = data["Expected_Free_energies"]
            self.agent.Length_trajectories = data["Length_trajectories"]
            self.agent.ECM_likelihood = data["ECM_likelihood"]
            self.agent.likelihood = data["likelihood"]
            self.agent.ECM_posterior = data["ECM_posterior"]
            self.agent.posterior = data["posterior"]
            self.agent.preferences = data["preferences"]
        
        if load_from_testing:
            self.agent.Free_energies = data["Free_energies"]
            self.agent.Expected_Free_energies = data["Expected_Free_energies"]
            self.agent.Length_trajectories = data["Length_trajectories"]
            self.agent.ECM_likelihood = data["ECM_likelihood"]
            self.agent.likelihood = data["likelihood"]
            self.agent.ECM_posterior = data["ECM_posterior"]
            self.agent.posterior = data["posterior"]
            self.agent.full_preferences = data["full_preferences"]
            self.agent.full_preferences_obs = data["full_preferences_obs"]
            self.agent.preferences = data["preferences"]
            self.success_rate = data["success_rate"]
            self.success_distance = data["success_distance"]
         
spec = [('reward_policy',int64),
        ('reward_posterior', float64),
        ('previous_belief_state', int64),
        ('previous_action', int64),
        ('belief_state', int64),
        ('N_states',  int64),
        ('fake_zero', float64),
        ('obs_environment', int64),
        ('T_episodes', int64),
        ('actions', int64[:]),
        ('observations', int64[:]),
        ('N_actions', int64),
        ('N_observations',  int64),
        ('N_clones',  int64),
        ('preferences', float64[:]),
        # ('ECM_emotions_policy', int64[:,:]),
        # ('ECM_emotions_posterior', int64[:,:]),
        # ('ECM_emotional_policy', float64[:,:]),
        # ('emotional_posterior', float64[:,:]),
        # ('emotional_policy', float64[:,:]),
        # ('boredom', float64[:,:]),
        # ('boredom_actions', float64[:,:]),
        ('ECM_g_policy', float64[:,:]),
        ('ECM_g_posterior', float64[:,:]),
        ('ECM_likelihood', float64[:,:]),
        ('ECM_policy', float64[:,:]),
        ('ECM_posterior', float64[:,:]),
        ('ECM_prior', float64[:,:]),
        ('ECM_rewards', float64[:,:]),
        ('EFE', float64),
        ('EFE_array', float64[:]),
        ('EFE_contributions', float64[:,:]),
        ('EFE_weights', float64[:,:]),
        ('FE', float64),
        ('FE_specific', float64[:]),
        ('failed_predictions', float64[:]),        
        ('full_preferences', float64[:,:]),
        ('full_preferences_obs', float64[:,:]),
        ('input_percept', int64),
        ('likelihood', float64[:,:]),
        ('obs_agent', int64),        
        ('past_trajectory', float64[:,:]),
        ('past_trajectory_actions', float64[:,:]),
        ('plausible_belief_states', int64[:]),
        ('plausible_previous_actions', int64[:]),        
        ('plausible_previous_belief_states', int64[:]),       
        ('policy', float64[:,:]),
        ('posterior', float64[:,:]),
        ('prior', float64[:,:]),
        ('reachability', float64[:,:]),
        ('rewards_task', float64[:]),
        ('success_distances', float64[:]),
        ('success_rates', float64[:]),
        ('success_distance', float64),
        ('success_rate', float64),
        ('observation', int64),
        ('duplicate', int64),
        ('Free_energies', float64[:]),
        ('Expected_Free_energies', float64[:]),
        ('Length_trajectories', float64[:])
]
# ('emotional_thinking', boolean),

@jitclass(spec)   
class Agent():
   reward_policy: int
   reward_posterior: int
   previous_belief_state: int
   previous_action: int
   belief_state: int
   N_states: int
   fake_zero: float
   # emotional_thinking: bool
   T_episodes: int
   actions: np.ndarray
   observations: np.ndarray
   N_actions: int
   N_observations: int
   N_clones: int
   preferences: np.ndarray
   # ECM_emotions_policy: np.ndarray
   # ECM_emotions_posterior: np.ndarray
   # emotional_posterior: np.ndarray
   ECM_g_policy: np.ndarray
   ECM_g_posterior: np.ndarray
   ECM_likelihood: np.ndarray
   ECM_policy: np.ndarray
   ECM_posterior: np.ndarray
   ECM_prior: np.ndarray
   ECM_rewards: np.ndarray
   EFE: float
   EFE_array: np.ndarray
   EFE_contributions: np.ndarray
   EFE_weights: np.ndarray
   FE: float
   FE_specific: np.ndarray
   failed_predictions: float
   full_preferences: np.ndarray
   full_preferences_obs: np.ndarray
   input_percept: int
   likelihood: np.ndarray
   obs_agent: int
   past_traj: float
   plausible_belief_states: np.ndarray  #  
   plausible_previous_actions: np.ndarray
   plausible_previous_belief_states: np.ndarray
   policy: np.ndarray
   posterior: np.ndarray
   prior: np.ndarray
   reachability: np.ndarray
   rewards_task: np.ndarray
   success_distance: np.ndarray
   success_rate: np.ndarray
   observation: int
   duplicate: int
   Free_energies: np.ndarray
   Expected_Free_energies: np.ndarray
   Length_trajectories: float


   def __init__(self, preferences, ECM_likelihood,
                N_clones, N_observations, N_actions, observations, actions, T_episodes,
                fake_zero): # emotional_thinking, 
      
      # Where the agent was before
      self.preferences = preferences
      self.N_clones = N_clones
      self.N_observations = N_observations
      self.N_actions = N_actions
      self.observations = observations
      self.actions = actions
      self.T_episodes = T_episodes
      # self.emotional_thinking = emotional_thinking
      self.fake_zero = fake_zero
      self.N_states = N_clones * N_observations
      
      self.ECM_likelihood = ECM_likelihood
      
      # Deliberation and training
      self.belief_state = 0
      self.previous_belief_state = 0
      self.previous_action = 0
   
      # Initialize the rewards
      self.reward_posterior = 0
      self.reward_policy = 0

   
   def ECM_to_probas(self, ECM, softmax=False, softmax_temperature=1, dimensions=np.array([0, 0])):
      """
      Takes an ECM and outputs the corresponding probability distribution, 
      using either the sum or the softmax rule
      """
      if np.sum(dimensions) < 1:
         dimensions = np.array((ECM.shape))
         
      if not softmax:
         distrib = ECM.copy() # / np.sum(ECM, axis=1)[:, np.newaxis]
      
      else:
         distrib = np.exp(np.minimum(softmax_temperature * ECM, 1000))
      
      for i in range(dimensions[0]):
         distrib[i,:] /= np.sum(distrib[i,:])
      
      return distrib
   
   
   def make_initial_ECMs(self, likelihood_given=True):
      """Makes the ECMs for the prior, posterior, likelihood and policy, with the 
      right connections"""
      
      # Prior
      self.ECM_prior = np.ones((self.N_states * self.N_actions, self.N_states), dtype=np.float64)
      #self.ECM_prior = np.random.choice(np.array([1.0, 2.0, 3.0]), size=self.ECM_prior.shape)
      
      # Posterior   
      self.ECM_posterior = self.ECM_prior.copy()
      self.ECM_g_posterior = np.zeros(self.ECM_posterior.shape, dtype=np.float64)
      self.ECM_rewards = np.zeros(self.ECM_posterior.shape, dtype=np.float64)
      # self.ECM_emotions_posterior = np.zeros(self.ECM_posterior.shape, dtype=np.int64)
      
      # # Likelihood
      # self.ECM_likelihood = np.full((self.N_states, self.N_observations), self.fake_zero)
      # for observation, duplicate in itertools.product(self.observations, range(self.N_clones)):
      #    self.ECM_likelihood[observation * self.N_clones + duplicate, observation] = 1
      
      # Policy
      self.ECM_policy = np.ones((self.N_states, self.N_actions), dtype=np.float64)
      self.ECM_g_policy = np.zeros(self.ECM_policy.shape, dtype=np.float64)
      # self.ECM_emotions_policy = np.zeros(self.ECM_policy.shape, dtype=np.int64)
      self.EFE_contributions = np.zeros(self.ECM_policy.shape, dtype=np.float64)
      
      return self.ECM_prior, self.ECM_posterior, self.ECM_g_posterior, self.ECM_likelihood, self.ECM_policy, self.ECM_g_policy
   
  
   def make_initial_distributions(self, softmax=False):
      self.prior = self.ECM_to_probas(self.ECM_prior, softmax, 1, np.array([self.N_states * self.N_actions, self.N_states]))
      self.posterior = self.ECM_to_probas(self.ECM_posterior, softmax, 1, np.array([self.N_states * self.N_actions, self.N_states]))
      self.likelihood = self.ECM_to_probas(self.ECM_likelihood, softmax, 1, np.array([self.N_states, self.N_observations]))
      self.policy = self.ECM_to_probas(self.ECM_policy, softmax, 1, np.array([self.N_states, self.N_actions]))
      
      # self.boredom_actions = np.ones((self.N_states, self.N_actions), dtype=np.float64) / self.N_actions
      # self.boredom = np.ones((self.N_states * self.N_actions, self.N_states), dtype = np.float64) / self.N_states
      
      return self.prior, self.posterior, self.likelihood, self.policy
   
   
   def select_first_state(self, observation):
      """
      Selects the first state that is consistent with the observation it received.
      """
      obs_idx = np.where(self.observations == observation)[0][0]
      #obs_idx = self.observations.index(observation)
      self.previous_belief_state = obs_idx * self.N_clones + np.random.choice(np.arange(self.N_clones))
      return self.previous_belief_state
   
   
   def select_first_action(self,):
      """
      Selects the first action, given the state it sampled
      """
      
      self.previous_action = rand_choice_nb(self.actions, self.policy[self.previous_belief_state,:])
      # self.previous_action = np.random.choice(self.actions, 
      #                                         p=self.policy[self.previous_belief_state,:])
      return self.previous_action
   
   
   def deliberate_next_state(self, # emotional_thinking, reflection_time=5, consistency_given=False):
                             ):
      """
      Samples the belief about the next state in the environment from the prior distrib

      Parameters
      -------
      obs_environment : int.
         DESCRIPTION index of the observation the agent received from the environment
      
      Returns
      -------
      belief_state : int.
         DESCRIPTION. Belief about the next state in the environment
      """
      
      self.input_percept = self.calculate_input_percept()
      #if not emotional_thinking:
      posterior_cdt = self.posterior[self.input_percept,:].copy()/np.sum(self.posterior[self.input_percept,:].copy())
      # elif emotional_thinking:
      #    posterior_cdt = self.emotional_posterior[self.input_percept,:].copy()/np.sum(self.emotional_posterior[self.input_percept,:].copy())
      
      self.belief_state = rand_choice_nb(np.arange(self.N_states), posterior_cdt)
      #self.belief_state = np.random.choice(list(range(self.N_states)), p=posterior_cdt)
      
      # R=0
      # while self.ECM_emotions_posterior[self.input_percept, self.belief_state] == 1 and R < reflection_time:
      #     #print("reflecting")
      #     # self.belief_state = np.random.choice(list(range(self.N_states)), 
      #     #                                       p=self.posterior[self.input_percept,:]/np.sum(self.posterior[self.input_percept,:]))
      #     if not emotional_thinking:
      #       self.belief_state = rand_choice_nb(np.arange(self.N_states),
      #                                           self.posterior[self.input_percept,:]/np.sum(self.posterior[self.input_percept,:]) )
      #     else:
      #       self.belief_state = rand_choice_nb(np.arange(self.N_states),
      #                                           self.emotional_posterior[self.input_percept,:]/np.sum(self.emotional_posterior[self.input_percept,:]) )

      #     R += 1
         
      return self.belief_state
   
   
   def calculate_input_percept(self):
      
      self.input_percept = self.previous_belief_state * self.N_actions + self.previous_action
      
      return self.input_percept
   
   
   def calculate_free_energy(self, obs_environment, big_constant=50.0):
      """
      Calculates the (specific) Free Energy corresponding to the current state
      of knowledge (ECMs) of the agent, given the observation it received 
      from the environment
      
      Parameters
      -------
      obs_environment: int.
         DESCRIPTION. Value of the observation provided by the environment

      Returns
      -------
      specific_FE: np.ndarray
         DESCRIPTION. Array of the contribution of each state to the Free Energy
      
      FE : float
         DESCRIPTION. value of the Free Energy (avg over all states)
      """
   
      input_percept = self.previous_belief_state * self.N_actions + self.previous_action
      prior_temp = self.prior[input_percept,:].copy()
      posterior_temp = self.posterior[input_percept, :].copy()
      likelihood_temp = self.likelihood[:, obs_environment].copy()
      
      specific_FE = np.zeros(self.N_states)
      # Make logarithm compatible with numba
      epsilon = np.exp(-big_constant)
      posterior_temp[posterior_temp <= self.fake_zero] = epsilon
      prior_temp[prior_temp <= self.fake_zero] = epsilon
      likelihood_temp[likelihood_temp<= self.fake_zero] = epsilon
      
      specific_FE = np.log(posterior_temp)
      specific_FE -= np.log(prior_temp)
      evidence = - np.log(likelihood_temp)
      
      # # Calculate the KL-divergence
      # specific_FE = np.log(posterior_temp, out=np.zeros_like(posterior_temp), 
      #                       where=(posterior_temp > self.fake_zero))
      # specific_FE -= np.log(prior_temp, out=np.ones_like(prior_temp) * (-big_constant), 
      #                       where=(prior_temp > self.fake_zero))
      # # Calculate the evidence
      # evidence = -np.log(likelihood_temp, out=np.ones_like(likelihood_temp) * (-big_constant), 
      #                       where=(likelihood_temp > self.fake_zero)).transpose()
      evidence[posterior_temp<self.fake_zero] = 0
      
      # Average over all transitions
      specific_FE = posterior_temp * (specific_FE + evidence)
      specific_FE = np.maximum(0, specific_FE)

      FE = np.sum(specific_FE)
      
      return specific_FE, FE
   
   
   def deliberate_observation(self):
      """
      Samples an observation based on the guess on the internal state

      Returns
      -------
      obs_agent : integer
         DESCRIPTION. describes the index of the observation it predicts

      """
      # self.obs_agent = np.random.choice(self.observations, p=self.likelihood[self.belief_state,:])
      
      self.obs_agent = rand_choice_nb(self.observations, self.likelihood[self.belief_state,:])
      return self.obs_agent
   
   
   def update_g_matrix(self, obs_environment, applies_to_actions=False, eta=0.5):
      """
      In the case of long-term rewards, keeps track of the good 
      guesses the agent makes and stops when it makes a wrong guess.

      Parameters
      ----------
      obs_environment : integer
         DESCRIPTION. Observation given by the environment

      Returns
      -------
      self.ECM_g_posterior : ndarray
         DESCRIPTION. Updated g-matrix for the current agent and interaction round

      """
      
      if self.obs_agent == obs_environment:
         # increase the g_values of previous right transitions
         where_update = np.greater(self.ECM_g_posterior, 
                                   self.fake_zero * np.ones((self.N_states*self.N_actions, self.N_states), dtype=np.float64))
         self.ECM_g_posterior += where_update * eta
         
         # turn on the tracking for the current guess
         input_percept = self.previous_belief_state * self.N_actions + self.previous_action
         self.ECM_g_posterior[input_percept, self.belief_state] += 1.
         
         if applies_to_actions:
            where_update = np.greater(self.ECM_g_policy, 
                                      self.fake_zero * np.ones(self.ECM_g_policy.shape, dtype=np.float64))
            self.ECM_g_policy += where_update * 1.
            self.ECM_g_policy[self.previous_belief_state, self.previous_action] += 1.   # += 1
         
      return self.ECM_g_posterior
   
    
   # def update_emotions(self, obs_environment, applies_to_actions=False):
   #    """
   #    Marks the transition that induced the failed transition.

   #    Parameters
   #    ----------
   #    obs_environment : int 
   #       DESCRIPTION. observation given by the environment (external to the agent)

   #    Returns
   #    -------
   #    self.ECM_emotions_posterior : ndarray
   #       DESCRIPTION. flags the problematic transition

   #    """
      
   #    if obs_environment != self.obs_agent:
   #       self.ECM_emotions_posterior = np.zeros((self.N_states * self.N_actions, self.N_actions), dtype=np.int64)
   #       self.ECM_emotions_posterior[self.input_percept, self.belief_state] = 1
         
   #       if applies_to_actions:
   #          self.ECM_emotions_policy = np.zeros(((self.N_states, self.N_actions)), dtype=np.int64)
   #          self.ECM_emotions_policy[self.previous_belief_state, self.previous_action] = 1
      
   #    return self.ECM_emotions_posterior
   
   
   def update_posterior_cumulative_rewards(self, obs_environment, 
                                           gamma=1/100, eta = 1/12, h_0 = 1.,
                                           #emotional_thinking = True, 
                                           alpha=1,
                                           #emotion_actions = True, 
                                           scale_rewards = 5.,
                                           force_update = False,
                                           softmax=False, softmax_temperature=1.):
      """
      Updates the posterior after a trajectory of interactions with the environment
      
      Parameters:
      -----------
      false_update: Boolean
         DESCRIPTION. Force an update (for example at the end of an episode)

      Returns
      -------
      self.ECM_posterior: ndarray
         DESCRIPTION. updated posterior for the transition probabilities
      """
      
      # self.ECM_posterior -= gamma * (self.ECM_posterior - h_0 * np.ones(self.ECM_posterior.shape))
      self.ECM_posterior -= gamma * (self.ECM_posterior - h_0 * np.ones(self.ECM_posterior.shape))

      if obs_environment != self.obs_agent or force_update:

         
         # Update the posterior
         self.ECM_posterior += self.ECM_g_posterior * scale_rewards
         # Record the final reward and the trajectory
         self.reward_posterior = np.max(self.ECM_g_posterior * scale_rewards)
         self.past_trajectory = self.ECM_g_posterior.copy()

         # Reset the g-matrix
         self.ECM_g_posterior = np.zeros(self.ECM_g_posterior.shape)
        
         
         self.posterior = self.ECM_to_probas(self.ECM_posterior, softmax, 
                                             softmax_temperature,
                                             np.array([self.N_states * self.N_actions, self.N_states]))
         # if emotional_thinking:
         #    g_max = np.float64(np.max(self.past_trajectory))
         #    self.boredom = np.zeros((self.N_states * self.N_actions, self.N_states), dtype=np.float64)
         #    # if g_max > 0:
         #    #    for i in range(self.N_states * self.N_actions):
         #    #       self.boredom[i,:] += self.past_trajectory[i,:]
         #    if g_max > 0:      
         #       # self.emotional_posterior = np.zeros((self.N_states * self.N_actions, self.N_states), dtype=np.float64)
         #       for i in range(self.N_states * self.N_actions):
         #          self.boredom[i,:] = (g_max - self.past_trajectory[i,:].copy())/g_max * np.mean(self.ECM_posterior[i,:])
         #       # self.boredom = (g_max - self.past_trajectory.copy())
         #    self.emotional_posterior = self.ECM_to_probas(self.ECM_posterior.copy() + alpha * self.boredom, 
         #                                                  False,1,
         #                                                  np.array([self.N_states * self.N_actions, self.N_states]))
      return self.ECM_posterior, self.posterior

   # def update_boredom_policy(self, emotion_actions = True):  
   
   #       if emotion_actions:
            
   #          # Calculate biased policy
   #          self.past_trajectory_actions = self.ECM_g_policy.copy()
   #          g_max_actions = np.max(self.past_trajectory_actions)
   #          # self.boredom_actions = np.zeros((self.N_states, self.N_actions)) #self.boredom_actions#
   #          self.boredom_actions *= 1/2
   #          if g_max_actions > 0:
   #             for i in range(self.N_states):
   #                sum_i = np.sum(self.past_trajectory_actions[i,:])
   #                if sum_i > 0:
   #                   for j in range(self.N_actions):
   #                      #self.boredom_actions[i,j] += (g_max_actions - self.past_trajectory_actions[i,j]) / g_max_actions
   #                      self.boredom_actions[i,j] += 1/2 * (sum_i - self.past_trajectory_actions[i,j])/sum_i
   #                      # if self.past_trajectory_actions[i,j] > self.fake_zero:
   #                         # self.boredom_actions[i,j] = self.past_trajectory_actions[i,j] / g_max_actions
         
   #       self.ECM_g_policy = np.zeros((self.N_states, self.N_actions))
         
         
      
   
  
   # def update_ECM_policy_cumulative_rewards(self, obs_environment,
   #                                      gamma=1/100, eta = 1/12,
   #                                      scale_rewards = 5,
   #                                      force_update = False,
   #                                      softmax=False, softmax_temperature = 1):
   #    """
   #    Updates the baseline h-values of the behavior ECM with the cumulated rewards collected through trajectories.
      
   #    Returns
   #    -------
   #    self.ECM_policy, self.policy
   #    """
      
   #    self.ECM_policy -= gamma * (self.ECM_policy - np.ones(self.ECM_policy.shape))
   #    if obs_environment != self.obs_agent or force_update:
   #       self.ECM_policy += self.ECM_g_policy * scale_rewards
   #       self.reward_policy = np.max(self.ECM_g_policy) * scale_rewards
         
   #       self.ECM_g_policy = np.zeros(self.ECM_g_policy.shape)
         
   #    return self.ECM_policy
   
  
   def EFE_filter(self, preference_target, preferences_marginal=True,
                  preferred_obs=3,
                  proba_preferred_obs=3/4, horizon=1, gamma=0.8,
                  give_reward=True, EFE_to_weights=True, curiosity_param = -10):
      """
      Makes the EFE_matrix to renormalize the baseline h-values with the expected free energy

      Parameters
      -----------
      preference_target: Boolean
         DESCRIPTION. whether or not the agent is given a goal

      Returns
      -------
      self.EFE_contributions

      """
      
      # previous_EFEs = self.EFE_contributions.copy()
      self.EFE_contributions = np.zeros((self.N_states, self.N_actions), dtype=np.float64)
      for state in range(self.N_states):
         EFE_array = self.calculate_expected_FE(state, 
                                                preference_target, 
                                                preferences_marginal,
                                                proba_preferred_obs, 
                                                50.0, 
                                                horizon,
                                                preferred_obs)
                                              
         self.EFE_contributions[state,:] = EFE_array.copy()
         
      if EFE_to_weights:
         self.EFE_weights = self.EFE_contributions.copy()
         for j in range(self.N_states):
            EFE_minimum = np.min(self.EFE_contributions[j,:])
            self.EFE_weights[j,:] -= EFE_minimum
            if np.sum(self.EFE_weights[j,:]) < self.fake_zero:   # All weights were equal to the minimal value
               self.EFE_weights[j,:] = 1/self.N_actions
         self.EFE_weights = self.ECM_to_probas(self.EFE_weights, False, 1,  np.array([self.N_states, self.N_actions]))
      
      return self.EFE_contributions
   
  
   # def reinforcement_policy(self, zeta, gamma_policy, eta_policy, scale_reward):
   #    """
   #    Updates the policy after the world model has been trained by reinforcing 
   #    the trajectories the agent picks

   #    Parameters
   #    ----------
   #    zeta : TYPE float
   #       DESCRIPTION. Curiosity parameter (negative: conservative policy, positive: explorative)
   #    scale_reward : TYPE float
   #       DESCRIPTION. Scale for the reinforcement of the edges

   #    Returns
   #    -------
   #    ECM_policy : TYPE np.array
   #       DESCRIPTION. h-values to calculate the policy for the agent

   #    """
      
   #    # Update the g_values
   #    self.ECM_g_policy *= (1-eta_policy)
   #    self.ECM_g_policy[self.belief_state, self.previous_action] = 1 
      
   #    # Calculate the reward
   #    weight = self.EFE_weights[self.belief_state, 
   #                                              self.previous_action]
   #    median = np.median(self.EFE_weights.copy(), axis=1)[self.belief_state]
      
   #    self.reward_policy = (weight <= median) * (1-weight) * scale_reward      
      
   #    # Update the h-values
   #    self.ECM_policy -= gamma_policy * (self.ECM_policy - np.ones(self.ECM_policy.shape))
   #    self.ECM_policy += self.ECM_g_policy * self.reward_policy
      
   #    # Build the policy distribution                                                           
   #    self.policy = self.ECM_to_probas(self.ECM_policy)
      
   
   # def update_policy_cumulative_rewards(self, obs_environment, temp_EFE = 5, certainty_seeking = False):
      
   #    if obs_environment != self.obs_agent:
      
   #       self.weighted_ECM_policy = np.zeros(self.policy.shape)
         
   #       weights_EFE = np.exp((-1) ** (certainty_seeking) * temp_EFE * self.EFE_contributions)
   #       weights_EFE /= np.sum(weights_EFE, axis=1)[:, np.newaxis]
   #       self.weighted_ECM_policy = self.ECM_policy * weights_EFE
         
   #       self.policy = self.weighted_ECM_policy / np.sum(self.weighted_ECM_policy, axis=1)[:, np.newaxis]
   #       self.ECM_g_policy = np.zeros(self.ECM_g_policy.shape)
         
   #    return self.policy
   
  
   # def calculate_reward_posterior(self, obs_environment, reward_scale=1.0, reward_with_gradient=False ):
   #    """
   #    Calculates the reward to apply to the edge sampled by the agent 
   #    during its deliberation
   #    It also updates the matrix of rewards averaged for each edge, over the course of an episode

   #    Parameters
   #    ----------
   #    obs_environment: int.
   #       DESCRIPTION. Value of the observation provided by the environment
         
   #    reward_scale : FLOAT, 
   #       DESCRIPTION. Scale of the rewards applied to the h-values of the ECM

   #    Returns
   #    -------
   #    reward_posterior : FLOAT
   #       DESCRIPTION. reward from sampling a state during the deliberation 
   #       calculated from a discrete gradient on the Free Energy
   #    """
      
   #    self.FE_specific, self.FE = self.calculate_free_energy(obs_environment)
   #    current_ECM_posterior = self.ECM_posterior.copy()
      
   #    # Give the hypothetical reward to the posterior
   #    input_percept = self.previous_belief_state * self.N_actions + self.previous_action
   #    if reward_with_gradient:
   #       self.ECM_posterior[input_percept, self.belief_state] += reward_scale
   #       self.posterior = self.ECM_to_probas(self.ECM_posterior).copy()
   #       specific_FE_rewarded, FE_rewarded = self.calculate_free_energy(obs_environment)
   #       gradient = (self.FE - FE_rewarded)/reward_scale #(self.FE + self.fake_zero)
   
   #       self.reward_posterior = gradient
      
            
   #    else:   # Reward with the contribution of the specific FE to the total FE
   #          self.reward_posterior = 1 
   #          if self.FE > 0:
   #             self.reward_posterior -= self.FE_specific[self.belief_state] / self.FE
         
   #    self.reward_posterior = np.maximum(0, self.reward_posterior)
      
   #    self.ECM_posterior = current_ECM_posterior.copy()
   #    self.posterior = self.ECM_to_probas(self.ECM_posterior)
      
   #    return self.reward_posterior
   
  
   # def update_posterior(self, obs_environment, gamma=1/100, eta = 1/12, 
   #                      scale_rewards = 5, reward_with_gradient=True,
   #                      softmax=False, softmax_temperature=1):
   #    """
   #    Applies the update to the posterior's ECM

   #    Parameters
   #    ----------
   #    obs_environment: int.
   #       DESCRIPTION. Value of the observation provided by the environment
   #    gamma: float.
   #       DESCRIPTION. Forgetting rate of the agent
   #    eta: float
   #       DESCRIPTION. decaying rate for the glow
         
   #    Returns
   #    -------
   #    ECM_posterior: ndarray
   #       DESCRIPTION. Array corresponding to the updated ECM for the posterior
         
   #    ECM_g_posterior: ndarray
   #       DESCRIPTION. Glow values of the agent
         
   #    posterior: ndarray
   #       DESCRIPTION. updated poosterior of the agent
   #    """
      
   #    input_percept = self.previous_belief_state * self.N_actions + self.previous_action
      
   #    # Update the ECMs
   #    self.ECM_g_posterior *= (1-eta)
   #    self.ECM_g_posterior[input_percept, self.belief_state] = 1
      
   #    self.reward_posterior = self.calculate_reward_posterior(obs_environment, 
   #                                                            reward_with_gradient=reward_with_gradient,
   #                                                            reward_scale = scale_rewards)
      
      
   #    self.ECM_posterior += -gamma * (self.ECM_posterior - np.ones(self.ECM_posterior.shape)) 
   #    self.ECM_posterior += self.ECM_g_posterior * self.reward_posterior * scale_rewards
      
   #    # Update the posterior
   #    self.posterior = self.ECM_to_probas(self.ECM_posterior, 
   #                                        softmax, softmax_temperature, 
   #                                        np.array([self.N_states * self.N_actions, self.N_states]))
   #    self.posterior = np.maximum(np.full(self.posterior.shape, self.fake_zero), 
   #                                self.posterior)
      
   #    return self.ECM_posterior, self.ECM_g_posterior, self.posterior, self.reward_posterior*scale_rewards, self.FE
   
   
   # def define_preferences_DR(self, temperature_pref, preferred_obs=1, horizon=3, 
   #                           return_full_pref=False):
   #    """
   #    Assigns a preference distribution to the agent, that targets having a full stomach. 
   #    It makes the assumption that the distribution factorizes over observations and internal states

   #    Parameters
   #    ----------
   #    temperature : TYPE float
   #       DESCRIPTION. sets the contrast in preferences between the different observations / internal states

   #    Returns
   #    -------
   #    self.preferences : ndarray
   #       DESCRIPTION preference distribution of the agent that encodes its goal
   #    """
      
   #    self.state_value = np.sum(self.posterior[::2,:] + self.posterior[1::2,:], axis=0)   # vector of values for each state
      
   #    self.pref_obs = np.ones(self.N_observations)
   #    self.pref_obs[preferred_obs] = temperature_pref
   #    self.pref_obs /= np.sum(self.pref_obs)
   #    self.pref_obs_clones = np.repeat(self.pref_obs, self.N_clones)
      
   #    self.state_value = self.state_value * self.pref_obs_clones
   #    self.state_value /= np.sum(self.state_value)
      
   #    self.preferences = np.kron(self.state_value, np.ones(self.N_observations))   #self.pref_obs)
   #    self.preferences /= np.sum(self.preferences)
      
   #    if not return_full_pref:
   #       return self.preferences
   #    else:
   #       return self.preferences, self.full_preferences
   
   
   def calculate_preferences(self, last_state, action, 
                             preference_target=False, 
                             discount=0.8, max_filter=False, softmax_scale = 1,
                             preferred_obs=3, proba_preferred_obs=0.75, 
                             horizon=1, gamma=0.8, 
                             softmax=False):
   
      """Calculates the array corresponding to the preferences distribution, either for passive learning,
      or given a target state. 
      """
      
      
      input_percept = last_state * self.N_actions + action
      
      if not preference_target:
         
         gen_model = np.zeros((self.N_states, self.N_states * self.N_observations))
         for state in range(self.N_states):
            input_percept = state * self.N_actions + action
            gen_model_state = self.posterior[input_percept,:][:, np.newaxis].copy() * self.likelihood.copy()
            gen_model_state = gen_model_state.copy().reshape((1, self.N_states * self.N_observations))
            gen_model[state,:] = gen_model_state.copy()
            gen_model[state,:] = gen_model_state.copy() / np.sum(gen_model_state.copy())
         self.full_preferences_obs = gen_model.copy()
         self.preferences = self.full_preferences_obs[last_state,:].copy()
      
      else:
         v_prev = np.ones(self.N_observations, dtype=np.float64) * (1.-proba_preferred_obs) / (self.N_observations-1)
         v_prev[preferred_obs] = proba_preferred_obs
         v_prev /= np.sum(v_prev)
         pref_obs = v_prev.copy()
         v_prev = np.repeat(v_prev.copy(), self.N_clones)
         
         
         # Calculate the reachability: how easy is it to reach the next states?
         self.reachability = self.posterior[0::self.N_actions].copy() * self.policy[:, 0].copy()#/ self.N_actions
         for a in range(1, self.N_actions):
            self.reachability += self.posterior[a::self.N_actions].copy() * self.policy[:, a].copy()#/self.N_actions

         
         t=1
         while t < horizon:
            v_t = self.reachability * v_prev.reshape((1,self.N_states))
            v_t_max = np.zeros(self.N_states)
            for b in range(self.N_states):
               v_t_max[b] = np.max(v_t[b,:])
               
            v_prev = np.maximum(v_prev, discount**(t-1) * v_t_max)
            t+=1
            
         
         # Determine the value of a state conditioned on the previous state:
         children_thresholds = np.zeros(self.N_states, dtype=np.float64)
         for i in range(self.N_states):
            children_thresholds[i] = np.mean(self.reachability[i,:])
         children_boolean = self.reachability > children_thresholds[np.newaxis,:]
         V_cond = children_boolean * v_prev.reshape((1, self.N_states))
         
         if max_filter==True:
             max_pref = np.argmax(V_cond.copy(), axis=1)
             self.full_preferences = np.ones(((self.N_states,self.N_states)), dtype=np.float64) * self.fake_zero
             for i in range(self.N_states):
                self.full_preferences[i, max_pref[i]] = 1. - self.fake_zero * (np.float64(self.N_states)-1.)
         else:
             self.full_preferences = self.ECM_to_probas(V_cond, softmax, softmax_scale, np.array([self.N_states, self.N_states]))
         self.full_preferences_obs = np.kron(self.full_preferences.copy(), pref_obs)#np.ones(self.N_observations)/self.N_observations)         
      
         
         self.preferences = self.full_preferences_obs[last_state, :]
         
         return self.preferences
   
 
   def calculate_expected_FE(self, last_state, preference_target=False, preferences_marginal=True,
                             proba_preferred_obs=3/4,
                             big_constant=40.0, horizon=1, preferred_obs=3):
      """
      Calculates the expected Free Energy, given the current state the agent believes it is in,
      for each possible action

      Returns
      -------
      EFE_array : ndarray
         DESCRIPTION. array of the values of the EFE for each action

      """
      
      EFE_array = np.zeros((self.N_actions))

      for action in range(self.N_actions):
         input_percept = last_state * self.N_actions + action
         gen_model = self.posterior[input_percept,:][:, np.newaxis] * self.likelihood
         gen_model = gen_model.reshape(self.N_states * self.N_observations)

         if not preference_target:
            if not preferences_marginal:
               preferences_temp = np.zeros(self.N_states * self.N_observations)
               preferences_temp = np.exp(np.minimum(400, self.ECM_posterior[input_percept,:]))
               preferences_temp /= np.sum(preferences_temp)
               preferences_temp = preferences_temp[:, np.newaxis] * self.likelihood
               self.preferences = preferences_temp.reshape(self.N_states * self.N_observations)
            elif preferences_marginal:
               self.preferences = np.zeros((self.N_states * self.N_observations))
               for act in range(self.N_actions):
                   percept = last_state * self.N_actions + act
                   gen_marg = self.posterior[percept,:][:, np.newaxis] * self.likelihood
                   gen_marg = gen_marg.reshape(self.N_states * self.N_observations)
                   gen_marg *= self.policy[last_state, act]
                   self.preferences += gen_marg
         else:
            self.preferences = self.full_preferences_obs[last_state,:].copy()

            
         non_zeros = 1. * np.greater(gen_model, 10*self.fake_zero)
         # pref_non_zeros = self.preferences > 10*self.fake_zero
         
         posterior_temp = self.posterior[input_percept,:].copy()
         posterior_temp_non_zero = posterior_temp > 10*self.fake_zero
         preferences_temp = self.preferences.copy()
         preferences_temp_non_zero = preferences_temp > 10*self.fake_zero
         
         posterior_temp = posterior_temp.copy() * posterior_temp_non_zero
         posterior_temp += (1-posterior_temp_non_zero)* self.fake_zero
         
         preferences_temp = preferences_temp.copy() * preferences_temp_non_zero
         preferences_temp += (1-preferences_temp_non_zero) * self.fake_zero
         
         # epsilon = np.exp(-big_constant)
         # posterior_temp[posterior_temp < self.fake_zero] = epsilon
         # preferences_temp[preferences_temp < self.fake_zero] = epsilon
         
         
         EFE = np.log(np.repeat(posterior_temp, self.N_observations)) \
               - np.log(preferences_temp)
         EFE = gen_model * EFE * (non_zeros)
         EFE = np.sum(EFE)
         if EFE <= 0:
            EFE = self.fake_zero
   
         EFE_array[action] = EFE
         # if np.isnan(np.sum(EFE)):
         #    print("EFE is nan")
         #    print("posterior", posterior_temp)
         #    print("preferences", preferences_temp)
         #    print("log posterior", np.log(np.repeat(posterior_temp, self.N_observations)))
         #    print("log preferences", - np.log(preferences_temp))
         
      return EFE_array
   

   # def update_policy(self, temp=-1, preference_target=False, temperature_pref=5, update_ON=True):
   #    """
   #    Updates the policy ECM with the relevant EFE and calculates the policy 
   #    as a softmax of the ECM

   #    Parameters
   #    ----------
   #    temp : float
   #       DESCRIPTION. Temperature for the softmax function. The default is -5.

   #    Returns
   #    -------
   #    ECM_policy : ndarray
   #       DESCRIPTION. Updated behavior ECM of the agent
   #    policy : ndarray
   #       DESCRIPTION. Updated policy calculated from the new ECM

   #    """
      
   #    self.EFE_array = self.calculate_expected_FE(self.belief_state, 
   #                                                preference_target=preference_target, 
   #                                                temperature_pref=temperature_pref)
   #    self.EFE = np.min(self.EFE_array)
      
   #    # update the ECM
   #    if update_ON:
   #       self.ECM_policy[self.belief_state,:] = self.EFE_array.copy()
         
   #       # update the policy
   #       self.policy = self.ECM_to_probas(self.ECM_policy, True, temp, np.array([self.N_states, self.N_actions]))
      
   #    return self.ECM_policy, self.policy, self.EFE


   def deliberate_next_action(self): #, R=5, emotion_actions=True):
      """
      Samples the next action from the policy, given the previous belief state

      Returns
      -------
      self.previous_action: int.
         DESCRIPTION. Index of the action the agent will take in the next step
      """
      
      # self.previous_action = np.random.choice(self.actions, p=self.policy[self.belief_state,:])
      # if not emotion_actions:
      self.previous_action = rand_choice_nb(self.actions, self.policy[self.belief_state,:])
      # elif emotion_actions:
      #    self.previous_action = rand_choice_nb(self.actions, self.emotional_policy[self.belief_state,:])
      
      # # If selected the previous erroneous action:
      #    # Reflect and try to sample another action
      # r=0
      # while self.ECM_emotions_policy[self.belief_state, self.previous_action] == 1 and r<= R:
      #    if not emotion_actions:
      #       self.previous_action = rand_choice_nb(self.actions, self.policy[self.belief_state,:])
      #    elif emotion_actions:
      #       self.previous_action = rand_choice_nb(self.actions, self.emotional_policy[self.belief_state,:])
      #    r+=1
      return self.previous_action
   

   # def test_DR(self, environment, N_trials = 100, trial_length = 1000, target_obs = 1):
   #    """
   #    Simulates the trained agent on a task based on the WM it has a learnt and collects the success rates      
   #    Return the sucess rates averaged over a number of trials
      
   #    Parameters
   #    ----------
   #    environment : object.
   #       DESCRIPTION. Object that defines the environment and provides the agent 
   #       with the observations relative to its WM. All parameters are already provided
         
   #    N_trials: int.
   #       DESCRIPTION. Number of attempts to get the target
         
   #    trial_length: int.
   #       DESCRIPTION. Length of each trial, or number of interactions with the environment for each trial
         
   #    target_obs: int.
   #       DESCRIPTION. target observation in the environment (delayed rewards: light off/stomach full)
      
   #    Returns
   #    -------
   #    success_rate: float.
   #       DESCRIPTION. Frequency at which the agent reaches the target amongst N_trials attempts
      
   #    avg_success_distance: float.
   #       DESCRIPTION. Average duration between two successful attempts

   #    """
      
   #    self.success_rate = np.zeros(N_trials)
   #    self.success_distance = np.zeros(N_trials)
   #    for trial in range(N_trials):
   #       # Initialize the agent in the environment
   #       environment.observation = environment.initial_conditions()
      
   #       # Initial deliberation
   #       self.previous_belief_state = self.select_first_state(environment.observation)
   #       self.previous_action = self.select_first_action()
         
   #       step_counter = 0   # Counts the number of steps between successes
         
   #       for step in range(trial_length):
            
   #          step_counter += 1
   #          # Apply action in environment
   #          environment.environment_state = environment.state_transition(self.previous_action)
   #          environment.observation = environment.give_observation(self.previous_action)
            
   #          if environment.observation == target_obs:
   #             self.success_rate[trial] += 1
   #             self.success_distance[trial] += step
   #             # step_counter = 0
   #             break
                  
   #          if environment.observation != self.obs_environment or step == trial_length-1:
   #             self.success_distance[trial] = trial_length
   #             break

   #          self.previous_action = self.deliberate_next_action()
   #          self.previous_belief_state = self.belief_state + 0
            
            
   #    self.success_rate = np.sum(self.success_rate) / N_trials #np.mean(self.success_rate)
   #    self.success_distance = np.mean(self.success_distance)
      
   #    return self.success_rate, self.success_distance
   
   def test_DR_superpostion(self, environment, N_trials, trial_length, target_obs):
      self.success_rates = np.zeros(N_trials, dtype=np.float64)
      self.success_distances = np.zeros(N_trials, dtype=np.float64)
      self.failed_predictions = np.zeros(N_trials, dtype=np.float64)
      for trial in range(N_trials):
          print(trial)
          # Initialize the agent in the environment
          environment.observation = environment.initial_conditions()
      
          self.plausible_previous_belief_states = np.zeros(self.N_states, dtype=np.int64)
          self.plausible_previous_belief_states[environment.observation * self.N_clones : environment.observation * self.N_clones + self.N_clones] = 1
         
          # Deliberate on the action to take
          self.plausible_previous_actions = np.zeros((self.N_clones), dtype=np.int64)
          # self.ECM_emotions_policy = np.zeros((self.N_states, self.N_actions), dtype=np.int64)
          frequencies_plausible_actions = np.zeros((self.N_actions), dtype=np.float64)
          for i in range(self.N_states):
            self.belief_state = self.plausible_previous_belief_states[i]
            act = np.int64(self.deliberate_next_action())
            frequencies_plausible_actions[act] += 1. / self.N_states
          self.previous_action = rand_choice_nb(self.actions, frequencies_plausible_actions)
         
          for step in range(1, trial_length):
      
            # Apply action in environment
            environment.environment_state = environment.state_transition(self.previous_action)
            environment.observation = environment.give_observation(self.previous_action)
            print("environment " + str(environment.environment_state))
               
            # Based on the observation, deliberate on the next plausible belief state:
            self.plausible_belief_states = np.zeros(self.N_states, dtype=np.int64)
            for state in range(self.N_states):
                if self.plausible_previous_belief_states[state] > 0:
                  self.previous_belief_state = self.plausible_previous_belief_states[state]
                  self.belief_state = self.deliberate_next_state(#False, 0, False
                                                                 )
                  self.obs_agent = self.deliberate_observation()
                  if self.obs_agent == environment.observation:
                      self.plausible_belief_states[self.belief_state] = 1
                     
            if np.sum(self.plausible_belief_states) == 0:
                self.plausible_belief_states = np.zeros(self.N_states, dtype=np.int64)
                self.plausible_belief_states[environment.observation * self.N_clones : environment.observation *self.N_clones +self.N_clones] = 1
            
            # Update the belief states when they confirmed the observation
            self.plausible_previous_belief_states = self.plausible_belief_states.copy()
            
            # Deliberate on the action to take
            self.plausible_previous_actions = np.zeros((self.N_clones), dtype=np.int64)
            # self.ECM_emotions_policy = np.zeros((self.N_states, self.N_actions), dtype=np.int64)
            frequencies_plausible_actions = np.zeros((self.N_actions), dtype=np.float64)
            for i in range(self.N_states):
                self.belief_state = self.plausible_previous_belief_states[i]
                act = int(self.deliberate_next_action())# 5))
                frequencies_plausible_actions[act] += 1. / self.N_states
            self.previous_action = rand_choice_nb(self.actions, frequencies_plausible_actions)
            
            print(step)
            
            if environment.observation == target_obs:
                self.success_rates[trial] = 1.
                self.success_distances[trial] = np.float64(step)
                break
               
            if step == trial_length - 1:
                self.success_distances[trial] = np.float64(step)
      
      self.success_rate = np.mean(self.success_rates)
      self.success_distance = np.mean(self.success_distances)
      
      
   # def test_DR_superposition(self, environment, causal_model=False,
   #                           N_trials = 100, trial_length = 50, target_obs = 1):
   #    """
   #    Simulates the trained agent on a task based on the WM it has a learnt and collects the success rates      
   #    Return the sucess rates averaged over a number of trials
      
   #    Parameters
   #    ----------
   #    environment : object.
   #       DESCRIPTION. Object that defines the environment and provides the agent 
   #       with the observations relative to its WM. All parameters are already provided
         
   #    N_trials: int.
   #       DESCRIPTION. Number of attempts to get the target
         
   #    trial_length: int.
   #       DESCRIPTION. Length of each trial, or number of interactions with the environment for each trial
         
   #    target_obs: int.
   #       DESCRIPTION. target observation in the environment (delayed rewards: light off/stomach full)
      
   #    Returns
   #    -------
   #    success_rate: float.
   #       DESCRIPTION. Frequency at which the agent reaches the target amongst N_trials attempts
      
   #    avg_success_distance: float.
   #       DESCRIPTION. Average duration between two successful attempts

   #    """
      
   #    self.success_rates = np.zeros(N_trials, dtype=np.float64)
   #    self.success_distances = np.zeros(N_trials, dtype=np.float64)
   #    self.failed_predictions = np.zeros(N_trials, dtype=np.float64)
      
   #    for trial in range(N_trials):
          
   #        # Initialize the agent in the environment
   #        environment.observation = environment.initial_conditions()
      
   #        # Initial deliberation
   #        self.plausible_previous_belief_states = np.zeros(self.N_states, dtype=np.int64)
   #        self.plausible_previous_belief_states[environment.observation * self.N_clones : environment.observation * self.N_clones + self.N_clones] = 1
          
   #        # Deliberate on the action to take
   #        self.ECM_emotions_policy = np.zeros((self.N_states, self.N_actions), dtype=np.int64)
   #        self.ECM_emotions_posterior = np.zeros((self.N_states*self.N_actions, self.N_states), dtype=np.int64)
   #        frequencies_plausible_actions = np.zeros((self.N_actions), dtype=np.float64)
   #        for i in range(self.N_states):
   #           self.belief_state = self.plausible_previous_belief_states[i]
   #           act = np.int64(self.deliberate_next_action())
   #           frequencies_plausible_actions[act] += 1. / self.N_states
   #        self.previous_action = rand_choice_nb(self.actions, frequencies_plausible_actions)
          
   #        for step in range(1, trial_length):
      
   #           # Apply action in environment
   #           environment.environment_state = environment.state_transition(self.previous_action)
   #           environment.observation = environment.give_observation(self.previous_action)
                
   #           # Based on the observation, deliberate on the next plausible belief state:
   #           self.plausible_belief_states = np.zeros(self.N_states, dtype=np.int64)
   #           for state in range(self.N_states):
   #              if self.plausible_previous_belief_states[state] > 0:
   #                 self.previous_belief_state = self.plausible_previous_belief_states[state]
   #                 self.belief_state = self.deliberate_next_state(#False, 0, False
   #                                                                  )
   #                 obs = self.deliberate_observation()
   #                 if obs == environment.observation:
   #                    self.plausible_belief_states[self.belief_state] = 1
                      
   #           if np.sum(self.plausible_belief_states) == 0:
   #              self.plausible_belief_states = np.zeros(self.N_states, dtype=np.int64)
   #              self.plausible_belief_states[environment.observation * self.N_clones : environment.observation * self.N_clones + self.N_clones] = 1
             
   #           # Update the belief states when they confirmed the observation
   #           self.plausible_previous_belief_states = self.plausible_belief_states.copy()
             
   #           # Deliberate on the action to take
   #           self.plausible_previous_actions = np.zeros((self.N_clones), dtype=np.int64)
   #           self.ECM_emotions_policy = np.zeros((self.N_states, self.N_actions), dtype=np.int64)
   #           frequencies_plausible_actions = np.zeros((self.N_actions), dtype=np.float64)
   #           for i in range(self.N_states):
   #              self.belief_state = self.plausible_previous_belief_states[i]
   #              act = np.int64(self.deliberate_next_action(5))
   #              frequencies_plausible_actions[act] += 1. / self.N_states
   #           self.previous_action = rand_choice_nb(self.actions, frequencies_plausible_actions)
      
   #           if environment.observation == target_obs:
   #              self.success_rates[trial] = 1.
   #              self.success_distances[trial] = np.float64(step)
   #              break
                
   #           if step == trial_length - 1:
   #              self.success_distances[trial] = np.float64(step)
      
   #    self.success_rate = np.mean(self.success_rates)
   #    self.success_distance = np.mean(self.success_distances)
      
      
   #    return self.success_rate, self.success_distance
   

   def deliberate_next_state_testing(self, obs_environment):
      """
      Samples the belief about the next state in the environment from the prior distrib

      Parameters
      -------
      obs_environment : int.
         DESCRIPTION index of the observation the agent received from the environment
      
      Returns
      -------
      belief_state : int.
         DESCRIPTION. Belief about the next state in the environment
      """
      
      self.input_percept = self.calculate_input_percept()
      invalid_states = [i for i in range(self.N_states) if (i < self.N_clones * obs_environment or i >= self.N_clones * (obs_environment+1))]
      posterior_cdt = self.posterior[self.input_percept,:].copy()
      posterior_cdt[np.array(invalid_states)] = 0
      posterior_cdt = posterior_cdt/np.sum(posterior_cdt)
     
      
      # self.belief_state = np.random.choice(list(range(self.N_states)), p=posterior_cdt)
      self.belief_state = rand_choice_nb(np.arange(self.N_states), posterior_cdt)
         
      return self.belief_state
   

   # def test_PGW(self, environment, N_trials = 100, trial_length = 20, 
   #              target_obs=3, target_coordinates =(2,2),
   #              emotional_thinking=False,
   #              periodic_boundaries = False):
      
   #    self.time_to_target = np.zeros(environment.dim ** 2)
   #    self.trajectory_to_target = []
      
   #    for initial_state in environment.grid_indices:
   #       for test in range(N_trials):
   #          t=0
   #          end = False
   #          environment.agent_position = initial_state + 0
   #          environment.observation = environment.give_observation()
   #          if environment.observation == target_obs:
   #             end=True
   #          if initial_state == 0:
   #             traj = [0]
               
   #          self.belief_state = self.select_first_state(environment.observation)
   #          while not end:
   #             # self.previous_action = np.random.choice(self.actions, p=self.policy[self.belief_state,:])
   #             self.previous_action = rand_choice_nb(self.actions, self.policy[self.belief_state,:])
   #             environment.agent_position = environment.move_the_agent(self.previous_action, 
   #                                                            periodic_boundaries=periodic_boundaries)
   #             environment.observation = environment.give_observation()
               
   #             if initial_state == 0:
   #                traj.append(environment.agent_position)
               
   #             self.observation = self.deliberate_observation()
               
   #             self.previous_belief_state = self.belief_state + 0
   #             self.belief_state = self.deliberate_next_state_testing(environment.observation)
   #             t += 1
   #             end = (t > trial_length or environment.observation == target_obs)
            
   #          if initial_state == 0 and end:
   #             self.trajectory_to_target.append(np.array(traj))
               
               
   #          self.time_to_target[initial_state] += t/N_trials
                       
   #    return self.time_to_target
   

   
   # def make_epsilon_machine(self, threshold=1):
   #    """
   #    Function that looks at the world model and determines whether it which states 
   #    can be kept and labelled as causal states of an epsilon machine. 
   #    It also calculates the causal posterior.
   
   #    Parameters
   #    ----------
   #    threshold : float
   #       DESCRIPTION. Threshold on the D_KL to identify two states as identical. =2 by default
   
   #    Returns
   #    -------
   #    None.
   
   #    """
   #    def L2_norm(distrib1, distrib2):
   #       dif = distrib1 - distrib2
   #       dif = dif.copy() * dif.copy()
   #       return np.sum(dif)
         
      
   #    self.causal_states = []
   #    # Identify the causal states
   #    self.divergences = np.zeros((self.N_observations, int(self.N_clones * (self.N_clones-1)/2)))
   #    for obs in self.observations:
   #       index = 0
   #       self.causal_states.append([])
         
   #       for clone_pair in itertools.combinations(range(self.N_clones), 2):
   #          memory = np.zeros(self.N_actions)
   #          state1 = obs * self.N_clones * self.N_actions + clone_pair[0] * self.N_actions
   #          state2 = obs * self.N_clones * self.N_actions + clone_pair[1] * self.N_actions
            
   #          for act in range(self.N_actions):
   #             memory[act] = L2_norm(self.posterior[state1 + act,:], self.posterior[state2 + act,:])
               
   #          self.divergences[obs, index] += np.mean(memory)
   #          if self.divergences[obs, index] < threshold:
   #             self.causal_states[obs].append(clone_pair)
   #          index += 1
      
   #    self.causal_posterior = self.posterior.copy()
   #    for obs in self.observations:
   #       for causal_pair in self.causal_states[obs]: 
   #           state1 = obs * self.N_clones + causal_pair[0]
   #           state2 = obs * self.N_clones + causal_pair[1]
   #           self.causal_posterior[:, state1] = 2 * (self.posterior[:, state1].copy())# + self.posterior[:, state2].copy())
   #           self.causal_posterior[:, state2] = 0
   #    self.causal_posterior /= np.sum(self.causal_posterior.copy(),axis=1)[:, np.newaxis]
   
   
   
   
class Normal_Agent():
   
   def __init__(self, preferences, ECM_likelihood,
                N_clones, N_observations, N_actions, observations, actions, T_episodes,
                fake_zero):
      self.preferences = preferences
      self.ECM_likelihood = ECM_likelihood
      self.N_clones = N_clones
      self.N_observations = N_observations
      self.N_actions = N_actions
      self.observations = observations
      self.actions = actions
      self.T_episodes = T_episodes
      self.fake_zero = fake_zero
      self.N_states = N_clones * N_observations
      
   def ECM_to_probas(self, ECM, softmax=False, softmax_temperature=1):
      """
      Takes an ECM and outputs the corresponding probability distribution, 
      using either the sum or the softmax rule
      """
      
      if not softmax:
         distrib = ECM / np.sum(ECM, axis=1)[:, np.newaxis]
      
      else:
         distrib = np.exp(np.minimum(softmax_temperature * ECM, 1000))
         distrib /= np.sum(distrib, axis=1)[:, np.newaxis]
      
      return distrib
   
   def select_first_state(self, observation):
      """
      Selects the first state that is consistent with the observation it received.
      """
      obs_idx = np.where(self.observations == observation)[0][0]
      #obs_idx = self.observations.index(observation)
      self.previous_belief_state = obs_idx * self.N_clones + np.random.choice(np.arange(self.N_clones))
      return self.previous_belief_state
   
   
   def select_first_action(self):
      """
      Selects the first action, given the state it sampled
      """
      
      self.previous_action = rand_choice_nb(self.actions, self.policy[self.previous_belief_state,:])
      # self.previous_action = np.random.choice(self.actions, 
      #                                         p=self.policy[self.previous_belief_state,:])
      return self.previous_action

   

   def calculate_preferences(self, last_state, action, 
                             preference_target=False, 
                             discount=0.8, max_filter=False, softmax_scale = 1,
                             preferred_obs=3, proba_preferred_obs=0.75, 
                             horizon=1, gamma=0.8, 
                             softmax=False):
   
      """Calculates the array corresponding to the preferences distribution, either for passive learning,
      or given a target state. 
      """
      
      
      input_percept = last_state * self.N_actions + action
      
      if not preference_target:
         
         gen_model = np.zeros((self.N_states, self.N_states * self.N_observations))
         for state in range(self.N_states):
            input_percept = state * self.N_actions + action
            gen_model_state = self.posterior[input_percept,:][:, np.newaxis].copy() * self.likelihood.copy()
            gen_model_state = gen_model_state.copy().reshape((1, self.N_states * self.N_observations))
            gen_model[state,:] = gen_model_state.copy()
            gen_model[state,:] = gen_model_state.copy() / np.sum(gen_model_state.copy())
         self.full_preferences_obs = gen_model.copy()
         self.preferences = self.full_preferences_obs[last_state,:].copy()
      
      else:
         
         v_prev = np.ones(self.N_observations) * (1-proba_preferred_obs) / (self.N_observations-1)
         v_prev[preferred_obs] = proba_preferred_obs
         v_prev /= np.sum(v_prev)
         pref_obs = v_prev.copy()
         v_prev = np.repeat(v_prev.copy(), self.N_clones)
         
         
         
         # Calculate the reachability: how easy is it to reach the next states?
         self.reachability = self.posterior[0::self.N_actions].copy() * self.policy[:, 0].copy()
         for a in range(1, self.N_actions):
            self.reachability += self.posterior[a::self.N_actions].copy() * self.policy[:, a].copy()
         self.reachability /= np.sum(self.reachability, axis=1)[:, np.newaxis]
         
         t=1
         while t < horizon:
            v_t = self.reachability * v_prev.reshape((1,self.N_states))
            v_t_max = np.max(v_t, axis=1)
            # v_t_max = np.zeros(self.N_states)
            # for b in range(self.N_states):
            #    v_t_max[b] = np.max(v_t[b,:])
               
            v_prev = np.maximum(v_prev, discount**(t-1) * v_t_max)
            t+=1
         
         # Determine the value of a state conditioned on the previous state:
         children_thresholds = np.mean(self.reachability, axis=1).reshape((self.N_states, 1))
         children_boolean = self.reachability > np.repeat(children_thresholds, self.N_states, axis=1)
         V_cond = children_boolean * v_prev.reshape((1, self.N_states))
         
         if max_filter==True:
             max_pref = np.argmax(V_cond.copy(), axis=1)
             self.full_preferences = np.ones(V_cond.shape) * self.fake_zero
             self.full_preferences[np.arange(self.N_states), max_pref] = 1 - self.fake_zero * (self.N_states-1)
         else:
             self.full_preferences = self.ECM_to_probas(V_cond, softmax=softmax, softmax_temperature = softmax_scale)
         self.full_preferences_obs = np.kron(self.full_preferences.copy(), pref_obs)#np.ones(self.N_observations)/self.N_observations)         
      
         
         self.preferences = self.full_preferences_obs[last_state, :]
         
         return self.preferences

   def calculate_expected_FE(self, last_state, preference_target=False, proba_preferred_obs=3/4,
                             big_constant=50.0, horizon=1, preferred_obs=3):
      """
      Calculates the expected Free Energy, given the current state the agent believes it is in,
      for each possible action

      Returns
      -------
      EFE_array : ndarray
         DESCRIPTION. array of the values of the EFE for each action

      """
      
      EFE_array = np.zeros((self.N_actions))

      for action in range(self.N_actions):
         input_percept = last_state * self.N_actions + action
         gen_model = self.posterior[input_percept,:][:, np.newaxis] * self.likelihood
         gen_model = gen_model.reshape(self.N_states * self.N_observations)

         if not preference_target:
            self.preferences = np.zeros(self.N_states * self.N_observations)
            for act in range(self.N_actions):
               percept = last_state * self.N_actions + act
               gen_marg = self.posterior[percept,:][:, np.newaxis] * self.likelihood
               gen_marg = gen_marg.reshape(self.N_states * self.N_observations)
               gen_marg *= self.policy[last_state, act]
               self.preferences += gen_marg
         else:
            self.preferences = self.full_preferences_obs[last_state,:].copy()

            
         non_zeros = gen_model > 10*self.fake_zero
         pref_non_zeros = self.preferences > 10*self.fake_zero
            
         EFE = np.log(np.repeat(self.posterior[input_percept, :], self.N_observations), out=np.zeros_like(gen_model) * (big_constant),
                       where=non_zeros) \
               - np.log(self.preferences, out=np.ones_like(self.preferences)*(-big_constant),
                        where = pref_non_zeros)
         EFE = gen_model * EFE * (non_zeros)
   
         EFE_array[action] = np.sum(EFE)
         
      return EFE_array

   
   def EFE_filter(self, preference_target, preferred_obs=3,
                  proba_preferred_obs=3/4, horizon=1, gamma=0.8,
                  give_reward=True, EFE_to_weights=True, curiosity_param = -10):
      """
      Makes the EFE_matrix to renormalize the baseline h-values with the expected free energy

      Parameters
      -----------
      preference_target: Boolean
         DESCRIPTION. whether or not the agent is given a goal

      Returns
      -------
      self.EFE_contributions

      """
      
      # previous_EFEs = self.EFE_contributions.copy()
      self.EFE_contributions = np.zeros((self.N_states, self.N_actions), dtype=np.float64)
      for state in range(self.N_states):
         EFE_array = self.calculate_expected_FE(state, 
                                                preference_target, 
                                                proba_preferred_obs, 
                                                40.0, 
                                                horizon,
                                                preferred_obs)
                                              
         self.EFE_contributions[state,:] = EFE_array.copy()
         
      if EFE_to_weights:
         EFE_minima = np.zeros((self.N_states), dtype=np.float64)
         for j in range(self.N_states):
            EFE_minima[j] = np.min(self.EFE_contributions[j,:])
         self.EFE_weights = self.EFE_contributions.copy() - EFE_minima[:, np.newaxis]
         self.EFE_weights = self.ECM_to_probas(self.EFE_weights,False, 1)
      else:
         self.EFE_weights = self.EFE_contributions.copy()
      
      return self.EFE_contributions
   
   def deliberate_next_action(self): #, R=5):
      """
      Samples the next action from the policy, given the previous belief state

      Returns
      -------
      self.previous_action: int.
         DESCRIPTION. Index of the action the agent will take in the next step
      """
      
      self.previous_action = np.random.choice(self.actions, p=self.policy[self.belief_state,:])
      # self.previous_action = rand_choice_nb(self.actions, self.policy[self.belief_state,:])
      
      # #### Reflect and try to sample another action if failed in the previous round
      # r=0
      # while self.ECM_emotions_policy[self.belief_state, self.previous_action] == 1 and r<= R:
      #    self.previous_action = rand_choice_nb(self.actions, self.policy[self.belief_state,:])
      #    r+=1
      
      return self.previous_action

   def deliberate_next_state(self #, emotional_thinking, reflection_time=5, consistency_given=False
                             ):
      """
      Samples the belief about the next state in the environment from the prior distrib

      Parameters
      -------
      obs_environment : int.
         DESCRIPTION index of the observation the agent received from the environment
      
      Returns
      -------
      belief_state : int.
         DESCRIPTION. Belief about the next state in the environment
      """
      
      self.input_percept = self.calculate_input_percept()
      # if not emotional_thinking:
      posterior_cdt = self.posterior[self.input_percept,:].copy()/np.sum(self.posterior[self.input_percept,:].copy())
      # elif emotional_thinking:
      #    posterior_cdt = self.emotional_posterior[self.input_percept,:].copy()/np.sum(self.emotional_posterior[self.input_percept,:].copy())
      
      #self.belief_state = rand_choice_nb(np.arange(self.N_states), posterior_cdt)
      self.belief_state = np.random.choice(list(range(self.N_states)), p=posterior_cdt)
      
      # R=0
      # while self.ECM_emotions_posterior[self.input_percept, self.belief_state] == 1 and R < reflection_time:
      #    # self.belief_state = np.random.choice(list(range(self.N_states)), 
      #    #                                       p=self.posterior[self.input_percept,:]/np.sum(self.posterior[self.input_percept,:]))
      #    if not emotional_thinking:
      #       self.belief_state = rand_choice_nb(np.arange(self.N_states),
      #                                          self.posterior[self.input_percept,:]/np.sum(self.posterior[self.input_percept,:]) )
      #    elif emotional_thinking:
      #       self.belief_state = rand_choice_nb(np.arange(self.N_states),
      #                                          self.emotional_posterior[self.input_percept,:]/np.sum(self.emotional_posterior[self.input_percept,:]) )
      #    R += 1
      #    #if R == 1:
      #       #print("reflection")
         
      return self.belief_state

   def deliberate_observation(self):
      

      """
      Samples an observation based on the guess on the internal state

      Returns
      -------
      obs_agent : integer
         DESCRIPTION. describes the index of the observation it predicts

      """
      self.obs_agent = np.random.choice(self.observations, p=self.likelihood[self.belief_state,:])
      
      # self.obs_agent = rand_choice_nb(self.observations, self.likelihood[self.belief_state,:])
      return self.obs_agent
       
   def calculate_input_percept(self):
      
      self.input_percept = self.previous_belief_state * self.N_actions + self.previous_action
      
      return self.input_percept

   def test_DR_superposition(self, environment, causal_model=False,
                             N_trials = 100, trial_length = 50, target_obs = 1):
      """
      Simulates the trained agent on a task based on the WM it has a learnt and collects the success rates      
      Return the sucess rates averaged over a number of trials
      
      Parameters
      ----------
      environment : object.
         DESCRIPTION. Object that defines the environment and provides the agent 
         with the observations relative to its WM. All parameters are already provided
         
      N_trials: int.
         DESCRIPTION. Number of attempts to get the target
         
      trial_length: int.
         DESCRIPTION. Length of each trial, or number of interactions with the environment for each trial
         
      target_obs: int.
         DESCRIPTION. target observation in the environment (delayed rewards: light off/stomach full)
      
      Returns
      -------
      success_rate: float.
         DESCRIPTION. Frequency at which the agent reaches the target amongst N_trials attempts
      
      avg_success_distance: float.
         DESCRIPTION. Average duration between two successful attempts

      """
      
      self.success_rate = np.zeros(N_trials)
      self.success_distance = np.zeros(N_trials)
      self.failed_predictions = np.zeros(N_trials)
      for trial in range(N_trials):
         #print(trial)
         # Initialize the agent in the environment
         environment.observation = environment.initial_conditions()
      
         # Initial deliberation
         if causal_model:
            collapsed_states = [pair[1] for pair in self.causal_states[environment.observation]]
            self.plausible_previous_belief_states = [environment.observation * self.N_clones + i for i in range(self.N_clones) if i not in collapsed_states]   
         else:
            self.plausible_previous_belief_states = list(range(environment.observation * self.N_clones, 
                                                           environment.observation * self.N_clones + self.N_clones))
         
         self.plausible_previous_actions = [self.deliberate_next_action() 
                                               for self.belief_state in self.plausible_previous_belief_states]                                         
         
         self.previous_action = Counter(self.plausible_previous_actions).most_common(1)[0][0]
         
         for step in range(1, trial_length):

            # Apply action in env
            environment.env_state = environment.state_transition(self.previous_action)
            environment.observation = environment.give_observation(self.previous_action)
               
            # Based on the observation, deliberate:
            self.plausible_belief_states = [self.deliberate_next_state(# emotional_thinking=False, reflection_time=0
                                                                       ) 
                                               for self.previous_belief_state in self.plausible_previous_belief_states]
            
            self.plausible_observations = [self.deliberate_observation() for self.belief_state 
                                               in self.plausible_belief_states]
            
            self.plausible_previous_belief_states = [self.plausible_belief_states[i] 
                                                        for i in range(len(self.plausible_belief_states))
                                                        if self.plausible_observations[i]==environment.observation]
            
            if len(self.plausible_previous_belief_states) == 0:
               self.plausible_previous_belief_states = list(range(environment.observation * self.N_clones,
                                                                  environment.observation * self.N_clones + self.N_clones))
            
            self.plausible_previous_actions = [self.deliberate_next_action() 
                                                  for self.belief_state in self.plausible_previous_belief_states] 
            
            self.previous_action = Counter(self.plausible_previous_actions).most_common(1)[0][0]
            
            if environment.observation == target_obs:
               self.success_rate[trial] = 1
               self.success_distance[trial] = step
               break
               
            if step == trial_length - 1:
               self.success_distance[trial] = step
      
      self.success_rate = np.mean(self.success_rate)
      self.success_distance = np.mean(self.success_distance)
      
      return self.success_rate, self.success_distance

   def test_PGW_superposition(self, environment, N_trials = 500, trial_length = 20, 
             target_obs=3, target_coordinates =(2,2),
             causal_model=False,
             emotional_thinking=False,
             periodic_boundaries = False):
   
   
      self.time_to_target = np.zeros(environment.dim ** 2)
   
      
      for initial_state in environment.grid_indices:
         for test in range(N_trials):
            environment.agent_position = initial_state + 0
            environment.observation = environment.give_observation()
            
            if environment.observation == target_obs:
               break
               
            # Initial deliberation
            if causal_model:
               collapsed_states = [pair[1] for pair in self.causal_states[environment.observation]]
               self.plausible_previous_belief_states = [environment.observation * self.N_clones + i for i in range(self.N_clones) if i not in collapsed_states]   
            else:
               self.plausible_previous_belief_states = list(range(environment.observation * self.N_clones, 
                                                              environment.observation * self.N_clones + self.N_clones))
            
            self.plausible_previous_actions = [self.deliberate_next_action() for self.belief_state in self.plausible_previous_belief_states]                                         
            self.previous_action = Counter(self.plausible_previous_actions).most_common(1)[0][0]
            
            for step in range(1, trial_length):
               
               
               # Apply action in environment
               environment.agent_position = environment.move_the_agent(self.previous_action, periodic_boundaries=periodic_boundaries)
               environment.observation = environment.give_observation()
   
               # Based on the observation, deliberate:
               self.plausible_belief_states = [self.deliberate_next_state(# emotional_thinking=False, reflection_time=0
                                                                          ) 
                                                  for self.previous_belief_state 
                                                  in self.plausible_previous_belief_states]
               
               self.plausible_observations = [self.deliberate_observation() 
                                                 for self.belief_state 
                                                 in self.plausible_belief_states]
               
               self.plausible_previous_belief_states = [self.plausible_belief_states[i] 
                                                           for i in range(len(self.plausible_belief_states)) 
                                                           if self.plausible_observations[i] == environment.observation]
                              
               if environment.observation == target_obs:
                  self.time_to_target[initial_state] += step / N_trials
                  break
                     
               if len(self.plausible_previous_belief_states) == 0:
                  # print("failed to predict")
                  if causal_model:
                     collapsed_states = [pair[1] for pair in self.causal_states[environment.observation]]
                     self.plausible_previous_belief_states = [environment.observation * self.N_clones + i 
                                                                 for i in range(self.N_clones) 
                                                                 if i not in collapsed_states]
                  else:
                     self.plausible_previous_belief_states = list(range(environment.observation * self.N_clones, 
                                                                    environment.observation * self.N_clones + self.N_clones))
   
   
               
               if environment.observation != target_obs and step == trial_length-1:
                  self.time_to_target[initial_state] += trial_length / N_trials
               
               self.plausible_previous_actions = [self.deliberate_next_action() 
                                                     for self.belief_state   
                                                     in self.plausible_previous_belief_states]                                         
               
               self.previous_action = Counter(self.plausible_previous_actions).most_common(1)[0][0]
   
      return self.time_to_target
   
   
   
class Plots:
   def __init__(self, N_episodes, N_agents, N_clones, N_observations, N_actions,
                actions, observations, environment_dimension,
                fontsize_titles, fontsize_labels, fontsize_annot,
                window_size):
      
      # Data relevant to dimensions of training
      self.N_episodes = N_episodes
      self.N_agents = N_agents
      self.N_clones = N_clones
      self.N_observations = N_observations
      self.environment_dimension = environment_dimension
      self.N_actions = N_actions
      self.actions = actions
      self.observations = observations
      
      # Aesthetic of the plots
      self.fontsize_titles = fontsize_titles
      self.fontsize_labels = fontsize_labels
      self.fontsize_annot = fontsize_annot
      self.window_size = window_size

         
   def moving_average(self, data):
       # Define the kernel for the moving window
       kernel = np.ones(self.window_size) / self.window_size
       
       # Convolve the data array with the kernel
       moving_avg = np.convolve(data, kernel, mode='valid')
       first_points = np.zeros((self.window_size-1))
       for t in range(self.window_size-1):
           first_points[t] = np.sum(data[:t+1])/(t+1)
       
       return np.concatenate((first_points, moving_avg))    

   def plot_energies_avg_evol(self, Free_energies, Expected_Free_energies, 
                              transparence=0.05, size=(10,7), std=False):
      
      if std:
         FE_mean = self.moving_average(np.mean(Free_energies, axis=0))
         FE_std = self.moving_average(np.std(Free_energies, axis=0))
   
         EFE_mean = self.moving_average(np.mean(Expected_Free_energies, axis=0))
         EFE_std = self.moving_average(np.std(Expected_Free_energies, axis=0))
   
         sns.set_theme(context="paper", style="whitegrid")
         fig_FE, ax_FE = plt.subplots(2, figsize=size, dpi=300)
         ax_FE[1].errorbar(list(range(self.N_episodes)), EFE_mean, yerr=EFE_std,
                           color="purple", fmt="+", alpha=transparence)
         ax_FE[1].plot(list(range(self.N_episodes)), EFE_mean,
                           color="purple", label="EFE")
         ax_FE[0].errorbar(list(range(self.N_episodes)), FE_mean, yerr=FE_std,
                           color="gold", fmt="+", alpha=transparence)
         ax_FE[0].plot(list(range(self.N_episodes)), FE_mean, 
                           color="gold",label="FE")


      else:
         sns.set_theme(context="paper", style="whitegrid")
         fig_FE, ax_FE = plt.subplots(2, figsize=size, dpi=300)
         colors_FE = sns.color_palette("crest", self.N_agents)
         colors_EFE = sns.color_palette("flare", self.N_agents)
         
         for actor in range(self.N_agents):
            
            ax_FE[0].plot(list(range(self.N_episodes)), 
                          self.moving_average(Free_energies[actor,:]), 
                          color=colors_FE[self.N_agents - 1 - actor],
                          alpha=0.5,
                          linewidth=1, 
                          label="agent " + str(actor + 1))
            
            ax_FE[1].plot(list(range(self.N_episodes)), 
                          self.moving_average(Expected_Free_energies[actor,:]), 
                          color=colors_EFE[self.N_agents - 1 - actor],
                          alpha=0.5,
                          linewidth=1,
                          label="agent " + str(actor + 1))
            
      # ax_FE[0].legend(ncols=2, fontsize=14)
      # ax_FE[1].legend(ncols=2, fontsize=14)
      ax_FE[0].set_ylabel("Free Energy", fontsize=self.fontsize_labels)
      ax_FE[1].set_ylabel("Expected Free Energy", fontsize=self.fontsize_labels)
      ax_FE[1].set_xlabel("episode", fontsize=self.fontsize_labels)
      
      ax_FE[0].tick_params(axis='both', labelsize=self.fontsize_labels)
      ax_FE[1].tick_params(axis="both", labelsize=self.fontsize_labels)
      
      
      return fig_FE
   
   def make_labels_GW(self,layer):
      
      signs_actions = ['right', 'left', 'up', 'down']
      if layer==1:   # Input layer of the world model
         labels_states = []
         #signs_actions = [r'\rightarrow', r'\leftarrow', r'\uparrow', r'\leftarrow']
         signs_actions = ['right', 'left', 'up', 'down']
         for obs, clone, action in itertools.product(self.observations, 
                                                     list(range(self.N_clones)), 
                                                     self.actions):
            label = 'c={}; a='.format(clone) + signs_actions[action]
            labels_states.append(label)
            
      if layer == 2:   # Latent layer of the world model
         labels_states = []
         for obs, clone in itertools.product(self.observations, list(range(self.N_clones))):
            label = 'c={}'.format(clone)
            labels_states.append(label)
      
      if layer == 3:   # Output layer of the world model
         labels_states = self.observations
      
      if layer == 4:   # Actions / Output layer of the behavior model
         labels_states = ['right', 'left', 'up', 'down']
         
      return labels_states
   
   def change_color_observations_ticks(self, g, x_or_y, layer):
      """
      Change the color of the ticks per observation to avoid visual clutter
      Parameters
      ----------
      g : TYPE object
         DESCRIPTION. Plot to change the color on
      x_or_y : Type. string
      layer : TYPE which axis needs recoloring
         DESCRIPTION.

      Returns
      -------
      None.

      """
      
      colors = cm.Reds(np.linspace(0, 1, self.N_observations+2))
      if layer == 2:
         k=1
         if x_or_y == 'y':
            for i, ticklabel in enumerate(g.yaxis.get_majorticklabels()):
                if i % (self.N_clones) == 0:
                    k += 1
                ticklabel.set_color(colors[k])
         elif x_or_y == "x":
            for i, ticklabel in enumerate(g.xaxis.get_majorticklabels()):
                if i % (self.N_clones) == 0:
                    k += 1
                ticklabel.set_color(colors[k])
                
      if layer == 1:
         j=1
         if x_or_y == 'y':
            for i, ticklabel in enumerate(g.yaxis.get_majorticklabels()):
                if i % (self.N_actions * self.N_clones) == 0:
                    j += 1
                ticklabel.set_color(colors[j])
                
         elif x_or_y == "x":
            for i, ticklabel in enumerate(g.xaxis.get_majorticklabels()):
                if i % (self.N_actions * self.N_clones) == 0:
                    j += 1
                ticklabel.set_color(colors[j])
                
      return g
            
   
   def plot_posterior(self, posterior):
      
      labels_states = self.make_labels_GW(layer=2)
      labels_states_actions = self.make_labels_GW(layer=1)
 
      # checkout the posteriors
      fig_posterior, ax_posterior = plt.subplots(1, figsize=(15,10))
      sns.set(font_scale=1.4)
      g = sns.heatmap(posterior.transpose(), ax=ax_posterior,
                  xticklabels=labels_states_actions,
                  yticklabels=labels_states,
                  cmap="crest",
                  linewidths=0.01, linecolor="black")
      
      g = self.change_color_observations_ticks(g, 'x', 1)
      g = self.change_color_observations_ticks(g, 'y', 2)
                
      ax_posterior.invert_yaxis()
      ax_posterior.set_xlabel("input states", fontsize=self.fontsize_labels)
      ax_posterior.set_ylabel("new belief states", 
                              fontsize=self.fontsize_labels)
      ax_posterior.set_yticklabels(labels_states, rotation=90)
      ax_posterior.set_xticklabels(labels_states_actions, rotation=45)
      
      return fig_posterior
   
   def plot_posterior_DR(self, posterior):
      labels_states = ['00', '01', '10']   # (Light on/off, empty/full)
      labels_states_clones = [item for item in labels_states for _ in range(self.N_clones)]
      labels_states_actions_clones = ["00, wait", "00, press", "00, wait", "00, press",
                               "01, wait", "01, press","01, wait", "01, press",
                               "10, wait", "10, press","10, wait", "10, press"]
 
      # checkout the posteriors
      fig_posterior, ax_posterior = plt.subplots(1, figsize=(15,10))
      sns.set(font_scale=1.4)
      g = sns.heatmap(posterior.transpose(), ax=ax_posterior,
                      xticklabels=labels_states_actions_clones,
                      yticklabels=labels_states_clones,
                      cmap="crest",
                      linewidths=0.01, linecolor="black")
          
      ax_posterior.invert_yaxis()
      ax_posterior.set_xlabel("input states", fontsize=self.fontsize_labels)
      ax_posterior.set_ylabel("new belief states", 
                              fontsize=self.fontsize_labels)
      ax_posterior.set_yticklabels(labels_states_clones, rotation=90)
      ax_posterior.set_xticklabels(labels_states_actions_clones, rotation=45)
      
      return fig_posterior
   
   def plot_policy_DR(self, policy):
      
      fig_policy, ax_policy = plt.subplots(1, figsize=(10,7))
      
      labels_states = ["OFF, hungry","OFF, full","ON, hungry"]
      labels_states_clones = [item for item in labels_states for _ in range(self.N_clones)]
      labels_actions = ["wait", "press"]
      
      h = sns.heatmap(policy.transpose(), ax=ax_policy,
                      cmap="rocket_r",
                      vmin=0, vmax=1,
                      xticklabels=labels_states_clones, yticklabels=labels_actions,
                      linewidths=0.01, linecolor="black")
      ax_policy.invert_yaxis()
      ax_policy.set_xticklabels(labels_states_clones,
                                fontsize=self.fontsize_labels)
      ax_policy.set_yticklabels(labels_actions, 
                                fontsize=self.fontsize_labels, rotation = 90)
      
      ax_policy.set_xlabel("states", fontsize=self.fontsize_labels)
      ax_policy.set_ylabel("actions", fontsize=self.fontsize_labels)

      return fig_policy
   
   def plot_WorldModel_Policy_DR(self, posterior, policy, title="", diagram=False):
      
      fig_agent, ax_agent = plt.subplots(1,1,figsize=(24,9)) #18))
      ax_agent.set_axis_off()
      width_ratios = [2] * (3 - diagram * 1 + (not diagram) * 1)
      if not diagram:
         width_ratios[self.N_actions] = 1 /8   # axis for the colorbar 
         width_ratios[-1] = 2.4
      gs = GridSpec(1, 3 - diagram * 1 + (not diagram) * 1, figure=fig_agent, 
                    width_ratios = width_ratios) 
      colors = ['blue', 'green']
      
      ######## Plot the posterior
      labels_states = ['c1(A)', 'c2(A)',
                       'c1(B)','c2(B)',
                       'c1(C)','c2(C)']
      labels_actions = ['wait', 'press']
      
      if diagram:
         ax_posterior = fig_agent.add_subplot(gs[0,0])
         ax_posterior.set_axis_off()
         for a in range(self.N_actions):
            mc = MarkovChain(posterior[a::2,:], labels_states, 
                             node_facecolors = ['red','red','orange','orange','yellow','yellow'],
                             node_edgecolors = ['red','red','orange','orange','yellow','yellow'],
                             node_fontsize=14, node_radius=1,
                             action_spacing=a,
                             arrow_edgecolor=colors[a], arrow_facecolor=colors[a],
                             title="Transition function")
            mc.draw(fig_agent, ax_posterior)
      
      else:
         # checkout the posteriors
         sns.set(font_scale=2)
         ax_colorbar = fig_agent.add_subplot(gs[0, self.N_actions])
         for a in range(self.N_actions):
            ax_posterior = fig_agent.add_subplot(gs[0,a])
            sns.heatmap(posterior[a::self.N_actions].transpose(), ax=ax_posterior,
                            vmin=0, vmax=1,
                            xticklabels=labels_states,
                            cmap="cividis", cbar = (a==self.N_actions-1), cbar_ax = ax_colorbar,
                            linewidths=0.01, linecolor="black", square=True)
            ax_posterior.invert_yaxis()
            ax_posterior.set_xlabel("input belief states", fontsize=28)
            
            if a == 0:
               ax_posterior.set_ylabel("new belief states", 
                                       fontsize=28)
               ax_posterior.set_yticklabels(labels_states, rotation=0)
            else:
               ax_posterior.set_yticklabels([" ", " ", " ", " ", " ", " " ])
            ax_posterior.set_xticklabels(labels_states, rotation=0)
            ax_posterior.set_title("World Model: " + labels_actions[a] )
            ax_colorbar.yaxis.set_ticks_position('left')
      
      ######## Plot the policy
      labels_actions = ["wait", "press"]
      ax_policy = fig_agent.add_subplot(gs[0, 2 - diagram * 1 + (not diagram) * 1])
      sns.set(font_scale=2)
      sns.heatmap(policy.transpose(), ax=ax_policy,
                      cmap="rocket_r",
                      vmin=0, vmax=1,
                      xticklabels=labels_states, yticklabels=labels_actions,
                      linewidths=0.01, linecolor="black",
                      square=False)
      ax_policy.invert_yaxis()
      ax_policy.set_xticklabels(labels_states, rotation=0)
      ax_policy.set_yticklabels(labels_actions, 
                                rotation = 90)
      
      ax_policy.set_xlabel("belief states", fontsize=28)
      ax_policy.set_ylabel("actions", fontsize=28)
      ax_policy.set_title("Policy")
      
      fig_agent.suptitle(title)
      fig_agent.tight_layout()
      return fig_agent
   
   def plot_models_square_DR(self, posterior, policy, preferences, title=""):
      
      fig_agent, ax_agent = plt.subplots(1,1,figsize=(20,18)) 
      ax_agent.set_axis_off()
      width_ratios = [2] * (self.N_actions + 1)
      width_ratios[self.N_actions] = 1 /8   # axis for the colorbar 
      gs = GridSpec(2,  (self.N_actions +1), figure=fig_agent, 
                    width_ratios = width_ratios) 
      colors = ['blue', 'green']
      labels_states = ['c1(A)', 'c2(A)',
                       'c1(B)','c2(B)',
                       'c1(C)','c2(C)']
      labels_actions = ['wait', 'press']
      
      
      cmap_WM = sns.cubehelix_palette(start=-.2,rot=-.5, dark=0.2, light=0.75, reverse=True, as_cmap=True)
      sns.set(font_scale=2.5)
      ax_colorbar_WM = fig_agent.add_subplot(gs[0, self.N_actions])
      for a in range(self.N_actions):
         ax_posterior = fig_agent.add_subplot(gs[0,a])
         sns.heatmap(posterior[a::self.N_actions].transpose(), ax=ax_posterior,
                         vmin=0, vmax=1,
                         xticklabels=labels_states,
                         cmap=cmap_WM,
                         cbar = (a==self.N_actions-1), 
                         cbar_ax = ax_colorbar_WM,
                         linewidths=0.01, linecolor="black", square=True)
         ax_posterior.invert_yaxis()
         
         if a == 0:
            ax_posterior.set_ylabel("next belief states", 
                                    fontsize=28)
            ax_posterior.set_yticklabels(labels_states, rotation=0)
         else:
            ax_posterior.set_yticklabels([" ", " ", " ", " ", " ", " " ])
         ax_posterior.set_xticklabels(labels_states, rotation=0)
         ax_posterior.set_title("World Model: " + labels_actions[a])
         ax_colorbar_WM.yaxis.set_ticks_position('left')
      
      # Plot the preferences
      cmap_actions = sns.cubehelix_palette(start=3, rot=0.45, dark=0.15, light=0.8, reverse=True, as_cmap=True)
      ax_preferences = fig_agent.add_subplot(gs[1, 0])
      ax_actions_cbar = fig_agent.add_subplot(gs[1,2])
      sns.set(font_scale=2.5)
      sns.heatmap(preferences.transpose(), ax=ax_preferences,
                  cmap=cmap_actions, 
                  vmin=0, vmax=1,
                  cbar=False,
                  xticklabels=labels_states,
                  yticklabels=labels_states,
                  linewidths=0.01, linecolor="black",
                  square=True)
      ax_preferences.invert_yaxis()
      ax_preferences.set_xticklabels(labels_states, rotation=0)
      ax_preferences.set_yticklabels(labels_states, rotation=0)
      ax_preferences.set_xlabel("current belief states", fontsize=28)
      ax_preferences.set_ylabel("future belief states", fontsize=28)
      ax_preferences.set_title("Preferences")
      
      # Plot the policy
      ax_policy = fig_agent.add_subplot(gs[1,1])
      sns.heatmap(policy.transpose(), ax=ax_policy,
                  cmap = cmap_actions,
                  cbar=True, 
                  cbar_ax = ax_actions_cbar,
                  vmin=0, vmax=1,
                  xticklabels=labels_states,
                  yticklabels=labels_actions,
                  linewidth=0.01, linecolor="k",
                  square=False)
      ax_actions_cbar.yaxis.set_ticks_position('left')
      ax_policy.set_xlabel("current belief states", fontsize=28)
      ax_policy.set_title("Policy")
      
      return fig_agent


   
   def plot_WorldModel_Policy_PGW(self, posterior, policy, title=""):
      fig_agent, ax_agent = plt.subplots(1,1,figsize=(21, 14))
      ax_agent.set_axis_off()
      gs = GridSpec(2, 3, figure=fig_agent)
      colors = ['purple','blue', 'green', 'grey']
      actions_strings = ['Right', 'Left', 'Up', 'Down']
      
      ######## Plot the posterior
      labels_states = ['c1(0)', 'c2(0)', 'c3(0)',
                       'c1(1)', 'c2(1)', 'c3(1)',
                       'c1(2)', 'c2(2)', 'c3(2)',
                       'c1(3)', 'c2(3)', 'c3(3)']
      
      obs_colors = ['silver', 'yellow', 'orange', 'red']
      node_colors = []
      for o, c in itertools.product(range(self.N_observations), range(self.N_clones)):
         node_colors.append(obs_colors[o])
         
      
      for a in range(self.N_actions):
         ax_posterior = fig_agent.add_subplot(gs[a%2,a//2])
         
         sns.heatmap(posterior[a::self.N_actions,:].transpose(), ax=ax_posterior,
                     vmin=0, vmax=1,
                     xticklabels = labels_states, yticklabels=labels_states,
                     cmap="cividis",
                     linewidths=0.01, linecolor="black")
         ax_posterior.invert_yaxis()
         ax_posterior.set_xlabel("input states", fontsize=self.fontsize_labels)
         ax_posterior.set_ylabel("new belief states", 
                                  fontsize=self.fontsize_labels)
         ax_posterior.set_title(actions_strings[a])
      
      
      
      labels_states = ['0', '1', '2', '3']   # (Light on/off, empty/full)
      labels_states_clones = [item for item in labels_states for _ in range(self.N_clones)]

      
      ######## Plot the policy
      labels_actions = ["right", "left", "up", "down"]
      ax_policy = fig_agent.add_subplot(gs[:, 2])
      sns.heatmap(policy.transpose(), ax=ax_policy,
                      cmap='YlOrRd',
                      vmin=0, vmax=1,
                      xticklabels=labels_states_clones, yticklabels=labels_actions,
                      linewidths=0.01, linecolor="black")   # "rocket_r"
      ax_policy.invert_yaxis()
      ax_policy.set_xticklabels(labels_states_clones,
                                fontsize=self.fontsize_labels)
      ax_policy.set_yticklabels(labels_actions, 
                                fontsize=self.fontsize_labels, rotation = 90)
      
      ax_policy.set_xlabel("states", fontsize=self.fontsize_labels)
      ax_policy.set_ylabel("actions", fontsize=self.fontsize_labels)
      ax_policy.set_title("Policy")
      
      fig_agent.suptitle(title)
      return fig_agent
  
   def plot_models_square_PGW(self, posterior, policy, preferences,
                              size=(30,25),vmax=1, title=""):
      
      fig_agent, ax_agent = plt.subplots(1,1,figsize=size, dpi=150) 
      ax_agent.set_axis_off()
      width_ratios = [2] * (2 + 1)
      width_ratios[2] = 1 /8   # axis for the colorbar 
      gs = GridSpec(3,  (2+1), figure=fig_agent, 
                    width_ratios = width_ratios) 
      colors = ['blue', 'green']
      labels_states = ['c1(0)', 'c2(0)', 'c3(0)',
                       'c1(1)', 'c2(1)', 'c3(1)',
                       'c1(2)', 'c2(2)', 'c3(2)',
                       'c1(3)', 'c2(3)', 'c3(3)']
      labels_actions = ['right', 'left', 'up', 'down']
      
      
      cmap_WM = sns.cubehelix_palette(start=-.2,rot=-.5, dark=0.2, light=0.75, reverse=True, as_cmap=True)
      sns.set(font_scale=2.7)
      ax_colorbar_WM = fig_agent.add_subplot(gs[0:2, 2])
      for a in range(self.N_actions):
         ax_posterior = fig_agent.add_subplot(gs[a%2,a//2])
         sns.heatmap(posterior[a::self.N_actions,:].transpose(), ax=ax_posterior,
                         vmin=0, vmax=vmax,
                         xticklabels=labels_states,
                         cmap=cmap_WM,
                         cbar = (a==self.N_actions-1), 
                         cbar_ax = ax_colorbar_WM,
                         linewidths=0.01, linecolor="black", 
                         square=False)
         ax_posterior.invert_yaxis()
         
         if a // 2 == 0:
            ax_posterior.set_ylabel("next belief states", 
                                    fontsize=28)
            ax_posterior.set_yticks(np.arange(self.N_observations*self.N_clones)+0.5)
            ax_posterior.set_yticklabels(labels_states, rotation=0)
         else:
            ax_posterior.set_yticks(np.arange(self.N_observations*self.N_clones)+0.5)
            ax_posterior.set_yticklabels([" ", " ", " ", " ", " ", " ",
                                          " ", " ", " ", " ", " ", " "])
         ax_posterior.set_xticklabels(labels_states, rotation=0)
         ax_posterior.set_title("World Model: " + labels_actions[a], fontsize=30)
         ax_colorbar_WM.yaxis.set_ticks_position('left')
      
      # Plot the preferences
      cmap_actions = sns.cubehelix_palette(start=3, rot=0.45, dark=0.15, light=0.8, reverse=True, as_cmap=True)
      ax_preferences = fig_agent.add_subplot(gs[2, 0])
      ax_actions_cbar = fig_agent.add_subplot(gs[2,2])
      sns.set(font_scale=2.5)
      sns.heatmap(preferences.transpose(), ax=ax_preferences,
                  cmap=cmap_actions, 
                  vmin=0, vmax=1,
                  cbar=False,
                  xticklabels=labels_states,
                  yticklabels=labels_states,
                  linewidths=0.01, linecolor="black",
                  square=False)
      ax_preferences.invert_yaxis()
      ax_preferences.set_xticklabels(labels_states, rotation=0)
      ax_preferences.set_yticklabels(labels_states, rotation=0)
      ax_preferences.set_xlabel("current belief states", fontsize=28)
      ax_preferences.set_ylabel("future belief states", fontsize=28)
      ax_preferences.set_title("Preferences", fontsize=30)
      
      # Plot the policy
      ax_policy = fig_agent.add_subplot(gs[2,1])
      sns.heatmap(policy.transpose(), ax=ax_policy,
                  cmap = cmap_actions,
                  cbar=True, 
                  cbar_ax = ax_actions_cbar,
                  vmin=0, vmax=1,
                  xticklabels=labels_states,
                  yticklabels=labels_actions,
                  linewidth=0.01, linecolor="k",
                  square=False)
      ax_actions_cbar.yaxis.set_ticks_position('left')
      ax_policy.set_xlabel("current belief states", fontsize=28)
      ax_policy.set_xticklabels(labels_states, rotation=0)
      ax_policy.set_title("Policy", fontsize=30)
      
      fig_agent.tight_layout()
      return fig_agent

  
   
   def plot_length_trajectories(self, lengths):
      fig_lengths, ax_lengths = plt.subplots(1, dpi=150)
      lengths = np.mean(lengths, axis=0)    # avg over the agents
      lengths = self.moving_average(lengths)   # running average over time
      ax_lengths.plot(range(self.N_episodes), lengths)
      ax_lengths.set_xlabel("episodes")
      ax_lengths.set_ylabel("consecutive right guesses")
      return fig_lengths
   

      
   def plot_policy(self, policy):
      
      fig_policy, ax_policy = plt.subplots(1, figsize=(10,7))
      
      labels_states = self.make_labels(layer=2)
      labels_actions = self.make_labels(layer=4)
      
      h = sns.heatmap(policy.transpose(), ax=ax_policy,
                      vmin=0, vmax=1,
                      xticklabels=labels_states, yticklabels=labels_actions,
                      linewidths=0.01, linecolor="black")
      ax_policy.invert_yaxis()
      ax_policy.set_xticklabels(labels_states,
                                fontsize=self.fontsize_labels)
      ax_policy.set_yticklabels(labels_actions, 
                                fontsize=self.fontsize_labels, rotation = 90)
      
      ax_policy.set_xlabel("states", fontsize=self.fontsize_labels)
      ax_policy.set_ylabel("actions", fontsize=self.fontsize_labels)
      
      h = self.change_color_observations_ticks(h, "x", 2)
            
      return fig_policy
   
   def plot_testing_DR(self, success_rates, success_distances, N_agents):
      fig_testing, ax_testing = plt.subplots(1,2, figsize=(15, 8))
      
      colors_rates = mcp.gen_color(cmap = "Blues", n=self.N_agents + 3) 
      colors_rates = colors_rates[-self.N_agents:]+ ["red"]
      ax_testing[0].bar(range(N_agents), success_rates)
      ax_testing[0].set_ylabel("Success rate")
      ax_testing[0].set_xlabel("Agents")

      
      ax_testing[1].bar(range(N_agents), success_distances)
      ax_testing[1].set_xlabel("Agents")
      ax_testing[1].set_ylabel("Success distances")
      for actor in range(N_agents):
         ax_testing[0].get_children()[actor].set_color(colors_rates[actor])
         ax_testing[1].get_children()[actor].set_color(colors_rates[actor])
         
      
      fig_testing.suptitle("Testing results")
      fig_testing.tight_layout()
      return 
   
   def plot_testing_PGW(self, agents, targets, size=(7,5), 
                        avg_only=False, min_max=False, subtract_random=False, ylim=20):
      
      N_targets = len(targets)
      dim = self.environment_dimension
      #### Gather the data in a single array
      performances = np.zeros((N_targets, self.N_agents, dim**2))
      for actor in range(self.N_agents):
         for target_idx in range(N_targets):
             performances[target_idx, actor,:] = agents[actor].performances[target_idx]
      
      
      
      # random agent
      random_performances = pd.DataFrame({"Time to target" : np.array(agents[-1].performances).reshape((N_targets * dim ** 2)),
                                          "Target" : np.repeat(np.array(targets), dim ** 2),
                                          "Initial position" : np.tile(np.arange(dim**2), N_targets)})

      #### Prepare a data frame
      df_performances = pd.DataFrame({"Time to target": performances.reshape((N_targets * self.N_agents * dim**2)),
                                      "Target": np.repeat(np.array(targets), self.N_agents * dim**2),
                                      "Agent": np.tile(np.repeat(np.arange(self.N_agents), dim ** 2), N_targets),
                                      "Initial position": np.tile(np.arange(dim**2), self.N_agents * N_targets)})
      df_performances['Dataset'] = "trained"
      random_performances['Dataset'] = "random"
      
      df_full_performances = pd.concat([df_performances, random_performances])
      df_full_performances["Scenario"] = df_full_performances.apply(lambda x: f"{x['Target']} {x['Dataset']}", axis=1)
      hue_order = ["0 trained", "0 random", "3 trained", "3 random"]
      
      #### Plot the performances
      colors = sns.color_palette('RdBu_r', as_cmap=True)
      palette = [colors(10), colors(0.3), colors(220), colors(170)]
      
      sns.set_theme(context="paper", style="whitegrid", font_scale = 3)
      fig, ax = plt.subplots()
      sns.catplot(data=df_full_performances, x="Initial position", y="Time to target", hue="Scenario",
                  hue_order = hue_order,
                  kind="bar", errorbar=("pi", 0), ax=ax,
                  palette = palette,
                  height=10, aspect=2, legend_out=False)
      
             
      plt.show()
            
      # fig_testing, ax_testing = plt.subplots(1, dpi=150, figsize=size)
      # ax_testing.grid(visible=True, which="both", axis='y')
      
      # width = 0.1 + (avg_only or min_max) * 0.2 # the width of the bars
      # multiplier = 0
      # N_states = self.environment_dimension ** 2
      # colors_FE = sns.color_palette("crest_r", self.N_agents)
      # colors_EFE = sns.color_palette("flare_r", self.N_agents)
      # if min_max:
      #    times = []
      #    for actor in range(len(agents)):
      #       times.append(agents[actor].time_to_target)
      #    times = np.array(times)
      #    min_array = np.min(times, axis=0)
      #    max_array = np.max(times, axis=0)

      # for actor in range(self.N_agents + 3):
      #    # Individual agents
      #     if actor < self.N_agents and not avg_only:
      #         label = "agent " + str(actor)
      #         offset = width * (multiplier)
   
      #         rects = ax_testing.bar(np.arange(N_states) + offset ,
      #                                agents[actor].time_to_target - subtract_random * agents[-1].time_to_target, 
      #                         width, label=label,
      #                         color=colors_EFE[actor])
      #         multiplier += 1
   
          
      #     # elif actor == self.N_agents + 2 and not subtract_random:
      #     #     label = 'random'
      #     #     offset = width * multiplier
      #     #     rects = ax_testing.bar(np.arange(N_states) + offset,
      #     #                            agents[actor-1].time_to_target.copy()  - subtract_random * agents[-1].time_to_target, 
      #     #                    width, label=label,
      #     #                    color="darkcyan")
      #     #     multiplier += 1      

      #     # Average agent
      #     elif actor == self.N_agents:
      #        avg_times = np.zeros(self.environment_dimension ** 2)
      #        for actor_2 in range(self.N_agents):
      #          avg_times += (agents[actor_2].time_to_target.copy() - subtract_random * agents[-1].time_to_target) / self.N_agents

      #        label = "average agent"
      #        offset = 0 * min_max + (not min_max) * (width * (avg_only) + (not avg_only) * width * multiplier)
      #        rects = ax_testing.bar(np.arange(N_states) + offset, avg_times, 
      #                      width, label=label,
      #                      color="gold", alpha=1)
      #        if min_max:
      #           offset_min =  width 
      #           rects_min = ax_testing.bar(np.arange(N_states) + offset_min , min_array, 
      #                         width, label="minimum",
      #                         color=colors_EFE[0], alpha=1)
      #           offset_max = 2 * width
      #           rects_max = ax_testing.bar(np.arange(N_states) + offset_max , max_array, 
      #                         width, label="maximum",
      #                         color=colors_EFE[len(colors_EFE) // 2], alpha=1)
             
      #        multiplier += 1
             
      #        if subtract_random:
      #           ax_testing.plot(np.arange(N_states + 1), np.mean(avg_times) * np.ones(N_states + 1),
      #                           color="gold")
          
      
      #     # Add some text for labels, title and custom x-axis tick labels, etc.
      #     ax_testing.set_ylabel('time to target', fontsize=14)
      #     ax_testing.set_xlabel("initial positions", fontsize=14)
      #     ax_testing.set_xlim(left=-0.1, right=self.environment_dimension**2)
      #     ax_testing.set_xticks(np.arange(N_states) , np.arange(N_states), fontsize=12)
      #     ax_testing.set_ylim(top=ylim)
      #     ax_testing.set_yscale('linear')
      #     ax_testing.legend(loc='upper right', ncols=4,fontsize=8)
      #     ax_testing.tick_params(axis='both', labelsize=12)
          
   def plot_generalization_PGW(self, performances, size=(7,5), number_of_tasks=2,
                               task_labels=[],
                               avg_only=False, compare_random=False):
      
      sns.set_theme(context="paper", style="darkgrid")
      fig_testing, ax_testing = plt.subplots(1, dpi=150, figsize=size)
      ax_testing.grid()
      
      width = 0.4 # the width of the bars
      multiplier = 0
      N_states = self.environment_dimension ** 2
      colors_FE = sns.color_palette("crest_r", self.N_agents)
      colors_EFE = sns.color_palette("flare_r", self.N_agents)
      colors = ["gold", "darkcyan"]
      for task in range(number_of_tasks):
         avg_times = np.zeros(self.environment_dimension ** 2)
         for actor in range(self.N_agents):
           avg_times += (performances[actor,task,:] - compare_random * performances[-1,task,:]) / self.N_agents

         label = "average agent"
         offset = width * task
         
         rects = ax_testing.bar(np.arange(N_states) + offset, avg_times, 
                                 width, label=task_labels[task],
                                 color=colors[task], alpha=1)

                  
         if compare_random:
            ax_testing.plot(np.arange(N_states + 1), np.mean(avg_times) * np.ones(N_states + 1),linestyle=":",
                            color=colors[task])
            
  
      # Add some text for labels, title and custom x-axis tick labels, etc.
      ax_testing.grid()
      ax_testing.set_ylabel('time to target', fontsize=14)
      ax_testing.set_xlabel("initial positions", fontsize=14)
      ax_testing.set_xlim(left=-1, right=36)
      ax_testing.set_xticks(np.arange(N_states) , np.arange(N_states), fontsize=10)
      # ax_testing.set_yticks(np.arange(20), yticks, fontsize=12)
      ax_testing.legend(loc='upper right', ncols=4,fontsize=8)
      ax_testing.tick_params(axis='both', labelsize=12)
