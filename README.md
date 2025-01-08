<h1 align="center">
Free Energy Projective Simulation (FEPS)
</h1>
<h4 align="center">
Active Inference with interpretability
</h4>

This repository gathers all the code and data files that were used to produce the figures in the paper ["Free Energy Projective Simulation: Active Inference with Interpretability"] by J. Pazem, M. Krumm, A. Vining, L.J. Fiderer and H.J. Briegel. The preprint can be found at the following URL: (https://arxiv.org/abs/2411.14991).

# How to use these files?
### Create and test the FEPS agents
* The FEPS agents were defined as classes: all functions necessary to the training and testing of the agents are in the file ["GW_FEPS_functions_parallel.py"]. Basic plotting of the result is also included in this file.
* The agents were trained and tested using numba and in parallel using the files ["Skinner_Box_parallel.py"] and ["GW_parallel.py"] for the delayed reward environment and navigation task, respectively.
* In order to compare the two belief state estimation strategies for all agents and different hyperparameter scenarios, the prediction lengths for each scenario were calculated with a separate file ["Test_GW_WM.py"].
### Plot the results
* The training can be monitored with the evolution of the free and expected free energies using the file ["Plot_Evolution_Energies.py"] for both environments. To switch, simply indicate the right folder with the relevant data.
* The comparison of the prediction lengths during the training for different hyperparameter settings in the navigation task was generated with the file ["Plot_Length_trajectories.py"].

# Pre-requisites
The code was written and run in Python 3.11.8. It uses the following libraries:
- numpy
- itertools
- numba
- joblib
- collections
- tqdm
- dill
- pandas
- matplotlib
- seaborn
- mycolorpy

# Citation
``` latex
@article{Pazem2024_FEPS,
      title={Free Energy Projective Simulation (FEPS): Active inference with interpretability}, 
      author={Joséphine Pazem and Marius Krumm and Alexander Q. Vining and Lukas J. Fiderer and Hans J. Briegel},
      year={2024},
      eprint={2411.14991},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2411.14991}, 
}
```
