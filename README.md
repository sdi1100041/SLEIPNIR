# SLEIPNIR
Implementation of SLEIPNIR in python 3, based on the paper "SLEIPNIR: Deterministic and Provably Accurate Feature Expansion for
Gaussian Process Regression with Derivatives" by Emmanouil Angelis, Philippe Wenk, Bernhard Schölkopf, Stefan Bauer, Andreas Krause. This code is based on https://github.com/Mojusko/QFF (published alongside https://papers.nips.cc/paper/8115-efficient-high-dimensional-bayesian-optimization-with-additivity-and-quadrature-fourier-features) as well as on https://github.com/gabb7/ODIN (published alongside https://arxiv.org/abs/1902.06278) .If you use this code, please consider citing our publication (https://arxiv.org/abs/2003.02658).

## Out of the Box Experiments
Currently, there are six experiments implemented that should run out of the box. In order to run an experiment, three stages are required:
1) Generate the data: Create the synthetic datasets of the corresponding experiment by specifying the number of different datasets that will be generated in total, the number of observation points and the noise level. E.g in the case of Lotka Volterra you run from current (working) directory: python3 generate_lv_data.py --noise_var 0.1 --n_obs 2000 --n_realizations 100

2) Copy the data: Copy the files "state_observations.npy" and "time_points.npy" from the corresponding folder (created in step 1), to working directory. 

3) Run ODIN_interface.py: From working directory run the inference algorithm, specifying the model corresponding to the input datasets (of the previous step), the approximation method, e.t.c. E.g in the case of Lotka Volterra you run from working directory: python3.6 ODIN_interface.py --model LV --GP_approx_method QFF --Risk_min_approx_method RFF --n_features 40 --state_normalization --train_gamma
