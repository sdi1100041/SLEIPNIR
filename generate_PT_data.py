from odin.utils.dynamical_systems import ProteinTransduction
import numpy as np
from tensorflow.python.platform import gfile

main_seed = 123
np.random.seed(main_seed)

import argparse
parser = argparse.ArgumentParser()

parser.add_argument( "--n_realizations", type=int, required=True, help="number of realizations of experiments")
parser.add_argument( "--final_time", type=float, default=50, help="final time of integration")
parser.add_argument( "--n_obs", type=int, required=True, help="number of observation")
parser.add_argument( "--noise_var", type=float, required=True, help="noise variance")
args = parser.parse_args()

delta_obs = args.final_time / float(args.n_obs)

PT = ProteinTransduction(true_param=[0.07, 0.6, 0.05, 0.3, 0.017, 0.3],
                         noise_variance=args.noise_var, stn_ratio=0.0)

if not gfile.Exists(str(args.noise_var)):
    gfile.MakeDirs(str(args.noise_var))

if not gfile.Exists(str(args.noise_var) + "/"+str(args.n_obs)):
    gfile.MakeDirs(str(args.noise_var) + "/"+str(args.n_obs))

for i in range(1,args.n_realizations+1):
    if not gfile.Exists(str(args.noise_var) + "/"+str(args.n_obs) + str(i)):
        gfile.MakeDirs(str(args.noise_var) + "/"+str(args.n_obs)+"/" + str(i))

    system_obs, t_obs = PT.observe(initial_state=(1.0, 0.0, 1.0, 0.0, 0.0),
                                   initial_time=0.0,
                                   final_time=50,
                                   t_delta_integration=min(0.01,delta_obs),
                                   t_delta_observation=delta_obs)
    np.save(str(args.noise_var) + "/"+str(args.n_obs)+"/" + str(i) + "/state_observations",system_obs)
    np.save(str(args.noise_var) + "/"+str(args.n_obs)+"/" + str(i) + "/time_points",t_obs)
