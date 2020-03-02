from odin.utils.dynamical_systems import Lorenz63
import numpy as np
from tensorflow.python.platform import gfile

#main_seed = 123
#np.random.seed(main_seed)

import argparse
parser = argparse.ArgumentParser()

parser.add_argument( "--n_realizations", type=int, required=True, help="number of realizations of experiments")
parser.add_argument( "--final_time", type=float, default=1, help="final time of integration")
parser.add_argument( "--n_obs", type=int, required=True, help="number of observation")
parser.add_argument( "--SNR", type=float, required=True, help="signal to noise ratio")
args = parser.parse_args()

delta_obs = args.final_time / float(args.n_obs)

Lorenz = Lorenz63(true_param=(10.0, 28.0, 8.0/3.0),
                     noise_variance=0.0,
                     stn_ratio=args.SNR)

if not gfile.Exists(str(args.SNR)):
    gfile.MakeDirs(str(args.SNR))

if not gfile.Exists(str(args.SNR) + "/"+str(args.n_obs)):
    gfile.MakeDirs(str(args.SNR) + "/"+str(args.n_obs))

for i in range(1,args.n_realizations+1):
    if not gfile.Exists(str(args.SNR) + "/"+str(args.n_obs) + str(i)):
        gfile.MakeDirs(str(args.SNR) + "/"+str(args.n_obs)+"/" + str(i))

    system_obs, t_obs = Lorenz.observe(initial_state=(1.0, 1.0, 1.0),
                                      initial_time=0.0,
                                      final_time=1.0,
                                      t_delta_integration=min(0.001, delta_obs),
                                      t_delta_observation=delta_obs)
    np.save(str(args.SNR) + "/"+str(args.n_obs)+"/" + str(i) + "/state_observations",system_obs)
    np.save(str(args.SNR) + "/"+str(args.n_obs)+"/" + str(i) + "/time_points",t_obs)
