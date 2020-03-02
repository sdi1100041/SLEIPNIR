from odin.core.GP_risk_minimization import GPRiskMinimization
from odin.core.GP_approx_risk_minimization import GPApproxRiskMinimization
from odin.core.ODE_risk_minimization import ODERiskMinimization
from odin.core.ODE_approx_risk_minimization import ODEApproxRiskMinimization
import numpy as np
import tensorflow as tf

import argparse
parser = argparse.ArgumentParser()

parser.add_argument( "--gp_npoints", type=int, default=-1, help="number of points to train the Gaussian Process")
parser.add_argument( "--gp_sampling", type=str, default="first", choices={"random","uniform","first"}, help="method of sampling for the Gaussian Process")
parser.add_argument( "--profiling", action='store_true', help="profiling")
parser.add_argument( "--tensorboard", action='store_true', help="tensorboard profiling")
parser.add_argument( "--load_hyperparameters", action='store_true',help="load hyperparameters from file hyperparams")
parser.add_argument( "--GP_approx_method", type=str, default="QFF",choices={"QFF","RFF","RFF_bias"}, help="approximation method for Gaussian Process")
parser.add_argument( "--Risk_min_approx_method", type=str, default="QFF",choices={"QFF","RFF","RFF_bias"}, help="approximation method for Risk Minimization")
parser.add_argument( "--n_features", type=int, default=-1, help="number of features for QFF approximation")
parser.add_argument( "--state_normalization", help="normalize the states", action='store_true')
parser.add_argument( "--train_gamma", help="gamma training", action='store_true')
parser.add_argument("--model", type=str, required = True, choices={"LV","Lorenz63","PT","QUADRO","FHN","Glucose"}, help="ODEs model")
args = parser.parse_args()

system_obs = np.load("state_observations.npy")
t_obs = np.load("time_points.npy")

t_obs = np.array(t_obs).reshape(-1, 1)
state_bounds = None

profiling_dir = None
tensorboard_dir = None
logging_dir= args.model + "_"  + str(t_obs.shape[0]-1) + "_" + args.Risk_min_approx_method + "_"+ str(args.n_features)
if args.profiling:
    profiling_dir = logging_dir
if args.tensorboard:
    tensorboard_dir = logging_dir

if args.gp_npoints == -1:
    args.gp_npoints = t_obs.shape[0]

if (args.n_features != -1) and ((args.n_features %2) == 1):
    args.n_features+=1

if not args.load_hyperparameters:
    if args.gp_sampling == "first":
        sample_indices=np.arange(np.min([t_obs.shape[0],args.gp_npoints]))
    elif args.gp_sampling == "uniform":
        sample_indices=np.arange(0,t_obs.shape[0],np.int16(np.ceil(t_obs.shape[0]/args.gp_npoints)))
    else:
        sample_indices=np.random.choice(t_obs.shape[0], np.min([t_obs.shape[0],args.gp_npoints]), replace=False)

    if args.n_features != -1 :
        gp_risk_minimizer = GPApproxRiskMinimization(system_obs[:,sample_indices],
                                               t_obs[sample_indices,:], gp_kernel='RBF',
                                               single_gp=False,
                                               time_normalization=False,
                                               state_normalization=args.state_normalization,
                                               QFF_approx = args.n_features,
                                               Approx_method = args.GP_approx_method)
    else:
        gp_risk_minimizer = GPRiskMinimization(system_obs[:,sample_indices],
                                               t_obs[sample_indices,:], gp_kernel='RBF',
                                               single_gp=False,
                                               time_normalization=False,
                                               state_normalization=args.state_normalization)

    gp_risk_minimizer.build_model()
    gp_parameters=gp_risk_minimizer.train()
    hyperparameter_training_time = gp_parameters[0]
    gp_parameters = gp_parameters[1:]
else:
    hyperparameter_training_time = 0
    gp_parameters = np.expand_dims( np.expand_dims( np.loadtxt("hyperparams"),-1 ), -1)

if args.model == "LV":
    from odin.utils.trainable_models import TrainableLotkaVolterra
    theta_bounds = np.array([[0.0, 100.0], [0.0, 100.0], [0.0, 100.0], [0.0, 100.0]])
    trainable_model = TrainableLotkaVolterra(system_obs, t_obs, bounds=theta_bounds)
    state_bounds = np.array([[0.0, 10.0], [0.0, 10.0]])
elif args.model == "Lorenz63" :
    from odin.utils.trainable_models import TrainableLorenz63
    theta_bounds = np.array([[0.0, 100.0], [0.0, 100.0], [0.0, 10.0]])# Trainable object
    trainable_model = TrainableLorenz63(system_obs, t_obs, bounds=theta_bounds)
elif args.model == "PT":
    from odin.utils.trainable_models import TrainableProteinTransduction
    theta_bounds = np.array([[1e-8, 10.0], [1e-8, 10.0], [1e-8, 10.0], [1e-8, 10.0],[1e-8, 10.0], [1e-8, 10.0]])
    trainable_model = TrainableProteinTransduction(system_obs, t_obs, bounds=theta_bounds)
    state_bounds = np.array([[0.0, 2.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0]])
elif args.model == "QUADRO":
    from odin.utils.trainable_models import TrainableQuadrocopter
    theta_bounds = np.array([[1e-8, 10.0]]*7)
    trainable_model = TrainableQuadrocopter(system_obs, t_obs, bounds=theta_bounds)
    state_bounds = np.array([[-200, 150], [-200, 150], [-200, 150], [-2, 2], [-2, 2], [-2, 2], [-20, 20], [-20, 20], [-20, 20],
                         [-800, 100], [-800, 100], [-800, 100]])
elif args.model == "FHN":
    from odin.utils.trainable_models import TrainableFitzHughNagumo
    theta_bounds = np.array([[0.0, 100.0], [0.0, 100.0], [0.0, 100.0]])# Trainable object
    trainable_model = TrainableFitzHughNagumo(system_obs, t_obs, bounds=theta_bounds)
elif args.model == "Glucose":
    from odin.utils.trainable_models import TrainableGlucose
    theta_bounds = np.array([[-1, 5]]*10)
    trainable_model = TrainableGlucose(system_obs, t_obs, bounds=theta_bounds)

if args.n_features == -1 :
    test_risk_minimizer = ODERiskMinimization(trainable_model, system_obs,
                                       t_obs, gp_kernel='RBF',
                                       optimizer='L-BFGS-B',
                                       initial_gamma=0.3,
                                       train_gamma=args.train_gamma,
                                       state_bounds=state_bounds,
                                       single_gp=False,
                                       basinhopping=False,
                                       time_normalization=False,
                                       state_normalization=args.state_normalization,
                                       runtime_prof_dir=profiling_dir,
				       tensorboard_summary_dir=tensorboard_dir)
    test_risk_minimizer.build_model()
    theta, secs = test_risk_minimizer.train(gp_parameters=gp_parameters)
else:
    test_approx_risk_minimizer = ODEApproxRiskMinimization(trainable_model, system_obs,
                                       t_obs, gp_kernel='RBF',
                                       optimizer='L-BFGS-B',
                                       initial_gamma=0.3,
                                       train_gamma=args.train_gamma,
                                       state_bounds=state_bounds,
                                       single_gp=False,
                                       basinhopping=False,
                                       time_normalization=False,
                                       state_normalization=args.state_normalization,
                                       QFF_features = args.n_features,
                                       Approx_method = args.Risk_min_approx_method, 
                                       runtime_prof_dir=profiling_dir,
                                       tensorboard_summary_dir=tensorboard_dir)
    test_approx_risk_minimizer.build_model()
    theta, secs = test_approx_risk_minimizer.train(gp_parameters=gp_parameters)

np.savetxt("hyperparameter_training_times.csv",np.array([hyperparameter_training_time]))
np.savetxt("optimization_times.csv",np.array([secs]))
np.savetxt("thetas.csv",theta,delimiter=',')
