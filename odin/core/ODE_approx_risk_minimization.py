"""
Implementation of ODE Risk minimization

Emmanouil Angelis, ETH Zurich 

based on code from

Gabriele Abbati, Machine Learning Research Group, University of Oxford
February 2019
"""


# Libraries
from odin.utils.trainable_models import TrainableModel
from odin.utils.gaussian_processes import GaussianProcess
from odin.utils.tensorflow_optimizer import ExtendedScipyOptimizerInterface
import numpy as np
import tensorflow as tf
from typing import Union, Tuple
import time

omegas = None
biases = None

def RFF_embeding(m,gamma,X):
    """
    Returns a numpy array of dimensions (n_states,n_points,m) that has as rows the RFFs for the RBF Kernel
    :param m: int, m/2 is the sample size for RFF approx (note that RFF vectors are of length m, as we use both sines and cosines);
    :param gamma: numpy array of dimensions (n_states,1,1) containing the lengthscales;
    :param X: numpy array of dimensions (n_points,1) with the time points
    """
    global omegas
    omegas = np.random.normal(size=m//2)
    nodes = np.reshape(omegas, [1,1,-1] )/ gamma
    X=np.reshape(X,[1,-1,1])
    nodes = nodes *X
    cos_nodes = np.cos(nodes)/np.sqrt(m//2)
    sin_nodes = np.sin(nodes)/np.sqrt(m//2)
    return np.concatenate([cos_nodes,sin_nodes],axis=2)

def RFF_embeding_derivative(m,gamma,X):
    """
    Returns a numpy array of dimensions (n_states,n_points,m) that has as rows the derivatives of RFFs for the RBF Kernel
    :param m: int, m/2 is the sample size for RFF approx (note that RFF vectors are of length m, as we use both sines and cosines);
    :param gamma: numpy array of dimensions (n_states,1,1) containing the lengthscales;
    :param X: numpy array of dimensions (n_points,1) with the time points
    """
    global omegas
    nodes_ = np.reshape(omegas, [1,1,-1] )/ gamma
    X=np.reshape(X,[1,-1,1])
    nodes = nodes_ *X
    cos_nodes = -nodes_*np.sin(nodes)/np.sqrt(m//2)
    sin_nodes = nodes_*np.cos(nodes)/np.sqrt(m//2)
    return np.concatenate([cos_nodes,sin_nodes],axis=2)

def RFF_embeding_bias(m,gamma,X):
    """
    Returns a numpy array of dimensions (n_states,n_points,m) that has as rows the RFFs for the RBF Kernel
    :param m: int, the sample size for RFF approx;
    :param gamma: numpy array of dimensions (n_states,1,1) containing the lengthscales;
    :param X: numpy array of dimensions (n_points,1) with the time points
    """
    global omegas
    global biases
    omegas = np.random.normal(size=m)
    biases = np.random.uniform(0,2*np.pi,size=m)
    nodes = np.reshape(omegas, [1,1,-1] )/ gamma
    X=np.reshape(X,[1,-1,1])
    nodes = nodes *X
    cos_nodes = np.sqrt(2)*np.cos(nodes+biases)/np.sqrt(m)
    return cos_nodes

def RFF_embeding_derivative_bias(m,gamma,X):
    """
    Returns a numpy array of dimensions (n_states,n_points,m) that has as rows the derivatives of RFFs for the RBF Kernel
    :param m: int, the sample size for RFF approx;
    :param gamma: numpy array of dimensions (n_states,1,1) containing the lengthscales;
    :param X: numpy array of dimensions (n_points,1) with the time points
    """
    global omegas
    global biases
    nodes_ = np.reshape(omegas, [1,1,-1] )/ gamma
    X=np.reshape(X,[1,-1,1])
    nodes = nodes_ *X
    cos_nodes = -np.sqrt(2)*nodes_*np.sin(nodes+biases)/np.sqrt(m)
    return cos_nodes


def hermite_embeding(m,gamma,X):
    """
    Returns a numpy array of dimensions (n_states,n_points,m) that has as rows the QFFs for the RBF Kernel
    :param m: int, m/2 is the order of the Quadrature Scheme (note that QFF vectors are of length m, as we use both sines and cosines);
    :param gamma: numpy array of dimensions (n_states,1,1) containing the lengthscales;
    :param X: numpy array of dimensions (n_points,1) with the time points
    """
    (nodes, weights) = np.polynomial.hermite.hermgauss(m//2)
    nodes = np.reshape(np.sqrt(2) * nodes, [1,1,-1] )/ gamma
    X=np.reshape(X,[1,-1,1])
    nodes = nodes *X
    weights = np.sqrt(np.reshape( weights/np.sqrt([np.pi]), [1,1,-1]))
    cos_nodes = weights*np.cos(nodes)
    sin_nodes = weights*np.sin(nodes)
    return np.concatenate([cos_nodes,sin_nodes],axis=2)

def hermite_embeding_derivative(m,gamma,X):
    """
    Returns a numpy array of dimensions (n_states,n_points,m) that has as rows the derivatives of QFFs for the RBF Kernel
    :param m: int, m/2 the order of the Quadrature Scheme (note that QFF vectors are of length m, as we use both sines and cosines);
    :param gamma: numpy array of dimensions (n_states,1,1) containing the lengthscales;
    :param X: numpy array of dimensions (n_points,1) with the time points
    """
    (nodes, weights) = np.polynomial.hermite.hermgauss(m//2)
    nodes = np.reshape(np.sqrt(2) * nodes, [1,1,-1] )/ gamma
    weights = np.sqrt(np.reshape( weights/np.sqrt([np.pi]), [1,1,-1]))*nodes
    X=np.reshape(X,[1,-1,1])
    nodes = nodes *X
    cos_nodes = -weights*np.sin(nodes)
    sin_nodes = weights*np.cos(nodes)
    return np.concatenate([cos_nodes,sin_nodes],axis=2)

class ODEApproxRiskMinimization(object):
    """
    Class that implements approximate ODIN risk minimization
    """

    def __init__(self, trainable: TrainableModel,
                 system_data: np.array, t_data: np.array,
                 gp_kernel: str = 'RBF',
                 optimizer: str = 'L-BFGS-B',
                 initial_gamma: float = 0.3,
                 train_gamma: bool = True,
                 gamma_bounds: Union[np.array, list, Tuple] = (1e-6+1e-4, 10.0),
                 state_bounds: np.array = None,
                 basinhopping: bool = True,
                 basinhopping_options: dict = None,
                 single_gp: bool = False,
                 state_normalization: bool = True,
                 time_normalization: bool = False,
                 tensorboard_summary_dir: str = None,
                 runtime_prof_dir: str = None,
                 QFF_features: int = 40,
                 Approx_method: str = "QFF"):
        """
        Constructor.
        :param trainable: Trainable model class, as explained and implemented in
        utils.trainable_models;
        :param system_data: numpy array containing the noisy observations of
        the state values of the system, size is [n_states, n_points];
        :param t_data: numpy array containing the time stamps corresponding to
        the observations passed as system_data;
        :param gp_kernel: string indicating which kernel to use in the GP.
        Valid options are 'RBF', 'Matern52', 'Matern32', 'RationalQuadratic',
        'Sigmoid';
        :param optimizer: string indicating which scipy optimizer to use. The
        valid ones are the same that can be passed to scipy.optimize.minimize.
        Notice that some of them will ignore bounds;
        :param initial_gamma: initial value for the gamma parameter.
        :param train_gamma: boolean, indicates whether to train of not the
        variable gamma;
        :param gamma_bounds: bounds for gamma (a lower bound of at least 1e-6
        is always applied to overcome numerical instabilities);
        :param state_bounds: bounds for the state optimization;
        :param basinhopping: boolean, indicates whether to turn on the scipy
        basinhopping;
        :param basinhopping_options: dictionary containing options for the
        basinhooping algorithm (syntax is the same as scipy's one);
        :param single_gp: boolean, indicates whether to use a single set of GP
        hyperparameters for each state;
        :param state_normalization: boolean, indicates whether to normalize the
        states values before the optimization (notice the parameter values
        theta won't change);
        :param time_normalization: boolean, indicates whether to normalize the
        time stamps before the optimization (notice the parameter values
        theta won't change);
        :param QFF_features: int, the order of the quadrature scheme
        :param tensorboard_summary_dir, runtime_prof_dir: str, logging directories
        """
        # Save arguments
        self.Approx_method = Approx_method
        self.QFF_approx=QFF_features
        self.lamda=1e-4
        self.trainable = trainable
        self.system_data = np.copy(system_data)
        self.t_data = np.copy(t_data).reshape(-1, 1)
        self.dim, self.n_p = system_data.shape
        self.gp_kernel = gp_kernel
        if self.gp_kernel != 'RBF':
            raise NotImplementedError("Only RBF kernel is currently implemented for use with QFFs")
        self.optimizer = optimizer
        self.initial_gamma = initial_gamma
        self.train_gamma = train_gamma
        self.basinhopping = basinhopping
        self.basinhopping_options = {'n_iter': 10,
                                     'temperature': 1.0,
                                     'stepsize': 0.05}
        self.state_normalization = state_normalization
        if basinhopping_options:
            self.basinhopping_options.update(basinhopping_options)
        self.single_gp = single_gp
        # Build bounds for the states and for gamma
        self._compute_state_bounds(state_bounds)
        self._compute_gamma_bounds(gamma_bounds)
        # Initialize utils
        self._compute_standardization_data(state_normalization,
                                           time_normalization)
        # Build the necessary TensorFlow tensors
        self._build_tf_data()
        # Initialize the Gaussian Process for the derivative model
        self.gaussian_process = GaussianProcess(self.dim, self.n_p,
                                                self.gp_kernel, self.single_gp)

        #initialize logging variables
        if tensorboard_summary_dir:
            self.writer = tf.summary.FileWriter(tensorboard_summary_dir)
            self.theta_sum=tf.summary.histogram('Theta_summary',self.trainable.theta)
        else:
          self.writer = None

        self.runtime_prof_dir= runtime_prof_dir
        # Initialization of TF operations
        self.init = None
        return

    def _compute_gamma_bounds(self, bounds: Union[np.array, list, Tuple])\
            -> None:
        """
        Builds the numpy array that defines the bounds for gamma.
        :param bounds: of the form (lower_bound, upper_bound).
        """
        self.gamma_bounds = np.array([1.0, 1.0])
        if bounds is None:
            self.gamma_bounds[0] = np.log(1e-6+1e-4)
            self.gamma_bounds[1] = np.inf
        else:
            self.gamma_bounds[0] = np.log(np.array(bounds[0]))
            self.gamma_bounds[1] = np.log(np.array(bounds[1]))
        return


    def _compute_state_bounds(self, bounds: np.array) -> None:
        """
        Builds the numpy array that defines the bounds for the states.
        :param bounds: numpy array, sized [n_dim, 2], in which for each
        dimensions we can find respectively lower and upper bounds.
        """
        if bounds is None:
            self.state_bounds = np.inf * np.ones([self.dim, 2])
            self.state_bounds[:, 0] = - self.state_bounds[:, 0]
        else:
            self.state_bounds = np.array(bounds)
        return

    def _compute_standardization_data(self, state_normalization: bool,
                                      time_normalization: bool) -> None:
        """
        Compute the means and the standard deviations for data standardization,
        used in the GP regression.
        """
        # Compute mean and std dev of the state and time values
        if state_normalization:
            self.system_data_means = np.mean(self.system_data,
                                             axis=1).reshape(self.dim, 1)
            self.system_data_std_dev = np.std(self.system_data,
                                              axis=1).reshape(self.dim, 1)
        else:
            self.system_data_means = np.zeros([self.dim, 1])
            self.system_data_std_dev = np.ones([self.dim, 1])
        if time_normalization:
            self.t_data_mean = np.mean(self.t_data)
            self.t_data_std_dev = np.std(self.t_data)
        else:
            self.t_data_mean = 0.0
            self.t_data_std_dev = 1.0
        # Normalize states and time
        self.normalized_states = (self.system_data - self.system_data_means) / \
            self.system_data_std_dev
        self.normalized_t_data = (self.t_data - self.t_data_mean) / \
            self.t_data_std_dev
        return

    def _build_tf_data(self) -> None:
        """
        Initialize all the TensorFlow constants needed in the pipeline.
        """
        self.system = tf.constant(self.normalized_states, dtype=tf.float64)
        self.t = tf.constant(self.normalized_t_data, dtype=tf.float64)
        self.system_means = tf.constant(self.system_data_means,
                                        dtype=tf.float64,
                                        shape=[self.dim, 1])
        self.system_std_dev = tf.constant(self.system_data_std_dev,
                                          dtype=tf.float64,
                                          shape=[self.dim, 1])
        self.t_mean = tf.constant(self.t_data_mean, dtype=tf.float64)
        self.t_std_dev = tf.constant(self.t_data_std_dev, dtype=tf.float64)
        self.n_points = tf.constant(self.n_p, dtype=tf.int32)
        self.dimensionality = tf.constant(self.dim, dtype=tf.int32)
        return

    def _build_states_bounds(self) -> None:
        """
        Builds the tensors for the normalized states that will containing the
        bounds for the constrained optimization.
        """
        # Tile the bounds to get the right dimensions
        state_lower_bounds = self.state_bounds[:, 0].reshape(self.dim, 1)
        state_lower_bounds = np.tile(state_lower_bounds, [1, self.n_p])
        state_lower_bounds = (state_lower_bounds - self.system_data_means)\
            / self.system_data_std_dev
        state_lower_bounds = state_lower_bounds.reshape([self.dim,
                                                         self.n_p])
        state_upper_bounds = self.state_bounds[:, 1].reshape(self.dim, 1)
        state_upper_bounds = np.tile(state_upper_bounds, [1, self.n_p])
        state_upper_bounds = (state_upper_bounds - self.system_data_means)\
            / self.system_data_std_dev
        state_upper_bounds = state_upper_bounds.reshape([self.dim,
                                                         self.n_p])
        self.state_lower_bounds = state_lower_bounds
        self.state_upper_bounds = state_upper_bounds
        return



    def _build_variables(self) -> None:
        """
        Builds the TensorFlow variables with the state values and the gamma
        that will later be optimized.
        """
        self.Z=tf.Variable(tf.zeros([self.dim,self.n_p,self.QFF_approx],dtype=tf.float64),dtype=tf.float64, trainable=False,name='Z')
        self.Z_prime=tf.Variable(tf.zeros([self.dim,self.n_p,self.QFF_approx],dtype=tf.float64),dtype=tf.float64, trainable=False,name='Z_prime')
        with tf.variable_scope('risk_main'):
            self.x = tf.Variable(self.system,
                                 dtype=tf.float64, trainable=True,
                                 name='states')
            if self.single_gp:
                self.log_gamma_single = tf.Variable(np.log(self.initial_gamma),
                                                    dtype=tf.float64,
                                                    trainable=self.train_gamma,
                                                    name='gamma')
                self.gamma =\
                    tf.exp(self.log_gamma_single)\
                    * tf.ones([self.dimensionality, 1, 1], dtype=tf.float64)
            else:
                self.log_gamma = tf.Variable(
                    np.log(self.initial_gamma)
                    * tf.ones([self.dimensionality, 1, 1],
                              dtype=tf.float64),
                    trainable=self.train_gamma,
                    dtype=tf.float64,
                    name='log_gamma')
                self.gamma = tf.exp(self.log_gamma)
        return

    def _build_regularization_risk_term(self) -> tf.Tensor:
        """
        Build the first term of the risk, connected to regularization.
        :return: the TensorFlow Tensor that contains the term.
        """
        a_vector = tf.matmul(self.Z_t_x,self.inv_Z_t_x,transpose_a=True,name='reg_risk_main_term')
        risk_term = 0.5/self.lamda * (tf.reduce_sum(self.x * self.x)-tf.reduce_sum(a_vector))
        return risk_term

    def _build_states_risk_term(self) -> tf.Tensor:
        """
        Build the second term of the risk, connected with the value of the
        states.
        :return: the TensorFlow Tensor that contains the term.
        """
        states_difference = self.system - self.x
        risk_term = tf.reduce_sum(states_difference * states_difference, 1)
        risk_term = risk_term * 0.5 / tf.squeeze(
            self.gaussian_process.likelihood_variances)
        return tf.reduce_sum(risk_term)

    def _build_derivatives_risk_term(self) -> tf.Tensor:
        """
        Build the third term of the risk, connected with the value of the
        derivatives.
        :return: the TensorFlow Tensor that contains the term.
        """
        # Compute model and data-based derivatives
        unnormalized_states = self.x * self.system_std_dev + self.system_means
        model_derivatives = tf.expand_dims(self.trainable.compute_gradients(
            unnormalized_states) / self.system_std_dev * self.t_std_dev, -1)

        self.data_derivatives=tf.matmul(self.Z_prime,self.inv_Z_t_x,name='Dx')

        derivatives_difference = model_derivatives - self.data_derivatives

        Z_prime_t_der_dif=tf.matmul(self.Z_prime,derivatives_difference ,transpose_a=True,name='Z_prime_t_der_dif')
        self.Hess_inner_dim=tf.matmul(self.Z_prime,self.Z_prime,transpose_a=True,name='Hess_inner_dim')
        temp=self.Hess_inner_dim+self.gamma*self.Z_t_Z_lamda/self.lamda
        temp1=tf.linalg.solve(temp,Z_prime_t_der_dif,name='inverse_of_der_risk_term')
        second_term=tf.matmul(Z_prime_t_der_dif,temp1,transpose_a=True)
        first_term=tf.matmul(derivatives_difference,derivatives_difference,transpose_a=True)
        risk_term= (first_term -second_term)/self.gamma
        risk_term = 0.5 * tf.reduce_sum(risk_term)
        return risk_term

    def _build_gamma_risk_term(self) -> tf.Tensor:
        """
        Build the terms associated with gamma
        :return: the TensorFlow Tensor that contains the terms
        """
        # Compute log_variance on the derivatives
        self.A_inner_dim=tf.Variable(tf.zeros([self.dim,self.QFF_approx,self.QFF_approx],dtype=tf.float64),dtype=tf.float64, trainable=False,name='A_inner_dim')
        risk_term = 0.5 * (tf.linalg.logdet(self.A_inner_dim + self.gamma *tf.eye(self.QFF_approx,dtype=tf.float64) ) + (self.n_p - self.QFF_approx) * tf.squeeze(tf.log(self.gamma)) )
        return tf.reduce_sum(risk_term)

    def _build_risk(self) -> None:
        """
        Build the risk tensor by summing up the single terms.
        """
        self.risk_term1 = self._build_regularization_risk_term()
        self.risk_term2 = self._build_states_risk_term()
        self.risk_term3 = self._build_derivatives_risk_term()
        self.risk_term4 = self._build_gamma_risk_term()
        self.risk = self.risk_term1 + self.risk_term2 + self.risk_term3
        if self.train_gamma:
            self.risk += self.risk_term4
        if self.writer:
            loss_sum=tf.summary.scalar(name='loss_sum', tensor=self.risk)
        return

    def _build_optimizer(self) -> None:
        """
        Build the TensorFlow optimizer, wrapper to the scipy optimization
        algorithms.
        """
        # Extract the TF variables that get optimized in the risk minimization
        t_vars = tf.trainable_variables()
        risk_vars = [var for var in t_vars if 'risk_main' in var.name]
        # Dictionary containing the bounds on the TensorFlow Variables
        var_to_bounds = {risk_vars[0]: (self.trainable.parameter_lower_bounds,
                                        self.trainable.parameter_upper_bounds),
                         risk_vars[1]: (self.state_lower_bounds,
                                        self.state_upper_bounds)}
        if self.train_gamma:
            var_to_bounds[risk_vars[2]] = (self.gamma_bounds[0],
                                           self.gamma_bounds[1])

        self.risk_optimizer = ExtendedScipyOptimizerInterface(
            loss=self.risk, method=self.optimizer, var_list=risk_vars,
            var_to_bounds=var_to_bounds,file_writer=self.writer,dir_prof_name=self.runtime_prof_dir)
        return

    def build_model(self) -> None:
        """
        Builds Some common part of the computational graph for the optimization.
        """
        #self.gaussian_process.build_supporting_covariance_matrices(self.t, self.t)
        self._build_states_bounds()
        self._build_variables()

        self.Z_t_x=tf.matmul(self.Z,tf.expand_dims(self.x,-1),transpose_a=True,name='Z_t_x')
        #self.prox=tf.matmul(self.Z,self.Z,transpose_b=True)
        self.Kernel_inner_dim=tf.matmul(self.Z,self.Z,transpose_a=True,name='Kernel_inner_dim')
        self.Z_t_Z_lamda=self.Kernel_inner_dim+self.lamda*tf.eye(self.QFF_approx,dtype=tf.float64)
        self.inv_Z_t_x=tf.linalg.solve(self.Z_t_Z_lamda,self.Z_t_x,name='inv_Z_t_x')

        self._build_risk()
        if self.writer:
            self.merged_sum = tf.summary.merge_all()
        self._build_optimizer()
        return

    def _initialize_variables(self) -> None:
        """
        Initialize all the variables and placeholders in the graph.
        """
        self.init = tf.global_variables_initializer()
        return

    def _initialize_states_with_mean_gp(self, session: tf.Session, compute_dict:dict) -> None:
        """
        Before optimizing the risk, we initialize the x to be the mean
        predicted by the Gaussian Process for an easier task later.
        :param session: TensorFlow session, used in the fit function.
        """
        #self.mean_prediction = self.gaussian_process.compute_posterior_mean(self.system)
        assign_states_mean = tf.assign(self.x, tf.squeeze(self.mean_prediction))
        session.run(assign_states_mean,feed_dict=compute_dict)
        self.X = self.x
        self.x = tf.clip_by_value(
            self.x, clip_value_min=tf.constant(self.state_lower_bounds),
            clip_value_max=tf.constant(self.state_upper_bounds))
        return

    def _initialize_constants_for_risk(self,lengthscales,variances,noise_var):
        lengthscales = np.reshape(lengthscales,[-1,1,1])
        variances = np.reshape(variances,[-1,1,1])
        if self.Approx_method == "QFF":
            Z=hermite_embeding(self.QFF_approx,lengthscales,self.t_data) * np.sqrt(variances)
            Z_prime=hermite_embeding_derivative(self.QFF_approx,lengthscales,self.t_data) * np.sqrt(variances)
        elif self.Approx_method == "RFF":
            Z=RFF_embeding(self.QFF_approx,lengthscales,self.t_data) * np.sqrt(variances)
            Z_prime=RFF_embeding_derivative(self.QFF_approx,lengthscales,self.t_data) * np.sqrt(variances)
        elif self.Approx_method == "RFF_bias":
            Z=RFF_embeding_bias(self.QFF_approx,lengthscales,self.t_data) * np.sqrt(variances)
            Z_prime=RFF_embeding_derivative_bias(self.QFF_approx,lengthscales,self.t_data) * np.sqrt(variances)

        Kernel_inner_dim = np.matmul(np.transpose(Z,[0,2,1]),Z)
        u,s,v = np.linalg.svd(Kernel_inner_dim)
        D= np.array([ np.diag( 1/ np.sqrt(s[i] + self.lamda) ) for i in range(s.shape[0]) ])
        inv_sqrt_Kernel_inner_dim = np.matmul(u,np.matmul(D,v))
        U=np.matmul(Z_prime,inv_sqrt_Kernel_inner_dim)
        A_inner_dim = self.lamda*np.matmul(np.transpose(U,[0,2,1]),U)

        #np.save("A_inner_dim",A_inner_dim)

        self.mean_prediction = np.matmul(Z,np.linalg.solve( Kernel_inner_dim + noise_var * np.eye(self.QFF_approx)  ,np.matmul( np.transpose(Z,[0,2,1]) , np.expand_dims( self.normalized_states,-1) ) ))
        
        comp_dict={self.Z : Z,self.Z_prime : Z_prime,self.Kernel_inner_dim : Kernel_inner_dim ,self.Hess_inner_dim : np.matmul(np.transpose(Z_prime,[0,2,1]),Z_prime), self.A_inner_dim : A_inner_dim}
        return comp_dict

    def train(self,gp_parameters):
        """
        Trains the model and returns thetas
        :param gp_parameters: values of hyperparameters of GP
        """
        compute_dict={self.gaussian_process.kernel.lengthscales: gp_parameters[0], self.gaussian_process.kernel.variances:gp_parameters[1], self.gaussian_process.likelihood_variances:gp_parameters[2]}
        compute_dict.update(self._initialize_constants_for_risk(gp_parameters[0],gp_parameters[1],gp_parameters[2]))

        self._initialize_variables()
        session = tf.Session()
        with session:
            # Start the session
            session.run(self.init)

            # Initialize x as the mean of the GP
            self._initialize_states_with_mean_gp(session,compute_dict=compute_dict)

            # Print initial theta
            theta = session.run(self.trainable.theta,feed_dict=compute_dict)
            print("Initialized Theta", theta)

            # Print initial gamma
            gamma = session.run(self.gamma,feed_dict=compute_dict)
            print("Initialized Gamma", gamma)

            # Print the terms of the Risk before the optimization
            print("Risk 1: ", session.run(self.risk_term1,feed_dict=compute_dict))
            print("Risk 2: ", session.run(self.risk_term2,feed_dict=compute_dict))
            print("Risk 3: ", session.run(self.risk_term3,feed_dict=compute_dict))
            print("Risk: ", session.run(self.risk,feed_dict=compute_dict))

            if self.writer:
                self.writer.add_graph(session.graph)

                def summary_funct(merged_sum):
                    summary_funct.step+=1
                    self.writer.add_summary(merged_sum, summary_funct.step)

                summary_funct.step=-1

            result=[]
            # Optimize
            if self.basinhopping:
                secs=time.time()
                result=self.risk_optimizer.basinhopping(session,feed_dict=compute_dict,
                                                 **self.basinhopping_options)
                secs=time.time() -secs
            else:
                if self.writer:
                    secs=time.time()
                    result=self.risk_optimizer.minimize(session,feed_dict=compute_dict,loss_callback=summary_funct,fetches=[self.merged_sum])
                    secs=time.time() -secs
                else:
                    secs=time.time()
                    result=self.risk_optimizer.minimize(session,feed_dict=compute_dict)
                    secs=time.time() -secs
            print("Elapsed time is ",secs)
            # Print the terms of the Risk after the optimization
            print("risk 1: ", session.run(self.risk_term1,feed_dict=compute_dict))
            print("risk 2: ", session.run(self.risk_term2,feed_dict=compute_dict))
            print("risk 3: ", session.run(self.risk_term3,feed_dict=compute_dict))
            if self.train_gamma:
                print("risk 4: ", session.run(self.risk_term4,feed_dict=compute_dict))
            found_risk=session.run(self.risk,feed_dict=compute_dict)
            print("risk: ", found_risk)

            unnormalized_states = tf.squeeze(self.x) * self.system_std_dev + \
                self.system_means
            states_after = session.run(unnormalized_states,feed_dict=compute_dict)

            # Print final theta
            theta = session.run(self.trainable.theta,feed_dict=compute_dict)
            print("Final Theta", theta)

            # Print final gamma
            gamma = session.run(self.gamma,feed_dict=compute_dict)
            print("Final Gamma", gamma)

        tf.reset_default_graph()
        return theta, secs
