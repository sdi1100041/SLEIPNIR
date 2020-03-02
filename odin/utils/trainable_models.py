"""
Implementations of trainable ODE models, the classes that contain the theta
variables optimized during training.

Gabriele Abbati, Machine Learning Research Group, University of Oxford
February 2019
"""


# Libraries
from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf


class TrainableModel(ABC):
    """
    Abstract class of a trainable dynamical system. The parameters to be
    estimated are contained in the attribute self.theta as TensorFlow Variables
    and will be optimized during training.
    Implementing new model consists in overriding the two abstract functions
    below.
    """

    def __init__(self,
                 n_states: int,
                 n_points: int,
                 bounds: np.array = None):
        """
        Constructor.
        :param n_states: number of states in the system;
        :param n_points: number of observation points;
        :param bounds for the optimization of theta.
        """
        self.n_states = n_states
        self.n_points = n_points
        with tf.variable_scope('risk_main'):
            self._initialize_parameter_variables()
        self.n_params = tf.constant(self.theta.shape[0], dtype=tf.int32)
        self.theta = tf.reshape(self.theta, shape=[self.n_params, 1])
        if bounds is None:
            bounds = np.inf * np.ones([self.theta.shape[0], 2])
            bounds[:, 0] = - bounds[:, 0]
        if self.theta.shape[0] == 1:
            self.parameter_lower_bounds = np.asscalar(bounds[0, 0])
            self.parameter_upper_bounds = np.asscalar(bounds[0, 1])
        else:
            self.parameter_lower_bounds = np.reshape(bounds[:, 0],
                                                     self.theta.shape)
            self.parameter_upper_bounds = np.reshape(bounds[:, 1],
                                                     self.theta.shape)
        return

    @abstractmethod
    def _initialize_parameter_variables(self) -> None:
        """
        Abstract method to be implemented. Initialize the TensorFlow variables
        containing the parameters theta of the ODE system. This will be 1D
        vector called 'self.theta', tensorflow.Variable type.
        """
        self.theta = tf.Variable(0.0)
        return

    @abstractmethod
    def compute_gradients(self, x: tf.Tensor) -> tf.Tensor:
        """
        Abstract method to be implemented. Compute the gradients of the ODE,
        meaning f(X, self.theta).
        :param x: values of the time series observed, whose shape is
        [n_states, n_points].
        :return: TensorFlow Tensor containing the gradients, whose shape is
        [n_states, n_points].
        """
        return tf.constant()


class TrainableLotkaVolterra(TrainableModel):
    """
    Trainable Lotka-Volterra model.
    """

    def _initialize_parameter_variables(self) -> None:
        """
        Initialize the TensorFlow variables containing the parameters theta of
        the ODE system. This will be 1D vector called 'self.theta',
        tensorflow.Variable type.
        """
        self.theta = tf.Variable(tf.abs(tf.random_normal([4, 1],
                                                         mean=0.0,
                                                         stddev=1.0,
                                                         dtype=tf.float64)),
                                 name='theta',
                                 trainable=True)
        return

    def compute_gradients(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute the gradients of the ODE, meaning f(X, self.theta).
        :param x: values of the time series observed, whose dimensions are
        [n_states, n_points].
        :return: TensorFlow Tensor containing the gradients, whose shape is
        [n_states, n_points].
        """
        grad1 = self.theta[0] * x[0:1, :]\
            - self.theta[1] * x[0:1, :] * x[1:2, :]
        grad2 = - self.theta[2] * x[1:2, :]\
            + self.theta[3] * x[0:1, :] * x[1:2, :]
        gradient_samples = tf.concat([grad1, grad2], 0)
        return gradient_samples


class TrainableFitzHughNagumo(TrainableModel):
    """
    Trainable FitzHug-Nagumo model.
    """

    def _initialize_parameter_variables(self) -> None:
        """
        Initialize the TensorFlow variables containing the parameters theta of
        the ODE system. This will be 1D vector called 'self.theta',
        tensorflow.Variable type.
        """
        self.theta = tf.Variable(tf.abs(tf.random_normal([3, 1],
                                                         mean=0.0,
                                                         stddev=1.0,
                                                         dtype=tf.float64)),
                                 name='theta',
                                 trainable=True)
        return

    def compute_gradients(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute the gradients of the ODE, meaning f(X, self.theta).
        :param x: values of the time series observed, whose dimensions are
        [n_states, n_points].
        :return: TensorFlow Tensor containing the gradients, whose shape is
        [n_states, n_points].
        """
        grad1 = self.theta[2] * (x[0:1, :] - tf.pow(x[0:1, :], 3.0) / 3.0 +
                                 x[1:2, :])
        grad2 = - (x[0:1, :] - self.theta[0] + self.theta[1] * x[1:2, :])\
            / self.theta[2]
        gradient_samples = tf.concat([grad1, grad2], 0)
        return gradient_samples


class TrainableProteinTransduction(TrainableModel):
    """
    Trainable 5D Protein transduction model.
    """

    def _initialize_parameter_variables(self) -> None:
        """
        Initialize the TensorFlow variables containing the parameters theta of
        the ODE system. This will be 1D vector called 'self.theta',
        tensorflow.Variable type.
        """
        self.theta = tf.Variable(tf.abs(tf.random_normal([6, 1],
                                                         mean=0.0,
                                                         stddev=1.0,
                                                         dtype=tf.float64)),
                                 name='theta',
                                 trainable=True)
        return

    def compute_gradients(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute the gradients of the ODE, meaning f(X, self.theta).
        :param x: values of the time series observed, whose dimensions are
        [n_states, n_points].
        :return: TensorFlow Tensor containing the gradients, whose shape is
        [n_states, n_points].
        """
        grad1 = - self.theta[0] * x[0:1, :]\
            - self.theta[1] * x[0:1, :] * x[2:3, :] + self.theta[2] * x[3:4, :]
        grad2 = self.theta[0] * x[0:1, :]
        grad3 = - self.theta[1] * x[0:1, :] * x[2:3, :]\
            + self.theta[2] * x[3:4, :]\
            + self.theta[4] * x[4:5, :] / (self.theta[5] + x[4:5, :])
        grad4 = self.theta[1] * x[0:1, :] * x[2:3, :]\
            - self.theta[2] * x[3:4, :] - self.theta[3] * x[3:4, :]
        grad5 = self.theta[3] * x[3:4, :]\
            - self.theta[4] * x[4:5, :] / (self.theta[5] + x[4:5, :])
        gradient_samples = tf.concat([grad1, grad2, grad3, grad4, grad5], 0)
        return gradient_samples


class TrainableLorenz96(TrainableModel):
    """
    Trainable Lorenz '96 model.
    """

    def _initialize_parameter_variables(self) -> None:
        """
        Initialize the TensorFlow variables containing the parameters of the
        ODE system. This will be 1D vector called 'self.theta',
        tensorflow.Variable type.
        """
        self.theta = tf.Variable(tf.abs(tf.random_normal([1],
                                                         mean=0.0,
                                                         stddev=1.0,
                                                         dtype=tf.float64)),
                                 name='theta',
                                 trainable=True)
        self.theta = tf.reshape(self.theta, [1, 1])
        return

    def compute_gradients(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute the gradients of the ODE, meaning f(X, self.theta).
        :param x: values of the time series observed, whose dimensions are
        [n_states, n_points].
        :return: TensorFlow Tensor containing the gradients, whose shape is
        [n_states, n_points].
        """
        grad1 = (x[1:2, :] - x[self.n_states - 2:self.n_states - 1, :])\
            * x[self.n_states - 1:self.n_states, :] - x[0:1, :] + self.theta
        grad2 = (x[2:3, :] - x[self.n_states - 1:self.n_states, :])\
            * x[0:1, :] - x[1:2, :] + self.theta
        grad_list = [grad1, grad2]
        for n in range(2, self.n_states - 1):
            state_derivative = (x[n + 1:n + 2, :] - x[n - 2:n - 1, :])\
                * x[n - 1:n, :] - x[n:n + 1, :] + self.theta
            grad_list.append(state_derivative)
        state_derivative = \
            (x[0:1, :] - x[self.n_states - 3:self.n_states - 2, :]) \
            * x[self.n_states - 2:self.n_states - 1, :]\
            - x[self.n_states - 1:self.n_states, :] + self.theta
        grad_list.append(state_derivative)
        gradients = tf.concat(grad_list, axis=0)
        return gradients

class TrainableLorenz63(TrainableModel):
    """
    Trainable Lorenz63 model.
    """

    def _initialize_parameter_variables(self) -> None:
        """
        Initialize the TensorFlow variables containing the parameters theta of
        the ODE system. This will be 1D vector called 'self.theta',
        tensorflow.Variable type.
        """
        self.theta = tf.Variable(tf.abs(tf.random_normal([3, 1],
                                                         mean=0.0,
                                                         stddev=1.0,
                                                         dtype=tf.float64)),
                                 name='theta',
                                 trainable=True)
        return

    def compute_gradients(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute the gradients of the ODE, meaning f(X, self.theta).
        :param x: values of the time series observed, whose dimensions are
        [n_states, n_points].
        :return: TensorFlow Tensor containing the gradients, whose shape is
        [n_states, n_points].
        """
        grad1 = self.theta[0]*(x[1:2, :] - x[0:1, :])
        grad2 = x[0:1, :]*(self.theta[1] - x[2:3, :]) - x[1:2, :]
        grad3 = x[0:1, :]*x[1:2, :] - self.theta[2]*x[2:3, :]
        gradient_samples = tf.concat([grad1, grad2, grad3], 0)
        return gradient_samples

class TrainableQuadrocopter(TrainableModel):
    """
    Trainable Quadrocopter model.
    """

    def _initialize_parameter_variables(self) -> None:
        """
        Initialize the TensorFlow variables containing the parameters theta of
        the ODE system. This will be 1D vector called 'self.theta',
        tensorflow.Variable type.
        """
        self.theta = tf.Variable(tf.abs(tf.random_normal([7, 1],
                                                         mean=0.0,
                                                         stddev=1.0,
                                                         dtype=tf.float64)),
                                 name='theta',
                                 trainable=True)
        return

    def compute_gradients(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute the gradients of the ODE, meaning f(X, self.theta).
        :param x: values of the time series observed, whose dimensions are
        [n_states, n_points].
        :return: TensorFlow Tensor containing the gradients, whose shape is
        [n_states, n_points].
        """
        np.random
        ub = x[0:1, :]
        vb = x[1:2, :]
        wb = x[2:3, :]
        p = x[3:4, :]
        q = x[4:5, :]
        r = x[5:6, :]
        phi = x[6:7, :]
        theta_sys = x[7:8, :]
        psi = x[8:9, :]
        xE = x[9:10, :]
        yE = x[10:11, :]
        hE = x[11:12, :]
        
        m = self.theta[0]         #kg
        Ixx = self.theta[1]*1e-3   #kg-m^2
        Iyy = self.theta[2]*1e-3   #kg-m^2
        Izz = self.theta[3]*(Ixx + Iyy) #kg-m^2 (Assume nearly flat object, z=0)
        dx = self.theta[4]      #m
        dy = self.theta[5]*1e-2     #m
        g = self.theta[6]  #m/s/s

        # Directly get forces as inputs
        [F1, F2, F3, F4] = TrainableQuadrocopter.controlForces(x)
        Fz = F1 + F2 + F3 + F4
        L = (F2 + F3) * dy - (F1 + F4) * dy
        M = (F1 + F3) * dx - (F2 + F4) * dx
        N = 0 #-T(F1,dx,dy) - T(F2,dx,dy) + T(F3,dx,dy) + T(F4,dx,dy)
        
        # Pre-calculate trig values
        cphi = tf.cos(phi);   sphi = tf.sin(phi)
        cthe = tf.cos(theta_sys); sthe = tf.sin(theta_sys)
        cpsi = tf.cos(psi);   spsi = tf.sin(psi)
        
        # Calculate the derivative of the state matrix using EOM
        grads = []
        
        grad0 = -g * sthe + r * vb - q * wb  # = udot
        grad1 = g * sphi*cthe - r * ub + p * wb # = vdot
        grad2 = 1/m * (-Fz) + g*cphi*cthe + q * ub - p * vb # = wdot
        grad3 = 1/Ixx * (L + (Iyy - Izz) * q * r)  # = pdot
        grad4 = 1/Iyy * (M + (Izz - Ixx) * p * r)  # = qdot
        grad5 = 1/Izz * (N + (Ixx - Iyy) * p * q)  # = rdot
        grad6 = p + (q*sphi + r*cphi) * sthe / cthe  # = phidot
        grad7 = q * cphi - r * sphi  # = thetadot
        grad8 = (q * sphi + r * cphi) / cthe  # = psidot
        grad9 = cthe*cpsi*ub + (-cphi*spsi + sphi*sthe*cpsi) * vb + \
            (sphi*spsi+cphi*sthe*cpsi) * wb  # = xEdot
        grad10 = cthe*spsi * ub + (cphi*cpsi+sphi*sthe*spsi) * vb + \
            (-sphi*cpsi+cphi*sthe*spsi) * wb # = yEdot
        grad11 = -1*(-sthe * ub + sphi*cthe * vb + cphi*cthe * wb) # = hEdot
                
        grads = [grad0,
                 grad1,
                 grad2,
                 grad3,
                 grad4,
                 grad5,
                 grad6,
                 grad7,
                 grad8,
                 grad9,
                 grad10,
                 grad11
                 ]

        return tf.concat(grads, 0)

    @staticmethod
    def controlForces(x):
        trim = 0.24525  # just enough force to keep the quadrocopter stable
        
        pitch_cmd = 0
        roll_cmd = 0
        climb_cmd = 0
        yaw_cmd=0
        
        climb_cmd = 0.01
        pitch_cmd = 0.0005
        roll_cmd = 0.0005
        
        u0 = trim + ( pitch_cmd + roll_cmd + climb_cmd - yaw_cmd) / 4
        u1 = trim + (-pitch_cmd - roll_cmd + climb_cmd - yaw_cmd) / 4
        u2 = trim + ( pitch_cmd - roll_cmd + climb_cmd + yaw_cmd) / 4
        u3 = trim + (-pitch_cmd + roll_cmd + climb_cmd + yaw_cmd) / 4
        return [u0, u1, u2, u3]
    
class TrainableGlucose(TrainableModel):
    """
    Trainable Lotka-Volterra model.
    """

    def _initialize_parameter_variables(self) -> None:
        """
        Initialize the TensorFlow variables containing the parameters theta of
        the ODE system. This will be 1D vector called 'self.theta',
        tensorflow.Variable type.
        """
        self.theta = tf.Variable(tf.abs(tf.random_normal([10, 1],
                                                         mean=0.0,
                                                         stddev=1.0,
                                                         dtype=tf.float64)),
                                 name='theta',
                                 trainable=True)
        return

    def compute_gradients(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute the gradients of the ODE, meaning f(X, self.theta).
        :param x: values of the time series observed, whose dimensions are
        [n_states, n_points].
        :return: TensorFlow Tensor containing the gradients, whose shape is
        [n_states, n_points].
        """
        p = self.theta
        grad0 = p[1]*x[5:6, :] - p[0]*x[0]*x[7:8, :]
        grad1 = p[3]*x[6:7, :] - p[2]*x[1:2, :]*x[8:9, :]
        grad2 = -p[7]*x[2:3, :] + p[6]*x[4:5, :]*x[8:9, :]
        grad3 = -p[5]*x[3:4, :] + p[4]*x[4:5, :]*x[6:7, :]
        grad4 = p[5]*x[3:4, :] + p[7]*x[2:3, :] - p[4]*x[4:5, :]*x[6:7, :] - p[6]*x[4:5, :]*x[8:9, :]
        grad5 = -p[1]*x[5:6, :] - p[8]*x[5:6, :] + p[8]*x[6:7, :] + p[0]*x[0:1, :]*x[7:8, :]
        grad6 = p[5]*x[3:4, :] - p[3]*x[6:7, :] + p[8]*x[5:6, :] - p[8]*x[6:7, :] \
              + p[2]*x[1:2, :]*x[8:9, :] - p[4]*x[4:5, :]*x[6:7, :]
        grad7 = p[1]*x[5:6, :] - p[9]*x[7:8, :] + p[9]*x[8:9, :] - p[0]*x[0:1, :]*x[7:8, :]
        grad8 = p[3]*x[6:7, :] + p[7]*x[2:3, :] + p[9]*x[7:8, :] - p[9]*x[8:9, :] \
              - p[2]*x[1:2, :]*x[8:9, :] - p[6]*x[4:5, :]*x[8:9, :]
        
        gradient_samples = tf.concat(
                [grad0, grad1, grad2, grad3, grad4, grad5, grad6, grad7, grad8],
                0)
        return gradient_samples
