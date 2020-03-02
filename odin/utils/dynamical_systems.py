"""
Dynamical Systems needed to create the data for the experiments.

Gabriele Abbati, Machine Learning Research Group, University of Oxford
February 2019
"""


# Libraries
from abc import ABC, abstractmethod
import numpy as np
from typing import Union, Tuple
from scipy.integrate import ode
from scipy.optimize import fsolve


class DynamicalSystem(ABC):
    """
    Abstract class for a dynamical system. Includes an ODE solver based on
    scipy.
    """

    def __init__(self, dimensionality: int,
                 true_param: Union[list, np.array],
                 noise_variance: float = 0.0,
                 stn_ratio: float = None):
        """
        General Constructor.
        :param dimensionality: dimension of the state of the system;
        :param true_param: true parameters of the system;
        :param noise_variance: variance of the observation noise, if different
        from zero it overwrites the signal to noise ratio;
        :param stn_ratio: signal to noise ratio (variance should be set to zero
        if the stn_ratio is different than None).
        """
        self.dim = dimensionality
        self.theta = np.array(true_param)
        self.mean = 0.0
        self.variance = noise_variance
        self.system_ode = ode(self._system_ode).set_integrator('vode',
                                                               method='bdf')
        self.stn_ratio = stn_ratio
        return

    @staticmethod
    @abstractmethod
    def _system_ode(t: float, y: np.array,
                    theta: np.array) -> list:
        """
        Describes the overall evolution of the system in the form:
                dy / dt = f( t, y, args)
        Needed by scipy.
        :param t: time, needed in arguments even if it's not directly used;
        :param y: current state;
        :param theta: arguments and parameters of the system.
        :return: the f function so built.
        """
        return []

    def simulate(self, initial_state: Union[list, np.array],
                 initial_time: float, final_time: float,
                 t_delta_integration: float) -> Tuple[np.array, np.array]:
        """
        Integrate the system using an scipy built-in ODE solver.
        :param initial_state: initial state of the system;
        :param initial_time: initial time of the simulation;
        :param final_time: final time of the simulation;
        :param t_delta_integration: time between integration intervals.
        :return: a numpy array containing the integrated dynamical system (of
        size [n_states, n_points]) and a numpy array containing the time stamps.
        """
        system = np.copy(initial_state).reshape(self.dim, 1)
        t = [initial_time]
        self.system_ode.set_initial_value(initial_state,
                                          initial_time).set_f_params(self.theta)
        while self.system_ode.successful() and self.system_ode.t < final_time:
            self.system_ode.integrate(self.system_ode.t + t_delta_integration)
            system = np.c_[system, self.system_ode.y.reshape(self.dim, 1)]
            t.append(self.system_ode.t)
        return system, np.array(t)

    def observe(self, initial_state: Union[list, np.array],
                initial_time: float, final_time: float,
                t_delta_integration: float,
                t_delta_observation: float) -> Tuple[np.array, np.array]:
        """
        Integrate the system using an scipy built-in ODE solver and extract the
        noisy observations.
        :param initial_state: initial state of the system;
        :param initial_time: initial time of the simulation;
        :param final_time: final time of the simulation;
        :param t_delta_integration: time between integration intervals;
        :param t_delta_observation: time between observation intervals.
        :return: a numpy array containing the noisy observations of the
        integrated dynamical system (of size [n_states, n_points])
        and a numpy array containing the time stamps.
        """
        [system, t] = self.simulate(initial_state,
                                    initial_time,
                                    final_time,
                                    t_delta_integration)
        t_obs = np.arange(initial_time, final_time + t_delta_observation,
                          t_delta_observation)
        observed_system = np.zeros([self.dim, t_obs.shape[0]])
        for n in range(self.dim):
            observed_system[n, :] = np.interp(t_obs, t, system[n, :])
        if self.variance != 0.0:
            noise = np.random.normal(loc=0.0, scale=np.sqrt(self.variance),
                                     size=observed_system.shape)
            observed_system += noise.reshape(observed_system.shape)
        if self.stn_ratio:
            std_devs_signal = np.std(observed_system, axis=1)
            std_devs_noise = std_devs_signal / np.sqrt(self.stn_ratio)
            noise = np.random.normal(loc=0.0, scale=1.0,
                                     size=observed_system.shape)
            for n in range(self.dim):
                noise[n, :] = noise[n, :] * std_devs_noise[n]
            observed_system += noise.reshape(observed_system.shape)
        return observed_system, t_obs.reshape(-1, 1)

    def observe_at_t(self, initial_state: Union[list, np.array],
                     initial_time: float, final_time: float,
                     t_delta_integration: float,
                     t_observations: np.array):
        """"
        Integrate the system using an scipy built-in ODE solver and extract the
        noisy observations, computed at the time stamps specified in
        t_observations.
        :param initial_state: initial state of the system;
        :param initial_time: initial time of the simulation;
        :param final_time: final time of the simulation;
        :param t_delta_integration: time between integration intervals;
        :param t_observations: time stamps at which observe the system.
        :return: a numpy array containing the noisy observations of the
        integrated dynamical system (of size [n_states, n_points])
        and a numpy array containing the time stamps.
        """
        [system, t] = self.simulate(initial_state,
                                    initial_time,
                                    final_time,
                                    t_delta_integration)
        t_obs = t_observations
        observed_system = np.zeros([self.dim, t_obs.shape[0]])
        for n in range(self.dim):
            observed_system[n, :] = np.interp(t_obs, t, system[n, :])
        if self.variance != 0.0:
            noise = np.random.normal(loc=0.0, scale=np.sqrt(self.variance),
                                     size=observed_system.shape)
            observed_system += noise.reshape(observed_system.shape)
        if self.stn_ratio:
            std_devs_signal = np.std(observed_system, axis=1)
            std_devs_noise = std_devs_signal / np.sqrt(self.stn_ratio)
            noise = np.random.normal(loc=0.0, scale=1.0,
                                     size=observed_system.shape)
            for n in range(self.dim):
                noise[n, :] = noise[n, :] * std_devs_noise[n]
            observed_system += noise.reshape(observed_system.shape)
        return observed_system, t_obs.reshape(-1, 1)


class LotkaVolterra(DynamicalSystem):
    """
    2D Lotka-Volterra ODE.
    """

    def __init__(self,
                 true_param: Union[list, np.array] = (2.0, 1.0, 4.0, 1.0),
                 noise_variance: float = 0.1 ** 2,
                 stn_ratio: float = None):
        """
        Constructor.
        :param true_param: true parameters of the system;
        :param noise_variance: variance of the observation noise, if different
        from zero it overwrites the signal to noise ratio;
        :param stn_ratio: signal to noise ratio (variance should be set to zero
        if the stn_ratio is different than None).
        """
        super(LotkaVolterra, self).__init__(2,
                                            true_param,
                                            noise_variance,
                                            stn_ratio)
        assert self.theta.shape[0] == 4,\
            "Error: length of true_param should be 4"
        return

    @staticmethod
    def _system_ode(t: float, y: np.array,
                    theta: np.array) -> list:
        """
        Describes the overall evolution of the system in the form:
                dy / dt = f( t, y, args)
        Needed by scipy.
        :param t: time, needed in arguments even if it's not directly used;
        :param y: current state;
        :param theta: arguments and parameters of the system.
        :return: the f function so built.
        """
        f = [theta[0] * y[0] - theta[1] * y[0] * y[1],
             - theta[2] * y[1] + theta[3] * y[0] * y[1]]
        return f

    def simulate(self,
                 initial_state: Union[list, np.array] = (5.0, 3.0),
                 initial_time: float = 0.0,
                 final_time: float = 2.0,
                 t_delta_integration: float = 0.01)\
            -> Tuple[np.array, np.array]:
        """
        Integrate the system using an scipy built-in ODE solver.
        :param initial_state: initial state of the system;
        :param initial_time: initial time of the simulation;
        :param final_time: final time of the simulation;
        :param t_delta_integration: time between integration intervals.
        :return: a numpy array containing the integrated dynamical system (of
        size [n_states, n_points]) and a numpy array containing the time stamps.
        """
        system, t = super(LotkaVolterra, self).simulate(initial_state,
                                                        initial_time,
                                                        final_time,
                                                        t_delta_integration)
        return system, t

    def observe(self, initial_state: Union[list, np.array] = (5.0, 3.0),
                initial_time: float = 0.0,
                final_time: float = 2.0,
                t_delta_integration: float = 0.01,
                t_delta_observation: float = 0.1) -> Tuple[np.array, np.array]:
        """
        Integrate the system using an scipy built-in ODE solver and extract the
        noisy observations.
        :param initial_state: initial state of the system;
        :param initial_time: initial time of the simulation;
        :param final_time: final time of the simulation;
        :param t_delta_integration: time between integration intervals;
        :param t_delta_observation: time between observation intervals.
        :return: a numpy array containing the noisy observations of the
        integrated dynamical system (of size [n_states, n_points])
        and a numpy array containing the time stamps.
        """
        observed_system, t = super(LotkaVolterra,
                                   self).observe(initial_state,
                                                 initial_time,
                                                 final_time,
                                                 t_delta_integration,
                                                 t_delta_observation)
        return observed_system, t


class FitzHughNagumo(DynamicalSystem):
    """
    2D FitzHugh-Nagumo ODE.
    """

    def __init__(self,
                 true_param: Union[list, np.array] = (0.2, 0.2, 3.0),
                 noise_variance: float = 0.0,
                 stn_ratio: float = None):
        """
        Constructor.
        :param true_param: true parameters of the system;
        :param noise_variance: variance of the observation noise, if different
        from zero it overwrites the signal to noise ratio;
        :param stn_ratio: signal to noise ratio (variance should be set to zero
        if the stn_ratio is different than None).
        """
        super(FitzHughNagumo, self).__init__(2,
                                             true_param,
                                             noise_variance,
                                             stn_ratio)
        assert self.theta.shape[0] == 3,\
            "Error: length of true_param should be 3"
        return

    @staticmethod
    def _system_ode(t: float, y: np.array,
                    theta: np.array) -> list:
        """
        Describes the overall evolution of the system in the form:
                dy / dt = f( t, y, args)
        Needed by scipy.
        :param t: time, needed in arguments even if it's not directly used;
        :param y: current state;
        :param theta: arguments and parameters of the system.
        :return: the f function so built.
        """
        f = [theta[2] * (y[0] - y[0]**3 / 3.0 + y[1]),
             - 1.0 / theta[2] * (y[0] - theta[0] + theta[1] * y[1])]
        return f

    def simulate(self,
                 initial_state: Union[list, np.array] = (-1.0, 1.0),
                 initial_time: float = 0.0,
                 final_time: float = 20.0,
                 t_delta_integration: float = 0.01)\
            -> Tuple[np.array, np.array]:
        """
        Integrate the system using an scipy built-in ODE solver.
        :param initial_state: initial state of the system;
        :param initial_time: initial time of the simulation;
        :param final_time: final time of the simulation;
        :param t_delta_integration: time between integration intervals.
        :return: a numpy array containing the integrated dynamical system (of
        size [n_states, n_points]) and a numpy array containing the time stamps.
        """
        system, t = super(FitzHughNagumo, self).simulate(initial_state,
                                                         initial_time,
                                                         final_time,
                                                         t_delta_integration)
        return system, t

    def observe(self,
                initial_state: Union[list, np.array] = (-1.0, 1.0),
                initial_time: float = 0.0,
                final_time: float = 20.0,
                t_delta_integration: float = 0.01,
                t_delta_observation: float = 0.5) -> Tuple[np.array, np.array]:
        """
        Integrate the system using an scipy built-in ODE solver and extract the
        noisy observations.
        :param initial_state: initial state of the system;
        :param initial_time: initial time of the simulation;
        :param final_time: final time of the simulation;
        :param t_delta_integration: time between integration intervals;
        :param t_delta_observation: time between observation intervals.
        :return: a numpy array containing the noisy observations of the
        integrated dynamical system (of size [n_states, n_points])
        and a numpy array containing the time stamps.
        """
        observed_system, t = super(FitzHughNagumo,
                                   self).observe(initial_state,
                                                 initial_time,
                                                 final_time,
                                                 t_delta_integration,
                                                 t_delta_observation)
        return observed_system, t


class ProteinTransduction(DynamicalSystem):
    """
    Protein transduction system, Vyshemirsky and Girolami, 2008. With coherent
    notation, y = [S, dS, R, R_s, R_pp]
    """

    def __init__(self,
                 true_param: np.array = np.array([0.07, 0.6, 0.05, 0.3,
                                                 0.017, 0.3]),
                 noise_variance: float = 0.001**2,
                 stn_ratio: float = 0.0):
        """
        Constructor.
        :param true_param: true parameters of the system;
        :param noise_variance: variance of the observation noise, if different
        from zero it overwrites the signal to noise ratio;
        :param stn_ratio: signal to noise ratio (variance should be set to zero
        if the stn_ratio is different than None).
        """
        super(ProteinTransduction, self).__init__(5,
                                                  true_param,
                                                  noise_variance,
                                                  stn_ratio)
        assert self.theta.shape[0] == 6,\
            "Error: length of true_param should be 6"
        return

    @staticmethod
    def _system_ode(t: float, y: np.array,
                    theta: np.array) -> list:
        """
        Describes the overall evolution of the system in the form:
                dy / dt = f( t, y, args)
        Needed by scipy.
        :param t: time, needed in arguments even if it's not directly used;
        :param y: current state;
        :param theta: arguments and parameters of the system.
        :return: the f function so built.
        """
        f = [- theta[0] * y[0] - theta[1] * y[0] * y[2] + theta[2] * y[3],
             theta[0] * y[0],
             - theta[1] * y[0] * y[2] + theta[2] * y[3] +
             theta[4] * y[4] / (theta[5] + y[4]),
             theta[1] * y[0] * y[2] - theta[2] * y[3] - theta[3] * y[3],
             theta[3] * y[3] - theta[4] * y[4] / (theta[5] + y[4])]
        return f

    def simulate(self, initial_state: Union[list, np.array] = (1.0, 0.0, 1.0,
                                                               0.0, 0.0),
                 initial_time: float = 0.0,
                 final_time: float = 100.0,
                 t_delta_integration: float = 0.1) -> Tuple[np.array, np.array]:
        """
        Integrate the system using an scipy built-in ODE solver.
        :param initial_state: initial state of the system;
        :param initial_time: initial time of the simulation;
        :param final_time: final time of the simulation;
        :param t_delta_integration: time between integration intervals.
        :return: a numpy array containing the integrated dynamical system (of
        size [n_states, n_points]) and a numpy array containing the time stamps.
        """
        system, t = super(ProteinTransduction,
                          self).simulate(initial_state,
                                         initial_time,
                                         final_time,
                                         t_delta_integration)
        return system, t

    def observe(self, initial_state: Union[list, np.array] = (1.0, 0.0, 1.0,
                                                              0.0, 0.0),
                initial_time: float = 0.0,
                final_time: float = 100.0,
                t_delta_integration: float = 0.1,
                t_delta_observation: float = 10.0) -> Tuple[np.array, np.array]:
        """
        Integrate the system using an scipy built-in ODE solver and extract the
        noisy observations.
        :param initial_state: initial state of the system;
        :param initial_time: initial time of the simulation;
        :param final_time: final time of the simulation;
        :param t_delta_integration: time between integration intervals;
        :param t_delta_observation: time between observation intervals.
        :return: a numpy array containing the noisy observations of the
        integrated dynamical system (of size [n_states, n_points])
        and a numpy array containing the time stamps.
        """
        observed_system, t = super(ProteinTransduction,
                                   self).observe(initial_state,
                                                 initial_time,
                                                 final_time,
                                                 t_delta_integration,
                                                 t_delta_observation)
        return observed_system, t


class Lorenz96(DynamicalSystem):
    """
    Lorenz 96 Attractor, n_states different ODEs.
    """

    def __init__(self,
                 n_states: int = 40,
                 true_param: float = 8.0,
                 noise_variance: float = 1.0,
                 stn_ratio: float = None):
        """
        Constructor.
        :param true_param: true parameters of the system;
        :param noise_variance: variance of the observation noise, if different
        from zero it overwrites the signal to noise ratio;
        :param stn_ratio: signal to noise ratio (variance should be set to zero
        if the stn_ratio is different than None).
        """
        self.n_states = n_states
        super(Lorenz96, self).__init__(n_states,
                                       true_param,
                                       noise_variance,
                                       stn_ratio)
        return

    @staticmethod
    def _system_ode(t: float, y: np.array,
                    theta: np.array) -> list:
        """
        Describes the overall evolution of the system in the form:
                dy / dt = f( t, y, args)
        Needed by scipy.
        :param t: time, needed in arguments even if it's not directly used;
        :param y: current state;
        :param theta: arguments and parameters of the system.
        :return: the f function so built.
        """
        n_states = y.shape[0]
        f = [(y[1] - y[n_states - 2]) * y[n_states - 1] - y[0] + theta,
             (y[2] - y[n_states - 1]) * y[0] - y[1] + theta]
        for n in range(2, n_states - 1):
            state_derivative = (y[n + 1] - y[n - 2]) * y[n - 1] - y[n] + theta
            f.append(state_derivative)
        state_derivative = (y[0] - y[n_states - 3])\
            * y[n_states - 2] - y[n_states - 1] + theta
        f.append(state_derivative)
        return f

    def simulate(self, initial_state: Union[list, np.array] = 1.0,
                 initial_time: float = 0.0,
                 final_time: float = 4.0,
                 t_delta_integration: float = 0.01)\
            -> Tuple[np.array, np.array]:
        """
        Integrate the system using an scipy built-in ODE solver.
        :param initial_state: initial state of the system;
        :param initial_time: initial time of the simulation;
        :param final_time: final time of the simulation;
        :param t_delta_integration: time between integration intervals.
        :return: a numpy array containing the integrated dynamical system (of
        size [n_states, n_points]) and a numpy array containing the time stamps.
        """
        initial_state_vector = initial_state * np.ones(self.n_states)\
            + np.random.normal(0.0, 0.01, self.n_states)
        system, t = super(Lorenz96, self).simulate(initial_state_vector,
                                                   initial_time,
                                                   final_time,
                                                   t_delta_integration)
        return system, t

    def observe(self, initial_state: Union[list, np.array] = 1.0,
                initial_time: float = 0.0,
                final_time: float = 20.0,
                t_delta_integration: float = 0.01,
                t_delta_observation: float = 0.2) -> Tuple[np.array, np.array]:
        """
        Integrate the system using an scipy built-in ODE solver and extract the
        noisy observations.
        :param initial_state: initial state of the system;
        :param initial_time: initial time of the simulation;
        :param final_time: final time of the simulation;
        :param t_delta_integration: time between integration intervals;
        :param t_delta_observation: time between observation intervals.
        :return: a numpy array containing the noisy observations of the
        integrated dynamical system (of size [n_states, n_points])
        and a numpy array containing the time stamps.
        """
        observed_system, t = super(Lorenz96,
                                   self).observe(initial_state,
                                                 initial_time,
                                                 final_time,
                                                 t_delta_integration,
                                                 t_delta_observation)
        return observed_system, t

class Lorenz63(DynamicalSystem):
    """
    2D FitzHugh-Nagumo ODE.
    """

    def __init__(self,
                 true_param: Union[list, np.array] = (10.0, 28.0, 8.0/3.0),
                 noise_variance: float = 0.0,
                 stn_ratio: float = None):
        """
        Constructor.
        :param true_param: true parameters of the system;
        :param noise_variance: variance of the observation noise, if different
        from zero it overwrites the signal to noise ratio;
        :param stn_ratio: signal to noise ratio (variance should be set to zero
        if the stn_ratio is different than None).
        """
        super(Lorenz63, self).__init__(3,
                                       true_param,
                                       noise_variance,
                                       stn_ratio)
        assert self.theta.shape[0] == 3,\
            "Error: length of true_param should be 3"
        return

    @staticmethod
    def _system_ode(t: float, y: np.array,
                    theta: np.array) -> list:
        """
        Describes the overall evolution of the system in the form:
                dy / dt = f( t, y, args)
        Needed by scipy.
        :param t: time, needed in arguments even if it's not directly used;
        :param y: current state;
        :param theta: arguments and parameters of the system.
        :return: the f function so built.
        """
        f = [theta[0]*(y[1] - y[0]),
             y[0]*(theta[1] - y[2]) - y[1],
             y[0]*y[1] - theta[2]*y[2]]
        return f

    def simulate(self,
                 initial_state: Union[list, np.array] = (1.0, 1.0, 1.0),
                 initial_time: float = 0.0,
                 final_time: float = 10.0,
                 t_delta_integration: float = 0.01)\
            -> Tuple[np.array, np.array]:
        """
        Integrate the system using an scipy built-in ODE solver.
        :param initial_state: initial state of the system;
        :param initial_time: initial time of the simulation;
        :param final_time: final time of the simulation;
        :param t_delta_integration: time between integration intervals.
        :return: a numpy array containing the integrated dynamical system (of
        size [n_states, n_points]) and a numpy array containing the time stamps.
        """
        system, t = super(Lorenz63, self).simulate(initial_state,
                                                         initial_time,
                                                         final_time,
                                                         t_delta_integration)
        return system, t

    def observe(self,
                initial_state: Union[list, np.array] = (1.0, 1.0, 1.0),
                initial_time: float = 0.0,
                final_time: float = 10.0,
                t_delta_integration: float = 0.01,
                t_delta_observation: float = 0.5) -> Tuple[np.array, np.array]:
        """
        Integrate the system using an scipy built-in ODE solver and extract the
        noisy observations.
        :param initial_state: initial state of the system;
        :param initial_time: initial time of the simulation;
        :param final_time: final time of the simulation;
        :param t_delta_integration: time between integration intervals;
        :param t_delta_observation: time between observation intervals.
        :return: a numpy array containing the noisy observations of the
        integrated dynamical system (of size [n_states, n_points])
        and a numpy array containing the time stamps.
        """
        observed_system, t = super(Lorenz63,
                                   self).observe(initial_state,
                                                 initial_time,
                                                 final_time,
                                                 t_delta_integration,
                                                 t_delta_observation)
        return observed_system, t
    
class Quadrocopter(DynamicalSystem):
    """
    m = 0.1         #kg
    Ixx = 0.00062   #kg-m^2
    Iyy = 0.00113   #kg-m^2
    Izz = 0.9*(Ixx + Iyy) #kg-m^2 (Assume nearly flat object, z=0)
    dx = 0.114      #m
    dy = 0.0825     #m
    g = 9.81  #m/s/s
    DTR = 1/57.3; RTD = 57.3
    
    code and parameters based on https://github.com/charlestytler/QuadcopterSim
    """

    def __init__(self,
                 true_param: Union[list, np.array] = \
                     (0.1, 0.62, 1.13, 0.9, 0.114, 8.25, 9.85),
                 noise_variance: float = 0.0,
                 stn_ratio: float = 100):
        """
        Constructor.
        :param true_param: true parameters of the system;
        :param noise_variance: variance of the observation noise, if different
        from zero it overwrites the signal to noise ratio;
        :param stn_ratio: signal to noise ratio (variance should be set to zero
        if the stn_ratio is different than None).
        """
        super(Quadrocopter, self).__init__(12,
                                           true_param,
                                           noise_variance,
                                           stn_ratio)
        assert self.theta.shape[0] == 7,\
            "Error: length of true_param should be 7"
        return
    
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
        
        u = np.zeros(4)
        u[0] = trim + ( pitch_cmd + roll_cmd + climb_cmd - yaw_cmd) / 4
        u[1] = trim + (-pitch_cmd - roll_cmd + climb_cmd - yaw_cmd) / 4
        u[2] = trim + ( pitch_cmd - roll_cmd + climb_cmd + yaw_cmd) / 4
        u[3] = trim + (-pitch_cmd + roll_cmd + climb_cmd + yaw_cmd) / 4
        return u
    
    @staticmethod
    def _system_ode(t: float, y: np.array,
                    theta: np.array) -> list:
        """
        Describes the overall evolution of the system in the form:
                dy / dt = f( t, y, args)
        Needed by scipy.
        :param t: time, needed to calculate inputs;
        :param y: current state;
        :param theta: arguments and parameters of the system.
        :return: the f function so built.
        """
        ub = y[0]
        vb = y[1]
        wb = y[2]
        p = y[3]
        q = y[4]
        r = y[5]
        phi = y[6]
        theta_sys = y[7]
        psi = y[8]
        xE = y[9]
        yE = y[10]
        hE = y[11]
        
        m = theta[0]         #kg
        Ixx = theta[1]*1e-3   #kg-m^2
        Iyy = theta[2]*1e-3   #kg-m^2
        Izz = theta[3]*(Ixx + Iyy) #kg-m^2 (Assume nearly flat object, z=0)
        dx = theta[4]      #m
        dy = theta[5]*1e-2     #m
        g = theta[6]  #m/s/s

        # Directly get forces as inputs
        [F1, F2, F3, F4] = Quadrocopter.controlForces(y)
        Fz = F1 + F2 + F3 + F4
        L = (F2 + F3) * dy - (F1 + F4) * dy
        M = (F1 + F3) * dx - (F2 + F4) * dx
        N = 0 #-T(F1,dx,dy) - T(F2,dx,dy) + T(F3,dx,dy) + T(F4,dx,dy)
        
        # Pre-calculate trig values
        cphi = np.cos(phi);   sphi = np.sin(phi)
        cthe = np.cos(theta_sys); sthe = np.sin(theta_sys)
        cpsi = np.cos(psi);   spsi = np.sin(psi)
        
        # Calculate the derivative of the state matrix using EOM
        xdot = np.zeros(12)
        
        xdot[0] = -g * sthe + r * vb - q * wb  # = udot
        xdot[1] = g * sphi*cthe - r * ub + p * wb # = vdot
        xdot[2] = 1/m * (-Fz) + g*cphi*cthe + q * ub - p * vb # = wdot
        xdot[3] = 1/Ixx * (L + (Iyy - Izz) * q * r)  # = pdot
        xdot[4] = 1/Iyy * (M + (Izz - Ixx) * p * r)  # = qdot
        xdot[5] = 1/Izz * (N + (Ixx - Iyy) * p * q)  # = rdot
        xdot[6] = p + (q*sphi + r*cphi) * sthe / cthe  # = phidot
        xdot[7] = q * cphi - r * sphi  # = thetadot
        xdot[8] = (q * sphi + r * cphi) / cthe  # = psidot
        
        xdot[9] = cthe*cpsi*ub + (-cphi*spsi + sphi*sthe*cpsi) * vb + \
            (sphi*spsi+cphi*sthe*cpsi) * wb  # = xEdot
            
        xdot[10] = cthe*spsi * ub + (cphi*cpsi+sphi*sthe*spsi) * vb + \
            (-sphi*cpsi+cphi*sthe*spsi) * wb # = yEdot
            
        xdot[11] = -1*(-sthe * ub + sphi*cthe * vb + cphi*cthe * wb) # = hEdot
                
        f = xdot
        return f

    def simulate(self,
                 initial_state: Union[list, np.array] = (0,0,0,0,0,0,0,0,0,0,0,0),
                 initial_time: float = 0.0,
                 final_time: float = 30.0,
                 t_delta_integration: float = 0.01)\
            -> Tuple[np.array, np.array]:
        """
        Integrate the system using an scipy built-in ODE solver.
        :param initial_state: initial state of the system;
        :param initial_time: initial time of the simulation;
        :param final_time: final time of the simulation;
        :param t_delta_integration: time between integration intervals.
        :return: a numpy array containing the integrated dynamical system (of
        size [n_states, n_points]) and a numpy array containing the time stamps.
        """
        system, t = super(Quadrocopter, self).simulate(initial_state,
                                                       initial_time,
                                                       final_time,
                                                       t_delta_integration)
        return system, t

    def observe(self, initial_state: Union[list, np.array] = (0,0,0,0,0,0,0,0,0,0,0,0),
                initial_time: float = 0.0,
                final_time: float = 30.0,
                t_delta_integration: float = 0.01,
                t_delta_observation: float = 0.1) -> Tuple[np.array, np.array]:
        """
        Integrate the system using an scipy built-in ODE solver and extract the
        noisy observations.
        :param initial_state: initial state of the system;
        :param initial_time: initial time of the simulation;
        :param final_time: final time of the simulation;
        :param t_delta_integration: time between integration intervals;
        :param t_delta_observation: time between observation intervals.
        :return: a numpy array containing the noisy observations of the
        integrated dynamical system (of size [n_states, n_points])
        and a numpy array containing the time stamps.
        """
        observed_system, t = super(Quadrocopter,
                                   self).observe(initial_state,
                                                 initial_time,
                                                 final_time,
                                                 t_delta_integration,
                                                 t_delta_observation)
        return observed_system, t


class Glucose(DynamicalSystem):
    """
    2D Lotka-Volterra ODE.
    """

    def __init__(self,
                 true_param: Union[list, np.array] = (0.1, 0.0, 0.4, 0.0, 0.3, 0.0, 0.7, 0.0, 0.1, 0.2),
                 noise_variance: float = 0.1 ** 2,
                 stn_ratio: float = None):
        """
        Constructor.
        :param true_param: true parameters of the system;
        :param noise_variance: variance of the observation noise, if different
        from zero it overwrites the signal to noise ratio;
        :param stn_ratio: signal to noise ratio (variance should be set to zero
        if the stn_ratio is different than None).
        """
        super(Glucose, self).__init__(9,
                                            true_param,
                                            noise_variance,
                                            stn_ratio)
        assert self.theta.shape[0] == 10,\
            "Error: length of true_param should be 10"
        return

    @staticmethod
    def _system_ode(t: float, y: np.array,
                    theta: np.array) -> list:
        """
        Describes the overall evolution of the system in the form:
                dy / dt = f( t, y, args)
        Needed by scipy.
        :param t: time, needed in arguments even if it's not directly used;
        :param y: current state;
        :param theta: arguments and parameters of the system.
        :return: the f function so built.
        """
        p = theta
        x = y
        f = np.zeros(9)
        f[0]= p[1]*x[5] - p[0]*x[0]*x[7]
        f[1]= p[3]*x[6] - p[2]*x[1]*x[8]
        f[2]= -p[7]*x[2] + p[6]*x[4]*x[8]
        f[3]= -p[5]*x[3] + p[4]*x[4]*x[6]
        f[4]= p[5]*x[3] + p[7]*x[2] - p[4]*x[4]*x[6] - p[6]*x[4]*x[8]
        f[5]= -p[1]*x[5] - p[8]*x[5] + p[8]*x[6] + p[0]*x[0]*x[7]
        f[6]= p[5]*x[3] - p[3]*x[6] + p[8]*x[5] - p[8]*x[6] + p[2]*x[1]*x[8] - p[4]*x[4]*x[6]
        f[7]= p[1]*x[5] - p[9]*x[7] + p[9]*x[8] - p[0]*x[0]*x[7]
        f[8]= p[3]*x[6] + p[7]*x[2] + p[9]*x[7] - p[9]*x[8] - p[2]*x[1]*x[8] - p[6]*x[4]*x[8]
        return f

    def simulate(self,
                 initial_state: Union[list, np.array] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                 initial_time: float = 0.0,
                 final_time: float = 100.0,
                 t_delta_integration: float = 0.01)\
            -> Tuple[np.array, np.array]:
        """
        Integrate the system using an scipy built-in ODE solver.
        :param initial_state: initial state of the system;
        :param initial_time: initial time of the simulation;
        :param final_time: final time of the simulation;
        :param t_delta_integration: time between integration intervals.
        :return: a numpy array containing the integrated dynamical system (of
        size [n_states, n_points]) and a numpy array containing the time stamps.
        """
        system, t = super(Glucose, self).simulate(initial_state,
                                                        initial_time,
                                                        final_time,
                                                        t_delta_integration)
        return system, t

    def observe(self, initial_state: Union[list, np.array] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                initial_time: float = 0.0,
                final_time: float = 100.0,
                t_delta_integration: float = 0.01,
                t_delta_observation: float = 0.1) -> Tuple[np.array, np.array]:
        """
        Integrate the system using an scipy built-in ODE solver and extract the
        noisy observations.
        :param initial_state: initial state of the system;
        :param initial_time: initial time of the simulation;
        :param final_time: final time of the simulation;
        :param t_delta_integration: time between integration intervals;
        :param t_delta_observation: time between observation intervals.
        :return: a numpy array containing the noisy observations of the
        integrated dynamical system (of size [n_states, n_points])
        and a numpy array containing the time stamps.
        """
        observed_system, t = super(Glucose,
                                   self).observe(initial_state,
                                                 initial_time,
                                                 final_time,
                                                 t_delta_integration,
                                                 t_delta_observation)
        return observed_system, t