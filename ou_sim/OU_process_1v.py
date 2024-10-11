import numpy as np

class OU_process_1v(object):
    """
    This class defines one variable, stationary Ornstein–Uhlenbeck process
    and allows data simulation.

    Follows symbols from Gardiner (2009).

    The process can be initialized with any two out of the arguments.

    Args:
        `k`: linear drift term, has to be > 0.
        `D`: diffusion term, has to be > 0.
        `sigma`: stationary variance of the process, has to be > 0.

    During data simulation parameter `mu` may be provided.
        It is the stationary mean of the process, it is 0 by default.

    The simulation uses discrete timesteps of equal length `d`
        and at each timestep samples X(t) from a conditional
        distribution of X(t) given X(t-d).

    Simulation runs for `total_time` units
        with 1/'d' samples per unit.

    Initial value of the process X(0) is sampled from bivariate normal
        distribution with mean `mu` and covariance `sigma2`.
    """

    def __init__(self, **kwargs):
        assert len(kwargs) == 2, """You should provide exactly two arguments."""

        self.k = kwargs.get('k', None)
        self.D = kwargs.get('D', None)
        self.sigma = kwargs.get('sigma', None)

        if self.k is not None and self.D is not None and self.sigma is None:
            self.sigma = self.D/(2*self.k)
        elif self.k is not None and self.D is None and self.sigma is not None:
            self.D = self.sigma*2*self.k
        elif self.k is None and self.D is not None and self.sigma is not None:
            self.k = self.D/(2*self.sigma)

    def __str__(self):
        return """One variable Ornstein–Uhlenbeck process;
        `use get_k()`, `get_sigma()` and 'get_D()' to access parameters."""

    # ------ GETTERS ------

    def get_k(self):
        return self.k

    def get_sigma(self):
        return self.sigma

    def get_D(self):
        return self.D

    # ------ DISCRETE TIME SIMULATION METHODS ------

    def get_sim_discrete_k(self, d=0.1):
        """
        This method calculates the discrete time k.

        It uses timestep `d`, with default 0.1.
        """
        return np.exp(-self.k * d)

    def get_sim_condPDF_variance(self, d=0.1):
        """
        Calculates the variance of the distribution of X(t) given X(t-d).

        Uses timestep `d`, with default 0.1.

        The conditional distribution of X(t) given X(t-d) depends
        on k and sigma. For the proof of the formula see for example
        Gillespie, Daniel T. “Exact Numerical Simulation of
        the Ornstein-Uhlenbeck Process and Its Integral.” Physical Review E 54, no. 2 (August 1, 1996): 2084–91.
        https://doi.org/10.1103/PhysRevE.54.2084.

        """

        cov = self.sigma*(1-np.exp(-2*self.k*d))
        return cov

    def get_stationary_autocov(self, d=0.1):
        """Compute autocovariance for a given time step in a stationary state.

        Based on eq. 3.8.83 from Gardiner, C. (2009). Stochastic Methods.

        Args:
            d: timestep. Default 0.1.

        Returns: autocovariance for the given process and time.
        """

        auto_covariance = self.sigma*np.exp(-self.k*d)

        return auto_covariance

    def get_stationary_autocorrelation(self, d=0.1):
        """Compute autocorrelation for a given time step in a stationary state.
        This is equivalent to getting simulation discrete k.

        Args:
            d: timestep. Default 0.1.

        Returns: time correlation for the given process and t.
        """

        return self.get_sim_discrete_k(d=d)

    def sim_data(self, d=0.1, total_time=10, mu=0, initial_value=None):
        """
        This function simulates a one variable Ornstein–Uhlenbeck process.

        Args:
            d: timestep. Default 0.1.
            total_time: total simulation time. Default 10.
            mu: stationary mean. Default 0.
            initial_value: initial value of the process. If None (default),
            it is sampled from a normal distribution with mean mu and stationary
            variance of the process.

        Returns: numpy array containing simulated data
        with length equal to len(np.arange(0, total_time, d)).

        Let X(t) be a state of the process at time t.
        Each X(t) is a gaussian with mean `mu` and variance `sigma`.

        The simulation uses discrete timesteps of equal length d
        and at each timestep calculates X(t) = X(t-d) + samples X(t) from a conditional
        distribution of X(t) given X(t-d). Sampling distribution is
        normal with parameters calculated using eq. 3.5a from Gilesspie (1996).

        Simulation runs for `total_time` units
            with 1/'d' samples per unit.

        Important: initial value of the process is sampled from bivariate normal
            distribution with mean mu and covariance matrix sigma.
        """

        rng = np.random.default_rng()

        # initialize data array
        n_datapoints = len(np.arange(0, total_time, d))
        X = np.empty((n_datapoints))

        # compute parameters for the given d
        condPDF_var = self.get_sim_condPDF_variance(d=d)
        condPDF_sd = np.sqrt(condPDF_var)
        cond_k = self.get_sim_discrete_k(d=d)

        # set process start X(t=0)
        if initial_value is None:
            X[0] = rng.normal(mu, self.sigma)
        else:
            X[0] = initial_value

        # simulate data
        for i in np.arange(n_datapoints - 1):
            X[i + 1] = mu + (X[i] - mu)*cond_k + rng.normal(0, condPDF_sd)

        return X
