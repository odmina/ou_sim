import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import expm

# You can have non-symetric positive definite B, because you are
# using parametrizatnion form oravecz2011hierarchical

# btw people have different covariance matices! And different prcesses!! OMG!
# compare it separately

"""
Possible predictors:
- simulated predictor
- mean predictors
- predictors of the shape of the underlying process - knowing these 
    predictors allow better sa prediction
"""

"""
- everyone has the same process but different means
    - sparse sampling vs intensive sampling
- processes differ and we do not know the predictors of who has which process
    - only intensive sampling, compare to same process 
- we know predictors of who gets which process and sample sparsely
- we know predictors of who gets which process and sample intensivelly
"""

"""
To get simulation you need:
- mean, assume 0, then allow variation (mu.
- stationary covariance matrix (GAMMA)
- drift matrix (B), informing the underlying process
"""


class OU_process_2v(object):
    """
    This class defines two variable Ornstein–Uhlenbeck process 
    and allows data simulation.


    Declaration of a variable of class OU_process_2v requires:
    `B` - centralizing (drift) matrix `B` links variables in
        the process, it is asserted to be a real matrix with
        with positive eigenvalues.
    `Gamma` - stationary covariance matrix of the OU process;
    `mu` - mean of the process, by default 0, but can be changed.
    
    The simulation uses discrete timesteps of equal length d
        and at each timestep samples X(t) from a conditional
        distibution of X(t) given X(t-d). The distrubution is
        normal with parameters calculated using eq (9) from
        Oravecz et al. 2011.

    Simulation runs for `total_time` units
        with 1/'d' samples per unit.
    The function uses a conditional distribution of X(t).

    Cholesky decomposition of instantaneus covariance (difussion)
        matrix appears in the stochastic term of the equation of X(t).
        For the simulation, the conditional distribution of X(t) 
        given X(t-d) is reparametrized so that it depends on B and Gamma.
        Thus the following condition has to be satisfied:
        Sigma = B*Gamma + Gamma*B.T 
        is a Hermitian, positive-definite marix.
        Since for the simulation only real valued matrices are used,
        Sigma has to be symetric.

    Initial value of the process X(0) is sampled from bivariate normal
        distribution with mean mu and covariance matrix Gamma.
    """

    def __init__(self, B, Gamma, mu=[0, 0]):
        self.B = np.array(B)
        self.Gamma = np.array(Gamma)
        self.mu = np.array(mu)
        # B is assumed to be 2x2, real and have positive eigenvalues
        assert np.all(np.isreal(self.B)), "B is not a real matrix"
        assert self.B.shape == (2, 2), "B is not 2x2 matrix"
        assert np.all(np.linalg.eigvals(self.B) >
                      0), "B has negative eigenvalues"
        # and Gamma are assumed to be 2x2, real, symmetric
        # and positive semi-definite
        # since it is symmertic, positive semidefiniteness is
        # assured when eigenvalues are >= 0 
        assert np.all(np.isreal(self.Gamma)), "Gamma is not a real matrix"
        assert self.Gamma.shape == (2, 2), "Gamma is not 2x2 matrix"
        assert np.all(self.Gamma == self.Gamma.T), "Gamma not symetric"
        assert np.all(np.linalg.eigvals(self.Gamma) >=
                      0), "Gamma has negative eigenvalues"
        # Sigma = B*Gamma + Gamma*B.T is positive-definite
        self.Sigma = np.matmul(self.B, self.Gamma) + np.matmul(self.Gamma, self.B.T)
        assert np.all(self.Sigma == self.Sigma.T) and \
            np.all(np.linalg.eigvals(self.Gamma) >= 0), \
            "Diffusion matrix not possitive-definite"
        # mu nas to be a vector of length 2
        assert self.mu.shape == (2,), "mu is not (2,) array"

    def __str__(self):
        return """Two variable Ornstein–Uhlenbeck process;
        `use get_B()`, `get_Gamma()` and 'get_mu()' to print parameters. """

    def get_B(self):
        return (self.B)

    def get_Gamma(self):
        return (self.Gamma)

    def get_Sigma(self):
        return (self.Sigma)

    def get_mu(self):
        return (self.mu)

    def get_sim_discreteB(self, d=0.1):
        """
        This method calculates the discrete time version of B.

        It uses timestep `d`, with default 0.1.
        """
        return expm(-self.B*d)

    def get_sim_condPDF_covariance(self, d=0.1):
        """
        Calculates the variance of the distribution of X(t) given X(t-d).

        Uses timestep `d`, with default 0.1. 

        The covariance is determined by the stochastic term of the 
        X(t) equation describing the process. 
        Cholesky decomposition of instantaneus covariance (difussion) 
        matrix appears in this equation. The conditional distribution 
        of X(t) given X(t-d) is reparametrized so that it depends
        on B and Gamma.See for example oravecz2011hierarchical eq (5).
        """
        cov = self.Gamma - np.matmul(
            np.matmul(expm(-self.B*d), self.Gamma), 
            expm(-self.B.T*d)
            )
        return cov

    def sim_data(self, d=0.1, total_time=10):
        """
        This function numerically simulates a two vairable Ornstein–Uhlenbeck process.
        Let vector X(t) be a vector of two variables at time t.
        Covariance matrix of the OU process is `Gamma` (and is stationary).
        Mean of the process `mu` is by default 0, but can be changed.
        The simulation uses discrete timesteps of equal length d
        and at each timestep samples X(t) from a conditional
        distibution of X(t) given X(t-d). The distrubution is
        normal with parameters calculated using eq (9) from
        Oravecz et al. 2011.
        Variables in X are linked by centralizing (drift) matrix `B` 

        Simulation runs for `total_time` units 
            with 1/'d' samples per unit.
    
        Initial value of the process is sampled from bivariate normal
            distribution with mean mu and covariance matrix Gamma.
        """

        rng = np.random.default_rng()

        #initialize data array
        n_datapoints = int(total_time / d)
        X = np.empty((2, n_datapoints))

        #compute parameters for the given d
        condPDF_cov = self.get_sim_condPDF_covariance(d = d)
        cond_B = self.get_sim_discreteB(d = d)

        #set process start X(t=0)
        X[:, 0] = rng.multivariate_normal(self.mu, self.Gamma)  # set process start
        for i in np.arange(n_datapoints - 1):
            this_mu = self.mu + cond_B*(X[:, i] - self.mu)
            X[:, i + 1] = rng.multivariate_normal(this_mu, condPDF_cov)
        return X

my_ou = OU_process_2v([[2, 0], [1, 2]], [[2, 0], [0, 2]])

