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
        the process, it is asserted to be positive definite
        and real.
    `Gamma` - stationary covariance matrix of the OU process;
    `mu` - mean of the process, by default 0, but can be changed.
    
    The simulation uses discrete timesteps of equal length d
        and at each timestep samples X(t) from a conditional
        distibution of X(t) given X(t-d). The distrubution is
        normal with parameters calculated using eq (9) from
        Oravecz et al. 2011.

    Simulation runs for `total_time` units
        with 1/'d' samples per unit.
    The function uses a conditional distribution of X(t)
    Cholesky decomposition of instantaneus covariance (difussion)
        matrix appears in the stochastic term of the equation of X(t).
        The conditional distribution of X(t) given X(t-d) can
        be reparametrized so that it depends on B and Gamma.
    Initial value of the process X(0) is sampled from bivariate normal
        distribution with mean mu and covariance matrix Gamma.
    """

    def __init__(self, B, Gamma, mu=[0, 0]):
        self.B = np.array(B)
        self.Gamma = np.array(Gamma)
        self.mu = np.array(mu)
        # B and Gamma are assumed to be real
        assert np.all(np.isreal(self.B)), "B is not a real matrix"
        assert np.all(np.isreal(self.Gamma)), "Gamma is not a real matrix"
        # B has to be 2x2 real matrix with positive eigenvalues
        assert self.B.shape == (2, 2) and self.Gamma.shape == (
            2, 2), "not 2x2 matrix"
        assert np.all(np.linalg.eigvals(self.B) >
                      0), "B has negative eigenvalues"
        # Gamma has to be symmetric and positive semi-definite
        assert np.all(self.Gamma == self.Gamma.T), "Gamma not symetric"
        assert np.all(np.linalg.eigvals(self.Gamma) >=
                      0), "Gamma not positive semidefinite"
        # mu nas to be a vector of length 2
        assert self.mu.shape == (2,)

    def __str__(self):
        return """Two variable Ornstein–Uhlenbeck process;
        `use get_B()`, `get_Gamma()` and 'get_mu()' to print parameters. """

    def get_B(self):
        return (self.B)

    def get_Gamma(self):
        return (self.Gamma)

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
        matrix appears in this equation.
        The conditional distribution of X(t) given X(t-d) can
        be reparametrized so that it depends on B and Gamma.
        See for example oravecz2011hierarchical eq (5).
        """
        var = self.Gamma - np.matmul(
            np.matmul(expm(-self.B*d), self.Gamma), 
            expm(-self.B.T*d)
            )
        return var

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
        (which is asserted to be positive definite 
        and for this simulation real). This matrix defines the
        cross
    Simulation runs for `total_time` units 
        with 1/'d' samples per unit.
    The function uses a conditional distribution of X(t)

    Initial value of the process X(0) is sampled from bivariate normal
        distribution with mean mu and covariance matrix Gamma.
        """

        return data

my_ou = OU_process_2v([[2, 0], [1, 2]], [[2, 0], [0, 2]])

# -------------------------------------------------------------------------

def ou_two_vars(B, Gamma, mu =0, total_time=100, d = 0.1):
    '''
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
        (which is asserted to be positive definite 
        and for this simulation real). This matrix defines the
        cross
    Simulation runs for `total_time` units 
        with 1/'d' samples per unit.
    The function uses a conditional distribution of X(t)

    Initial value of the process X(0) is sampled from bivariate normal
        distribution with mean mu and covariance matrix Gamma.
    '''
    
    n_datapoints = int(total_time / dt)
    # a thousand datapoints for two vars per person
    person = np.empty((2, n_datapoints))
    w_init = np.array([0, 0])
    person[:, 0] = w_init  # set process start
    for i in np.arange(n_datapoints - 1):
        person[:, i + 1] = np.matmul(auto_cross_lagged, person[:, i]) + \
            rng.multivariate_normal([0, 0], error_cov)
    return person

def set_conditional_PDF_mean():
    return 0;
def set_conditional_PDF_variance(B, Gamma):
    return 0;


3
array([[ 0.00247875,  0.        ],
       [-0.00743626,  0.00247875]])