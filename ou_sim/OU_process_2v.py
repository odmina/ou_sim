import numpy as np
from scipy.linalg import expm


class OU_process_2v(object):
    """
    This class defines two variable Ornstein–Uhlenbeck process
    and allows data simulation.


    Declaration of a variable of class OU_process_2v requires:

    `B` - centralizing (drift) matrix `B` links variables in
        the process, it is asserted to be a real matrix with
        with positive eigenvalues.
    `Gamma` - stationary covariance matrix of the OU process;

    During data simulation parameter `mu` may be provided.
        It is the mean of the process, it is 0 by default.

    The simulation uses discrete timesteps of equal length d
        and at each timestep samples X(t) from a conditional
        distibution of X(t) given X(t-d). The distrubution is
        normal with parameters calculated using eq (9) from
        Oravecz et al. 2011.

    Simulation runs for `total_time` units
        with 1/'d' samples per unit.
    The function uses a conditional distribution of X(t).

    Cholesky decomposition of instantaneus covariance
        matrix appears in the stochastic term of the equation of X(t).
        For the simulation, the conditional distribution of X(t)
        given X(t-d) is reparametrized so that it depends on B and Gamma.
        Thus, the following condition has to be satisfied:
        Sigma = B*Gamma + Gamma*B.T
        is a Hermitian, positive-definite matrix.

        Since for the simulation only real valued matrices are used,
        Sigma has to be symetric (otherwise it could be Hermitian).

        If Gamma is provided, Sigma is calculated to check whether
        it is positive definite (so that we get proper covariance matrices).

    Initial value of the process X(0) is sampled from bivariate normal
        distribution with mean mu and covariance matrix Gamma.
    """

    def __init__(self, B, Gamma=None, Sigma=None):
        assert Gamma is not None or Sigma is not None, \
            """You should provide either stationary covariance matrix (Gamma)
            or diffusion matrix (Sigma)"""

        # set and check B
        # B is assumed to be 2x2, real and have positive eigenvalues
        self.B = np.array(B)
        assert np.all(np.isreal(self.B)), "B is not a real matrix"
        assert self.B.shape == (2, 2), "B is not 2x2 matrix"
        assert np.all(np.linalg.eigvals(self.B) >
                      0), "B has zero or negative eigenvalues"

        # if Gamma is provided set and check it and set Sigma
        # Gamma is assumed to be 2x2, real, symmetric and positive semi-definite
        # Sigma = B*Gamma + Gamma*B.T should be positive-definite
        # (symetric with positive eigenvalues)
        if Gamma is not None and Sigma is None:
            self.Gamma = np.array(Gamma)
            assert np.all(np.isreal(self.Gamma)), "Gamma is not a real matrix"
            assert self.Gamma.shape == (2, 2), "Gamma is not 2x2 matrix"
            assert np.all(self.Gamma == self.Gamma.T), "Gamma not symetric"
            assert np.all(np.linalg.eigvals(self.Gamma) >=
                          0), "Gamma has negative eigenvalues"
            # set Sigma
            self.Sigma = np.matmul(self.B, self.Gamma) + \
                         np.matmul(self.Gamma, self.B.T)
            assert np.all(self.Sigma == self.Sigma.T) and \
                   np.all(np.linalg.eigvals(self.Sigma) > 0), \
                "Sigma not possitive-definite"

        # if Sigma is provided, check it and set Gamma
        # Sigma is assumed to be 2x2, symetric and positive semidefinite
        # Sigma could be Hermitian, but for now simulation uses only real values
        if Sigma is not None and Gamma is None:
            self.Sigma = np.array(Sigma)
            assert self.Sigma.shape == (2, 2), "Sigma is not 2x2 matrix"
            assert np.all(self.Sigma == self.Sigma.T), "Sigma not symetric"
            assert np.all(np.linalg.eigvals(self.Sigma) > 0), "Sigma has negative eigenvalues"
            # set Gamma
            # equation for Gamma is eq. 4.4.53 from
            # https://archive.org/details/handbookofstocha0000gard/page/110/mode/2up
            det_B = np.linalg.det(self.B)
            tr_B = np.trace(self.B)
            B_no_trace = self.B - tr_B * np.identity(2)
            self.Gamma = (det_B * self.Sigma + np.matmul(np.matmul(B_no_trace, self.Sigma),
                                                         B_no_trace.T)) / \
                         (2 * det_B * tr_B)
            assert np.all(np.isreal(self.Gamma)), "Gamma is not a real matrix"
            assert self.Gamma.shape == (2, 2), "Gamma is not 2x2 matrix"
            assert np.all(self.Gamma == self.Gamma.T), "Gamma not symetric"
            assert np.all(np.linalg.eigvals(self.Gamma) >=
                          0), "Gamma has negative eigenvalues"

    def __str__(self):
        return """Two variable Ornstein–Uhlenbeck process;
        `use get_B()`, `get_Gamma()` and 'get_Sigma()' to access parameters."""

    # ------ GETTERS ------

    def get_B(self):
        return self.B

    def get_Gamma(self):
        return self.Gamma

    def get_Sigma(self):
        return self.Sigma

    # ------ DISCRETE TIME SIMULATION METHODS ------

    def get_sim_discreteB(self, d=0.1):
        """
        This method calculates the discrete time version of B.

        It uses timestep `d`, with default 0.1.
        """
        return expm(-self.B * d)

    def get_sim_condPDF_covariance(self, d=0.1):
        """
        Calculates the variance of the distribution of X(t) given X(t-d).

        Uses timestep `d`, with default 0.1.

        The covariance is determined by the stochastic term of the
        integral equation X(t) describing the process.
        Cholesky decomposition of instantaneus covariance
        matrix appears in this equation. The conditional distribution
        of X(t) given X(t-d) is reparametrized so that it depends
        on B and Gamma.See for example oravecz2011hierarchical eq (5).
        """

        cov = self.Gamma - np.matmul(
            np.matmul(expm(-self.B * d), self.Gamma),
            expm(-self.B.T * d)
        )
        return cov

    def sim_data(self, d=0.1, total_time=10, mu=(0, 0), mu_fun='linear', **kwargs):
        """
        This function simulates a two vairable Ornstein–Uhlenbeck process.

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

        mu = np.array(mu)
        assert mu.shape == (2,), "mu is not (2,) array"

        rng = np.random.default_rng()

        # initialize data array
        n_datapoints = int(total_time / d)
        X = np.empty((2, n_datapoints))

        # compute parameters for the given d
        condPDF_cov = self.get_sim_condPDF_covariance(d=d)
        cond_B = self.get_sim_discreteB(d=d)

        # set process start X(t=0)
        X[:, 0] = rng.multivariate_normal(mu, self.Gamma)

        # simulate data
        for i in np.arange(n_datapoints - 1):
            this_mu = mu + np.matmul(cond_B, (X[:, i] - mu))
            X[:, i + 1] = rng.multivariate_normal(this_mu, condPDF_cov)

        return X

# # helper function to introduce mean trends
# def sine(mean, cycles, amplitude, offset, total_time, dt):
#     n_datapoints = int(total_time/dt)
#     time = np.linspace(0, 2*np.pi, n_datapoints) + (2*np.pi/cycles)*offset
#     mus = mean + amplitude*np.sin(time*cycles)
#     return mus
