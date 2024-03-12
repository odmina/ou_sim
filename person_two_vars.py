import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import expm


# TODO: create a class person and add methods:
#     init
#     add disturbance
#     produce plot
#     display attributes (drift matrix, expected value etc)

def get_residual_cov_matrix(G, A, delta_t):  # formula from Voelkle et. al 2012
    if np.not_equal(np.tril(G), G).any():
        print("Diffusion matrix not lower triangular")
        return
    Q = np.matmul(G, G.transpose())
    A_hash = np.kron(A, np.identity(
        A.shape[0])) + np.kron(np.identity(A.shape[0]), A)
    cov_vec = np.matmul(np.matmul(np.linalg.inv(A_hash),
                                  expm(A_hash * delta_t) - np.identity(A_hash.shape[0])),
                        np.reshape(Q, (Q.size,)))
    cov_mtx = np.reshape(cov_vec, (2, 2))
    return cov_mtx


def get_within_cov_matrix(A, G):  # formula from Shuurman 2023 (preprint)
    auto_cross_lagged = expm(A)
    error_cov = get_residual_cov_matrix(G, A, 1)
    phi_kron_product = np.kron(auto_cross_lagged, auto_cross_lagged)
    left_matrix = np.linalg.inv(np.identity(
        phi_kron_product.shape[0]) - phi_kron_product)
    right_vector = np.reshape(error_cov, error_cov.size)
    within_cov_vec = np.matmul(left_matrix, right_vector)
    return within_cov_vec


def person_two_vars(dt, auto_cross_lagged, error_cov, total_time=500):
    n_datapoints = int(total_time / dt)
    # a thousand datapoints for two vars per person
    person = np.empty((2, n_datapoints))
    w_init = np.array([0, 0])
    person[:, 0] = w_init  # set process start
    for i in np.arange(n_datapoints - 1):
        person[:, i + 1] = np.matmul(auto_cross_lagged, person[:, i]) + \
            rng.multivariate_normal([0, 0], error_cov)
    return person


def covariance(a):
    mean = np.array([np.mean(a[:, 0]), np.mean(a[:, 1])])
    difference = a - mean
    cov = np.sum(difference[:, 0] * difference[:, 1]) / difference.shape[0]
    return cov


# set seed
rng = np.random.default_rng()


# inter-individual distribution
inter_cov_matrix = np.array([[1, 0.5], [0.5, 1]])
ppl = np.zeros((10000, 100, 2))
for j in range(10000):
    for i in range(100):
        ppl[j, i] = rng.multivariate_normal([0, 0], inter_cov_matrix)
# fig, ax = plt.subplots()
# ax.scatter(ppl[:, 0], ppl[:, 1])
# plt.show()

covariances = np.zeros(10000)
for j in range(10000):
    covariances[j] = covariance(ppl[j])
fig, ax = plt.subplots()
ax.hist(covariances)
plt.show()


variances = np.zeros((10000, 2))

for j in range(10000):
    variances[j] = np.array([np.var(ppl[j, :, 0]), np.var(ppl[j, :, 1])])

fig, ax = plt.subplots()
ax.hist(variances[:, 0])
plt.show()


# drift and diffusion matrices
drift = np.array([[-0.5, 0.1], [0, -1]])
# diffusion should be lower-triangular
diffusion = np.array([[0.5, 0], [0, 0.5]])

# illustrate the drift
window = 5
resolution = 100
dx = 1/resolution
phi_illustration = expm(dx * drift)
x = np.arange(0, window, dx)
y = np.empty((2, x.size))
y_init = np.array([0, 1])
y[:, 0] = y_init  # set process start

for i in np.arange(x.size-1):
    y[:, i + 1] = np.matmul(phi_illustration, y[:, i])

fig, ax = plt.subplots()
ax.plot(x, y[0, :], c="orange")
ax.plot(x, y[1, :], c="grey")

plt.show()


cov_within = get_within_cov_matrix(drift, diffusion)
sampling_int = 0.1
phi = expm(sampling_int * drift)
zeta = get_residual_cov_matrix(diffusion, drift, sampling_int)


# PLOT A PERSON
this_person = person_two_vars(
    dt=sampling_int, auto_cross_lagged=phi, error_cov=zeta, total_time=10)
fig, ax = plt.subplots()
ax.plot(this_person[0, :], c="orange")
ax.plot(this_person[1, :], c="grey")
# ax.scatter(np.where(this_person[1] == 1), np.ones(this_person[1].sum()) * np.min(this_person[0]))
plt.show()


# # generate files to plot means and variances of 500 / 1000 people
# vars = np.zeros((3,1000,2))
# for j in range(3):
#     # matrix of autoregression and cross-lagged coefficients
#     sampling_int = 1/10**j
#     phi = expm(sampling_int * drift)
#     zeta = get_residual_cov_matrix(diffusion, drift, sampling_int)
#     for i in range(500):
#         this_person = person_two_vars(sampling_int, phi, zeta, total_time=1000)
#         this_person_mean = np.mean(this_person, axis=1)
#         this_person_var = np.var(this_person, axis=1)
#         means[j, i, :] = this_person_mean
#         vars[j, i, :] = this_person_var

# load files with means and variances for 500 people
with open("two_vars_means_variances/means.npy", "rb") as f:
    means = np.load(f)
with open("two_vars_means_variances/variances.npy", "rb") as f:
    vars = np.load(f)


fig, axs = plt.subplots(nrows=3, ncols=2, sharex=False, layout='constrained')
axs[0, 0].set_title("Means", fontsize='small', loc='left')
axs[0, 0].hist(means[0, 0:500, 0], bins=21)
axs[1, 0].hist(means[1, 0:500, 0], bins=21)
axs[2, 0].hist(means[2, 0:500, 0], bins=21)
axs[0, 1].set_title("Variances", fontsize='small', loc='left')
axs[0, 1].hist(vars[0, 0:500, 0], bins=21)
axs[1, 1].hist(vars[1, 0:500, 0], bins=21)
axs[2, 1].hist(vars[2, 0:500, 0], bins=21)
fig.suptitle('Dependent var', fontsize=14)
plt.show()

fig, axs = plt.subplots(nrows=3, ncols=2, sharex=False, layout='constrained')
axs[0, 0].set_title("Means", fontsize='small', loc='left')
axs[0, 0].hist(means[0, 0:500, 1], bins=21)
axs[1, 0].hist(means[1, 0:500, 1], bins=21)
axs[2, 0].hist(means[2, 0:500, 1], bins=21)
axs[0, 1].set_title("Variances", fontsize='small', loc='left')
axs[0, 1].hist(vars[0, 0:500, 1], bins=21)
axs[1, 1].hist(vars[1, 0:500, 1], bins=21)
axs[2, 1].hist(vars[2, 0:500, 1], bins=21)
fig.suptitle('Independent var', fontsize=14)
plt.show()
