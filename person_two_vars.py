import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import expm

# TODO: create a class person and add methods:
#     init
#     add disturbance
#     produce plot
#     display atrubutes (drift matrix, expected value etc)

# set seed
rng = np.random.default_rng()


# matrix of autoregression and cross-lagged coefficients
drift = np.array([[-0.144, -0.029], [1.795, -1.832]]) # drift for desire and intent from Coppersmith et al. 2023
scale = 0.1
phi = expm(0.1 * drift)


def person_two_vars():
    n_datapoints = int(50 / scale)
    person = np.empty((2, n_datapoints))  # a thousand datapoints for two vars per person
    w_init = rng.normal(0, 1, (2,))  # atm process starts at some random point
    person[:, 0] = w_init  # set process start
    ind_disturb_rate = rng.poisson(5, 1)
    disturb_prob = ind_disturb_rate / n_datapoints
    disturb = rng.binomial(1, disturb_prob, n_datapoints)
    for i in np.arange(n_datapoints-1):
        person[:, i+1] = np.matmul(phi, person[:, i]) + np.array((disturb[i], 0))
    return person, disturb


this_person = person_two_vars()


fig, ax = plt.subplots()
ax.plot(this_person[0][0, :])
ax.plot(this_person[0][1, :])
ax.scatter(np.where(this_person[1] == 1), np.ones(this_person[1].sum()) * np.min(this_person[0]))
plt.show()

