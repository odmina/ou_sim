import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import expm

# set seed
rng = np.random.default_rng()
drift = -1.1
phi = np.exp(0.01 * drift)
x = np.arange(0, 5, 0.01)
series_len = x.size
disturbance = np.zeros(series_len)
disturbance[0] = 1
y = np.zeros(series_len)
for i in np.arange(series_len - 1):
    y[i + 1] = phi * y[i] + disturbance[i]  # variance of
fig, ax = plt.subplots()
ax.plot(x, y, c="red")
ax.scatter([0], [- 0.1], c="grey")
plt.show()


# run a plot that shows the time course of the risk after event occurence


def person_pred_binomial():
    n_units = 1
    n_datapoints = int(n_units / scale)
    person = np.empty((2, n_datapoints))  # a thousand datapoints for two vars per person
    # w_init = rng.normal(0, 1, (2,))  # atm process starts at some random point
    person[:, 0] = [0, 0]  # set process start
    ind_disturb_rate = 1  # rng.poisson(5, 1) (5 over the whole sim time - i.e. year)
    disturb_prob = ind_disturb_rate / n_datapoints
    disturb = rng.binomial(1, disturb_prob, n_datapoints)
    person[1] = disturb
    for i in np.arange(n_datapoints - 1):
        person[0, i + 1] = phi[0, 1] * person[0, i] + person[1, i]
    return person


# matrix of autoregression and cross-lagged coefficients
drift = np.array([[-1.832, 5], [0, -3]])
scale = 0.01
phi = expm(0.1 * drift)

this_person = person_pred_binomial()

# fig, ax = plt.subplots()
# ax.plot(this_person[0, :], c="red")
# # ax.plot(this_person[1, :], c="grey")
# ax.scatter(np.where(this_person[1, :] == 1), np.zeros(int(this_person[1, :].sum())) - 0.1, c="grey")
# plt.show()
