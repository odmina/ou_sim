import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import expm

from ou_sim.OU_process_2v import OU_process_2v
from ou_sim.person import person


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

"""# %% comment
commands
To get simulation you need:
- mean, assume 0, then allow variation (mu.
- stationary covariance matrix (GAMMA)
- drift matrix (B), informing the underlying process
"""


def example_plot(person, figure_name):
    timevec = np.arange(0,
                        person.get_parameters("ou")["Total time"],
                        person.get_parameters("ou")["Delta t"])
    this_person = person.get_data("ou")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(timevec, this_person[0, :], c="orange")
    ax.plot(timevec, this_person[1, :], c="grey")
    ax.hlines(0, 0, person.get_parameters("ou")["Total time"], colors="lightgrey", linestyles="dashed")
    plt.savefig(figure_name)


my_ou_0 = OU_process_2v(B=[[2, -1.7], [0, 0.1]],
                        Gamma=[[1, 0.5], [0.5, 1]])

my_ou_1 = OU_process_2v(B=[[0.5, 0], [0, 0.5]],
                        Gamma=[[1, 0.6], [0.6, 1]])

my_ou_2 = OU_process_2v(B=[[1, 0], [0, 1]],
                        Gamma=[[1, 0.6], [0.6, 1]])

my_ou_3 = OU_process_2v(B=[[2, 0], [0, 2]],
                        Gamma=[[1, 0.6], [0.6, 1]])


kasia = person({'płeć': "k", 'wiek': "23"})
tomek = person({'płeć': "m", 'wiek': "23"})
zosia = person({'płeć': "k", 'wiek': "23"})
wiesio = person({'płeć': "k", 'wiek': "23"})

kasia.simulate_OU_process_2v(set_name="ou", ou_process=my_ou_0, mu=[0, 0], dt=0.25, time=30)
tomek.simulate_OU_process_2v(set_name="ou", ou_process=my_ou_1, mu=[0, 0], dt=0.25, time=30)
zosia.simulate_OU_process_2v(set_name="ou", ou_process=my_ou_2, mu=[0, 0], dt=0.25, time=30)
wiesio.simulate_OU_process_2v(set_name="ou", ou_process=my_ou_3, mu=[0, 0], dt=0.25, time=30)

example_plot(kasia, "temp/Figure.png")
example_plot(tomek, "temp/Figure1.png")
example_plot(zosia, "temp/Figure2.png")
example_plot(wiesio, "temp/Figure3.png")


d = 0.01
n_datapoints = int(20 / d)
X = np.empty((2, n_datapoints))
mu = np.array([2, 2])

condPDF_cov = my_ou.get_sim_condPDF_covariance(d=d)
cond_B = my_ou.get_sim_discreteB(d=d)

rng = np.random.default_rng()

# X[:, 0] = [rng.multivariate_normal(my_ou.mu, my_ou.Gamma)]  # set process start
X[:, 0] = [0, 0]
for i in np.arange(n_datapoints - 1):
    this_mu = mu + np.matmul(cond_B, (X[:, i] - mu))
    X[:, i + 1] = rng.multivariate_normal(this_mu, condPDF_cov)

this_person = X
fig, ax = plt.subplots()
ax.plot(this_person[0, :], c="orange")
ax.plot(this_person[1, :], c="grey")
ax.hlines(0, 0, n_datapoints, colors="lightgrey", linestyles="dashed")
# ax.scatter(np.where(this_person[1] == 1), np.ones(this_person[1].sum()) * np.min(this_person[0]))
plt.savefig("temp/Figure2.png")


kasia.get_ind_diffs()

kasia.get_observations()

kasia.available_datasets()

kasia.add_ind_diffs({'wzrost': 170})

kasia.add_observations("krowa", added_obs=["duza krowa", "z cielaczkiem"])
kasia.get_data("krowa")
kasia.get_parameters("krowa")

kasia.add_observations("krowa", {"nr oberwacji": 1}, ["duza krowa", "z cielaczkiem"])

np.cov(X)

condPDF_cov

cond_B

my_ou.get_Sigma()


"""
To get the same variance of the variable, if you add stationary covariance from the other var, you have to decrease the own noise variance of the var
my_ou = OU_process_2v(B=[[1, -1.73], [0, 1]],
                      Gamma=[[1, 0.5], [0.5, 1]],
                      mu=[0, 0])

cond_pdf
array([[ 0.01512503, -0.0510058 ],
       [-0.0510058 ,  0.18126925]])
array([[ 0.15676545, -0.14164042],
       [-0.14164042,  0.18126925]])

In [360]: cond_B
Out[360]:
array([[ 0.90483742,  0.15653687],
       [-0.        ,  0.90483742]])
array([[ 0.90483742,  0.15653687],
       [-0.        ,  0.90483742]])

Sigma
array([[ 0.27, -0.73],
       [-0.73,  2.  ]])
array([[ 2.  , -1.73],
       [-1.73,  2.  ]])
"""


for i in range(10):
    a = i % 2
    try:
        b = 10 / a
    except ZeroDivisionError as e:
        print(e)
        continue

    print(i, " ", b)


for i, cov in enumerate(stat_covs):
    for j, N in enumerate(sample_sizes):
        for run in range(n_runs):
            if isinstance(data_dict[str(cov)][str(N)][str(run)], dict):
                data_dict[str(cov)][str(N)][str(run)]["False positive rates"] = list(data_dict[str(cov)][str(N)][str(run)]["False positive rates"])
                data_dict[str(cov)][str(N)][str(run)]["True positive rates"] = list(data_dict[str(cov)][str(N)][str(run)]["True positive rates"])
                data_dict[str(cov)][str(N)][str(run)]["AUC"] = float(data_dict[str(cov)][str(N)][str(run)]["AUC"])
                data_dict[str(cov)][str(N)][str(run)]["Cases in training set"] = int(data_dict[str(cov)][str(N)][str(run)]["Cases in training set"])
                data_dict[str(cov)][str(N)][str(run)]["Cases in test set"] = int(data_dict[str(cov)][str(N)][str(run)]["Cases in test set"])
            else:
                data_dict[str(cov)][str(N)][str(run)] = str(data_dict[str(cov)][str(N)][str(run)])

