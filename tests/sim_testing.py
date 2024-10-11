import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import expm

from ou_sim.OU_process_2v import OU_process_2v
from ou_sim.OU_process_1v import OU_process_1v
from data_simulation.person import person

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
- we know predictors of who gets which process and sample intensively
"""

"""# %% comment
To get simulation you need:
- mean, assume 0, then allow variation (mu.
- stationary covariance matrix (GAMMA)
- drift matrix (B), informing the underlying process
"""

"""
PROCESS PARAMETERS TO TEST
stationary covariances 0.3 0.5 or 02. 04. 06
How fast risk centralizes
How fast predictor centralizes
How predictor affects risk (always decentralizing)
How risk affects predictor - not at all or a little bit, show a pair!
When suicide attempt occurs - different rules: 
    - daily mean above certain threshold
    - risk at some time-point above certain threshold
    - risk above a certain threshold for the whole day
    - risk above a certain threshold for some % time during the day
"""

"""
DATA FITTING STRATEGIES

"""

### ONE VAR PROCESS TESTS ###

Ks = np.array([0.1, 0.2, 0.5, 0.7, 1.0, 2, 5])
d = 0.1
total_time = 30
timevec = np.arange(0, total_time, d)

for k in Ks:
    this_ou = OU_process_1v(sigma=1, k=k)
    data = this_ou.sim_data(d=d, total_time=total_time, initial_value=0)
    fig = plt.figure(figsize=(10, 12))
    gs = plt.GridSpec(2, 2)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax1.plot(timevec, data, c="orange")
    ax2.hist(data, bins=10, color='orange')
    discrete_k = this_ou.get_sim_discrete_k()
    y = np.zeros((timevec.size))
    y[0] = 1
    last = timevec.size
    for i in np.arange(last - 1):
        y[i + 1] = y[i]*discrete_k
        if y[i+1] < 0.001:
            last = int(i + 1 + np.ceil(1/d))
            y = y[:last]
            break
    ax3.plot(timevec[:last], y, c="blue")
    plt.show()

    n_reps = 1
    means = np.zeros(n_reps)
    vars = np.zeros(n_reps)
    d_av = 1
    total_time_av = 100000
    for i in range(n_reps):
        data = this_ou.sim_data(d=d_av, total_time=total_time_av)
        means[i] = np.mean(data)
        vars[i] = np.var(data)
    av_mean = np.mean(means)
    av_var = np.mean(vars)
    print("SIGMA = ", this_ou.get_sigma(), "k = ", this_ou.get_k(),
          "\nmean of means = ", av_mean, " mean of vars = ", av_var)


# ou1_0 = OU_process_1v(sigma=1, k=0.1)
# ou1_0 = OU_process_1v(sigma=1, k=0.1)
# ou1_0 = OU_process_1v(sigma=1, k=0.1)
# ou1_0 = OU_process_1v(sigma=1, k=0.1)
# ou1_0 = OU_process_1v(sigma=1, k=0.1)
# ou1_0 = OU_process_1v(sigma=1, k=0.1)
#
# #### TWO VARS PROCESS TESTS ###
#
# def example_plot(which_person, figure_name):
#     timevec = np.arange(0,
#                         which_person.get_parameters("ou")["Total time"],
#                         which_person.get_parameters("ou")["Delta t"])
#     person_data = which_person.get_data("ou")
#     params = which_person.get_parameters("ou")
#
#     fig = plt.figure(figure_name, figsize=(10, 12))
#     gs = plt.GridSpec(4, 2)
#     ax1 = fig.add_subplot(gs[0, :])
#     ax2 = fig.add_subplot(gs[1, 0])
#     ax3 = fig.add_subplot(gs[1, 1])
#     ax4 = fig.add_subplot(gs[2, 0])
#     ax5 = fig.add_subplot(gs[2, 1])
#     ax6 = fig.add_subplot(gs[3, 0])
#     ax7 = fig.add_subplot(gs[3, 1])
#
#     ax1.plot(timevec, person_data[0, :], c="orange")
#     ax1.plot(timevec, person_data[1, :], c="grey")
#     ax1.hlines(0, 0, which_person.get_parameters("ou")["Total time"], colors="lightgrey", linestyles="dashed")
#     props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#     text_discr_A = 'Discrete A\n' + np.array2string(params["Discrete A"],
#                                                    formatter={'float_kind': lambda x: "%.2f" % x})
#     ax1.text(0.6, 1, text_discr_A, transform=ax1.transAxes, fontsize=10, fontweight='bold',
#              verticalalignment='bottom', bbox=props)
#     text_condPDF = 'Simulation cond. cov\n' + np.array2string(params["Simulation cond. cov"],
#                                                               formatter={'float_kind': lambda x: "%.2f" % x})
#     ax1.text(0.75, 1, text_condPDF, transform=ax1.transAxes, fontsize=10, fontweight='bold',
#              verticalalignment='bottom', bbox=props)
#     text_BB_T = 'BB_T (diffusion)\n' + np.array2string(params["BB_T (diffusion)"],
#                                                          formatter={'float_kind': lambda x: "%.2f" % x})
#     ax1.text(0.4, 1, text_BB_T, transform=ax1.transAxes, fontsize=10, fontweight='bold',
#              verticalalignment='bottom', bbox=props)
#     text_A = 'A (drift)\n' + np.array2string(params["A (drift)"],
#                                              formatter={'float_kind': lambda x: "%.2f" % x})
#     ax1.text(0, 1, text_A, transform=ax1.transAxes, fontsize=10, fontweight='bold',
#              verticalalignment='bottom', bbox=props)
#     text_sigma = 'sigma (stat.cov)\n' + np.array2string(params["sigma (stationary cov)"],
#                                                         formatter={'float_kind': lambda x: "%.2f" % x})
#     ax1.text(0.15, 1, text_sigma, transform=ax1.transAxes, fontsize=10, fontweight='bold',
#              verticalalignment='bottom', bbox=props)
#
#     emp_cov_matrix = np.cov(person_data)
#     corr = np.corrcoef(person_data)
#     covariance = 'cov =' + np.array2string(emp_cov_matrix[0, 1], formatter={'float_kind': lambda x: "%.3f" % x}) + \
#                  '\ncor =' + np.array2string(corr[0, 1], formatter={'float_kind': lambda x: "%.3f" % x})
#     variances = 'var(orange) = ' + np.array2string(emp_cov_matrix[0, 0],
#                                                    formatter={'float_kind': lambda x: "%.3f" % x}) + \
#                 '\nvar(gray) = ' + np.array2string(emp_cov_matrix[1, 1], formatter={'float_kind': lambda x: "%.3f" % x})
#
#     ax2.scatter(person_data[0, :], person_data[1, :])
#     ax2.text(0.01, 0.99, covariance, transform=ax2.transAxes, fontsize=10, fontweight='bold',
#              verticalalignment='top', bbox=props)
#
#     ax3.hist(person_data[0, :], bins=30, color='orange', alpha=0.5)
#     ax3.hist(person_data[1, :], bins=30, color='grey', alpha=0.5)
#     ax3.text(0.01, 0.99, variances, transform=ax3.transAxes, fontsize=10, fontweight='bold',
#              verticalalignment='top', bbox=props)
#
#     discreteA = params["Discrete A"]
#     window = params["Total time"]
#     dt = params["Delta t"]
#
#     # for ax4
#     y = np.zeros((2, timevec.size))
#     y_init = np.array([0, 1])
#     y[:, 0] = y_init  # set process start
#     last = timevec.size
#     for i in np.arange(timevec.size - 1):
#         y[:, i + 1] = np.matmul(discreteA, y[:, i])
#         if y[0, i+1] < 0.001 and y[1, i+1] < 0.001:
#             last = int(i + 1 + np.ceil(1/dt))
#             y = y[:, :last]
#             break
#
#     ax4.plot(timevec[:last], y[0, :], c="orange")
#     ax4.plot(timevec[:last], y[1, :], c="grey")
#     ax4.set_title('Drift illustration grey = 1')
#
#     # for ax5
#     y = np.zeros((2, timevec.size))
#     y_init = np.array([1, 0])
#     y[:, 0] = y_init  # set process start
#     last = timevec.size
#     for i in np.arange(timevec.size - 1):
#         y[:, i + 1] = np.matmul(discreteA, y[:, i])
#         if y[0, i+1] < 0.001 and y[1, i+1] < 0.001:
#             last = int(i + 1 + np.ceil(1/dt))
#             y = y[:, :last]
#             break
#
#     ax5.plot(timevec[:last], y[0, :], c="orange")
#     ax5.plot(timevec[:last], y[1, :], c="grey")
#     ax5.set_title('Drift illustration orange = 1')
#
#     # for ax6
#     x = np.arange(0, params["Total time"], 0.01)
#     y = np.zeros((x.size, 2, 2))
#     last = x.size
#     for index, value in enumerate(x):
#         y[index] = params["Process"].get_stationary_time_cov(d=value)
#         if np.all(y[index] <= 0.005):
#             last = int(index + 1 + np.ceil(1/dt))
#             y = y[:last]
#             break
#
#     ax6.plot(x[:last], y[:last, 0, 0], c="orange", label="autocovariance of var0")
#     ax6.plot(x[:last], y[:last, 1, 1], c="grey", label="autocovariance of var1")
#     ax6.plot(x[:last], y[:last, 0, 1], c="lime", label="crosscovariance: var1 -> var0")
#     ax6.plot(x[:last], y[:last, 1, 0], c="purple", label="crosscovariance: var0 -> var1")
#     ax6.legend(loc='upper right')
#     ax6.set_title('Cross-covariance over time')
#
#     # for ax7 (builds up on ax6!)
#     x = x[:last]
#     y = np.zeros((x.size, 2, 2))
#     last = x.size
#     for index, value in enumerate(x):
#         y[index] = params["Process"].get_stationary_time_correlation(d=value)
#
#     ax7.plot(x[:last], y[:last, 0, 0], c="orange", label="autocorrelation of var0")
#     ax7.plot(x[:last], y[:last, 1, 1], c="grey", label="autocorrelation of var1")
#     ax7.plot(x[:last], y[:last, 0, 1], c="lime", label="crosscorrelation: var1 -> var0")
#     ax7.plot(x[:last], y[:last, 1, 0], c="purple", label="crosscorrelation: var0 -> var1")
#     ax7.legend(loc='upper right')
#     ax7.set_title('Cross-correlation over time')
#
#     plt.show()
#     fig.savefig("temp/" + figure_name + ".png")
#
#
#
# my_ou_0 = OU_process_2v(A=[[5, 0], [0, 5]],
#                        BB_T=[[1, 0.0], [0.0, 1]])
#
# my_ou_1 = OU_process_2v(A=[[2, 0], [0, 2]],
#                         BB_T=[[1, 0.0], [0.0, 1]])
#
# my_ou_2 = OU_process_2v(A=[[1, 0], [0, 1]],
#                         BB_T=[[1, 0.0], [0.0, 1]])
#
# my_ou_3 = OU_process_2v(A=[[0.5, 0], [0, 0.5]],
#                         BB_T=[[1, 0.0], [0.0, 1]])
#
# my_ou_4 = OU_process_2v(A=[[0.2, 0], [0, 0.2]],
#                         BB_T=[[1, 0.0], [0.0, 1]])
#
# my_ou_5 = OU_process_2v(A=[[0.1, 0], [0, 0.1]],
#                         BB_T=[[1, 0.0], [0.0, 1]])
#
# kasia = person({'płeć': "k", 'wiek': "23"})
# tomek = person({'płeć': "m", 'wiek': "23"})
# zosia = person({'płeć': "k", 'wiek': "23"})
# wiesio = person({'płeć': "k", 'wiek': "23"})
# zenek = person()
# ziuta = person()
#
# simulation_time = 30
# delta_t = 0.1
# kasia.simulate_OU_process_2v(set_name="ou", ou_process=my_ou_0, mu=[0, 0], dt=delta_t, time = simulation_time)
# tomek.simulate_OU_process_2v(set_name="ou", ou_process=my_ou_1, mu=[0, 0], dt=delta_t, time = simulation_time)
# zosia.simulate_OU_process_2v(set_name="ou", ou_process=my_ou_2, mu=[0, 0], dt=delta_t, time = simulation_time)
# wiesio.simulate_OU_process_2v(set_name="ou", ou_process=my_ou_3, mu=[0, 0], dt=delta_t, time = simulation_time)
# zenek.simulate_OU_process_2v(set_name="ou", ou_process=my_ou_4, mu=[0, 0], dt=delta_t, time = simulation_time)
# ziuta.simulate_OU_process_2v(set_name="ou", ou_process=my_ou_5, mu=[0, 0], dt=delta_t, time = simulation_time)
#
# example_plot(kasia, "kasia")
# example_plot(tomek, "tomek")
# example_plot(zosia, "zosia")
# example_plot(wiesio, "wiesio")
# example_plot(zenek, "zenek")
# example_plot(ziuta, "ziuta")
#
# # results = np.zeros((4, 5))
# # processes = [my_ou_0, my_ou_1, my_ou_2, my_ou_3]
#
# # for j, p in enumerate(processes):
# #     #test the process
# #     n_runs = 1
# #     means = np.zeros((2, n_runs))
# #     variances = np.zeros((2, n_runs))
# #     covariances = np.zeros(n_runs)
# #     process = p
# #
# #     for i in range(n_runs):
# #         some_dude = person()
# #         some_dude.simulate_OU_process_2v(set_name="ou", ou_process=process, mu=[0, 0], dt=delta_t, time = simulation_time)
# #         dude_data = some_dude.get_data("ou")
# #         means[:, i] = np.mean(dude_data, axis=1)
# #         variances[:, i] = np.var(dude_data, axis=1)
# #         covariances[i] = np.cov(dude_data)[0, 1]
# #
# #     results[j, 0] = (np.mean(means[0]))
# #     results[j, 1] = (np.mean(means[1]))
# #     results[j, 2] = (np.mean(variances[0]))
# #     results[j, 3] = (np.mean(variances[1]))
# #     results[j, 4] = (np.mean(covariances))


