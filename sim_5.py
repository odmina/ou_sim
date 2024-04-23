import numpy as np
from functions.get_variances_from_icc import get_variances_from_icc
from functions.OU_process_2v import OU_process_2v
from person import person
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
import json


####################################################################################################
# 1: Accuracy of cross sectional prediction is driven by stationary covariance
####################################################################################################

####################################################################################################
# 1.2: When interindividual differences are present, accuracy of the predictions depends on
# cross sectional correlation which is a weighted average of within and between person correlation
# (show it for different ICCs and between-within correlations of different levels)
####################################################################################################

# %% set simulation parameters
# within person variances of the variables are allways 1
# as a result, for different levels of iccs there are different
# inter-individual variances
iccs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
within_cors = [0.5]
between_cors = [0.3, 0.5, 0.7]
sample_sizes = [500, 1000, 5000]  # show only 1000 in the paper, put the rest in the supplement
n_runs = 10

# %% set simulation parameters for testing
# stat_covs = [0.1, 0.3, 0.5, 0.7, 0.9]
# sample_sizes = [500, 1000, 5000]
# n_runs = 3

# %% prepare data dictionary
data_dict = {}

# %% rng set state
rng = np.random.default_rng(3456)

# %% simulate the data and fit ROC curves
for N in sample_sizes:
    data_dict["N"] = {}

    # each sample size has own ROC curves plot
    # other plots are created from simulation data
    fig, ax = plt.subplots(nrows=len(within_cors),
                           ncols=len(iccs),
                           figsize=(12, 20),
                           sharex=True,
                           sharey=True)
    fig_name = "fig-01-02_icc_between_sim_N" + str(N) + ".png"

    for i, icc in enumerate(iccs):
        data_dict["N"]["icc"] = {}
        get_variances_from_icc([0.5, 0.5], within = [1, 1])

        for j, within_cor in enumerate(within_cors):
            # variables have var 1 and covariance cov
            this_Gamma = [[1, cov], [cov, 1]]

            # dictionary to store data with this cov
            data_dict[str(cov)] = {}

            # run this cov for all sample sizes
