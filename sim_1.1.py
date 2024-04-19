import numpy as np
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
# 1.1: In the absence of interindividual differences, higher covariance = better predictions
# Predictor and dependent variable are sampled from bivariate normal distribution with a given
# covariance matrix.
# For each covariance level three sample sizes (500, 1000 and 5000) are simulated
# Each sample size-covariance pair is simulated n_runs times
# During each run:
#   data is simulated
#   model is fit
#   roc curve is fitted
#   auc is calculated
####################################################################################################


# %% set simulation parameters
stat_covs = [0.1, 0.3, 0.5, 0.7, 0.9]
sample_sizes = [500, 1000, 5000]
n_runs = 1000

# %% set simulation parameters for testing
# stat_covs = [0.1, 0.3, 0.5, 0.7, 0.9]
# sample_sizes = [500, 1000, 5000]
# n_runs = 3

# %% prepare plot & data dictionary
fig, ax = plt.subplots(nrows=len(stat_covs),
                       ncols=len(sample_sizes),
                       figsize=(12, 20),
                       sharex=True,
                       sharey=True)
data_dict = {}

# %% rng set state
rng = np.random.default_rng(3456)

# %% simulate the data and fit ROC curves
for i, cov in enumerate(stat_covs):
    # variables have var 1 and covariance cov
    this_Gamma = [[1, cov], [cov, 1]]

    # dictionary to store data with this cov
    data_dict[str(cov)] = {}

    # run this cov for all sample sizes
    for j, N in enumerate(sample_sizes):
        # dictionary to store data with this cov and N
        data_dict[str(cov)][str(N)] = {}

        # data stored to plot ROC curves for this cov and N
        tprs = []  # true positive rates for each run
        aucs = []  # aucs for each run
        mean_fpr = np.linspace(0, 1, 100)

        # run this cov and sample size multiple times
        for run in range(n_runs):
            # simulate data for this run
            data = rng.multivariate_normal(
                [0, 0],
                this_Gamma,
                N)
            X = data[:, 1].reshape(-1, 1)
            y = data[:, 0] > (this_Gamma[0][0] * 1.96)

            # divide into training data and data held out for testing for this run
            # If there are not enough True cases split will raise a ValueError
            # I such a case model will NOT be fitted and the loop will skip to next run
            # Details of the failed run will be saved to data_dict
            try:
                X_train, X_holdout, y_train, y_holdout = train_test_split(X,
                                                                          y,
                                                                          test_size=0.25,
                                                                          stratify=y)
            except ValueError as e:
                print(e)
                data_dict[str(cov)][str(N)][str(run)] = str(e)
                continue

            # run classifier for this run
            classifier = LogisticRegression(penalty=None, random_state=3456)
            classifier.fit(X_train, y_train)

            # fit roc curve or this run and store it for averaging
            if run == 0:
                lab = "Single fit"
            else:
                lab = "_nolab_"
            roc = RocCurveDisplay.from_estimator(classifier,
                                                 X_holdout,
                                                 y_holdout,
                                                 name="Singe sample ROC",
                                                 alpha=0.5,
                                                 ax=ax[i, j],
                                                 lw=0.5,
                                                 c="gray",
                                                 label=lab,
                                                 plot_chance_level=(run == n_runs - 1)
                                                 )

            # interpolate tprs to make them averageable for each fpr, store fpr and auc
            interp_tpr = np.interp(mean_fpr, roc.fpr, roc.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc.roc_auc)

            # save run details
            data_dict[str(cov)][str(N)][str(run)] = {
                "AUC": float(roc.roc_auc),
                "False positive rates": list(roc.fpr),
                "True positive rates": list(roc.tpr),
                "Cases in training set": int(np.sum(y_train)),
                "Cases in test set": int(np.sum(y_holdout))
            }

        # average tprs over all runs with a given N and cov
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        # plot averaged tpr and fpr
        ax[i, j].plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC \n(AUC = %0.2f$\pm$%0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=1,
        )

        # plot standard deviation
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax[i, j].fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="b",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        # set axes titles
        ax[i, j].set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"N={N}, cov(x,y)={cov}",
        )

        # set legend
        ax[i, j].legend(loc="lower right")

plt.savefig("_results/fig-01-01_N_cov_full_sim.png")

with open("_results/data-01-01_N_cov_full_sim.json", "w") as fp:
    fp.write(json.dumps(data_dict, indent=4, separators=(',', ': ')))
