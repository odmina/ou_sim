import numpy as np
import OU_process_2v
import person


####################################################################################################
# 2: When measurements are taken at different timepoints, predictions quality depends from:
# - Autocorrelation
# - Cross-covariance
# Use one day timewindow, take 0.5 correlation in Gamma
# Next step - analyze different timewindows
####################################################################################################

# dt - measurements are one day apart
dt = 1

# %% set simulation parameters
auto_corrs = [0.1, 0.3, 0.5, 0.7, 0.9]
cross_corrs = [0.0, 0.3, 0.5, 0.7]
sample_size = 1000
Gamma = np.array([[1, 0.5], [0.5, 1]])
n_runs = 1000

# %% prepare plot & data dictionary
fig, ax = plt.subplots(nrows=len(auto_corrs),
                       ncols=len(cross_corrs),
                       figsize=(16, 20),
                       sharex=True,
                       sharey=True)
data_dict = {}

# %% rng set state
rng = np.random.default_rng(3456)

# %% simulate the data and fit ROC curves
for i, auto_corr in enumerate(auto_corrs):
    for j, cross_corr