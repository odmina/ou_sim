import numpy as np
from OU_process_2v import OU_process_2v
from person import person
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt


####################################################################################################
# 1: Accuracy of cross sectional prediction is driven by stationary covariance
####################################################################################################

####################################################################################################
# 1.2: When interindividual differences are present, accuracy of the predictions depends on
# cross sectional correlation which is a weighted average of within and between person correlation
# (show it for different ICCs and between-within correlations of different levels)
####################################################################################################

