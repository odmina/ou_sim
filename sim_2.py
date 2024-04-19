import numpy as np
import OU_process_2v
import person


# SUPLEMENTARY MATERIALS #

####################################################################################################
# 2: Stationary non-zero correlation between variables puts a limit on positive cross-correlation
# between predictor and dependent variable! This limit is set by the fact, that difussion matrix
# has to be a valid covariance-matrix, so it has to be positive-definite.
# Funny byproduct is that the more a variable is usefull in cross-sectional predictions, the less
# usefull it is for time-lagged predictions. Or it has to have a really strong
####################################################################################################

