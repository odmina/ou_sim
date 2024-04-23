import numpy as np


def cor_to_cov(cor_matrix, variances):
    """
    Function to turn correlation matrix to covriance matrix.

    It will return valid covariance matrix if you provide
    valud correlation matrix and valid variances.
    It does not perform checks, so if your input is not valid
    it will return something, but it may not be a valid
    covariance matrix.
    """

    sds = np.sqrt(variances)
    sds_outer = np.outer(sds, sds)
    cov_matrix = cor_matrix * sds_outer
    return cov_matrix
