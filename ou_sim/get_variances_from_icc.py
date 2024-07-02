def get_variances_from_icc(icc, **kwargs):
    """@markdown
    This function calculates how much variance will produce provided ICC.

    It requires argument `icc` (between 0 and 1) and one of the two:

    `E_within` - expected value of the within set variance
    `between` - variance of the sets expected values
    """

    E_within = kwargs.get('E_within', None)
    between = kwargs.get('between', None)

    assert icc >= 0 and icc <= 1, "ICC should be between O and 1"

    if E_within is None:
        assert between > 0, "Variances should be > 0"
        E_within = between * ((1-icc)/icc)

    if between is None:
        assert E_within > 0, "Variances should be > 0"
        between = E_within * (icc/(1-icc))

    return {"icc": icc, "E_within": E_within, "between": between}
