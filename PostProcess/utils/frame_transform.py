import math
import numpy as np


def skew_symmetric(a):
    """
    skew_symmetric - Calculates skew-symmetric matrix

    This function created 10/9/2022 by Jun Choi

    Inputs:
        a       3-element(3x1) vector
    Outputs:
        A       3x3 matrix
    """
    A = np.zeros((3, 3))
    A[0, 1] = -a[2, 0]
    A[0, 2] = a[1, 0]
    A[1, 0] = a[2, 0]
    A[1, 2] = -a[0, 0]
    A[2, 0] = -a[1, 0]
    A[2, 1] = a[0, 0]

    return A
