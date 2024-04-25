# coding: utf-8
# Copyright (c) Materials Center Leoben Forschung GmbH (MCL)

import numpy as np

__author__ = "Franco Moitzi"
__copyright__ = "Copyright 2020-, Materials Center Leoben Forschung GmbH"
__credits__ = ["Oleg Peil"]
__license__ = "All rights reserved"
__email__ = "franco.moitzi@mcl.at"
__status__ = "Development"


def calc_c11_c12(c_prime: float, bulkmodulus: float):
    """
    Calculate the values of C11 and C12 given the value of C'.

    :param c_prime: Value of C'.
    :param bulkmodulus: Bulk modulus.
    :return: Tuple containing values of C11 and C12.
    """

    b = [bulkmodulus * 3, c_prime * 2]
    A_mat = np.array([[1, 2], [1, -1]])
    c = np.linalg.solve(A_mat, b)
    c_11 = c[0]
    c_12 = c[1]

    return c_11, c_12
