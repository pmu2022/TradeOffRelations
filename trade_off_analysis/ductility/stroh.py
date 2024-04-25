# coding: utf-8

import numpy as np
from numpy import linalg as la

__author__ = "Olga Kovalyova"
__credits__ = ["Max Hodapp", "Franco Moitzi", "Ivan Novikov", "Oleg Peil"]
__email__ = "franco.moitzi@mcl.at"
__status__ = "Development"


def stiffness_tensor_cubic(C11, C12, C44):
    """
    Compute the stiffness tensor in cubic coordinates.

    :param C11: Elastic constant C11.
    :param C12: Elastic constant C12.
    :param C44: Elastic constant C44.
    :return: Stiffness tensor C.
    """
    # Stiffness tensor in cubic coordiantes
    C = np.zeros((6, 6))

    C[0, 0] = C11
    C[0, 1] = C12
    C[3, 3] = C44

    C[1, 1] = C[0, 0]
    C[2, 2] = C[0, 0]

    C[1, 0] = C[0, 1]
    C[2, 0] = C[0, 1]
    C[0, 2] = C[0, 1]
    C[1, 2] = C[0, 1]
    C[2, 1] = C[0, 1]

    C[4, 4] = C[3, 3]
    C[5, 5] = C[3, 3]

    return C


def rot_tensor(n0, n1, n2):
    """
    Compute the rotation tensor.

    :param n0: First column vector of rotation tensor.
    :param n1: Second column vector of rotation tensor.
    :param n2: Third column vector of rotation tensor.
    :return: Rotation tensor.
    """
    n00 = n0 / la.norm(n0)
    n11 = n1 / la.norm(n1)
    n22 = n2 / la.norm(n2)

    e0 = np.array([1, 0, 0])
    e1 = np.array([0, 1, 0])
    e2 = np.array([0, 0, 1])

    Q = np.zeros((3, 3))

    Q[0, 0] = np.dot(n00, e0)
    Q[0, 1] = np.dot(n00, e1)
    Q[0, 2] = np.dot(n00, e2)

    Q[1, 0] = np.dot(n11, e0)
    Q[1, 1] = np.dot(n11, e1)
    Q[1, 2] = np.dot(n11, e2)

    Q[2, 0] = np.dot(n22, e0)
    Q[2, 1] = np.dot(n22, e1)
    Q[2, 2] = np.dot(n22, e2)

    return Q


def rotate_4th_rank_tensor(Q, C_ini):
    """

    :param Q:
    :param C_ini:
    :return:
    """
    K1 = np.array(
        (
            [Q[0, 0] * Q[0, 0], Q[0, 1] * Q[0, 1], Q[0, 2] * Q[0, 2]],
            [Q[1, 0] * Q[1, 0], Q[1, 1] * Q[1, 1], Q[1, 2] * Q[1, 2]],
            [Q[2, 0] * Q[2, 0], Q[2, 1] * Q[2, 1], Q[2, 2] * Q[2, 2]],
        )
    )
    K2 = np.array(
        (
            [Q[0, 1] * Q[0, 2], Q[0, 2] * Q[0, 0], Q[0, 0] * Q[0, 1]],
            [Q[1, 1] * Q[1, 2], Q[1, 2] * Q[1, 0], Q[1, 0] * Q[1, 1]],
            [Q[2, 1] * Q[2, 2], Q[2, 2] * Q[2, 0], Q[2, 0] * Q[2, 1]],
        )
    )

    K3 = np.array(
        (
            [Q[1, 0] * Q[2, 0], Q[1, 1] * Q[2, 1], Q[1, 2] * Q[2, 2]],
            [Q[2, 0] * Q[0, 0], Q[2, 1] * Q[0, 1], Q[2, 2] * Q[0, 2]],
            [Q[0, 0] * Q[1, 0], Q[0, 1] * Q[1, 1], Q[0, 2] * Q[1, 2]],
        )
    )

    K4 = np.array(
        (
            [
                Q[1, 1] * Q[2, 2] + Q[1, 2] * Q[2, 1],
                Q[1, 2] * Q[2, 0] + Q[1, 0] * Q[2, 2],
                Q[1, 0] * Q[2, 1] + Q[1, 1] * Q[2, 0],
            ],
            [
                Q[2, 1] * Q[0, 2] + Q[2, 2] * Q[0, 1],
                Q[2, 2] * Q[0, 0] + Q[2, 0] * Q[0, 2],
                Q[2, 0] * Q[0, 1] + Q[2, 1] * Q[0, 0],
            ],
            [
                Q[0, 1] * Q[1, 2] + Q[0, 2] * Q[1, 1],
                Q[0, 2] * Q[1, 0] + Q[0, 0] * Q[1, 2],
                Q[0, 0] * Q[1, 1] + Q[0, 1] * Q[1, 0],
            ],
        )
    )

    KK1 = np.concatenate((K1, 2 * K2), axis=1)
    KK2 = np.concatenate((K3, K4), axis=1)

    KK = np.concatenate((KK1, KK2), axis=0)

    C = KK @ C_ini @ KK.T  # rotated stiffness tensor

    return C


def stroh_formalism(C):
    """

    :param C:
    :return:
    """
    Qu = np.array(
        (
            [C[0, 0], C[0, 5], C[0, 4]],
            [C[0, 5], C[5, 5], C[4, 5]],
            [C[0, 4], C[4, 5], C[4, 4]],
        )
    )

    R = np.array(
        (
            [C[0, 5], C[0, 1], C[0, 3]],
            [C[5, 5], C[1, 5], C[3, 5]],
            [C[4, 5], C[1, 4], C[3, 4]],
        )
    )

    T = np.array(
        (
            [C[5, 5], C[1, 5], C[3, 5]],
            [C[1, 5], C[1, 1], C[1, 3]],
            [C[3, 5], C[1, 3], C[3, 3]],
        )
    )

    N1 = -1 * la.inv(T) @ R.T
    N2 = la.inv(T)
    N3 = R @ la.inv(T) @ R.T - Qu

    NN1 = np.concatenate((N1, N2), axis=1)
    NN2 = np.concatenate((N3, N1.T), axis=1)
    N = np.concatenate((NN1, NN2), axis=0)  # fundamental elasticity matrix

    u, v = la.eig(N)  # u - eigen values, v- eigen vectors

    kk1 = -1  # dummy index
    kk2 = 2  # dummy index

    p = np.zeros(len(u), dtype=complex)  # p from Stroh formalism
    AB = np.zeros(
        (np.shape(v)), dtype=complex
    )  # AB corresponds to [A;B] from Stroh formalism

    for i in range(len(u)):
        if np.imag(u[i]) > 0:
            kk1 = kk1 + 1
            p[kk1] = u[i]
            AB[::, kk1] = v[::, i]
        else:
            kk2 = kk2 + 1
            p[kk2] = u[i]
            AB[::, kk2] = v[::, i]

    for i in range(3):
        if np.absolute(np.real(p[i] - p[i + 3])) > 1e-6 * np.absolute(
            np.real(p[i])
        ) and p.absolute(np.imag(p[i] - p[i + 3])) > 1e-6 * np.absolute(np.imag(p[i])):
            raise TypeError(
                "wrong order of p entries; check it manually and consult "
                "Stroh formalism"
            )

    J = np.zeros(np.shape(v))
    for i in range(3):
        J[i, i + 3] = 1
        J[i + 3, i] = 1

    AB_n = np.zeros((np.shape(v)), dtype=complex)
    for i in range(len(J)):
        AB_n[:, i] = AB[:, i] / np.sqrt(AB[:, i].T @ J @ AB[:, i])

    A = np.zeros((3, 3), dtype=complex)
    B = np.zeros((3, 3), dtype=complex)
    for i in range(3):
        A[:, i] = AB_n[0:3, i]
        B[:, i] = AB_n[3:6, i]

    return A, B, p
