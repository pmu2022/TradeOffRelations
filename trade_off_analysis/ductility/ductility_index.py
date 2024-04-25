# coding: utf-8
# Copyright (c) Materials Center Leoben Forschung GmbH (MCL)

import copy
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .stroh import (rot_tensor, rotate_4th_rank_tensor, stiffness_tensor_cubic,
                    stroh_formalism)

__author__ = "Olga Kovalyova"
__credits__ = ["Max Hodapp", "Franco Moitzi", "Ivan Novikov", "Oleg Peil"]
__email__ = "franco.moitzi@mcl.at"
__status__ = "Development"


def compute_o(phi, theta, lambda_):
    """
    Compute the value of o given phi, theta, and lambda.

    :param phi: Angle phi.
    :param theta: Angle theta.
    :param lambda_: 3x3 matrix lambda.
    :return: Value of o.
    """
    s = np.array([np.cos(phi), 0.0, np.sin(phi)])
    Omega = np.array(
        [
            [np.cos(theta), np.sin(theta), 0.0],
            [-np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    lambda_theta = Omega.dot(lambda_.dot(Omega.transpose()))
    lambda_theta_inv = np.linalg.inv(lambda_theta)
    o = s.dot(lambda_theta_inv.dot(s))

    return o


def compute_F12(C, theta):
    """
    Compute the value of F12 given matrix C and angle theta.

    :param C: 6x6 matrix C.
    :param theta: Angle theta.
    :return: Value of F12.
    """

    # Compute a1,a2
    S = np.linalg.inv(C)
    # Plane strain
    S_copy = copy.deepcopy(S)
    for i in range(0, 6):
        for j in range(0, 6):
            S[i, j] = S_copy[i, j] - S_copy[i, 2] * S_copy[2, j] / S_copy[2, 2]
    a = np.roots(
        [S[0, 0], -2.0 * S[0, 5], 2.0 * S[0, 1] + S[5, 5], -2.0 * S[1, 5], S[1, 1]]
    )
    # Sort the roots
    a_ = np.zeros(len(a), dtype=complex)
    ii = 0
    jj = 0
    for i in range(0, len(a)):
        if np.imag(a[i]) > 0.0:
            a_[ii] = a[i]
            ii += 1
        else:
            a_[int(len(a) / 2) + jj] = a[i]
            jj += 1
    # a1 & a2 are those with positive imaginary parts, others are complex
    # conjugates
    a1 = a_[0]
    a2 = a_[1]

    # The following values are already normalized by the K-factor
    sigma_11 = ((a1 * a2) / (a1 - a2)) * (
        (a2 / np.sqrt(np.cos(theta) + a2 * np.sin(theta)))
        - (a1 / np.sqrt(np.cos(theta) + a1 * np.sin(theta)))
    )
    sigma_22 = (1.0 / (a1 - a2)) * (
        (a1 / np.sqrt(np.cos(theta) + a2 * np.sin(theta)))
        - (a2 / np.sqrt(np.cos(theta) + a1 * np.sin(theta)))
    )
    sigma_12 = ((a1 * a2) / (a1 - a2)) * (
        (1.0 / np.sqrt(np.cos(theta) + a1 * np.sin(theta)))
        - (1.0 / np.sqrt(np.cos(theta) + a2 * np.sin(theta)))
    )

    s11 = np.real(sigma_11)
    s22 = np.real(sigma_22)
    s12 = np.real(sigma_12)

    F12 = (s22 - s11) * np.sin(theta) * np.cos(theta) + s12 * (
        np.cos(theta) ** 2 - np.sin(theta) ** 2
    )
    return F12


def compute_F12_iso(theta):
    """

    :param theta:
    :return:
    """
    F12 = (np.cos(0.5 * theta) ** 2) * np.sin(0.5 * theta)
    return F12


def compute_chi(cubic_el_consts, axes, angles, angles_in_degrees=True):
    """
    Compute the chi parameter (eq. (4) in (Mak & Curtin, JMPS 152, 2021))

    :param cubic_el_consts: List of cubic elastic constants [C11, C12, C44].
    :param axes: List with crystallographic axes in the rotated coordinate system.
                 axes[2] should be the crack front, and axes[1] should be the normal of the cleavage plane.
    :param angles: List of angles.
                   angles[0] (= phi in the Mak & Curtin paper) is the angle between the Burgers vector and the crack front,
                   and angles[1] (= theta in the Mak & Curtin paper) is the angle of the slip plane with respect to the cleavage plane.
    :param angles_in_degrees: Boolean indicating whether angles are given in degrees (default True).
    :return: Value of the chi parameter.
    """

    C11 = cubic_el_consts[0]
    C12 = cubic_el_consts[1]
    C44 = cubic_el_consts[2]
    ar = 2.0 * C44 / (C11 - C12)  # anisotropy ratio

    C_ini = stiffness_tensor_cubic(C11, C12, C44)

    # 2D rotation tensor
    Q = rot_tensor(axes[0], axes[1], axes[2])

    # stiffness tensor rotatated in crystal orientation
    C = rotate_4th_rank_tensor(Q, C_ini)

    # calculate comples matrices - Stroh formalism
    A, B, p = stroh_formalism(C)

    lamb = 0.5 * np.real(1j * A @ np.linalg.inv(B))

    phi = angles[0]
    theta = angles[1]
    if angles_in_degrees:
        phi = np.radians(angles[0])
        theta = np.radians(angles[1])

    # Now we put everything together
    o = compute_o(phi, theta, lamb)
    F12 = compute_F12(C, theta)
    chi = np.sqrt(lamb[1, 1] * o) / (np.sqrt(2.0) * np.cos(phi) * F12)

    return chi, ar


def compute_D(surf, usf, cubic_el_consts, axes, angles, angles_in_degrees=True):
    """
    Compute the ductility index

    :param surf: Surface energy.
    :param usf: Stacking fault energy.
    :param cubic_el_consts: List of cubic elastic constants [C11, C12, C44].
    :param axes: List with crystallographic axes in the rotated coordinate system.
                 axes[2] should be the crack front, and axes[1] should be the normal of the cleavage plane.
    :param angles: List of angles.
                   angles[0] (= phi in the Mak & Curtin paper) is the angle between the Burgers vector and the crack front,
                   and angles[1] (= theta in the Mak & Curtin paper) is the angle of the slip plane with respect to the cleavage plane.
    :param angles_in_degrees: Boolean indicating whether angles are given in degrees (default True).
    :return: Value of the ductility index.
    """

    chi = compute_chi(
        cubic_el_consts, axes, angles, angles_in_degrees=angles_in_degrees
    )

    D = chi * np.sqrt(usf / surf)

    return D


@dataclass
class Orientation:
    """
    Represents crystallographic orientation with axes and angles.
    """

    axes: Sequence
    angles: Sequence


orient_100_110 = Orientation(
    [np.array([0, -1, 0]), np.array([1, 0, 0]), np.array([0, 0, 1])], [35.2643897, 45.0]
)

orient_100_112 = Orientation(
    [np.array([0, -1, 1]), np.array([1, 0, 0]), np.array([0, 1, 1])], [0.0, 35.2643897]
)

orient_110_110 = Orientation(
    [np.array([1, -1, 0]), np.array([1, 1, 0]), np.array([0, 0, 1])], [35.2643897, 90.0]
)

orient_110_112 = Orientation(
    np.array([[0, 0, -1], [1, 1, 0], [1, -1, 0]]), [0.0, 54.73561031724534]
)
