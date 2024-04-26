# coding: utf-8
# Copyright (c) Materials Center Leoben Forschung GmbH (MCL)

import numpy as np
from numpy.polynomial import Polynomial
from scipy.constants import Boltzmann as kB

__author__ = "Franco Moitzi"
__copyright__ = "Copyright 2020-, Materials Center Leoben Forschung GmbH"
__credits__ = ["Oleg Peil"]
__license__ = "All rights reserved"
__email__ = "franco.moitzi@mcl.at"
__status__ = "Development"

# Constants
A_to_m = 1e-10
prefactor = 0.04
e_prefactor = 2.0
alpha = 1.0 / 12.0
epsilon_zero = 10**4
epsilon = 10**-4
tau_prefactor = 0.6

elements = ["Ti", "Cr", "Zr", "Nb", "Mo", "Ta", "W", "V", "Hf"]
valences = np.array([2, 4, 2, 3, 4, 3, 4, 3, 2], dtype=int)
rows = np.array([3, 3, 4, 4, 4, 5, 5, 3, 5], dtype=int)

valences_dict = {e: v for e, v in zip(elements, valences)}
rows_dict = {e: v for e, v in zip(elements, rows)}
widths = np.array([0.414, 0.397, 0.614, 0.667, 0.651, 0.783, 0.773, 0.418, 0.715])

widths_dict = {e: v for e, v in zip(elements, widths)}

p_v11 = np.poly1d([-21.0273769, 131.73912553, -207.91994684])
p_v12 = np.poly1d([-71.30270617, 485.66738787, -808.43731851])
p_v13 = np.poly1d([19.75260076, -73.42767866, 20.46637474])

p_v21 = np.poly1d([11.62689882, -53.08459983, 130.52713348, -102.14263859])
p_v22 = np.poly1d([112.35643006, -874.28133674, 2094.44972552, -1353.36934647])
p_v23 = np.poly1d([86.0427864, -771.96067217, 2315.77910625, -2052.15884144])

p1 = [-350769.1038239886, -2853.5662932845244, -6232.327396369234]
p2 = [302561.1287094357, 53385.95837858341, -8777.124452570419, 1784.420589419902]
p3 = [-380115.49353108235, 10578.27821958051, -6521.148138822875]
p4 = [373742.39261398365, -6997.418121039415, 8973.25837044404, -244.444017792858]
p5 = [-5155323.351319714, 283425.3921217114, -27945.170436472486]
p6 = [6062115.141338895, -1227342.4895198627, 352728.97935198527, -35351.35603358079]
p7 = [-5070917.435821753, 178247.88892555656, -18396.75138995952]
p8 = [5673897.134274427, -767293.0208389125, 213224.31097534188, -20405.59966464692]

p_v1_usf_110 = Polynomial(p1)
p_v2_usf_110 = Polynomial(p2)

p_v1_usf_112 = Polynomial(p3)
p_v2_usf_112 = Polynomial(p4)

p_v1_surf_100 = Polynomial(p5)
p_v2_surf_100 = Polynomial(p6)

p_v1_surf_110 = Polynomial(p7)
p_v2_surf_110 = Polynomial(p8)


def width_corr_dict(elem, k1, k2):
    coef_row = {3: 1 / k1, 4: 1.0, 5: 1 / k2}

    return coef_row[rows_dict[elem]]


def calculate_tau_yield_zero(prefactor, alpha, mu, nu, delta):
    """
    Calculates the tau yield
    """
    return (
        prefactor
        * alpha ** (-1.0 / 3.0)
        * mu
        * ((1 + nu) / (1 - nu)) ** (4.0 / 3.0)
        * delta ** (4.0 / 3.0)
    )


def calculate_delta_e_b(prefactor, alpha, mu, nu, delta, burgers):
    """
    Calculated the energy barrier
    """

    burgers_converted = burgers * A_to_m  # Units m
    mu_converted = mu * 10**9  # Pascal

    return (
        prefactor
        * alpha ** (1.0 / 3.0)
        * mu_converted
        * burgers_converted**3.0
        * ((1 + nu) / (1 - nu)) ** (2.0 / 3.0)  # no units
        * delta ** (2.0 / 3.0)
    )  # no units


def high_stress_tau_yield(tau_zero, temperature, ene_b, epsilon_zero, epsilon):
    """High stress/lower temperature"""
    return tau_zero * (
        1.0
        - ((kB * temperature / ene_b) * np.log(epsilon_zero / epsilon)) ** (2.0 / 3.0)
    )


def low_stress_tau_yield(tau_zero, temperature, ene_b, epsilon_zero, epsilon):
    """Low stress/high temperature"""
    return tau_zero * np.exp(
        -1 / 0.55 * (kB * temperature / ene_b) * np.log(epsilon_zero / epsilon)
    )


def calculate_average_mu(c_44, c_11, c_12):
    """
    Calculate the average shear modulus (mu) using the Voigt-Reuss-Hill approximation.

    :param float c_44: C44
    :param float c_11: C11
    :param float c_12: C12

    :return: Average shear modulus.
    :rtype: float
    """
    return np.sqrt(0.5 * c_44 * (c_11 - c_12))


def calculate_average_nu(b_modulus, mu):
    """
    Calculate the average Poisson's ratio (nu) using the bulk modulus and shear modulus.

    :param float b_modulus: Bulk modulus.
    :param float mu: Shear modulus.

    :return: Average Poisson's ratio.
    :rtype: float
    """
    return (3.0 * b_modulus - 2.0 * mu) / (2.0 * (3.0 * b_modulus + mu))


def calc_vba_elastic(concs, eles):
    """
    Calculate the elastic parameters with VPA
    :param concs:
    :param eles:
    :return:
    """
    cp = 0
    c4 = 0
    bulk = 0

    k1 = 0.81
    k2 = 1.14

    for c, e in zip(concs, eles):
        nv1 = valences_dict[e]
        cp += c * p_v11(nv1) * widths_dict[e] * width_corr_dict(e, k1, k2)
        c4 += c * p_v12(nv1) * widths_dict[e] * width_corr_dict(e, k1, k2)
        bulk += c * p_v13(nv1) * widths_dict[e] * width_corr_dict(e, k1, k2)

    for c1, e1 in zip(concs, eles):
        for c2, e2 in zip(concs, eles):
            nv1 = valences_dict[e1]
            nv2 = valences_dict[e2]

            wab = np.sqrt(
                widths_dict[e1]
                * width_corr_dict(e1, k1, k2)
                * width_corr_dict(e2, k1, k2)
                * widths_dict[e2]
            )

            nv = (nv1 + nv2) / 2.0

            cp += c1 * c2 * p_v21(nv) * wab
            c4 += c1 * c2 * p_v22(nv) * wab
            bulk += c1 * c2 * p_v23(nv) * wab

    return cp, c4, bulk


def calc_vba_misfits(concs, eles):
    k1 = 8.019e-01
    k2 = 1.247e00

    v1 = np.poly1d([1.15149047, 8.99185795])
    v2 = np.poly1d([0.22283889, -2.67231669, -3.74311463])

    widths = np.array(
        [
            4.467e00,
            3.887e00,
            7.001e00,
            6.522e00,
            5.639e00,
            8.275e00,
            7.245e00,
            4.797e00,
            7.150e00,
        ]
    )

    widths_dict = {e: v for e, v in zip(elements, widths)}

    dvoldc = np.zeros_like(concs, dtype=np.float64)
    v_misfit = np.zeros_like(concs, dtype=np.float64)

    volume = 0.0

    for idx, ele in enumerate(eles):

        dval = valences_dict[ele]

        w_A = widths_dict[ele] * width_corr_dict(ele, k1, k2)

        dvoldc[idx] = v1(dval) * w_A

        volume += concs[idx] * v1(dval) * w_A

        for jdx, ele2 in enumerate(eles):
            d1 = valences_dict[ele]
            d2 = valences_dict[ele2]

            w_A = widths_dict[ele] * width_corr_dict(ele, k1, k2)
            w_B = widths_dict[ele2] * width_corr_dict(ele2, k1, k2)
            w_AB = np.sqrt(w_A * w_B)

            d12 = np.mean([d1, d2])
            dvoldc[idx] += 2.0 * concs[jdx] * v2(d12) * w_AB

            volume += concs[idx] * concs[jdx] * v2(d12) * w_AB

    for idx, ele in enumerate(eles):
        v_misfit[idx] = dvoldc[idx] - np.sum(concs * dvoldc)

    return v_misfit, volume


def calc_vba_tau_sss(concs, eles, temperature=300):
    concs = np.array(concs)
    eles = np.array(eles)

    args = np.ravel(np.argwhere(~np.isclose(concs, 0.0)))

    misfit_volumes, equilibrium_volume = calc_vba_misfits(concs[args], eles[args])

    delta = np.sqrt(np.sum(concs[args] @ misfit_volumes**2)) / (equilibrium_volume)

    burgers = np.cbrt(2.0 * equilibrium_volume) * np.sqrt(3) / 2

    cpr, c44, bulkmodulus = calc_vba_elastic(concs[args], eles[args])

    b = [bulkmodulus * 3, cpr * 2]
    A_mat = [[1, 2], [1, -1]]
    c = np.linalg.solve(A_mat, b)
    c11 = c[0]
    c12 = c[1]

    mu = calculate_average_mu(c44, c11, c12)
    nu = calculate_average_nu(bulkmodulus, mu)

    tau_zero = calculate_tau_yield_zero(prefactor, alpha, mu, nu, delta) * tau_prefactor

    ene_b = calculate_delta_e_b(e_prefactor, alpha, mu, nu, delta, burgers)

    tau_ht = (
        high_stress_tau_yield(tau_zero, temperature, ene_b, epsilon_zero, epsilon)
        * 1000
    )
    tau_lt = (
        high_stress_tau_yield(tau_zero, temperature, ene_b, epsilon_zero, epsilon)
        * 1000
    )

    factor = tau_ht / tau_zero

    if factor > 0.5:
        tau = tau_ht
    else:
        tau = tau_lt

    if np.isnan(tau):
        tau = 0.0

    mf = np.zeros_like(concs)
    mf[args] = misfit_volumes

    return np.concatenate(([tau], [nu], [mu], mf, [delta]))


def compute_vba_usf(concs, eles, p_v1, p_v2):
    """

    This will return \gamma * a^2

    :param concs:
    :param eles:
    :param p_v1:
    :param p_v2:
    :return:
    """
    k1, k2 = 0.66811995, 0.83232495
    energy = 0

    for c, e in zip(concs, eles):
        nv1 = valences_dict[e]
        energy += c * p_v1(nv1) * widths_dict[e] * width_corr_dict(e, k1, k2)

    for c1, e1 in zip(concs, eles):
        for c2, e2 in zip(concs, eles):
            nv1 = valences_dict[e1]
            nv2 = valences_dict[e2]

            wij = np.sqrt(
                widths_dict[e1]
                * width_corr_dict(e1, k1, k2)
                * width_corr_dict(e2, k1, k2)
                * widths_dict[e2]
            )

            nv = (nv1 + nv2) / 2.0

            energy += c1 * c2 * p_v2(nv) * wij

    return energy


def compute_vba_surf(concs, eles, p_v1, p_v2):
    """

    Important: This will return \gamma * a^2

    :param concs:
    :param eles:
    :param p_v1:
    :param p_v2:
    :return:
    """

    k1, k2 = 0.67591635, 1.00480069
    energy = 0

    for c, e in zip(concs, eles):
        nv1 = valences_dict[e]
        energy += c * p_v1(nv1) * widths_dict[e] * width_corr_dict(e, k1, k2)

    for c1, e1 in zip(concs, eles):
        for c2, e2 in zip(concs, eles):
            nv1 = valences_dict[e1]
            nv2 = valences_dict[e2]

            wij = np.sqrt(
                widths_dict[e1]
                * width_corr_dict(e1, k1, k2)
                * width_corr_dict(e2, k1, k2)
                * widths_dict[e2]
            )

            nv = (nv1 + nv2) / 2.0

            energy += c1 * c2 * p_v2(nv) * wij

    return energy
