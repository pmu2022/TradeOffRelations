# coding: utf-8
# Copyright (c) Materials Center Leoben Forschung GmbH (MCL)
from collections.abc import Callable

import numpy as np
from numpy.polynomial.polynomial import Polynomial

__author__ = "Franco Moitzi"
__copyright__ = "Copyright 2020-, Materials Center Leoben Forschung GmbH"
__credits__ = ["Oleg Peil"]
__license__ = "All rights reserved"
__email__ = "franco.moitzi@mcl.at"
__status__ = "Development"


class VBAModelFitter(Callable):

    def __init__(self, na_degree, nb_degree):

        elements = ['Ti', 'Cr', 'Zr', 'Nb', 'Mo', 'Ta', 'W', 'V', 'Hf']
        self.valences = np.array([2, 4, 2, 3, 4, 3, 4, 3, 2], dtype=int)
        self.rows = np.array([3, 3, 4, 4, 4, 5, 5, 3, 5], dtype=int)

        self.base_elements = elements

        self.na_degree = na_degree
        self.nb_degree = nb_degree

        self.p_v1 = None
        self.p_v2 = None

    def _fit(self, param, b, concs, eles):

        valences_dict = {e: v for e, v in zip(self.base_elements, self.valences)}
        rows_dict = {e: v for e, v in zip(self.base_elements, self.rows)}

        b = np.array(b)

        widths = np.array([0.414, 0.397, 0.614, 0.667, 0.651, 0.783, 0.773, 0.418, 0.560])

        # widths = param[2:]
        k1, k2, *_ = param

        self.widths_dict = {e: v for e, v in zip(self.base_elements, widths)}

        print(self.widths_dict)

        def width_corr_dict(elem, k1, k2):
            coef_row = {
                3: 1 / k1,
                4: 1.0,
                5: 1 / k2
            }

            return coef_row[rows_dict[elem]]

        n = len(concs)

        # Create v^(1) part
        A_v1 = np.zeros((n, self.na_degree + 1))

        # Create v^(2) part
        A_v2 = np.zeros((n, self.nb_degree + 1))

        # Fill the array
        for idx, (ele, conc) in enumerate(zip(eles, concs)):

            for i in range(self.na_degree + 1):

                for e, c in zip(ele, conc):
                    A_v1[idx, i] += c * valences_dict[e] ** i * self.widths_dict[e] * \
                                    width_corr_dict(e, k1, k2)

            for i in range(self.nb_degree + 1):

                for e1, c1 in zip(ele, conc):
                    for e2, c2 in zip(ele, conc):
                        valences = (valences_dict[e1] + valences_dict[e2]) / 2

                        wij = np.sqrt(self.widths_dict[e1] *
                                      width_corr_dict(e1, k1, k2) *
                                      width_corr_dict(e2, k1, k2) *
                                      self.widths_dict[e2])

                        A_v2[idx, i] += c1 * c2 * valences ** i * wij

        A = np.hstack((A_v1, A_v2))

        v_coef = np.linalg.lstsq(A, b, rcond=None)[0]
        p_v1 = Polynomial(v_coef[:self.na_degree + 1])
        p_v2 = Polynomial(v_coef[self.na_degree + 1:])

        self.p_v1 = p_v1
        self.p_v2 = p_v2

        return p_v1, p_v2

    def __call__(self, param, b, concs, eles, return_full=True):

        valences_dict = {e: v for e, v in zip(self.base_elements, self.valences)}
        rows_dict = {e: v for e, v in zip(self.base_elements, self.rows)}

        k1, k2, *_ = param

        def width_corr_dict(elem, k1, k2):
            coef_row = {3: 1 / k1, 4: 1.0, 5: 1 / k2}

            return coef_row[rows_dict[elem]]

        p_v1, p_v2 = self._fit(param, b, concs, eles)

        def eval_vba(concs, eles, p_v1, p_v2):

            energy = 0

            for c, e in zip(concs, eles):
                nv1 = valences_dict[e]
                energy += c * p_v1(nv1) * self.widths_dict[e] * width_corr_dict(e, k1,
                                                                                k2)

            for c1, e1 in zip(concs, eles):
                for c2, e2 in zip(concs, eles):
                    nv1 = valences_dict[e1]
                    nv2 = valences_dict[e2]

                    wij = np.sqrt(self.widths_dict[e1]
                                  * width_corr_dict(e1, k1, k2)
                                  * width_corr_dict(e2, k1, k2)
                                  * self.widths_dict[e2])

                    nv = (nv1 + nv2) / 2.0

                    energy += c1 * c2 * p_v2(nv) * wij

            return energy

        vba_energies = np.zeros_like(b)

        for idx, (ele, conc) in enumerate(zip(eles, concs)):
            vba_energies[idx] = eval_vba(conc, ele, p_v1, p_v2)

        return vba_energies

