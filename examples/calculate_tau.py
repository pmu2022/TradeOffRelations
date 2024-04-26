# coding: utf-8
# Copyright (c) Materials Center Leoben Forschung GmbH (MCL)


from vba_model.model import calc_vba_elastic, calc_vba_misfits, compute_vba_surf, compute_vba_usf, \
    calc_vba_tau_sss
from vba_model.model import p_v1_surf_110, p_v2_surf_110
from vba_model.model import p_v1_usf_112, p_v2_usf_112

__author__ = "Franco Moitzi"
__copyright__ = "Copyright 2020-, Materials Center Leoben Forschung GmbH"
__credits__ = ["Oleg Peil"]
__license__ = "All rights reserved"
__email__ = "franco.moitzi@mcl.at"
__status__ = "Development"

concs = [0.25, 0.25, 0.25, 0.25]

elements = ['Mo', 'Nb', 'Ti', 'Ta']

tau, *_ = calc_vba_tau_sss(concs, elements)

usf = compute_vba_usf(concs, elements, p_v1_usf_112, p_v2_usf_112)
surf = compute_vba_surf(concs, elements, p_v1_surf_110, p_v2_surf_110)

_, equilibrium_volume = calc_vba_misfits(concs, elements)

cp, c44, bulk = calc_vba_elastic(concs, elements)

rice_criterion = surf / usf

print("Rice", rice_criterion)
print("Tau", tau)
