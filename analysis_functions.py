#!/usr/bin/env python

##############################################################################
#
# QUANTINUUM LLC CONFIDENTIAL & PROPRIETARY.
# This work and all information and expression are the property of
# Quantinuum LLC, are Quantinuum LLC Confidential & Proprietary,
# contain trade secrets and may not, in whole or in part, be licensed,
# used, duplicated, disclosed, or reproduced for any purpose without prior
# written permission of Quantinuum LLC.
#
# In the event of publication, the following notice shall apply:
# (c) 2022 Quantinuum LLC. All Rights Reserved.
#
##############################################################################

''' Functions for plotting quantum volume data from Quantinuum. '''

from typing import Optional

import numpy as np
from scipy.special import erf


def original_bounds(success: float,
                    trials: int):
    ''' Returns bounds from original CI method. '''

    sigma = np.sqrt(success*(1 - success)/trials)
    lower_ci = success - 2*sigma
    upper_ci = success + 2*sigma

    return lower_ci, upper_ci


def pass_threshold(ntrials: int):
    ''' Returns minimum average success to pass with given ntrials. '''

    threshold = 0.5 * (
        36 + 12*ntrials + np.sqrt(1296 + 288*ntrials)
    )/(36 + 9*ntrials)
    
    return threshold


def bootstrap_bounds(qv_fitter,
                     reps: int = 1000,
                     ntrials: Optional[int] = None):
    ''' Returns bounds from bootstrap CI method. '''

    nqubits = len(qv_fitter.qubit_lists[0])

    success = bootstrap(
        qv_fitter,
        reps,
        ntrials
    )
    qv_mean = np.mean([
        qv_fitter.heavy_output_counts[f'qv_depth_{nqubits}_trial_{i}']/qv_fitter._circ_shots[f'qv_depth_{nqubits}_trial_{i}']
        for i in range(ntrials)
    ])
    lower_ci = 2*qv_mean - np.quantile(success, 1/2 + erf(np.sqrt(2))/2)
    upper_ci = 2*qv_mean - np.quantile(success, 1/2 - erf(np.sqrt(2))/2)

    return lower_ci, upper_ci
    

def bootstrap(qv_fitter,
              reps: int = 1000,
              ntrials: Optional[int] = None):
    ''' Semi-parameteric bootstrap QV data. '''
    nqubits = len(qv_fitter.qubit_lists[0])
        
    if not ntrials:
        ntrials = len(qv_fitter.heavy_output_counts)
        
    shot_list = np.array([
        qv_fitter._circ_shots[f'qv_depth_{nqubits}_trial_{i}']
        for i in range(ntrials)
    ])
    success_list = np.array([
        qv_fitter.heavy_output_counts[f'qv_depth_{nqubits}_trial_{i}']/shot_list[i]
        for i in range(ntrials)
    ])
    probs = success_list[
        np.random.randint(0, ntrials, size=(reps, ntrials))
    ]
    success_list = np.random.binomial(shot_list, probs)/shot_list
    success = np.mean(success_list, 1)
                
    return success