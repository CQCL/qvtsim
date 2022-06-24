#!/usr/bin/env python

#####################################################################################
#
# Copyright 2022 Quantinuum
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
#####################################################################################

"""Function to perform confidence interval analysis."""

from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import  erf

from qvtsim.analysis_functions import original_bounds
from qvtsim.transpiler_passes import preset_passes


ecolor = {i: plt.get_cmap('tab10').colors[i] for i in range(10)}
pm = preset_passes('low', None)


def analyze_confidence_intervals(act_success: dict,
                                 ns_list: list,
                                 nc_list: list,
                                 reps: int):
    """
    Estimate coverage and width of confidence interval methods for numeric data.

    Sample act_success data over grid of len(ns_list) * len(nc_list) with different 
    shots and circuits. Each sample represents a simulated QVT experiment with
    ns shots and nc circuits. For each sample we estimate the coverage and width of
    original and bootstrap confidence interval constructions.
    
    Args:
        act_success: NumericEstimate.act_success from scalable or numeric estimate
        ns_list: list of shots to sample act_success to simulate QVT experiment
        nc_list: list of circuits to sample act_success to simulate QVT experiment
        reps: Number of times to resample each [ns, nc] pair
    """

    success = {}
    coverage = {
        'original': {(n, e): {} for n,e in act_success},
        'bootstrap': {(n, e): {} for n,e in act_success}
    }
    lower = {
        'original': {(n, e): {} for n,e in act_success},
        'bootstrap': {(n, e): {} for n,e in act_success}
    }

    for (n, e), probs in act_success.items():
        mean_probs = np.mean(probs)
        probs = np.array(probs)
        for ntrials in nc_list:
            # resample from initial set to generate resampled set of experiments
            resampled_probs = probs[
                np.random.randint(0, len(probs), size=(reps, ntrials))
            ]
            resampled_probs[resampled_probs < 0] = 0
            resampled_probs[resampled_probs > 1] = 1
                
            for shots in ns_list:
                # binimial sample from resampled set to simulate finite sampled data
                success_list = np.random.binomial(shots, resampled_probs)/shots
                success[ntrials, shots] = np.sum(success_list, 1)/ntrials

                lower['original'][n, e][ntrials, shots] = original_bounds(
                    success[ntrials, shots], 
                    ntrials
                )[0]
                coverage['original'][n, e][ntrials, shots] = sum(
                    mean_probs >= lower['original'][n, e][ntrials, shots]
                )/reps
                        
                lower['bootstrap'][n, e][ntrials, shots] = np.array(
                    [lower_bootstrap(slist, shots) 
                     for slist in success_list]
                )
                coverage['bootstrap'][n, e][ntrials, shots] = sum(
                    mean_probs >= lower['bootstrap'][n, e][ntrials, shots]
                )/reps
                
    return success, lower, coverage


def lower_bootstrap(success_list,
                    shots: int,
                    reps: int = 1000) -> dict:
    """
    Simplified semi-parameteric bootstrap simulated QV data.
    
    Notes:
        -Assumes same shots for all circuits
        
    """
    ntrials = len(success_list)
    
    probs = success_list[np.random.randint(0, ntrials, size=(reps, ntrials))]
    success_list = np.random.binomial(shots, probs)/shots
    success = np.mean(success_list, 1)
    
    thresh = 1/2 + erf(np.sqrt(2))/2

    return 2 * np.mean(success_list) - np.quantile(success, thresh)