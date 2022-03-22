# -*- coding: utf-8 -*-
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

"""Functions for analyzing qiskit's QVFitter objects."""

import numpy as np


def success_lower(success: float,
                  trials: int):
    """Returns lower bound of confidence interval."""


    sigma = np.sqrt(success*(1 - success)/trials)
    lower_ci = success - 2*sigma
    
    return lower_ci
    
    
def pass_threshold(ntrials: int):
    """Returns minimum average success to pass with given ntrials."""

    threshold = 0.5 *(36 + 12*ntrials + np.sqrt(1296 + 288*ntrials))/(36 + 9*ntrials)
    
    return threshold
    
    
def bootstrap(qv_fitter,
              reps: int = 1000):
    """Semi-parameteric bootstrap QV fitter object."""
    shots = np.array(list(qv_fitter._circ_shots.values()))
    ntrials = len(qv_fitter.heavy_output_counts)
        
    success_list = np.array(
        [qv_fitter.heavy_output_counts[key]/qv_fitter._circ_shots[key] 
         for key in qv_fitter.heavy_output_counts]
    )
    
    probs = success_list[np.random.randint(0, ntrials, size=(reps, ntrials))]
    success_list = np.random.binomial(shots, probs)/shots
    success = np.mean(success_list, 1)
        
    return success
