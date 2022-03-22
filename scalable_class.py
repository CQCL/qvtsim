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

"""Scalable estimation class."""

import os
import pickle
import numpy as np
from scipy.special import comb, factorial
from scipy.interpolate import CubicSpline

from estimation_class import QVEstimate
from utils import convert, estimate_errors

class ScalableEstimate(QVEstimate):
    """
    Scalable estimation method for QVT heavy output probabilities.
    
    """
    def __init__(self,
                 qubit_list: list,
                 optimization_lvl: list,
                 error_name: str,
                 error_list: list):

        super().__init__(qubit_list, optimization_lvl, error_name, error_list)
        
        # initialized ideal success
        self.ideal_success = {}
        for n in self.qubit_list:
            if n <= 11:
                success_list = {            # found from numerical estimation 
                    2: 0.7926894245725548,
                    3: 0.8486307351563217,
                    4: 0.8398467848987355,
                    5: 0.8565091924431827,
                    6: 0.8513469535938657,
                    7: 0.8571307561174293,
                    8: 0.8508395392789203,
                    9: 0.8526381863855619,
                    10: 0.8484872555591324,
                    11: 0.8488043234898409
                }
                self.ideal_success[n] = success_list[n]
                
            else:
                self.ideal_success[n] =  (1 + np.log(2))/2
                
        # initialize predicted actual success
        self.act_success = {}
    
    def run(self, 
            method):
        """
        Run sample over axes parameters.
        
        """
        self.act_success = {
            (n, e): self.single(n, edict, method)
            for n in self.qubit_list
            for e, edict in self.error_dict.items()
        }
    
    def single(self,
               nqubits: int,
               error_dict: dict,
               method: str):
        """
        Returns probability of success based on analytic model.
        
        Args:
            axes: variable parameters
            
        Returns:
            success:    success probability from model
            prob_pass:  probility to pass QV test
            
        
        """
        tq_dep = (convert(1 - error_dict['tq_dep'], 4, 'avg', 'dep')
                  * convert(1 - error_dict['tq_coh'], 4, 'avg', 'dep'))
                  
        tq_other = (convert(1 - error_dict['tq_dph'], 4, 'avg', 'dep')
                    * convert(convert(1 - error_dict['tq_cross'], 2, 'avg', 'proc'), 4, 'proc', 'dep'))

        sq_dep = convert(convert(convert(1 - error_dict['sq_dep'], 2, 'avg', 'dep')
                                 * convert(1 - error_dict['sq_coh'], 2, 'avg', 'dep')
                                 * convert(1 - error_dict['sq_dph'], 2, 'avg', 'dep'), 2, 'dep', 'proc')**2,
                         4, 'proc', 'dep')

        # optimization determines gates per depth
        if self.optimization_lvl == 'low':
            block_depth = 3
            rounds = nqubits
            p_eff = 1  # effective depolarizing rate from approximate gates
            
        elif self.optimization_lvl == 'medium':
            block_depth = 3
            rounds = expected_gates(nqubits)/(3 * (nqubits//2))
            p_eff = 1
            
        elif self.optimization_lvl == 'high':
            block_depth, base_fid = num_approx_gates(estimate_errors(error_dict))
            rounds = expected_gates(nqubits)/(3 * (nqubits//2))
            p_eff = convert(base_fid, 4, 'avg', 'dep')
            
        else:
            raise NameError(f'No compiler option {self.optimization_lvl}')
            
        # block errors
        block_dep = p_eff * (sq_dep * tq_dep * tq_other) ** block_depth  # dep sq*tq for block (arbitrary SU(4))
        
        # prep errors
        prep_dep = (1 - error_dict['prep']) ** nqubits
        
        # measurement errors
        meas_dep = (1 - error_dict['meas']) ** nqubits
        
        if method == 'dep':
            # total gate errors
            depth1_proc = convert(block_dep, 4, 'dep', 'proc') ** (nqubits//2)  # block dep. -> depth-1 proc. fid        
            depth1_dep = convert(depth1_proc, 2**nqubits, 'proc', 'dep')  # depth-1 proc. fid -> depth-1 dep.
            circ_dep = sq_dep * depth1_dep ** rounds  # depth1 dep. -> total dep.
        
        elif method == 'avg':
            depth1_avg = convert(block_dep, 4, 'dep', 'avg') ** (nqubits//2)
            depth1_dep = convert(depth1_avg, 2**nqubits, 'avg', 'dep')  # depth-1 proc. fid -> depth-1 dep.
            circ_dep = sq_dep * depth1_dep ** rounds  # depth1 dep. -> total dep.
            
        success = self.ideal_success[nqubits] * circ_dep * prep_dep * meas_dep + (1 - circ_dep * meas_dep * prep_dep)/2  # heavy probability

        return success



def num_pairs(n):
    """Returns number of arrangments for nqubits."""
    
    return factorial(n)/(factorial(n//2) * 2**(n//2))


def no_repeats(n):
    """Returns number of arrangments with no repeats."""
    
    return sum((-1)**k * comb(n//2, k) * num_pairs(n - 2*k) for k in range(n//2+1))


def nrepeats(n, k):
    """Returns number of arrangements with exactly k repeats."""
    
    return comb(n//2, k) * no_repeats(n - 2*k)


def gate_dist(n):
    """Returns fraction of arrangments with each number of repeats."""
    
    tot_arangments = num_pairs(n)
    
    return {k: nrepeats(n, k)/tot_arangments for k in range(n//2+1)}


def expected_gates(n):
    """Returns number of expected gates."""
    
    tot_arangments = num_pairs(n)
    
    dist = gate_dist(n)
    
    return 3*(n//2) + (n - 1) * sum(3 * (n//2 - k) * dist[k] for k in range(n//2+1))


def total_gates(n):
    """Returns total number of gates without combines."""
    
    return 3*(n//2)*n
    
    
def num_approx_gates(fidelity: float) -> float:
    """Returns number of gates from approximate optimzation scheme."""

    script_dir = os.path.dirname(__file__)
    abs_file_path = os.path.join(script_dir + '/approx_gate_counts.pkl')
    with open(abs_file_path, 'rb') as f:
        approx_gates = pickle.load(f)
        fid = pickle.load(f)
        
    cs_gates = CubicSpline(list(approx_gates.keys()), list(approx_gates.values()))
    gates = 3 * cs_gates(1 - fidelity)
    
    cs_fid = CubicSpline(list(fid.keys()), list(fid.values()))
    fid_out = cs_fid(1 - fidelity)
    
    return gates, fid_out
    
    