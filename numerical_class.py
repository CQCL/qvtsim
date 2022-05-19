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

"""Numerical estimation class."""

import numpy as np
import pickle
from datetime import datetime

from qiskit import Aer, execute, QuantumCircuit
import qiskit.ignis.verification.quantum_volume as qv

from estimation_class import QVEstimate
from transpiler_passes import preset_passes
from error_models import arbitrary_noise
from utils import gate_counts, convert, qv_circuits, binstr


class NumericalEstimate(QVEstimate):
    """Numerical estimation method for QVT heavy output probabilities."""

    def __init__(self,
                 qubit_list: list,
                 optimization_lvl: list,
                 error_name: str,
                 error_list: list,
                 ntrials: int):

        super().__init__(qubit_list, optimization_lvl, error_name, error_list)
        
        self.ntrials = ntrials
        self.gate_counts = {}
        
    def run(self):
        """Runs numerical simulation over error_dict and qubit list. """

        self.act_success = {}
        for n in self.qubit_list:
            start = datetime.today()
            noise_model = {
                e: arbitrary_noise(edict, n, True) 
                for e, edict in self.error_dict.items()
            }
            qv_circs, heavy_outputs, self.ideal_success[n] = generate_ideal(
                n, 
                self.ntrials
            )
            if self.optimization_lvl != 'high':
                # Everything but 'high' is independent of the error rate so done 
                # outside next loop
                pm = preset_passes(self.optimization_lvl)
                qv_circs_new = [pm.run(qc) for qc in qv_circs]
                self.gate_counts[n] = gate_counts(qv_circs_new)
                
            for e in self.error_dict:
                if self.optimization_lvl == 'high':
                    # 'high' optimization is set based on error rate so done 
                    # inside loop
                    transpiler_options = {
                        'tol': estimate_errors(self.error_dict[e]), 
                        'mirror': True
                    }
                    pm = preset_passes('high', transpiler_options)
                    qv_circs_new = [pm.run(qc) for qc in qv_circs]
                    self.gate_counts[n] = gate_counts(qv_circs_new)

                self.act_success[n, e] = act_outcomes(
                    qv_circs_new,
                    noise_model[e],
                    heavy_outputs,
                    self.optimization_lvl,
                )            
                
            time = datetime.today() - start
            print(f'finished n={n}, time={time}')


def generate_ideal(nqubits: int,
                   reps: int,
                   savefile: bool = True):
    """
    Generate ideal circuits, heavy_outcomes, and success for all qubit numbers.
    
    Args:
        nqubits: number of qubits
        reps: number of random circuits
        savefile: if True then save ideal info to file
        
    """
    backend_ideal = Aer.get_backend('statevector_simulator')
    
    # circuit generation
    qv_circs, qv_circs_nomeas = qv_circuits(nqubits, reps)
    
    # circuit simulation
    ideal_results = execute(
        qv_circs_nomeas,
        backend=backend_ideal
    ).result()
                                  
    # identify heavy outcomes
    plist = [
        np.array([
            np.abs(s)**2 
            for s in ideal_results.get_statevector(i)
        ])
        for i in range(reps)
    ]
    heavy_outcomes = [np.argsort(p)[len(p)//2:] for p in plist]
    ideal_results = [np.sum(p[h]) for p, h in zip(plist, heavy_outcomes)]
    
    if savefile:
        with open(f'qv_ideal_n{nqubits}.pkl', 'wb') as f:
            pickle.dump([qc.qasm() for qc in qv_circs_nomeas], f)
            pickle.dump(ideal_results, f)
            pickle.dump(heavy_outcomes, f)
            
    return qv_circs, heavy_outcomes, ideal_results
    
    
def act_outcomes(qv_circs: list,
                 noise_model,
                 heavy_outputs: list,
                 optimization_level: str):
    """
    Returns actual state under noise_model.
    
    Notes:
        - only works when optimization is done before execute
    
    Args:
        qv_circs: list of qv circuits as qasm strings
        noise_model:    qiskit NoiseModel object
        heavy_outcomes: list of heavy outcomes for each circuits
        optimization_level: level of optimization of circuits
        backend_options: options used in execute for backend
    
    Returns:
        (list) list of probability of each outcome for each circuit
        
    """                    
    heavy_probs = []
    for i, qc in enumerate(qv_circs):
        if optimization_level == 'high':        
            meas_order = new_result_order(qc.num_qubits, qc)
            
        qc.remove_final_measurements()
        [qc.id(q) for q in range(qc.num_qubits)]
        qc.save_probabilities(label='end')

        backend = Aer.get_backend('qasm_simulator')
        ideal_results = execute(
            qc, 
            noise_model=noise_model,
            backend=backend, 
            optimization_level=0
        ).result()
        tmp_probs = ideal_results.results[0].data.end
        
        if optimization_level == 'high':
            heavy_probs.append(
                sum(
                    tmp_probs[h]
                    for h in np.argsort(meas_order)[heavy_outputs[i]]
                )
            )
                              
        else:
            heavy_probs.append(
                sum(
                    tmp_probs[h]
                    for h in heavy_outputs[i]
                )
            )

    return heavy_probs


def read_meas_order(nqubits, 
                    qc: QuantumCircuit):
    """Qubit order from measurement order of qasm str."""
    
    qubits = [0] * nqubits
    for n in range(1, nqubits + 1):
        qubits[qc[-n][2][0].index] = nqubits - 1 - qc[-n][1][0].index

    return qubits[::-1]
    
    
def new_result_order(nqubits, 
                     qc: QuantumCircuit):
    """Map for measurement index to new index."""
    
    morder = read_meas_order(nqubits, qc)

    str_list = [binstr(i, nqubits) for i in range(2**nqubits)]
    meas_map = [
        int(''.join(np.array([b for b in bstr])[morder]), 2) 
        for bstr in str_list
    ]
        
    return meas_map


def estimate_errors(error_dict: dict):
    """Estimate TQ errors based on error_dict."""

    tq_dep = 1
    sq_dep = 1
    if 'tq_dep' in error_dict:
        tq_dep *= convert(1 - error_dict['tq_dep'], 4, 'avg', 'dep')
        
    if 'tq_coh' in error_dict:
        tq_dep *= convert(1 - error_dict['tq_coh'], 4, 'avg', 'dep')
        
    if 'sq_dep' in error_dict:
        sq_dep *= convert(1 - error_dict['sq_dep'], 2, 'avg', 'dep')
        
    if 'sq_coh' in error_dict:
        sq_dep *= convert(1 - error_dict['sq_coh'], 2, 'avg', 'dep')

    if 'sq_dph' in error_dict:
        sq_dep *= convert(1 - error_dict['sq_dph'], 2, 'avg', 'dep')
    
    if 'tq_dph' in error_dict:
        tq_dep *= convert(1 - error_dict['tq_dph'], 2, 'avg', 'dep')
    
    sq_dep = convert(convert(sq_dep, 2, 'dep', 'proc') ** 2, 4, 'proc', 'dep')
    
    slice_fid = convert(sq_dep * tq_dep, 4, 'dep', 'avg')
    
    return slice_fid