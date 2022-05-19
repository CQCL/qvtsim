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

"""Functions to generate error models as qiskit NoiseModel."""

import numpy as np
from scipy.linalg import expm

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import pauli_error, depolarizing_error, coherent_unitary_error

sigmaz = np.array([[1,0],[0,-1]])

    
def arbitrary_noise(error_dict: dict,
                    nqubits: int,
                    act_outcomes: bool = False) -> NoiseModel:
    """
    Make arbitrary error model for QV simulations.
    
    Notes:
        - All error magnigutes are defined in terms of avg fidelity 
          except for 'meas' which is failur prob.
    
    Args:
        error_dict: dict of errors two include key=name, value=avg. fidelity
        nqubits: number of qubits (crosstalk only)
        act_outcomes: If true include meas error w/ identity
        
    Returns:
        (NoiseModel) qiskit noise model
    
    """   
    sq_errors = []
    tq_errors = []
    
    # Coherent errors
    if 'sq_coh' in error_dict and error_dict['sq_coh'] != 0:
        theta = 2*np.arccos(np.sqrt((2 - 3*error_dict['sq_coh'])/2))
        uni = expm(-1j * theta * sigmaz/2)
        sq_errors.append(coherent_unitary_error(uni))
    
    if 'tq_coh' in error_dict and error_dict['tq_coh'] != 0:
        theta = 2*np.arccos(np.sqrt((4 - 5*error_dict['tq_coh'])/4))
        uni = expm(-1j * theta * np.kron(sigmaz, sigmaz)/2)
        tq_errors.append(coherent_unitary_error(uni))
        
    # Depolarizing errors
    if 'sq_dep' in error_dict and error_dict['sq_dep'] != 0:
        sq_errors.append(depolarizing_error(2*error_dict['sq_dep'], 1))
        
    if 'tq_dep' in error_dict and error_dict['tq_dep'] != 0:
        tq_errors.append(depolarizing_error(4*error_dict['tq_dep']/3, 2))
        
    # Dephasing errorss
    if 'sq_dph' in error_dict and error_dict['sq_dph'] != 0:
        dph = 3*error_dict['sq_dph']/2
        sq_errors.append(pauli_error([('Z', dph), ('I', 1 - dph)]))
        
    if 'tq_dph' in error_dict and error_dict['tq_dph'] != 0:
        dph = 1 - np.sqrt(1 - (5/4)*error_dict['tq_dph'])
        sq_channel = pauli_error([('Z', dph), ('I', 1 - dph)])
        tq_errors.append(sq_channel.tensor(sq_channel))
        
    # Prep errors
    if 'prep' in error_dict and error_dict['prep'] != 0:
        prep_error = pauli_error(
            [('X', error_dict['prep']), ('I', 1 - error_dict['prep'])]
        )
        
    # Measurement errors
    if 'meas' in error_dict and error_dict['meas'] != 0:
        meas_error = pauli_error(
            [('X', error_dict['meas']), ('I', 1 - error_dict['meas'])
        ])
        
    # make noise model
    noise_model = NoiseModel()
    
    try:
        total_sq = sq_errors[0]
        for err in sq_errors[1:]:
            total_sq = total_sq.compose(err)
        noise_model.add_all_qubit_quantum_error(total_sq, ['u2', 'u3'])
        
    except IndexError:
        pass
    
    try:
        total_tq = tq_errors[0]
        for err in tq_errors[1:]:
            total_tq = total_tq.compose(err)
        noise_model.add_all_qubit_quantum_error(total_tq, ['cx', 'cz'])
        
    except IndexError:
        pass
        
    try:
        noise_model.add_all_qubit_quantum_error(meas_error, ['measure'])
        
    except UnboundLocalError:
        pass
        
    try:
        noise_model.add_all_qubit_quantum_error(prep_error, ['u1'])
        
    except UnboundLocalError:
        pass
        
    if act_outcomes and error_dict['meas'] != 0:
        noise_model.add_all_qubit_quantum_error(meas_error, 'id')
    
    # include crosstalk errors
    if 'sq_cross' in error_dict and error_dict['sq_cross'] != 0:
        dep = depolarizing_error(2*error_dict['sq_cross'], 1)
        for n in range(nqubits):
            noise_model.add_nonlocal_quantum_error(
                dep, 
                ['u2', 'u3'], 
                [n], 
                [(n + 1) % nqubits,
                 (n - 1) % nqubits]
            )

    if 'tq_cross' in error_dict and error_dict['tq_cross'] != 0:
        dep = depolarizing_error(2*error_dict['tq_cross'], 1)
        for n in range(nqubits):
            for m in range(nqubits):
                adjacent_list = [
                    (n+1)%nqubits, 
                    (n-1)%nqubits, 
                    (m+1)%nqubits, 
                    (m-1)%nqubits
                ]
                adjacent_list = [a for a in adjacent_list if a != n and a != m]
                noise_model.add_nonlocal_quantum_error(
                    dep, 
                    ['cx', 'cz'], 
                    [n, m], 
                    adjacent_list
                )
            
    return noise_model