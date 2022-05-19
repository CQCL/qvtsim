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

''' Functions for loading quantum volume data from Quantinuum. '''

from collections import Counter
import json

from qiskit.result import Result
from qiskit import QuantumCircuit
from qiskit import execute
from qiskit import Aer
import qiskit.ignis.verification.quantum_volume as qv


def data2qiskit(raw_results):
    """
    Convert HQS job_results to list of qiskit results.
    
    Args:
        raw_results: List of list of output counts

    Returns:
        Result: qiskit result object

    """
    nqubits = len(raw_results[0][0])
        
    results_out = []
    for i, res in enumerate(raw_results):
        results_dict = {
            'date': 'loaded', 
            'success': True, 
            'job_id': str(i),
            'backend_name': 'Quantinuum', 
            'backend_version': '0.0.1', 
            'qobj_id': 'None'
        }
        results_dict['results'] = [0]
        results_dict['results'][0] = {
            'success': True, 
            'time_taken': 0,
            'shots': len(res),
            'status': 'DONE',
            'meas_level': 2, 
            'seed_simulator': 0
        }
        results_dict['results'][0]['header'] = {
            'qreg_sizes': [['qr', nqubits]], 
            'n_qubits': nqubits, 
            'memory_slots': nqubits, 
            'name': f'qv_depth_{nqubits}_trial_{i}',
            'clbit_labels': [['cr', j] for j in range(nqubits)],
            'creg_sizes': [['cr', nqubits]],
            'qubit_labels': [['qr', j] for j in range(nqubits)]
        }
        results_dict['results'][0]['data'] = {
            'counts': {
                hex(int(cstr, 2)): count
                for cstr, count in Counter(res).items()
            }
        }
        results_out.append(Result.from_dict(results_dict))
    
    return results_out


def circ2qiskit(qasm_circs):

    qiskit_circs = [
        QuantumCircuit.from_qasm_str(qc)
        for qc in qasm_circs
    ]

    return qiskit_circs


def ideal2qiskit(nqubits,
                 qv_circs_nomeas):
    """
    Returns ideal state results as statevectors for qv_fitter.
    
    Args:
        nqubits: number of qubits
        qv_circs_nomeas: qv circuits without final measurement
       
    Returns:
        list: list of qiskit Results objects for ideal circuits
        
    """
    qv_circs_new = []
    for i, circ in enumerate(qv_circs_nomeas):
        qv_circs_new.append(QuantumCircuit.from_qasm_str(circ))
        qv_circs_new[-1].name = f'qv_depth_{nqubits}_trial_{i}'
        
    backend = Aer.get_backend('statevector_simulator')
    res = execute(
        qv_circs_new, 
        backend=backend, 
        optimization_level=0,
    ).result()
    
    return res


def load_fitter(machine, 
                nqubits):

    with open(f'data/n{nqubits}_{machine}_raw_results.json', 'r') as f:
        data = json.load(f)

    exp_results = data2qiskit(data['raw_results'])
    ideal_results = ideal2qiskit(nqubits, data['qv_circs_nomeas'])

    qv_fitter = qv.QVFitter(qubit_lists=[list(range(nqubits))])
    qv_fitter.add_statevectors(ideal_results)
    qv_fitter.add_data(exp_results)

    return qv_fitter
