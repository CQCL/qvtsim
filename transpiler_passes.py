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

"""Custom transpiler passes."""

import numpy as np

# transpile options
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import PassManager
from qiskit.circuit.library import SwapGate, CXGate, RXXGate, RYYGate, RZZGate
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info.synthesis import TwoQubitBasisDecomposer
from qiskit.quantum_info.synthesis.two_qubit_decompose import TwoQubitWeylDecomposition
from qiskit.extensions import UnitaryGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes import Unroller, Optimize1qGates
from qiskit.transpiler.passes import BarrierBeforeFinalMeasurements, RemoveDiagonalGatesBeforeMeasure
from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks



def preset_passes(optimization_level: str,
                  transpile_options: dict):
    """
    Preset pass options.
    
    Args:
        optimization_level: 'low' = combine SQ gates, relabel qubits to avoid first transport
                            'medium' = low + combine adjacent SU(4) blocks
                            'high' = medium + approximate blocks
        transpile_options: dictionary of other transpiler options for 'high'
        
    Returns:
        (PassManager): PassManager object for compilation
    
    """
    pm = PassManager()
    if optimization_level == 'low':
        pm.append(
            [Unroller(['u1','u2','u3','cx']), 
            Optimize1qGates(), 
            RemoveDiagonalGatesBeforeMeasure(), 
            BarrierBeforeFinalMeasurements()]
        )
    
    elif optimization_level == 'medium':
        pm.append(
            [Unroller(['u1','u2','u3','cx']), 
            Collect2qBlocks(),
            ConsolidateBlocks(), 
            Unroller(['u1','u2','u3','cx']), 
            Optimize1qGates(),
            RemoveDiagonalGatesBeforeMeasure(), 
            BarrierBeforeFinalMeasurements()]
        )
                   
    elif optimization_level == 'high':     
        pm.append(
            [Unroller(['u1','u2','u3','cx']), 
            Collect2qBlocks(), 
            ApproxBlocks(**transpile_options), 
            Optimize1qGates(),
            RemoveDiagonalGatesBeforeMeasure(), 
            BarrierBeforeFinalMeasurements()]
        )
    
    else:
        ValueError('Invalid optimization_level selected.')

    return pm
        
        
class ApproxBlocks(TransformationPass):
    """
    Replace SU(4) block with approximation

    Based of qiskit ConsolidateBlocks pass
        
    """
    def __init__(self, 
                 tol: float = 0.999, 
                 mirror: bool = False, 
                 arbitrary_angles: bool = False, 
                 comp_method: str = 'fid',
                 kak_basis_gate = CXGate(), 
                 force_consolidate: bool = False):
        """
        Args:
            tol: tolerance in approximation
            mirror: option to check SWAP decomposition
            arbitrary_angles: True=use arbitrary Rxx, Ryy, Rzz gates
            comp_method: method to compare SU(4) decompositions
            kak_basis_gate: Basis gate for KAK decomposition.
            force_consolidate: Force block consolidation
            
        """
        super().__init__()
        self.force_consolidate = force_consolidate
        self.decomposer = TwoQubitBasisDecomposer(kak_basis_gate)
        self.tol = tol
        self.mirror = mirror
        self.comp_method = comp_method
        self.arbitrary_angles = arbitrary_angles

    def run(self, dag):

        new_dag = DAGCircuit()
        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

        # compute ordered indices for the global circuit wires
        global_index_map = {wire: idx for idx, wire in enumerate(dag.qubits())}
        qubit_map = {wire: wire for wire in dag.qubits()}

        blocks = self.property_set['block_list']
        # just to make checking if a node is in any block easier
        all_block_nodes = {nd for bl in blocks for nd in bl}

        for node in dag.topological_op_nodes():
            if node not in all_block_nodes:
                # need to add this node to find out where in the list it goes
                preds = [nd for nd in dag.predecessors(node) if nd.type == 'op']

                block_count = 0
                while preds:
                    if block_count < len(blocks):
                        block = blocks[block_count]

                        # if any of the predecessors are in the block, remove them
                        preds = [p for p in preds if p not in block]
                    else:
                        # should never occur as this would mean not all
                        # nodes before this one topologically had been added
                        # so not all predecessors were removed
                        raise TranspilerError("Not all predecessors removed due to error"
                                              " in topological order")

                    block_count += 1

                # we have now seen all predecessors
                # so update the blocks list to include this block
                blocks = blocks[:block_count] + [[node]] + blocks[block_count:]

        # create the dag from the updated list of blocks
        basis_gate_name = self.decomposer.gate.name

        for block in blocks:
            if len(block) == 1 and block[0].name != 'cx':
                # an intermediate node that was added into the overall list
                new_dag.apply_operation_back(
                    block[0].op, 
                    [qubit_map[q] for q in block[0].qargs],
                    block[0].cargs, 
                    block[0].condition
                )
            else:
                # find the qubits involved in this block
                block_qargs = set()
                for nd in block:
                    block_qargs |= set(nd.qargs)
                # convert block to a sub-circuit, then simulate unitary and add
                block_width = len(block_qargs)
                q = QuantumRegister(block_width)
                subcirc = QuantumCircuit(q)
                block_index_map = self._block_qargs_to_indices(
                    block_qargs,
                    global_index_map
                )
                inv_block_index_map = {val:key for key, val in block_index_map.items()}
                
                basis_count = 0
                for nd in block:
                    if nd.op.name == basis_gate_name:
                        basis_count += 1
                    subcirc.append(nd.op, [q[block_index_map[i]] for i in nd.qargs])
                unitary = UnitaryGate(Operator(subcirc))  # simulates the circuit
                
                qc, flip_qubits = mirror_decomposer(
                    unitary.to_matrix(), 
                    self.tol, 
                    self.mirror, 
                    self.arbitrary_angles, 
                    self.comp_method, 
                    self.decomposer
                )
                if not (self.force_consolidate or unitary.num_qubits > 2):
                    for inst in qc.data:
                        qubit_list = [qubit.index for qubit in inst[1]]
                        new_dag.apply_operation_back(
                            inst[0], 
                            [qubit_map[inv_block_index_map[i]] for i in qubit_list]
                        )
                        
                    if flip_qubits:
                        qubits = qubit_map[inv_block_index_map[1]], qubit_map[inv_block_index_map[0]]
                        qubit_map.update(
                            {inv_block_index_map[0]: qubits[0],
                             inv_block_index_map[1]: qubits[1]}
                        )
                    
                else:
                    for nd in block:
                        new_dag.apply_operation_back(
                            nd.op, 
                            [qubit_map[q] for q in nd.qargs], 
                            nd.cargs, 
                            nd.condition
                        )

        return new_dag
        

    def _block_qargs_to_indices(self, block_qargs, global_index_map):

        block_indices = [global_index_map[q] for q in block_qargs]
        ordered_block_indices = sorted(block_indices)
        block_positions = {
            q: ordered_block_indices.index(global_index_map[q])
            for q in block_qargs
        }
                           
        return block_positions
        
        
def mirror_decomposer(U: np.ndarray,
                      fidelity: float,
                      mirror: bool,
                      arbitrary_angles: bool,
                      comp_method: str,
                      basis_decomp: TwoQubitBasisDecomposer):
    """
    Decompose unitary in standard and mirror option and return most efficient.
    
    Args:
        U: unitary to decomposer
        fidelity: tolerance fidelity of decomposition
        mirror: option to allow search over U * SWAP in addition
        arbitrary_angles: option to do decomposition with arbitrary angles
        comp_method: method used to compare decompositions
        decomp: SU(4) decomposition object

    Returns:
        (QuantumCircuit): circuit describing decoposition
    """   
    target_decomposed = TwoQubitWeylDecomposition(U)
    decomp_success = comparison_func(
        target_decomposed, 
        basis_decomp, 
        fidelity, 
        comp_method
    )
    
    if mirror:
        target_decomposed_rev = TwoQubitWeylDecomposition(SwapGate().to_matrix().dot(U))
        decomp_success += comparison_func(
            target_decomposed_rev, 
            basis_decomp, 
            fidelity, 
            comp_method
        )
    
    best_decomp = np.argmax(decomp_success)
    if best_decomp >= 4:
        flip_future = True
        best_decomp = best_decomp % 4
        target_decomposed = target_decomposed_rev
        
    else:
        flip_future = False

    return_circuit = write_circuit(
        best_decomp, 
        basis_decomp, 
        target_decomposed, 
        arbitrary_angles
    )
        
    return return_circuit, flip_future
    
    
def write_circuit(best_decomp: int,
                  decomp: TwoQubitBasisDecomposer,
                  target_decomposed: TwoQubitWeylDecomposition,
                  arbitrary_angles: bool):
    """
    Make qiskit circuit out of selected decomp.
    
    Args:
        best_decomp: index for decomposition based on fidelity criteria (also number of TQ gates)
        decomp: decomposition object
        target_decomposed: decomposition of target unitary
        arbitrary_angles: option if using arbitrary angles with different circuit decomp
       
    Returns: 
        (QuantumCircuit): Circuit describing target_decomposed
        
    """
    q = QuantumRegister(2)
    qc = QuantumCircuit(q)
    pm = PassManager()
    pm.append(Optimize1qGates())
        
    if not arbitrary_angles:
        decomposition = decomp.decomposition_fns[best_decomp](target_decomposed)
        decomposition_euler = [decomp._decomposer1q(x) for x in decomposition]

        for i in range(best_decomp):
            qc.compose(decomposition_euler[2*i], [q[0]], inplace=True)
            qc.compose(decomposition_euler[2*i+1], [q[1]], inplace=True)
            qc.append(decomp.gate, [q[0], q[1]])
        qc.compose(decomposition_euler[2*best_decomp], [q[0]], inplace=True)
        qc.compose(decomposition_euler[2*best_decomp+1], [q[1]], inplace=True)
        
    elif arbitrary_angles:
        gate_list = [RXXGate, RYYGate, RZZGate]
        tq_angles = [-target_decomposed.a, -target_decomposed.b, -target_decomposed.c]
        
        qc.compose(decomp._decomposer1q(target_decomposed.K2r), [q[0]], inplace=True)
        qc.compose(decomp._decomposer1q(target_decomposed.K2l), [q[1]], inplace=True)
        
        for i in range(best_decomp):
            qc.append(gate_list[i](2*tq_angles[i]), [q[0], q[1]])

        qc.compose(decomp._decomposer1q(target_decomposed.K1r), [q[0]], inplace=True)  
        qc.compose(decomp._decomposer1q(target_decomposed.K1l), [q[1]], inplace=True)
    
    qc_new = pm.run(qc)
    
    return qc_new
    
    
def comparison_func(target: TwoQubitWeylDecomposition,
                    basis: TwoQubitBasisDecomposer,
                    base_fid: float,
                    comp_method: str):
    """
    Decompose traces for arbitrary angle rotations.
    
    This assumes that the tq angles go from highest to lowest.
    
    """
    dep_param = (4 * base_fid - 1)/3
    
    if comp_method == 'fid':
        # fidelity estimate for fixed error per TQ operation
        traces = fixed_traces(target, basis)
        values = [
            ((abs(tr)**2 - 1) * dep_param**i + 1)/ 16
            for i, tr in enumerate(traces)
        ]
                
    elif comp_method == 'arb_fid':
        # fidelity estimate for arb angles
        traces = arb_traces(target)
        values = [
            ((abs(tr)**2 - 1) * dep_param**i + 1)/ 16
            for i, tr in enumerate(traces)
        ]
                    
    elif comp_method == 'arb_total':
        # total angle of rotation
        traces = arb_traces(target)
        total_angles = [
            0, 
            abs(target.a),
            abs(target.a) + abs(target.b),
            abs(target.a) + abs(target.b) + abs(target.c)
        ]
        values = [
            ((abs(tr)**2 - 1) * dep_param**(a/np.pi) + 1)/ 16
            for a, tr in zip(total_angles, traces)
        ]
                          
    return values
    

def arb_traces(target: TwoQubitWeylDecomposition):
    """Returns normalized traces for arbitrary angle decomps."""
    
    traces = [
        4*(np.cos(target.a)*np.cos(target.b)*np.cos(target.c)
        + 1j*np.sin(target.a)*np.sin(target.b)*np.sin(target.c)),
        4*(np.cos(target.b)*np.cos(target.c)),
        4*np.cos(target.c),
        4
    ]
                  
    return traces
    

def fixed_traces(target: TwoQubitWeylDecomposition,
                 basis: TwoQubitBasisDecomposer):
    """Returns traces for fixed angle decomps."""
    
    traces = [
        4*(np.cos(target.a)*np.cos(target.b)*np.cos(target.c)
        + 1j*np.sin(target.a)*np.sin(target.b)*np.sin(target.c)),
        4*(np.cos(np.pi/4-target.a)*np.cos(basis.basis.b-target.b)*np.cos(target.c)
        + 1j*np.sin(np.pi/4-target.a)*np.sin(basis.basis.b-target.b)*np.sin(target.c)),
        4*np.cos(target.c),
        4
    ]
    
    return traces
