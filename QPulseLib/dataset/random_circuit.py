from qiskit import QuantumCircuit
import random
from math import pi
import sys
sys.path.append('..')
from coupling_map.coupling_map import one_dim_couping

basis_single_gates = ['rx', 'ry']
basis_two_gates = ['cz']
random_params = [pi, -pi, pi / 2, -pi / 2, pi / 4, -pi / 4, 0]

def one_dim_couping(n_qubits):
    return [[i, i + 1] for i in range(n_qubits - 1)] + [[i + 1, i] for i in range(n_qubits - 1)]

def random_circuit(n_qubits, n_gates, two_qubit_prob = 0.5, reverse = True):
    if reverse:
        n_gates = n_gates//2
    circuit = QuantumCircuit(n_qubits)
    qubits = list(range(n_qubits))
    coupling_map = one_dim_couping(n_qubits)

    for qubit in qubits:
        gate_type = random.choice(basis_single_gates)
        if gate_type == 'i':
            getattr(circuit, gate_type)(qubit)
            continue
        getattr(circuit, gate_type)(random.choice(random_params), qubit)

    for _ in range(n_gates):
        if random.random() < two_qubit_prob:
            gate_type = basis_two_gates[0]
            assert len(basis_two_gates) == 1
        else:
            gate_type = random.choice(basis_single_gates)
        
        operated_qubits = list(random.choice(coupling_map))
        random.shuffle(operated_qubits)
        control_qubit = operated_qubits[0]
        target_qubit = operated_qubits[1]
        if gate_type == 'cz':
            circuit.cz(control_qubit, target_qubit)
        elif gate_type in ('rx', 'ry'):
            getattr(circuit, gate_type)(pi/2, random.choice(qubits))
        else:
            circuit.i(random.choice(qubits))

    if reverse:
        circuit = circuit.compose(circuit.inverse())
        # random.random()
    # print(circuit)
    # circuit_to_dag(circuit)
    # print(circuit)
    return circuit

from qiskit.circuit import QuantumCircuit, CircuitInstruction
from qiskit.circuit.library.generalized_gates import PermutationGate, UnitaryGate
import scipy.stats
import numpy as np

def su4_circuit(num_qubits, depth, seed=1):
    width = num_qubits // 2
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    unitaries = scipy.stats.unitary_group.rvs(4, depth * width, rng).reshape(depth, width, 4, 4)
    qc = QuantumCircuit(num_qubits)
    qubits = tuple(qc.qubits)
    for row in unitaries:
        perm = rng.permutation(num_qubits)
        for w, unitary in enumerate(row):
            gate = UnitaryGate(unitary, check_input=False)
            qubit = 2 * w
            qc._append(
                CircuitInstruction(gate, (qubits[perm[qubit]], qubits[perm[qubit + 1]]))
            )
    return qc

# from qiskit import transpile
# print(transpile(su4_circuit(5, 500), basis_gates=['rx', 'ry', 'cz']))