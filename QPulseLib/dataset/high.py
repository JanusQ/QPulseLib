import random
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit
from qiskit.circuit.random import random_circuit

import sys
import os
sys.path.append(os.path.dirname(__file__))

from algorithm2 import Algorithm
from dataset1 import hamiltonian_simulation, ising, qknn, qsvm, vqc, \
    QAOA_maxcut, w_state, grover
from dataset2 import deutsch_jozsa, qec_5_x, multiplier, qec_9_xyz, qnn, qugan, simon


def get_data(id, qiskit_circuit):

    circuit_info = {
        "id": id + '_' + str(qiskit_circuit.num_qubits),
        "circuit": qiskit_circuit
    }
    return circuit_info


def get_bitstr(n_qubits):
    b = ""
    for i in range(n_qubits):
        if random.random() > 0.5:
            b += '0'
        else:
            b += '1'
    return b


def get_dataset(min_qubit_num, max_qubit_num):
    dataset = []

    for n_qubits in range(min_qubit_num, max_qubit_num):
        al = Algorithm(n_qubits)
        dataset.append(get_data(f'hamiltonian_simulation', hamiltonian_simulation.get_cir(n_qubits)))
        dataset.append(get_data(f'ising', ising.get_cir(n_qubits)))
        dataset.append(get_data(f'QAOA_maxcut', QAOA_maxcut.get_cir(n_qubits)))
        if n_qubits > 2:
            dataset.append(get_data(f'qknn', qknn.get_cir(n_qubits)))
        dataset.append(get_data(f'qsvm', qsvm.get_cir(n_qubits)))
        dataset.append(get_data(f'vqc', vqc.get_cir(n_qubits)))
        dataset.append(get_data(f'qft', al.qft()))
        dataset.append(get_data(f'ghz', al.ghz()))
        dataset.append(
            get_data(f'qft_inverse', al.qft_inverse(QuantumCircuit(n_qubits), n_qubits)))
        # if min_qubit_num <= 10:
        #     dataset.append(get_data(f'grover', grover.get_cir(n_qubits)))
        al = Algorithm(n_qubits - 1)
        dataset.append(get_data(f'bernstein_vazirani', al.bernstein_vazirani(get_bitstr(n_qubits - 1))))
        dataset.append(get_data(f'w_state', w_state.get_cir(n_qubits)))
        dataset.append(get_data(f'deutsch_jozsa', deutsch_jozsa.get_cir(n_qubits - 1, get_bitstr(n_qubits - 1))))
        if n_qubits % 5 == 0:
            dataset.append(get_data(f'qec_5_x', qec_5_x.get_cir(n_qubits // 5)))
            dataset.append(get_data(f'multiplier', multiplier.get_cir(n_qubits // 5)))
        if n_qubits % 17 == 0:
            dataset.append(get_data(f'qec_9_xyz', qec_9_xyz.get_cir(n_qubits // 17)))
        if n_qubits % 2 == 1:
            dataset.append(get_data(f'qnn', qnn.get_cir(n_qubits)))
            dataset.append(get_data(f'qugan', qugan.get_cir(n_qubits)))
        if n_qubits % 2 == 0:
            dataset.append(get_data(f'simon', simon.get_cir(get_bitstr(n_qubits // 2))))

    return dataset


if __name__ == '__main__':
    pass
    # get_dataset()
