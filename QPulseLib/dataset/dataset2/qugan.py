import numpy as np
import qiskit
from dataset.dataset_consts import PARAMETERS
import random

class QuGAN:
    def __init__(self, qubit_count):
        self.qubit_count = qubit_count
        self.quantum_register = qiskit.circuit.QuantumRegister(self.qubit_count)
        self.classical_register = qiskit.circuit.ClassicalRegister((self.qubit_count - 1) // 2)
        self.circuit = qiskit.circuit.QuantumCircuit(self.quantum_register)
        self.data_load()
        self.swap_test()
        self.generate_samples()

    def data_load(self):
        for qubit in list(self.quantum_register)[1:]:
            self.circuit.ry(random.choice(PARAMETERS) * np.pi, qubit)
        qubit_list = list(self.quantum_register)[1:]
        s1 = qubit_list[:len(qubit_list) // 2]
        s1_a, s1_b = s1[:-1], s1[1:]
        s2 = qubit_list[len(qubit_list) // 2:]
        s2_a, s2_b = s2[:-1], s2[1:]
        for qa, qb in zip(s1_a, s1_b):
            self.circuit.ryy(random.choice(PARAMETERS) * np.pi, qa, qb)
            self.circuit.cry(random.choice(PARAMETERS) * np.pi, qa, qb)
        for qa, qb in zip(s2_a, s2_b):
            self.circuit.ryy(random.choice(PARAMETERS) * np.pi, qa, qb)
            self.circuit.cry(random.choice(PARAMETERS) * np.pi, qa, qb)

    def swap_test(self):
        self.circuit.h(0)
        qubit_list = list(self.quantum_register)[1:]
        q1 = qubit_list[:len(qubit_list) // 2]
        q2 = qubit_list[len(qubit_list) // 2:]
        for q_1, q_2 in zip(q1, q2):
            self.circuit.cswap(self.quantum_register[0], q_1, q_2)
        self.circuit.h(0)

    def generate_samples(self):
        qubit_list = list(self.quantum_register)[1:]
        s2 = qubit_list[len(qubit_list) // 2:]


# 大于等于3的奇数
def get_cir(n_qubits):
    return QuGAN(n_qubits).circuit

if __name__ == '__main__':
    print(get_cir(97).num_qubits)
