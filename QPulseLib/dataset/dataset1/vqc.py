import numpy as np
import qiskit
import random
from dataset.dataset_consts import PARAMETERS


class VQC:
    def __init__(self, qubit_count):
        self.qubit_count = qubit_count
        self.quantum_register = qiskit.circuit.QuantumRegister(self.qubit_count)
        self.classical_register = qiskit.circuit.ClassicalRegister(self.qubit_count)
        self.circuit = qiskit.circuit.QuantumCircuit(self.quantum_register)
        self.layer_count = 4
        self.hadamard_circuit()
        self.phase_addition()
        self.learnable_layers()

    def hadamard_circuit(self):
        for qubit in self.quantum_register:
            self.circuit.h(qubit)

    def phase_addition(self):
        for qubit in self.quantum_register:
            self.circuit.rz(random.choice(PARAMETERS) * np.pi, qubit)
        for cqubit, aqubit in zip(self.quantum_register[:-1], self.quantum_register[1:]):
            self.circuit.cx(cqubit, aqubit)
            self.circuit.rz(random.choice(PARAMETERS) * np.pi, aqubit)
            self.circuit.cx(cqubit, aqubit)

    def learnable_layers(self):
        for _ in range(self.layer_count):
            for qubit in self.quantum_register:
                self.circuit.ry(random.choice(PARAMETERS) * np.pi, qubit)
                self.circuit.rz(random.choice(PARAMETERS) * np.pi, qubit)
            qbs = list(self.quantum_register)
            for i, qb in enumerate(qbs):
                for j in range(i + 1, self.qubit_count):
                    self.circuit.cz(qb, qbs[j])


def get_cir(n_qubits):
    return VQC(n_qubits).circuit

if __name__ == '__main__':
    print(get_cir(100))
