import numpy as np
from qiskit import QuantumCircuit
from qiskit.algorithms import Grover
from qiskit.circuit.random import random_circuit


class Algorithm:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits

    # QFT
    def qft(self):
        from qiskit.circuit.library import QFT
        return QFT(self.n_qubits)

    # GHZ
    def ghz(self):
        qc = QuantumCircuit(self.n_qubits)
        qc.h(0)
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        return qc

    # 黑盒，随便设计的
    def grover_oracle(self, target: str):
        from qiskit.quantum_info import Statevector
        from qiskit.algorithms import AmplificationProblem
        qc = AmplificationProblem(Statevector.from_label(target)).grover_operator.oracle.decompose()
        return qc

    def amplitude_amplification(self, target: str):
        from qiskit.quantum_info import Statevector
        from qiskit.algorithms import AmplificationProblem
        problem = AmplificationProblem(Statevector.from_label(target))
        qc = problem.grover_operator.decompose()
        return qc, problem

    # Grover算法
    def grover(self, target: str, iterations=1):
        # grover
        circuit, problem = self.amplitude_amplification(target)
        qc = Grover(iterations=iterations).construct_circuit(problem)
        return qc

    @staticmethod
    def shor(N):
        from qiskit.algorithms.factorizers.shor import Shor
        shor = Shor()
        return shor.construct_circuit(N)

    def phase_estimation(self, U: QuantumCircuit):
        from qiskit.algorithms.phase_estimators.phase_estimation import PhaseEstimation
        pe = PhaseEstimation(self.n_qubits)
        return pe.construct_circuit(U)

    def bernstein_vazirani(self, s: str):
        bv_circuit = QuantumCircuit(self.n_qubits + 1)

        # put auxiliary in state |->
        bv_circuit.h(self.n_qubits)
        bv_circuit.z(self.n_qubits)

        # Apply Hadamard gates before querying the oracle
        for i in range(self.n_qubits):
            bv_circuit.h(i)

        # Apply the inner-product oracle
        s = s[::-1]  # reverse s to fit qiskit's qubit ordering
        for q in range(self.n_qubits):
            if s[q] == '0':
                bv_circuit.i(q)
            else:
                bv_circuit.cx(q, self.n_qubits)

        # Apply Hadamard gates after querying the oracle
        for i in range(self.n_qubits):
            bv_circuit.h(i)

        return bv_circuit

    def qft_inverse(self, circuit: QuantumCircuit, n: int) -> QuantumCircuit:
        """ Applies the inverse of the Quantum Fourier Transform on the first n qubits in the given circuit. """
        for qubit in range(n // 2):
            circuit.swap(qubit, n - qubit - 1)
        for j in range(n):
            for m in range(j):
                circuit.cp(-np.pi / float(2 ** (j - m)), m, j)
            circuit.h(j)

        return circuit

if __name__ == '__main__':
    a = Algorithm(3)
    from qiskit.quantum_info import Statevector
    print(Statevector(a.bernstein_vazirani('111111')).probabilities_dict())


