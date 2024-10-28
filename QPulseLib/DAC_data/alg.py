from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from dataset.dataset1.vqc import get_cir as vqc_get_cir
from dataset.dataset1.hamiltonian_simulation import get_cir as hs_get_cir


# alg bv
def bv(n_qubits, s: str = None):
    n_qubits -= 1
    if s is None:
        s = "1" * n_qubits
    bv_circuit = QuantumCircuit(n_qubits + 1)

    # put auxiliary in state |->
    bv_circuit.h(n_qubits)
    bv_circuit.z(n_qubits)

    # Apply Hadamard gates before querying the oracle
    for i in range(n_qubits):
        bv_circuit.h(i)

    # Apply the inner-product oracle
    s = s[::-1]  # reverse s to fit qiskit's qubit ordering
    for q in range(n_qubits):
        if s[q] == '0':
            bv_circuit.i(q)
        else:
            bv_circuit.cx(q, n_qubits)

    # Apply Hadamard gates after querying the oracle
    for i in range(n_qubits):
        bv_circuit.h(i)

    return bv_circuit


# alg GHZ
def ghz(n_qubits):
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    return qc


# alg qft
def qft(n_qubits):
    return QFT(n_qubits)


# alg vqc
def vqc(n_qubits):
    return vqc_get_cir(n_qubits)


# alg hs
def hs(n_qubits):
    return hs_get_cir(n_qubits)
