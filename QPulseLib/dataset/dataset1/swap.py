import math
import random
from dataset.dataset_consts import PARAMETERS
from qiskit import QuantumCircuit, QuantumRegister


def swapTest(qc, qr):
    qc.h(qr[0])
    n_swap = len(qr) // 2
    for i in range(n_swap):
        qc.cswap(qr[0], qr[i + 1], qr[i + n_swap + 1])
    qc.h(qr[0])


def init(qc, qr):
    for i in range(1, len(qr) // 2 + 1):
        ro = random.choice(PARAMETERS) * math.pi
        qc.rx(ro + random.choice(PARAMETERS) * math.pi, qr[i])
        qc.rx(ro + random.choice(PARAMETERS) * math.pi, qr[i + len(qr) // 2])


def get_cir(n_qubits):
    random.seed(555)

    qr = QuantumRegister(n_qubits)
    qc = QuantumCircuit(qr)

    init(qc, qr)
    swapTest(qc, qr)
    return qc
