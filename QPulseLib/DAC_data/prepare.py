import pickle

from alg import *
from qiskit import transpile
from common.cirb import ZxSimplifier, QiskitCircuit
import numpy as np

BASIS_GATES = ['rx', 'rz', 'cz', 'id']


def one_dim_couping(n_qubits):
    return [[i, i + 1] for i in range(n_qubits - 1)] + [[i + 1, i] for i in range(n_qubits - 1)]


def get_layer_gates(qc: QuantumCircuit):
    layer_gates = []
    current_qubits = []
    current_layer = []
    for ins in qc:
        if ins.operation.name in ['rz', 'id']:
            continue
        qubits = [qubit.index for qubit in ins.qubits]
        if any([qubit in current_qubits for qubit in qubits]):
            layer_gates.append(current_layer.copy())
            current_qubits.clear()
            current_layer.clear()
        current_qubits.extend(qubits)
        current_layer.append({
            'name': ins.operation.name,
            'para': float(ins.operation.params[0]) / np.pi if ins.operation.params else None,
            'qubits': qubits
        })
    if current_layer:
        layer_gates.append(current_layer.copy())
    return layer_gates


def process_m(m):
    # 已经是矩阵了，开始分块
    blocks = []
    print(m.shape)
    for t in range(0, m.shape[0], 2):
        layer = []
        for q in range(0, m.shape[1], 2):
            if t + 1 < m.shape[0] and q + 1 < m.shape[1]:
                if np.all(m[t: t + 2, q: q + 2] == 0):
                    continue
                layer.append({
                    'block': m[t: t + 2, q: q + 2],
                    'location': (t, q)
                })
            elif t + 1 < m.shape[0] and q + 1 >= m.shape[1]:
                if np.all(m[t: t + 2, q: q + 1] == 0):
                    continue
                layer.append({
                    'block': np.hstack((m[t: t + 2, q: q + 1], np.array([[0], [0]], dtype=int))),
                    'location': (t, q)
                })
            elif t + 1 >= m.shape[0] and q + 1 < m.shape[1]:
                if np.all(m[t: t + 1, q: q + 2] == 0):
                    continue
                layer.append({
                    'block': np.vstack((m[t: t + 1, q: q + 2], np.array([[0, 0]], dtype=int))),
                    'location': (t, q)
                })
            else:
                if m[t, q] == 0:
                    continue
                z = np.zeros((2, 2), dtype=int)
                z[0, 0] = m[t, q]
                layer.append({
                    'block': z,
                    'location': (t, q)
                })
        blocks.append(layer)
    # 现在分成块了，进行特殊情况的合并
    new_blocks = []
    for layer in blocks:
        new_layer = []
        pre_merged = False
        for index in range(len(layer)):
            if pre_merged:
                pre_merged = False
                continue
            if index + 1 >= len(layer):
                new_layer.append(layer[index])
                continue
            b1, b2 = layer[index], layer[index + 1]
            if np.all(b1['block'][:, 0] == 0) and np.all(b2['block'][:, 1] == 0):
                new_layer.append({
                    'block': np.hstack((b1['block'][:, 1: 2], b2['block'][:, 0: 1])),
                    'location': (b1['location'][0], b1['location'][1] + 1)
                })
                pre_merged = True
            else:
                new_layer.append(b1)
        new_blocks.append(new_layer)
    return new_blocks


def preprocess(qc):
    tqc = transpile(qc, coupling_map=one_dim_couping(qc.num_qubits), basis_gates=BASIS_GATES)
    print(tqc)
    zs = ZxSimplifier(QiskitCircuit(tqc))
    oqc = zs.qiskit_circuit_zx_optimized(one_dim_couping(qc.num_qubits))._circuit
    layer_gates = get_layer_gates(oqc)
    # print(layer_gates)
    # 接下来转换成矩阵
    m_rx = np.zeros((len(layer_gates), qc.num_qubits), dtype=int)
    m_cz = np.zeros((len(layer_gates), qc.num_qubits), dtype=int)
    for t, layer in enumerate(layer_gates):
        for gate in layer:
            if gate['name'] == 'rx':
                m_rx[t, gate['qubits'][0]] = int(gate['para'] * 10000)
            else:
                for qubit in gate['qubits']:
                    m_cz[t, qubit] = 100000
    rx_blocks = process_m(m_rx)
    cz_blocks = process_m(m_cz)
    rx_pcodes = get_pcodes(rx_blocks)
    cz_pcodes = get_pcodes(cz_blocks)
    return rx_pcodes, cz_pcodes


def get_pcodes(block):
    pcodes = []
    for layer in block:
        for b in layer:
            s = ""
            for row in range(2):
                for col in range(2):
                    temp = b['block'][row, col]
                    if temp < 0:
                        s += '1' + bin(-temp)[2:].zfill(31)
                    else:
                        s += bin(temp)[2:].zfill(32)
            pcodes.append({
                'pcode': s,
                'location': b['location']
            })
    return pcodes


# 搜索pattern
def search_pattern(qcs, is_output=False):
    rx_patterns, cz_patterns = [], []
    for qc in qcs:
        rx_pcodes, cz_pcodes = preprocess(qc)
        for pcode_item in rx_pcodes:
            for pattern in rx_patterns:
                if pattern['pcode'] == pcode_item['pcode']:
                    pattern['count'] += 1
                    break
            else:
                rx_patterns.append({
                    'pcode': pcode_item['pcode'],
                    'count': 1
                })
        for pcode_item in cz_pcodes:
            for pattern in cz_patterns:
                if pattern['pcode'] == pcode_item['pcode']:
                    pattern['count'] += 1
                    break
            else:
                cz_patterns.append({
                    'pcode': pcode_item['pcode'],
                    'count': 1
                })
    # 需要排序
    rx_patterns = sorted([pattern for pattern in rx_patterns], key=lambda x: x['count'], reverse=True)
    cz_patterns = sorted([pattern for pattern in cz_patterns], key=lambda x: x['count'], reverse=True)

    if is_output:
        with open('pattern/rx_pattern_top10', mode='w') as f:
            for pattern in rx_patterns[:10]:
                f.write(f"{pattern['pcode']}\n")
        with open('pattern/rx_pattern', mode='w') as f:
            for pattern in rx_patterns:
                f.write(f"{pattern['pcode']}\n")
        with open('pattern/cz_pattern', mode='w') as f:
            for pattern in cz_patterns:
                f.write(f"{pattern['pcode']}\n")
    return rx_patterns, cz_patterns


# 搜索pattern
def search_pattern_fu(qcs, is_output=False):
    res_rx = {}
    for qc in qcs:
        rx, cz = preprocess_fu(qc)
        for gate in rx:
            if gate['value'] not in res_rx:
                res_rx[gate['value']] = 1
            else:
                res_rx[gate['value']] += 1
    if is_output:
        with open('pattern/rx_fu_pattern', mode='w') as f:
            for k in res_rx:
                f.write(f'{k}\n')
    return res_rx


def process_m_fu(m):
    res = []
    for t in range(m.shape[0]):
        for q in range(m.shape[1]):
            e = m[t, q]
            if e != 0:
                res.append({
                    'value': e,
                    'location': (t, q)
                })
    return res


def preprocess_fu(qc):
    tqc = transpile(qc, coupling_map=one_dim_couping(qc.num_qubits), basis_gates=BASIS_GATES)
    zs = ZxSimplifier(QiskitCircuit(tqc))
    oqc = zs.qiskit_circuit_zx_optimized(one_dim_couping(qc.num_qubits))._circuit
    layer_gates = get_layer_gates(oqc)
    # print(layer_gates)
    # 接下来转换成矩阵
    m_rx = np.zeros((len(layer_gates), qc.num_qubits), dtype=int)
    m_cz = np.zeros((len(layer_gates), qc.num_qubits), dtype=int)
    for t, layer in enumerate(layer_gates):
        for gate in layer:
            if gate['name'] == 'rx':
                m_rx[t, gate['qubits'][0]] = int(gate['para'] * 10000)
            else:
                for qubit in gate['qubits']:
                    m_cz[t, qubit] = 100000

    rx_gates = process_m_fu(m_rx)
    cz_gates = process_m_fu(m_cz)
    return rx_gates, cz_gates



if __name__ == '__main__':
    alg = [bv, ghz, qft, vqc, hs]
    # alg = [bv]
    qcs = []
    for n_qubits in range(2, 6):
        for qc_func in alg:
            qcs.append(qc_func(n_qubits))
    # search_pattern_fu(qcs, is_output=True)
    for n_qubits in [5, 10, 15, 18]:
        for qc_func in alg:
            with open(f'alg_data/{qc_func.__name__}_{n_qubits}', mode='w') as f:
                qc = qc_func(n_qubits)
                rx_pcodes, cz_pcodes = preprocess(qc)
                f.write(f'{len(rx_pcodes)} {len(cz_pcodes)}\n')
                for pcode in rx_pcodes + cz_pcodes:
                    f.write(f"{pcode['pcode']} {pcode['location'][0]} {pcode['location'][1]}\n")
    # qc = ghz(15)
    # preprocess_fu(qc)
