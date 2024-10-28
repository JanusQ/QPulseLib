import numpy as np
from qiskit import transpile
from qiskit.converters import circuit_to_dag

from common.cirb import ZxSimplifier, QiskitCircuit
from common.parser1 import get_layer_type
from consts import BASIS_GATES
from common.my_formatter import *
from coupling_map.coupling_map import one_dim_couping


def divide_layer(layer2instructions):
    copy_list = []
    for layer in layer2instructions:
        list_s = []
        list_t = []
        for instruction in layer:
            if len(instruction.qubits) == 1:
                list_s.append(instruction)
            else:
                list_t.append(instruction)
        if len(list_s) != 0 and len(list_t) != 0:
            copy_list.append(list_t)
            copy_list.append(list_s)
        else:
            copy_list.append(layer)
    return copy_list


def merge_single(layer_type, layer_info):
    new_layer_info = [[] for _ in range(len(layer_info))]
    num_layer = len(layer_type)
    num_qubits = len(layer_info)
    vis = []
    for i in range(num_layer):
        if i in vis:
            continue
        vis.append(i)
        if layer_type[i] != 1:
            for qubit in range(num_qubits):
                new_layer_info[qubit].append(layer_info[qubit][i])
        else:
            for qubit in range(num_qubits):
                new_layer_info[qubit].append(layer_info[qubit][i])

            for j in range(i + 1, num_layer):
                if layer_type[j] == 1:
                    for qubit in range(num_qubits):
                        if new_layer_info[qubit][-1] is None:
                            new_layer_info[qubit][-1] = layer_info[qubit][j]
                        else:
                            if layer_info[qubit][j] is not None:
                                new_layer_info[qubit][-1][1] += layer_info[qubit][j][1]
                    vis.append(j)
                else:
                    break

    return layer_type, new_layer_info


def merge_single_xy(layer_type, layer_info):
    new_layer_info = [[] for _ in range(len(layer_info))]
    num_layer = len(layer_type)
    num_qubits = len(layer_info)
    vis = []
    for i in range(num_layer):
        if i in vis:
            continue
        vis.append(i)
        if layer_type[i] != 1:
            for qubit in range(num_qubits):
                new_layer_info[qubit].append(layer_info[qubit][i])
        else:
            for qubit in range(num_qubits):
                paras = [0, 0]
                if layer_info[qubit][i] is not None:
                    if layer_info[qubit][i][0] == 'rx':
                        paras[0] += layer_info[qubit][i][1]
                    else:
                        paras[1] += layer_info[qubit][i][1]
                for j in range(i + 1, num_layer):
                    if layer_type[j] == 2:
                        break
                    else:
                        if layer_info[qubit][j] is not None:
                            if layer_info[qubit][j][0] == 'rx':
                                paras[0] += layer_info[qubit][j][1]
                            else:
                                paras[1] += layer_info[qubit][j][1]
                    vis.append(j)
                new_layer_info[qubit].append(['u', paras])
    return layer_type, new_layer_info


def transform_to_matrix(circuit, xy=False):
    layer2instructions, instruction2layer, instructions, dagcircuit, nodes = get_layered_instructions(circuit)
    # 分层
    layer2instructions = divide_layer(layer2instructions)
    layer_type, layer_info = get_layer_type(layer2instructions, circuit.num_qubits)
    # 转成矩阵好了
    # 合并
    if xy:
        layer_type, layer_info = merge_single_xy(layer_type, layer_info)
    else:
        layer_type, layer_info = merge_single(layer_type, layer_info)
    return layer_type, layer_info


def matrix_to_float(layer_info):
    if not layer_info:
        return []
    num_rows, num_cols = len(layer_info), len(layer_info[0])
    res = np.zeros((num_rows, num_cols), dtype=float)
    for i in range(num_rows):
        for j in range(num_cols):
            gate = layer_info[i][j]
            if gate is None:
                res[i, j] = 0
                continue
            if gate[0] == 'rx':
                res[i, j] = round(gate[1], 4)
            elif gate[0] == 'ry':
                res[i, j] = round(gate[1], 4)
            elif gate[0] == 'rz':
                res[i, j] = 0
            elif gate[0] == 'cz':
                res[i, j] = 10 + gate[1]
    # 去掉RZ
    # 去掉全0列
    res = res[:, np.any(res != 0, axis=0)]
    return res


def matrix_to_complex(layer_info):
    if not layer_info:
        return []
    num_rows, num_cols = len(layer_info), len(layer_info[0])
    res = np.zeros((num_rows, num_cols), dtype=complex)
    for i in range(num_rows):
        for j in range(num_cols):
            gate = layer_info[i][j]
            if gate is None:
                res[i, j] = 0
                continue
            if gate[0] == 'u':
                res[i, j] = complex(gate[1][0], gate[1][1])
            elif gate[0] == 'cz':
                res[i, j] = -10 - 10j
    # 去掉RZ
    # 去掉全0列
    res = res[:, np.any(res != 0, axis=0)]
    return res


def circuit_preprocessing_matrix(circuit, coupling_map=None, rz_opti=True, xy=False):
    """
    输入电路和拓扑结构，输出float矩阵和每列的layer_type
    :param circuit: qc
    :param coupling_map:
    :return:
    """
    if xy:
        tqc = None
        try:
            tqc = transpile(circuit, basis_gates=['rx', 'ry', 'cz'], optimization_level=0, coupling_map=coupling_map)
        except Exception as e:
            print(circuit)
            exit(0)
        layer_type, layer_info = transform_to_matrix(circuit, xy=xy)
        float_matrix = matrix_to_complex(layer_info)
        layer_type = np.ones((float_matrix.shape[1],), dtype=int)
    else:
        tqc = None
        try:
            tqc = transpile(circuit, basis_gates=['rx', 'ry', 'rz', 'cz'], optimization_level=0, coupling_map=coupling_map)
        except Exception as e:
            print(circuit)
            exit(0)
        layer_type, layer_info = transform_to_matrix(circuit, xy=xy)
        if rz_opti:
            zs = ZxSimplifier(QiskitCircuit(tqc))
            circuit = zs.qiskit_circuit_zx_optimized(coupling_map)._circuit
        float_matrix = matrix_to_float(layer_info)
        layer_type = np.ones((float_matrix.shape[1],), dtype=int)
        for col in range(float_matrix.shape[1]):
            if np.any(float_matrix[:, col] >= 10):
                layer_type[col] = 2
    return float_matrix, layer_type


def circuit_preprocessing_dag(circuit, coupling_map=None):
    """
    输入电路和拓扑结构，输出dag和编译后电路，删除掉rz
    :param circuit:
    :param coupling_map:
    :return:
    """
    if coupling_map is None:
        coupling_map = one_dim_couping(circuit.num_qubits)
    tqc = transpile(circuit, coupling_map=coupling_map, basis_gates=BASIS_GATES)
    zs = ZxSimplifier(QiskitCircuit(tqc))
    circuit = zs.qiskit_circuit_zx_optimized(coupling_map)._circuit
    dag = circuit_to_dag(circuit)
    dag.remove_all_ops_named('rz')
    return dag, circuit


if __name__ == '__main__':
    from dataset.algorithm2 import Algorithm
    from All_pulse_generation import wave_construction

    qft = Algorithm(3).qft()
    pulse_array = []
    matrix = circuit_preprocessing_matrix(qft)[0]
    pad = np.zeros((10, ))
    pad_cz = np.zeros((10, ))
    pad_wave = np.zeros((30, ))
    wave_x = wave_construction('x')
    wave_cz = wave_construction('cz')
    for i in range(3):
        wave = np.zeros((0, ))
        for j in range(matrix.shape[1]):
            if matrix[i][j] != 0 and matrix[i][j] < 10:
                wave = np.hstack((wave, pad, wave_x.real, pad))
            else:
                wave = np.hstack((wave, pad_cz))
        pulse_array.append(wave)
    for i in range(3):
        wave = np.zeros((0, ))
        for j in range(matrix.shape[1]):
            if matrix[i][j] > 10:
                wave = np.hstack((wave, wave_cz))
            else:
                wave = np.hstack((wave, pad_wave))
        pulse_array.append(wave)

    import matplotlib.pyplot as plt

    fig = plt.figure(facecolor='none')

    for i in range(6):
        axi = plt.subplot(6, 1, i + 1)
        axi.plot(np.arange(0, len(pulse_array[i])), pulse_array[i], '-')
        plt.xticks([])
        plt.yticks([])
        axi.spines['top'].set_color('none')
        axi.spines['bottom'].set_color('none')
        axi.spines['left'].set_color('none')
        axi.spines['right'].set_color('none')
    plt.xlim([-50, 850])
    plt.show()