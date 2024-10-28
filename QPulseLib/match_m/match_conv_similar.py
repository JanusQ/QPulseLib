from common.circuit_preprocessing import circuit_preprocessing_matrix
from coupling_map.coupling_map import one_dim_couping
import numpy as np
from scipy import signal

kernel_list = [(3, 3), (4, 2), (2, 4), (3, 2), (2, 3), (4, 1), (1, 4), (2, 2), (3, 1), (1, 3), (2, 1), (1, 2)]
filters = {kernel_size: np.ones(kernel_size) for kernel_size in kernel_list}

def prepare(circuit):
    # 基于rx ry，且cz也混在其中
    matrix = circuit_preprocessing_matrix(circuit=circuit, coupling_map=one_dim_couping(circuit.num_qubits), basis_gates=['rx', 'ry', 'cz', 'i'], rz_opti=False, xy=True)[0]
    return matrix

def match(m, all_components):
    match_res = {}
    match_matrix = np.zeros(m.shape)
    for kernel in kernel_list:
        if kernel[0] > m.shape[0] or kernel[1] > m.shape[1]:
            continue
        # 卷积占坑
        # 卷积值判断
        conv_matrix = signal.convolve2d(m, filters[kernel], 'valid')
        if kernel == (3, 3):
            match_conv = np.zeros((m.shape[0] - 2, m.shape[1] - 2))
        else:
            match_conv = signal.convolve2d(match_matrix, filters[kernel], 'valid')
        for arr in all_components[kernel]:
            # 找到相同的且没被匹配过的
            find_idx = np.where((conv_matrix == arr[0]) & (match_conv == 0))
            for row, col in zip(find_idx[0], find_idx[1]):
                row_end, col_end = row + kernel[0], col + kernel[1]
                if (row, col) in match_res:
                    continue
                match_res[(row, col)] = [arr, m[row: row_end, col: col_end]]
                match_conv[row, col] = 1
                match_matrix[row: row_end, col: col_end] = 1
    return match_res

def find_shortest_path(m, mst):
    pass
    return 0


def generate(m, match_res):
    # find shortest path.
    path = find_shortest_path(match_res[1], match_res[3])
    # swap pulse
    # generate
    pass