import sys
sys.path.append('..')
import numpy as np

# extract pattern from (left, right)-qubits alg
left_range = 2
right_range = 11
# keep counts >= 8 patterns
threshold = 8
kernel_list = [(3, 3), (4, 2), (2, 4), (3, 2), (2, 3), (4, 1), (1, 4), (2, 2), (3, 1), (1, 3), (2, 1), (1, 2)]

def search(matrix, kernel_size):
    res = []
    if kernel_size[0] > matrix.shape[0] or kernel_size[1] > matrix.shape[1]:
        return [], kernel_size
    # get all slide windows
    windows = np.lib.stride_tricks.sliding_window_view(matrix, kernel_size)
    
    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):
            window = windows[i, j].copy()
            # r中，0为子矩阵，1为计数，2为卷积值，3为=看类型
            # 特征值捏
            for r in res:
                if np.array_equal(r[0], window):
                    r[1] += 1
                    break
            else:
                res.append([window, 1, np.sum(np.multiply(window, np.ones(kernel_size)))])
    return res, kernel_size

def search_single(matrix):
    res = {}
    for row in matrix.shape[0]:
        for col in matrix.shape[1]:
            e = matrix[row, col]
            if e not in res:
                res[e] = 1
            else:
                res[e] += 1
    return res

if __name__ == '__main__':
    from dataset.high import get_dataset
    from common.circuit_preprocessing import circuit_preprocessing_matrix

    dataset = get_dataset(left_range, right_range)

    circuits = [data['circuit'] for data in dataset]
    matrice = [circuit_preprocessing_matrix(circuit, xy=True)[0] for circuit in circuits]

    print('start search pattern.')
    res = {}
    for kernel_size in kernel_list:
        kernel_res = []
        for matrix in matrice:
            # 1 matrix
            matrix_res = search(matrix, kernel_size)
            if len(kernel_res) == 0:
                kernel_res = matrix_res
            else:
                # all
                not_exist_pattern = []
                for pattern in kernel_res:
                    for matrix_pattern in matrix_res:
                        # equal. sum counts
                        if np.equal(pattern[0], matrix_pattern[0]):
                            pattern[1] += matrix_pattern[1]
                    else:
                        # not found. add to all.
                        not_exist_pattern.append(matrix_pattern)
                kernel_res.extend(not_exist_pattern)
        res[kernel_size] = kernel_res
    import pickle
    with open(f'../result/search/search_conv_{left_range}_{right_range}_threshold{threshold}', mode='wb') as f:
        pickle.dump(res, f)
    print('finish search pattern.')



