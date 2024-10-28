import sys
import time

import numpy as np

sys.path.append('..')
import pickle
from common.circuit_preprocessing import circuit_preprocessing_matrix
from dataset.high import get_dataset
from coupling_map.coupling_map import one_dim_couping
from multiprocessing import Pool
from consts import *
from match_m.match_code import encode_matrix, encode_matrix_int


class SearchKernel:
    def __init__(self, kernel_list, low_qubits=2, high_qubits=11, threshold=2, basis_gates=None):
        """
        预处理一下
        :param low_qubits:
        :param high_qubits:
        """
        self.low_qubits = low_qubits
        self.high_qubits = high_qubits
        self.dataset = get_dataset(low_qubits, high_qubits)
        self.float_matrices = []
        self.layer_type = []
        self.circuit_names = []
        self.kernel_list = kernel_list
        self.threshold = threshold
        for data in self.dataset:
            self.circuit_names.append(data['id'])
            qc = data['circuit']
            n_qubits = qc.num_qubits
            print(data['id'])
            fm, lt = circuit_preprocessing_matrix(qc, one_dim_couping(n_qubits), basis_gates=basis_gates, xy=False, rz_opti=True)
            self.float_matrices.append(fm)
            self.layer_type.append(lt)
        print('init finish')

        self.all_components = {kernel: [] for kernel in kernel_list}

    @staticmethod
    def _search(matrix, kernel):
        res = []
        if kernel[0] > matrix.shape[0] or kernel[1] > matrix.shape[1]:
            return [], kernel
        windows = np.lib.stride_tricks.sliding_window_view(matrix, kernel)
        # 滑动窗口
        for i in range(windows.shape[0]):
            for j in range(windows.shape[1]):
                window = windows[i, j].copy()
                window[window >= 10] = 0
                flag = False
                # 有全零行或列直接清除
                for window_row in range(window.shape[0]):
                    if np.array_equal(window[window_row, :], np.zeros((window.shape[1],), dtype=float)):
                        flag = True
                        break
                if (window[0, 0] == 0) or (window[0, kernel[1] - 1] == 0):
                    flag = True
                if flag:
                    continue
                # r中，0为子矩阵，1为计数，2为卷积值，3为=看类型
                # 特征值捏
                for r in res:
                    if np.array_equal(r[0], window):
                        r[1] += 1
                        break
                else:
                    res.append([window, 1, encode_matrix(window)])
        print('finish', kernel)
        return res, kernel

    def _post_processing(self):
        new_all_components = {}
        for kernel in self.all_components:
            new_all_components[kernel] = []
            for component in self.all_components[kernel]:
                if component[1] >= self.threshold:
                    new_all_components[kernel].append(component)
            new_all_components[kernel].sort(key=lambda x: x[1], reverse=True)
        self.all_components = new_all_components

    def run(self):
        pool = Pool(10)
        futures = []
        for kernel in self.kernel_list:
            for matrix in self.float_matrices:
                futures.append(pool.apply_async(self._search, args=(matrix, kernel, self.one_row_kernel)))
        pool.close()
        pool.join()

        for future in futures:
            res, kernel = future.get()
            for r in res:
                for c in self.all_components[kernel]:
                    if r[2] != c[2]:
                        continue
                    if np.array_equal(r[0], c[0]):
                        c[1] += 1
                        break
                else:
                    self.all_components[kernel].append(r.copy())
            print('finish one')
        self._post_processing()
        filename = f'../result/search/search_{self.low_qubits}_{self.high_qubits}_1_{MAX_KERNEL}_threshold{self.threshold}_no_cz'
        print(filename)
        with open(filename, mode='wb') as f:
            pickle.dump(self.all_components, f)

    def run_single(self):
        res = {}
        for matrix in self.float_matrices:
            for row in range(matrix.shape[0]):
                for col in range(matrix.shape[1]):
                    it = matrix[row, col]
                    if it == 0 or it >= 10:
                        continue
                    if it in res:
                        res[it] += 1
                    else:
                        res[it] = 1

        with open(f'../result/search/search_single_{self.low_qubits}_{self.high_qubits}', mode='wb') as f:
            pickle.dump(res, f)


if __name__ == '__main__':
    sk = SearchKernel(KERNEL_SIZE_LIST_NUM, 2, 11, 8, one_row_kernel=False, basis_gates=CONV_BASIS_GATES, xy=True, rz_opti=False)
    sk.run()
    # with open(f'../result/search/search_2_11_6_6_threshold8_cz', mode='rb') as f:
    #     a = pickle.load(f)
    # print(a)
