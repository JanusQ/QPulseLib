import time
import random

from common.All_pulse_generation import wave_construction
from common.common_class import Method
from common import circuit_preprocessing
import numpy as np
import pickle


# rx门的dt
RX_DT = 60
# cz门的dt
CZ_DT = 120
# 每层的dt
LAYER_DT = 200
# 单门的library
with open('../result/search/search_single_2_11', mode='rb') as f:
    SINGLE_LIBRARY = pickle.load(f)
WX = np.pad(wave_construction('x').real, (0, LAYER_DT - RX_DT), 'constant', constant_values=(0, 0))
WCZ = np.pad(wave_construction('cz').real, (0, LAYER_DT - CZ_DT), 'constant', constant_values=(0, 0))
SINGLE_PULSE_LIBRARY = {k: WX for k in SINGLE_LIBRARY}


class MatrixMethod(Method):
    def data_loader(self):
        pass

    def match(self):
        pass

    def gen_pulse(self, m):
        pass

    def __init__(self, thresholds=None):
        super().__init__(thresholds)
        self.libraries = []


class Single1MatrixMethod(MatrixMethod):
    """
    假装自己提前加好cz偏置，只需要加cz底波的缓存方法
    """
    def __init__(self, thresholds=None):
        super().__init__(thresholds)
        self.data_loader()

    def data_loader(self):
        super().data_loader()
        if self.thresholds is not None:
            for threshold in self.thresholds:
                self.libraries.append({k: WX for k, v in SINGLE_LIBRARY.items() if v >= threshold})
        self.libraries.append(SINGLE_PULSE_LIBRARY)

    def gen_pulse(self, m):
        sm = circuit_preprocessing.transform_sparse_matrix(m)
        n_qubits = m.shape[0]
        depth = m.shape[1]

        t = time.time()
        pulse_array = np.zeros((n_qubits, depth * LAYER_DT), dtype=float)
        for element in sm:
            pos = element[0]
            value = element[1]
            cs = pos[1] * LAYER_DT
            cd = cs + LAYER_DT
            if value >= 10:
                pulse_array[pos[0], cs: cd] += WCZ
            elif value < 10 and value != 0:
                for library in self.libraries:
                    if value in library:
                        pulse_array[pos[0], cs: cd] = library[value]
                        break
                else:
                    pulse_array[pos[0], cs: cs + RX_DT] = wave_construction('x').real
        self.time = time.time() - t
        return pulse_array

    def gen_pulse_one_qubit(self, m):
        sm_qubits_rx, sm_qubits_cz = circuit_preprocessing.transform_sparse_matrix_qubit(m)
        n_qubits = m.shape[0]
        depth = m.shape[1]

        t = time.time()
        pulse_array = np.zeros((1, depth * LAYER_DT), dtype=float)
        for qubit in range(n_qubits):
            for element in sm_qubits_rx[qubit]:
                pos = element[0]
                value = element[1]
                cs = pos[1] * LAYER_DT
                cd = cs + LAYER_DT
                for library in self.libraries:
                    if value in library:
                        pulse_array[pos[0], cs: cd] = library[value]
                        break
                else:
                    pulse_array[pos[0], cs: cs + RX_DT] = wave_construction('x').real

            for element in sm_qubits_cz[qubit]:
                pos = element[0]
                pulse_array[0, pos[1] * LAYER_DT: pos[1] * LAYER_DT + LAYER_DT] += WCZ
        self.time = time.time() - t
        return pulse_array


class Single2MatrixMethod(MatrixMethod):
    """
    并没有提前+好偏置，直接使用获得的cachelibrary，里边有2-11比特的偏置，高的要自己再bias中搜
    """
    def __init__(self, thresholds=None, bias_max=300):
        super().__init__(thresholds)
        self.data_loader()
        self.bias_library = {}
        for i in range(bias_max):
            self.bias_library[i] = random.random()

    def data_loader(self):
        super().data_loader()
        if self.thresholds is not None:
            for threshold in self.thresholds:
                self.libraries.append({k: WX for k, v in SINGLE_LIBRARY.items() if v >= threshold})
        self.libraries.append(SINGLE_PULSE_LIBRARY)
        for library in self.libraries:
            for i in range(8):
                library[10 + i] = WCZ

    def gen_pulse(self, m):
        sm = circuit_preprocessing.transform_sparse_matrix(m)
        n_qubits = m.shape[0]
        depth = m.shape[1]

        t = time.time()
        pulse_array = np.zeros((n_qubits, depth * LAYER_DT), dtype=float)
        for element in sm:
            pos = element[0]
            value = element[1]
            cs = pos[1] * LAYER_DT
            cd = cs + LAYER_DT
            for library in self.libraries:
                if value in library:
                    pulse_array[pos[0], cs: cd] = library[value]
                    break
            else:
                if value >= 10:
                    pulse_array[pos[0], cs: cd] = WCZ + self.bias_library[value]
                else:
                    pulse_array[pos[0], cs: cs + RX_DT] = wave_construction('x').reals
        self.time = time.time() - t
        return pulse_array

    def gen_pulse_one_qubit(self, m):
        sm_qubits_rx, sm_qubits_cz = circuit_preprocessing.transform_sparse_matrix_qubit(m)
        n_qubits = m.shape[0]
        depth = m.shape[1]

        pulse_array = np.zeros((1, depth * LAYER_DT), dtype=float)
        for qubit in range(n_qubits):
            for element in sm_qubits_rx[qubit]:
                pos = element[0]
                value = element[1]
                cs = pos[1] * LAYER_DT
                cd = cs + LAYER_DT
                if value in SINGLE_PULSE_LIBRARY:
                    pulse_array[0, cs: cd] = SINGLE_PULSE_LIBRARY[value]
                else:
                    pulse_array[0, cs: cs + RX_DT] = wave_construction('x').real

            for element in sm_qubits_cz[qubit]:
                pos = element[0]
                pulse_array[0, pos[1] * LAYER_DT: pos[1] * LAYER_DT + LAYER_DT] += WCZ
        return pulse_array


if __name__ == '__main__':
    m = Single1MatrixMethod()
    print(m.single_pulse_library)




