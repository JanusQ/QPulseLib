# import
import sys

sys.path.append('..')

from consts import *
import match_tree
import pickle
import random
from common.All_pulse_generation import wave_construction
from scipy import signal
from match_code import encode, encode_int

RX_DT = 60
CZ_DT = 120
LAYER_DT = 200


# unsolve
# 1. id替代arr
# 2. 卷积方法的优化


class GenPulse:
    def __init__(self, all_array, all_array_num, cache_threshold=100):
        with open('../result/search/search_single_2_11', mode='rb') as f:
            single_library = pickle.load(f)
        # mode1 library
        self.wx = np.pad(wave_construction('x').real, (0, LAYER_DT - RX_DT), 'constant', constant_values=(0, 0))
        self.wcz = np.pad(wave_construction('cz').real, (0, LAYER_DT - CZ_DT), 'constant', constant_values=(0, 0))
        self.single_pulse_library = {k: self.wx for k in single_library}
        self.single_pulse_library[10] = self.wcz
        for i in range(8):
            self.single_pulse_library[10 + i] = self.wcz

        # mode2 library
        self.cache_library_min = {k: self.wx for k in single_library if
                                  single_library[k] >= cache_threshold}
        self.cache_library = self.single_pulse_library
        self.bias_library = {k: random.random() for k in range(301)}

        # mode3 library
        self.pattern_pulse_library = [self.gen_pulse_for_kernel(arr[1]) for arr in all_array]
        self.pattern_pulse_library_num = [self.gen_pulse_for_kernel(arr[1]) for arr in all_array_num]

    def gen_pulse_for_kernel(self, kernel):
        kernel_pulse = np.zeros((kernel.shape[0], kernel.shape[1] * LAYER_DT), dtype=float)
        for col in range(kernel.shape[1]):
            for row in range(kernel.shape[0]):
                e = kernel[row, col]
                cs = LAYER_DT * col
                cd = cs + LAYER_DT
                if e == 10:
                    kernel_pulse[row, cs: cd] = self.wcz
                elif e != 10 and e != 0:
                    kernel_pulse[row, cs: cd] = self.wx
        return kernel_pulse

    def gen_mode_single_1(self, matrix_info):
        """
        加好cz偏置，只用单门library来进行波形生成
        :param matrix_info:
        :return:
        """
        sm = matrix_info['sm']
        depth = matrix_info['depth']
        n_qubits = matrix_info['n_qubits']

        pulse_array = np.zeros((n_qubits, depth * LAYER_DT), dtype=float)
        for element in sm:
            pos = element[0]
            value = element[1]
            cs = pos[1] * LAYER_DT
            cd = cs + LAYER_DT
            if value >= 10:
                pulse_array[pos[0], cs: cd] += self.single_pulse_library[10.0]
            elif value < 10 and value != 0:
                if value in self.single_pulse_library:
                    pulse_array[pos[0], cs: cd] = self.single_pulse_library[value]
                else:
                    pulse_array[pos[0], cs: cs + RX_DT] = wave_construction('x').real
        return pulse_array

    def gen_mode_single_1_one_qubit(self, sm_qubit_rx, sm_qubit_cz, matrix_info):
        """
        加好cz偏置，只用单门library来进行波形生成
        :param matrix_info:
        :return:
        """
        depth = matrix_info['depth']

        pulse_array = np.zeros((1, depth * LAYER_DT), dtype=float)
        for element in sm_qubit_rx:
            pos = element[0]
            value = element[1]
            cs = pos[1] * LAYER_DT
            cd = cs + LAYER_DT
            if value in self.single_pulse_library:
                pulse_array[0, cs: cd] = self.single_pulse_library[value]
            else:
                pulse_array[0, cs: cs + RX_DT] = wave_construction('x').real

        for element in sm_qubit_cz:
            pos = element[0]
            pulse_array[0, pos[1] * LAYER_DT: pos[1] * LAYER_DT + LAYER_DT] += self.single_pulse_library[10]

    def gen_mode_single_2(self, matrix_info):
        """
        并没有提前+好偏置，直接使用获得的cachelibrary，里边有2-11比特的偏置，高的要自己再bias中搜
        :param matrix_info:
        :return:
        """
        sm = matrix_info['sm']
        depth = matrix_info['depth']
        n_qubits = matrix_info['n_qubits']

        pulse_array = np.zeros((n_qubits, depth * LAYER_DT), dtype=float)
        for element in sm:
            pos = element[0]
            value = element[1]
            cs = pos[1] * LAYER_DT
            cd = cs + LAYER_DT
            if value in self.cache_library_min:
                pulse_array[pos[0], cs: cd] = self.cache_library_min[value]
            elif value in self.cache_library:
                pulse_array[pos[0], cs: cd] = self.cache_library[value]
            else:
                if value >= 10:
                    pulse_array[pos[0], cs: cd] = self.cache_library[10.0] + \
                                                  self.bias_library[pos[0]]
                else:
                    pulse_array[pos[0], cs: cs + RX_DT] = wave_construction('x').real
        return pulse_array

    def gen_mode_single_2_one_qubit(self, sm_qubit_rx, sm_qubit_cz, matrix_info):
        """
        并没有提前+好偏置，直接使用获得的cachelibrary，里边有2-11比特的偏置，高的要自己再bias中搜
        :param matrix_info:
        :return:
        """
        depth = matrix_info['depth']

        pulse_array = np.zeros((1, depth * LAYER_DT), dtype=float)
        for element in sm_qubit_rx:
            pos = element[0]
            value = element[1]
            cs = pos[1] * LAYER_DT
            cd = cs + LAYER_DT
            if value in self.cache_library_min:
                pulse_array[0, cs: cd] = self.cache_library_min[value]
            elif value in self.cache_library:
                pulse_array[0, cs: cd] = self.cache_library[value]
            else:
                pulse_array[0, cs: cs + RX_DT] = wave_construction('x').real
        for element in sm_qubit_cz:
            pos = element[0]
            value = element[1]
            cs = pos[1] * LAYER_DT
            cd = cs + LAYER_DT
            if value in self.cache_library_min:
                pulse_array[0, cs: cd] = self.cache_library_min[value]
            elif value in self.cache_library:
                pulse_array[0, cs: cd] = self.cache_library[value]
            else:
                pulse_array[0, cs: cd] = self.cache_library[10.0] + self.bias_library[pos[0]]

        return pulse_array

    def gen_mode_match(self,
                       matrix_info,
                       match_result):
        sm = matrix_info['sm']
        depth = matrix_info['depth']
        n_qubits = matrix_info['n_qubits']

        pulse_array = np.zeros((n_qubits, depth * LAYER_DT), dtype=float)
        for element in sm:
            pos = element[0]
            value = element[1]
            if pos in match_result:
                rs, rd = pos[0], pos[0] + match_result[pos][1].shape[0]
                cs = pos[1] * LAYER_DT
                cd = cs + match_result[pos][1].shape[1] * LAYER_DT
                pulse_array[rs: rd, cs: cd] = self.pattern_pulse_library_num[match_result[pos][0]]
            else:
                cs = pos[1] * LAYER_DT
                cd = cs + LAYER_DT
                if value >= 10:
                    pulse_array[pos[0], cs: cd] += self.single_pulse_library[10]
                elif value < 10 and value != 0:
                    if value in self.single_pulse_library:
                        pulse_array[pos[0], cs: cd] = self.single_pulse_library[value]
                    else:
                        pulse_array[pos[0], cs: cs + RX_DT] = wave_construction('x').real
        return pulse_array

    def gen_mode_match_one_qubit(self, sm_qubit_rx, sm_qubit_cz, matrix_info, match_result):
        depth = matrix_info['depth']

        pulse_array = np.zeros((1, depth * LAYER_DT), dtype=float)
        for element in sm_qubit_rx:
            pos = element[0]
            value = element[1]
            if pos in match_result:
                cs = pos[1] * LAYER_DT
                cd = cs + match_result[pos][1].shape[1] * LAYER_DT
                pulse_array[0, cs: cd] = self.pattern_pulse_library_num[match_result[pos][0]]
            else:
                cs = pos[1] * LAYER_DT
                cd = cs + LAYER_DT
                if value in self.single_pulse_library:
                    pulse_array[0, cs: cd] = self.single_pulse_library[value]
                else:
                    pulse_array[0, cs: cs + RX_DT] = wave_construction('x').real

        for element in sm_qubit_cz:
            pos = element[0]
            pulse_array[0, pos[1] * LAYER_DT: pos[1] * LAYER_DT + LAYER_DT] += self.single_pulse_library[10]
        return pulse_array


    def gen_mode_withoutcache(self, matrix_info):
        sm = matrix_info['sm']
        depth = matrix_info['depth']
        n_qubits = matrix_info['n_qubits']

        pulse_array = np.zeros((n_qubits, depth * LAYER_DT), dtype=float)
        for element in sm:
            pos = element[0]
            value = element[1]
            cs = pos[1] * LAYER_DT
            cd = cs + LAYER_DT
            if value >= 10:
                pulse_array[pos[0], cs: cs + CZ_DT] = wave_construction('cz').real
            elif value != 0:
                pulse_array[pos[0], cs: cs + RX_DT] = wave_construction('x').real


class MatchPattern:
    def __init__(self,
                 all_components,  # 0为id，1为矩阵，2为频次，3为卷积值
                 all_array,
                 all_components_num,
                 all_array_num,
                 threshold_min=5,
                 need_max=True):
        """
        过滤100后的all_
        :param all_components:
        :param threshold_min:
        """
        self.threshold_min = threshold_min
        self.all_components = all_components
        self.all_array = all_array
        self.need_max = need_max

        m_tree_min, m_tree, single = match_tree.build_match_tree(self.all_array, threshold=threshold_min,
                                                                 need_max=self.need_max)

        self.m_tree_min = m_tree_min
        self.m_tree = m_tree
        self.single = single

        self.all_components_num = all_components_num
        self.all_array_num = all_array_num
        all_components_num_dict = {}
        for kernel in self.all_components_num:
            all_components_num_dict[kernel] = {}
            for arr in self.all_components_num[kernel]:
                all_components_num_dict[kernel][arr[3]] = arr
        self.all_components_num = all_components_num_dict

    def _build_slide_list(self, m, sm):
        num_row, num_col = m.shape

        def slide_list_from_miss_array(miss_array):
            return [pos for pos, x in np.ndenumerate(miss_array) if not x]

        pre_pos, pre_element = None, None
        miss_array = np.zeros(m.shape, dtype=bool)
        for pos, element in sm.items():
            if pre_pos is None:
                pre_pos, pre_element = pos, element
                continue
            else:
                # 根本没在字典内，然后一大块全都没必要滑了
                if element not in self.single and element < 10:
                    miss_array[pos] = True
                # 两个元素间也是超过max_kernel格超过的部分也不用滑了
                # 在同一行
                if pre_pos[0] == pos[0]:
                    if pos[1] - pre_pos[1] > MAX_COL_KERNEL:
                        miss_array[pos[0], pre_pos[1] + 1: pos[1] - MAX_COL_KERNEL + 1] = True
                # 在不同行
                else:
                    if pre_pos[1] + 1 != num_col:
                        miss_array[pre_pos[0], pre_pos[1] + 1: num_col] = True
                    if pos[1] > MAX_COL_KERNEL - 1:
                        miss_array[pos[0], 0: pos[1] - MAX_COL_KERNEL + 1] = True
        return slide_list_from_miss_array(miss_array)

    def _match_one_window(self, m, pos, mode='tree'):
        def match_one_window_tree():
            if self.m_tree is None:
                return []
            sub_m = m[pos[0]: pos[0] + MAX_ROW_KERNEL, pos[1]: pos[1] + MAX_COL_KERNEL]
            # 返回的是所有匹配上的矩阵
            res = match_tree.search_from_tree(sub_m, self.m_tree)
            if not res:
                res = match_tree.search_from_tree(sub_m, self.m_tree_min)
            return res

        def match_one_window_conv():
            res = []
            for kernel_size in KERNEL_SIZE_LIST:
                sub_m = m[pos[0]: pos[0] + kernel_size[0], pos[1]: pos[1] + kernel_size[1]]
                conv_value = np.sum(np.multiply(sub_m, FILTERS[kernel_size]))
                for arr in self.all_array:
                    if arr[3] == conv_value and np.array_equal(arr[1], sub_m):
                        res.append(arr)
            return res

        if mode == 'tree':
            return match_one_window_tree()
        else:
            return match_one_window_conv()

    def match_pattern_sm(self, m, sm, mode='tree'):
        match_res = {}
        slide_list = self._build_slide_list(m, sm)
        for pos in slide_list:
            if pos[0] + MAX_ROW_KERNEL >= m.shape[0] or pos[1] + MAX_COL_KERNEL >= m.shape[1]:
                continue
            res = self._match_one_window(m, pos, mode)
            if res:
                max_shape = 0
                max_r = None
                for r in res:
                    s = r[1].shape[0] * r[1].shape[1]
                    if s > max_shape:
                        max_shape = s
                        max_r = r
                match_res[pos] = max_r
        return match_res

    @staticmethod
    def _encode_sm(pre_ele, ele_list):
        current_num = pre_ele[1]
        for ele in ele_list:
            current_num = encode_int(current_num << 16 * (ele[0][1] - pre_ele[0][1] - 1), ele[1])
            pre_ele = ele
        return current_num

    def match_pattern_sm_num(self, sm_rx_int):
        match_res = {}
        # for
        num_ele = len(sm_rx_int)
        i = 0
        while i < num_ele:
            # 搜索8个元素
            one_match = []
            for j in range(i + 1, num_ele):
                # 在同一行
                if sm_rx_int[j][0][0] == sm_rx_int[i][0][0]:
                    if sm_rx_int[j][0][1] - sm_rx_int[i][0][1] < MAX_KERNEL:
                        one_match.append(sm_rx_int[j])
                    else:
                        break
                else:
                    break
            if one_match:
                # 编码部分
                code = self._encode_sm(sm_rx_int[i], one_match)
                num_j = len(one_match)
                len_code = one_match[-1][0][1] - sm_rx_int[i][0][1] + 1
                for kernel in KERNEL_SIZE_LIST_NUM_REVERSE[(MAX_KERNEL - len_code):]:
                    if code % (2 ** 16) == 0:
                        code = code >> 16
                        continue
                    num_j -= 1
                    if code in self.all_components_num[kernel]:
                        # 此时匹配上了
                        match_res[sm_rx_int[i][0]] = self.all_components_num[kernel][code]
                        i = num_j + i + 2
                        break
                    else:
                        code = code >> 16
                else:
                    i += 1
            else:
                i += 1
        return match_res

    def match_pattern_sm_num_parallel_one_qubit(self, sm_rx_int):
        """
        此时传进来的是一个qubit的所有门
        :param sm_rx_int:
        :return:
        """
        match_res = {}
        # for
        num_ele = len(sm_rx_int)
        i = 0
        while i < num_ele:
            # 搜索8个元素
            one_match = []
            for j in range(i + 1, num_ele):
                if sm_rx_int[j][0][1] - sm_rx_int[i][0][1] < MAX_KERNEL:
                    one_match.append(sm_rx_int[j])
                else:
                    break
            if one_match:
                # 编码部分
                code = self._encode_sm(sm_rx_int[i], one_match)
                num_j = len(one_match)
                len_code = one_match[-1][0][1] - sm_rx_int[i][0][1] + 1
                for kernel in KERNEL_SIZE_LIST_NUM_REVERSE[(MAX_KERNEL - len_code):]:
                    if code % (2 ** 16) == 0:
                        code = code >> 16
                        continue
                    num_j -= 1
                    if code in self.all_components_num[kernel]:
                        # 此时匹配上了
                        match_res[sm_rx_int[i][0]] = self.all_components_num[kernel][code]
                        i = num_j + i + 2
                        break
                    else:
                        code = code >> 16
                else:
                    i += 1
            else:
                i += 1
        return match_res


    def match_pattern_conv(self, m, judge_equal=True):
        match_res = {}
        for kernel in KERNEL_CONV_LIST:
            if kernel[0] > m.shape[0] or kernel[1] > m.shape[1]:
                continue
            conv_matrix = signal.convolve2d(m, FILTERS[kernel], 'valid')
            for arr in self.all_components[kernel]:
                find_idx = np.where(conv_matrix == arr[3])
                for row, col in zip(find_idx[0], find_idx[1]):
                    if (row, col) in match_res:
                        continue
                    sub_m = m[row: row + kernel[0], col: col + kernel[1]]
                    if judge_equal:
                        if np.array_equal(sub_m, arr[1]):
                            match_res[(row, col)] = arr
                    else:
                        match_res[(row, col)] = arr
        return match_res

    def match_pattern_conv_speed_up(self, m, judge=True):
        match_res = {}
        match_matrix = np.zeros(m.shape, dtype=int)
        for kernel in KERNEL_CONV_LIST:
            if kernel[0] > m.shape[0] or kernel[1] > m.shape[1]:
                continue
            conv_matrix = signal.convolve2d(m, FILTERS[kernel], 'valid')
            conv_match = signal.convolve2d(match_matrix, FILTERS[kernel], 'valid')
            for arr in self.all_components[kernel]:
                find_idx = np.where((conv_matrix == arr[3]) & (conv_match == 0))
                for row, col in zip(find_idx[0], find_idx[1]):
                    if (row, col) in match_res or conv_match[row, col] != 0:
                        continue
                    row_end, col_end = row + kernel[0], col + kernel[1]
                    # 这里是精确比对了
                    if np.array_equal(m[row: row_end, col: col_end], arr[1]):
                        match_res[(row, col)] = arr
        return match_res


if __name__ == '__main__':
    # exp = Experiment(qubits_list=[15, 35], threshold=5, matrix_dir='../transpiled_circuit')
    # exp.run()
    from dataset.high import get_dataset
    dataset = get_dataset(10, 11)
    print(dataset[3])