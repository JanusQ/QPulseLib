# import
import os
import sys
sys.path.append('')

import time
import datetime
import pickle
import numpy as np
import pandas as pd
from multiprocessing import Pool

from common.circuit_preprocessing import circuit_preprocessing_matrix

from match_m.match_pattern_parallel import GenPulse, MatchPattern


def transform_sparse_matrix(m: np.ndarray):
    """
    转换为稀疏矩阵同时生成预制的波形矩阵，
    :param m:
    :return:
    """
    num_row, num_col = m.shape
    # 转为稀疏矩阵
    sm = []
    sm_rx = []
    sm_cz = []
    sm_rx_int = []
    sm_qubits_rx = []
    sm_qubits_cz = []
    sm_qubits_rx_int = []
    for row in range(num_row):
        sm_qubits_rx.append([])
        sm_qubits_cz.append([])
        sm_qubits_rx_int.append([])
        for col in range(num_col):
            if m[row, col] != 0:
                if m[row, col] >= 10:
                    sm.append(((row, col), m[row, col]))
                    sm_cz.append(((row, col), m[row, col]))
                    sm_qubits_cz[row].append(((row, col), m[row, col]))
                else:
                    sm.append(((row, col), m[row, col]))
                    sm_rx.append(((row, col), m[row, col]))
                    sm_rx_int.append(((row, col), int(m[row, col] * 10000)))
                    sm_qubits_rx[row].append(((row, col), m[row, col]))
                    sm_qubits_rx_int[row].append(((row, col), int(m[row, col] * 10000)))

    return {
        'sm': sm,
        'sm_rx': sm_rx,
        'sm_cz': sm_cz,
        'sm_rx_int': sm_rx_int,
        'sm_qubits_rx': sm_qubits_rx,
        'sm_qubits_cz': sm_qubits_cz,
        'sm_qubits_rx_int': sm_qubits_rx_int
    }


class Experiment:
    def __init__(self, qubits_list, threshold=100, threshold_num=2, matrix_dir=None):
        """
        事先做个过滤
        :param qubits_list:
        :param threshold:
        :param matrix_dir:
        """
        self.qubits_list = qubits_list
        self.matrix_dir = matrix_dir
        self.matrix_list = {qubit: [] for qubit in self.qubits_list}
        self.matrix_name_list = {qubit: [] for qubit in self.qubits_list}
        if matrix_dir != 'no':
            self._load_matrix()

        self.threshold = threshold
        self.threshold_num = threshold_num
        self.all_array = []
        self.all_components = {}
        self.all_array_num = []
        self.all_components_num = {}
        self._load_pattern()

        print('load pulse generation')
        self.gp = GenPulse(self.all_array, self.all_array_num)
        print('finish pulse generation')
        print('load match')
        self.mp = MatchPattern(self.all_components, self.all_array,
                               self.all_components_num, self.all_array_num,
                               need_max=False)
        print('finish match')

    def _load_pattern(self):
        print('load components')
        new_all_components = {}
        with open('result/search/search_2_11_6_6_threshold2_cz', mode='rb') as f:
            all_components = pickle.load(f)
        x_id = 0
        for kernel in all_components:
            new_all_components[kernel] = []
            for arr in all_components[kernel]:
                if arr[1] >= self.threshold:
                    new_all_components[kernel].append([x_id] + arr)
                    self.all_array.append([x_id] + arr)
                    x_id += 1
        self.all_components = new_all_components

        new_all_components_num = {}
        with open('result/search/search_2_11_1_8_threshold2_no_cz', mode='rb') as f:
            all_components = pickle.load(f)
        x_id = 0
        for kernel in all_components:
            new_all_components_num[kernel] = []
            for arr in all_components[kernel]:
                if arr[1] >= self.threshold_num:
                    new_all_components_num[kernel].append([x_id] + arr)
                    self.all_array_num.append([x_id] + arr)
                    x_id += 1
        self.all_components_num = new_all_components_num
        print('finish load components')

    def _load_matrix(self):
        """
        生成矩阵，如果没有矩阵文件，则重新生成，巨慢
        :return:
        """
        print('load matrix')
        if self.matrix_dir is None:
            from dataset.high import get_dataset

            pool = Pool(10)
            futures = []
            for qubit in self.qubits_list:
                dataset = get_dataset(qubit, qubit + 1)
                for data in dataset:
                    self.matrix_name_list[qubit].append(data['id'])
                    futures.append(pool.apply_async(circuit_preprocessing_matrix, (data['circuit'],)))
            pool.close()
            pool.join()
            for future in futures:
                matrix = future.get()[0]
                self.matrix_list[matrix.shape[0]].append(matrix)
        else:
            all_cir = os.listdir(self.matrix_dir)

            for cir in all_cir:
                if int(cir.split('_')[-1]) in self.qubits_list:
                    with open(os.path.join(self.matrix_dir, cir), mode='rb') as f:
                        matrix = pickle.load(f)[0]
                        self.matrix_list[int(cir.split('_')[-1])].append(matrix)
                        self.matrix_name_list[int(cir.split('_')[-1])].append(cir)
        print('finish load matrix')

    def mix_match_and_gen_pulse(self, sm_qubit_rx_int, sm_qubit_rx, sm_qubit_cz, matrix_info):
        match_result = self.mp.match_pattern_sm_num_parallel_one_qubit(sm_qubit_rx_int)
        self.gp.gen_mode_match_one_qubit(sm_qubit_rx, sm_qubit_cz, matrix_info, match_result)

    def run_for_one_matrix(self, m, pool):
        sm_dict = transform_sparse_matrix(m)
        sm = sm_dict['sm']
        sm_rx_int = sm_dict['sm_rx_int']
        sm_qubits_rx = sm_dict['sm_qubits_rx']
        sm_qubits_cz = sm_dict['sm_qubits_cz']
        sm_qubits_rx_int = sm_dict['sm_qubits_rx_int']

        result_t = []
        matrix_info = {
            'sm': sm,
            'depth': m.shape[1],
            'n_qubits': m.shape[0]
        }
        t = time.time()
        self.gp.gen_mode_single_1(matrix_info)
        result_t.append(time.time() - t)

        t = time.time()
        self.gp.gen_mode_single_2(matrix_info)
        result_t.append(time.time() - t)

        # t = time.time()
        # match_result = self.mp.match_pattern_sm(m, sm, mode='tree')
        # self.gp.gen_mode_match(matrix_info, match_result)
        # result_t.append(time.time() - t)
        #
        # t = time.time()
        # match_result = self.mp.match_pattern_conv(m, judge_equal=False)
        # self.gp.gen_mode_match(matrix_info, match_result)
        # result_t.append(time.time() - t)
        #
        # t = time.time()
        # match_result = self.mp.match_pattern_conv(m, judge_equal=True)
        # self.gp.gen_mode_match(matrix_info, match_result)
        # result_t.append(time.time() - t)

        t = time.time()
        match_result = self.mp.match_pattern_sm_num(sm_rx_int)
        self.gp.gen_mode_match(matrix_info, match_result)
        result_t.append(time.time() - t)


        t = time.time()
        for qubit in range(len(sm_qubits_rx)):
            self.gp.gen_mode_single_1_one_qubit(sm_qubits_rx[qubit], sm_qubits_cz[qubit],matrix_info)
        result_t.append(time.time() - t)

        t = time.time()
        for qubit in range(len(sm_qubits_rx)):
            self.gp.gen_mode_single_2_one_qubit(sm_qubits_rx[qubit], sm_qubits_cz[qubit], matrix_info)
        result_t.append(time.time() - t)

        t = time.time()
        for qubit in range(len(sm_qubits_rx)):
            match_result = self.mp.match_pattern_sm_num_parallel_one_qubit(sm_qubits_rx_int[qubit])
            self.gp.gen_mode_match_one_qubit(sm_qubits_rx[qubit], sm_qubits_cz[qubit], matrix_info, match_result)
        result_t.append(time.time() - t)

        ######################################################
        # 多进程开启了##########################################
        #####################################################
        t = time.time()
        futures = []
        for qubit in range(len(sm_qubits_rx)):
            futures.append(pool.apply_async(self.gp.gen_mode_single_1_one_qubit, args=(sm_qubits_rx[qubit],
                                                                                       sm_qubits_cz[qubit],
                                                                                       matrix_info)))
        for future in futures:
            future.get()
        result_t.append(time.time() - t)

        t = time.time()
        futures = []
        for qubit in range(len(sm_qubits_rx)):
            futures.append(pool.apply_async(self.gp.gen_mode_single_2_one_qubit, args=(sm_qubits_rx[qubit],
                                                                                       sm_qubits_cz[qubit],
                                                                                       matrix_info)))
        for future in futures:
            future.get()
        result_t.append(time.time() - t)



        t = time.time()
        futures = []
        for qubit in range(len(sm_qubits_rx)):
            futures.append(pool.apply_async(self.mix_match_and_gen_pulse, args=(sm_qubits_rx_int[qubit],
                                                                           sm_qubits_rx[qubit],
                                                                           sm_qubits_cz[qubit],
                                                                           matrix_info)))
        for future in futures:
            future.get()
        result_t.append(time.time() - t)

        # t = time.time()
        # self.gp.gen_mode_withoutcache(matrix_info)
        # result_t.append(time.time() - t)

        return result_t

    def run(self):
        num_methods = 9
        current_date = datetime.datetime.now()
        td = str(current_date.year) + '_' + str(current_date.month) + '_' + str(current_date.day)
        if not os.path.exists(f'../result/match/{td}'):
            os.makedirs(f'../result/match/{td}')
        save_path = f'../result/match/{td}/{str(current_date).replace(":", "_")}'
        os.makedirs(save_path)
        pool = Pool(100)
        for qubit in self.qubits_list:
            res_t = [[] for _ in range(num_methods)]
            print('run', qubit)
            for m in self.matrix_list[qubit]:
                m_t = self.run_for_one_matrix(m, pool)
                assert len(m_t) == num_methods
                for i in range(num_methods):
                    res_t[i].append(m_t[i])
            print('finish', qubit)
            df = pd.DataFrame([self.matrix_name_list[qubit]] + res_t)
            df = df.rename(index={
                0: '算法名',
                1: '提前加好偏置',
                2: '并没有提前加好偏置',
                3: 'num',
                4: '1的单比特',
                5: '2的单比特',
                6: '3的单比特',
                7: '4的并行',
                8: '5的并行',
                9: '6的并行'
                # 3: 'tree',
                # 4: '卷积不精确比对',
                # 5: '卷积精确比对',
                # 6: '稀疏卷积',
                # 4: '最原始的卷积'
            })
            df.to_excel(f'{save_path}/time_{1000}_{qubit}_{num_methods}methods.xlsx', index=True, header=False)


if __name__ == '__main__':
    # exp = Experiment(qubits_list=[15, 35, 50, 100], threshold=2, matrix_dir='no')
    # from dataset.high import get_dataset
    # from line_profiler import LineProfiler

    # dataset = get_dataset(10, 11)
    # data = dataset[6]
    # qc = data['circuit']

    # sm_dict = transform_sparse_matrix(circuit_preprocessing_matrix(qc)[0])
    # sm_rx_int = sm_dict['sm_rx_int']

    # lp = LineProfiler()
    # lp_wrapper = lp(exp.mp.match_pattern_sm_num)
    # lp_wrapper(sm_rx_int)
    # lp.print_stats()

    exp = Experiment(qubits_list=[15, 35, 50, 100], threshold=5, matrix_dir='transpiled_circuit')
    exp.run()

    # todo: float改为整数矩阵
    # todo: all_components改为字典形式，省略查找时间
    # todo: 波形生成并行
