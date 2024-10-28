import pickle
import sys

sys.path.append('..')
import numpy as np
from scipy import signal

kernels = [(3, 3), (4, 2), (2, 4), (3, 2), (2, 3), (2, 2), (1, 4), (1, 3), (1, 2)]


def get_key(dct, value):
    return list(filter(lambda k: np.array_equal(dct[k], value), dct))


def components_filter(all_components, threshold):
    for kernel in all_components:
        res = []
        for component in all_components[kernel]:
            if component[1] < threshold:
                break
            res.append(component[0])
        all_components[kernel] = res
    return all_components


def convert_value_in_dic(temp):
    # 字典处理
    mapping = temp.copy()
    for key in mapping:
        current_matrix = mapping[key]
        dic_matrix = [['I' for i in range(len(current_matrix[0]))] for j in range(len(current_matrix))]
        for row in range(len(current_matrix)):
            for col in range(len(current_matrix[0])):
                current_m = current_matrix[row][col]
                if current_m == -10 - 10j:
                    dic_matrix[row][col] = 'cz'
                elif current_m == 0 + 0j:
                    dic_matrix[row][col] = 'I'
                else:
                    dic_matrix[row][col] = f'u({current_m.real}, {current_m.imag})'
        mapping[key] = dic_matrix
    return mapping


def process_matrix(matrix_in_complex):
    circuit_matrix = [['I' for _ in range(len(matrix_in_complex[0]))] for __ in range(len(matrix_in_complex))]
    for row in range(len(matrix_in_complex)):
        for col in range(len(matrix_in_complex[0])):
            current_m = matrix_in_complex[row][col]
            if matrix_in_complex[row][col] == -10 - 10j:
                circuit_matrix[row][col] = 'cz'
            elif matrix_in_complex[row][col] == 0 + 0j:
                circuit_matrix[row][col] = 'I'
            else:
                circuit_matrix[row][col] = f'u({current_m.real}, {current_m.imag})'
    return circuit_matrix


def post_process(matrix, match_matrix):
    circuit_matrix = [[] for _ in range(matrix.shape[0])]
    # 将复数矩阵转换回字符串矩阵
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            current_m = matrix[row, col]
            if match_matrix[row, col] == 0:
                if current_m == -10 - 10j:
                    circuit_matrix[row].append('cz')
                elif current_m == 0 + 0j:
                    circuit_matrix[row].append('I')
                else:
                    circuit_matrix[row].append(f'u({current_m.real}, {current_m.imag})')
            else:
                circuit_matrix[row].append(match_matrix[row, col])
    return circuit_matrix


def kernel_process(kernel_res):
    res = {}
    import re
    pattern = r'[0-9]+.[0-9]+'
    for k in kernel_res:
        a = np.zeros(k, dtype=complex)
        for i, l1 in enumerate(kernel_res[k][0]):
            for j, l2 in enumerate(l1):
                current = kernel_res[k][0][i][j]
                if current == 'cz':
                    a[i, j] = -10 - 10j
                elif current[0] == 'u':
                    xy = re.findall(pattern, current)
                    a[i, j] = complex(float(xy[0]), float(xy[0]))
                else:
                    a[i, j] = 0 + 0j
        res[k] = a
    return res


def match_this_conv_1(matrix, gate_library={}, frequency={}, pulse_index=1, threshold=5, name=None, kernel_count={}):
    with open('../../old/result/search.result', mode='rb') as f:
        all_components = pickle.load(f)
    with open('kernel_8.res', mode='rb') as f:
        max_kernel = kernel_process(pickle.load(f))
    all_components = components_filter(all_components, threshold)
    if isinstance(matrix, list):
        return 0, gate_library, frequency, pulse_index, 0
    if 0 in matrix.shape:
        return 0, gate_library, frequency, pulse_index, 0
    max_match_percent = 0
    max_match_matrix = np.zeros(matrix.shape, dtype=int)
    max_gate_library, max_frequency, max_pulse_index = {}, {}, 0
    max_kernel_count = {}
    for i in range(3):
        current_gate_library, current_frequency, current_pulse_index = gate_library.copy(), frequency.copy(), pulse_index
        current_kernel_count = kernel_count.copy()
        # 进行三次实验
        match_percent = 0
        match_matrix = np.zeros(matrix.shape, dtype=int)
        # 计算块数和匹配率
        for kernel in kernels[i:]:
            if kernel not in current_kernel_count[name]:
                current_kernel_count[name][kernel] = 0
            if kernel[0] > match_matrix.shape[0] or kernel[1] > match_matrix.shape[1]:
                continue
            match_wave_filter = np.ones(kernel)
            # 卷积判断相等
            flat_size = kernel[0] * kernel[1]
            conv_matrix = signal.convolve2d(matrix, match_wave_filter, 'valid')
            match_conv = signal.convolve2d(match_matrix, match_wave_filter, 'valid')

            for component in all_components[kernel]:
                unused = False
                component_count = 0
                sum_component = np.sum(component)
                find_component_index = np.where((conv_matrix == sum_component) & (match_conv == 0))
                for row_start, col_start in zip(find_component_index[0], find_component_index[1]):
                    if match_conv[row_start, col_start] != 0:
                        continue
                    row_end, col_end = row_start + kernel[0], col_start + kernel[1]

                    if np.array_equal(component, matrix[row_start: row_end, col_start: col_end]):
                        match_percent += flat_size
                        component_count += 1
                        for key in current_gate_library.keys():
                            if np.array_equal(component, current_gate_library[key]):
                                match_matrix[row_start: row_end, col_start: col_end] = key
                                match_conv[row_start, col_start] = 1
                                current_frequency[key] += 1
                                break
                        else:
                            match_matrix[row_start: row_end, col_start: col_end] = current_pulse_index
                            unused = True
                        left_r, right_r = row_start - kernel[0] + 1, row_start + 2 * kernel[0] - 1
                        left_c, right_c = col_start - kernel[1] + 1, col_start + 2 * kernel[1] - 1
                        rows = [left_r if left_r >= 0 else 0,
                                right_r if right_r <= matrix.shape[0] else matrix.shape[0]]
                        cols = [left_c if left_c >= 0 else 0,
                                right_c if right_c <= matrix.shape[1] else matrix.shape[1]]
                        match_conv[rows[0]: rows[1] - kernel[0] + 1, cols[0]: cols[1] - kernel[1] + 1] = 1
                if np.array_equal(max_kernel[kernel], component):
                    current_kernel_count[name][kernel] += component_count
                if unused:
                    current_gate_library[current_pulse_index] = component
                    current_frequency[current_pulse_index] = component_count
                    current_pulse_index += 1
        if match_percent > max_match_percent:
            max_match_percent = match_percent
            max_match_matrix = match_matrix
            max_frequency = current_frequency
            max_gate_library = current_gate_library
            max_pulse_index = current_pulse_index
            max_kernel_count = current_kernel_count

    circuit_matrix = np.zeros(matrix.shape, dtype=object)
    # 将复数矩阵转换回字符串矩阵
    find_0_res = np.where(max_match_matrix == 0)
    find_not_0_res = np.where(max_match_matrix != 0)
    count_I = 0
    for row, col in zip(find_0_res[0], find_0_res[1]):
        current_e = matrix[row, col]
        if current_e == -10 - 10j:
            circuit_matrix[row, col] = 'cz'
        elif current_e == 0 + 0j:
            circuit_matrix[row, col] = 'I'
            count_I += 1
        else:
            circuit_matrix[row, col] = f'u({current_e.real}, {current_e.imag})'
    for row, col in zip(find_not_0_res[0], find_not_0_res[1]):
        circuit_matrix[row, col] = max_match_matrix[row, col]
    match_percent = max_match_percent / (matrix.shape[0] * matrix.shape[1] - count_I)
    return circuit_matrix, max_gate_library, max_frequency, max_pulse_index, match_percent, max_kernel_count


def count_gates(matrix):
    num_single = len(np.where((matrix != 0 + 0j) & (matrix != -10 - 10j))[0])
    num_cz = len(np.where(matrix == -10 - 10j)[0])
    return num_single, num_cz / 2


if __name__ == '__main__':
    import pandas as pd
    import os
    import re
    from common.circuit_preprocessing import circuit_preprocessing_matrix

    # profile_match(is_file=True)

    def load_matrix(is_rb, nums_rb, is_big_circuit, lower, upper, step):
        all_matrix = []
        circuit_name = []
        circuits = []
        pre_process_time = []
        from dataset.high import get_dataset

        if is_rb:
            from dataset.random_circuit import random_circuit
            for qubit in range(lower, upper, step):
                for i in range(nums_rb):
                    all_matrix.append(random_circuit(qubit, 100)[1])
                    circuit_name.append(f'rb_circuit_{qubit}_{100}_{i}')
        else:
            if is_big_circuit and lower == 0:
                dir_name = '../transpile'
                res = {}
                files = os.listdir(dir_name)
                pattern = r'[0-9][0-9]+'
                for file_name in files:
                    current_name = file_name.replace('.circuit', '')
                    qubit = re.findall(pattern, current_name)[0]
                    current_name = current_name.replace(f'_{qubit}', '')
                    if qubit not in res:
                        res[qubit] = [current_name]
                    else:
                        res[qubit].append(current_name)
                qubits = [str(i) for i in range(50, 301, 50)]

                for qubit in qubits:
                    for filename in res[qubit]:
                        with open(f'{dir_name}/{filename}_{qubit}.circuit', mode='rb') as f:
                            all_matrix.append(pickle.load(f))
                        circuit_name.append(f'{filename}_{qubit}')
            elif not is_big_circuit and lower != 0:
                dataset = []
                for i in range(lower, upper, step):
                    dataset.extend(get_dataset(i, i + 1))
                    circuits.extend([data['circuit'] for data in dataset])
                    circuit_name.extend([data['id'] for data in dataset])
                # 矩阵预处理
                import time
                for circuit in circuits:
                    t = time.time()
                    matrix = circuit_preprocessing_matrix(circuit, xy=True)
                    all_matrix.append(matrix)
                    pre_process_time.append(time.time() - t)
            elif is_big_circuit and lower != 0:
                dataset = []
                for i in range(lower, upper, step):
                    dataset.extend(get_dataset(i, i + 1))
                    circuits = [data['circuit'] for data in dataset]
                    circuit_name = [data['id'] for data in dataset]
                # 矩阵预处理
                import time
                for circuit in circuits:
                    t = time.time()
                    matrix = circuit_preprocessing_matrix(circuit, xy=True)
                    all_matrix.append(matrix)
                    pre_process_time.append(time.time() - t)

                dir_name = '../transpile'
                res = {}
                files = os.listdir(dir_name)
                pattern = r'[0-9][0-9]+'
                for file_name in files:
                    current_name = file_name.replace('.circuit', '')
                    qubit = re.findall(pattern, current_name)[0]
                    current_name = current_name.replace(f'_{qubit}', '')
                    if qubit not in res:
                        res[qubit] = [current_name]
                    else:
                        res[qubit].append(current_name)
                qubits = [str(i) for i in range(50, 301, 50)]

                for qubit in qubits:
                    for filename in res[qubit]:
                        with open(f'{dir_name}/{filename}_{qubit}.circuit', mode='rb') as f:
                            all_matrix.append(pickle.load(f))
                        circuit_name.append(f'{filename}_{qubit}')
        return all_matrix, circuit_name, pre_process_time


    def count_matrix_time(all_matrix, frequency, transfer, circuit_name, gates_library, pulse_index, is_add_pre_process,
                          pre_process_time, lower, upper, threshold, is_big_circuit):
        import time
        from common import All_pulse_generation
        pulse_library = {}
        for key in frequency:
            key_name = 'U' + str(key)
            pulse_library[key_name] = [transfer[key], frequency[key]]
        time_for_xixi_1, time_for_xixi_2, time_for_kechuang = [0.0 for _ in range(len(circuit_name))], [0.0 for _ in
                                                                                                        range(
                                                                                                            len(circuit_name))], [
                                                                  0.0 for _ in range(len(circuit_name))]

        wave_pulse_library = {}
        for key in pulse_library:
            row, col = len(pulse_library[key][0]), len(pulse_library[key][0][0])
            unitary_wave = []
            for i in range(row):
                current_qubit_wave = []
                for j in range(col):
                    gate_type = pulse_library[key][0][i][j]
                    current_wave_result = All_pulse_generation.wave_construction(gate_type)
                    current_qubit_wave.append(current_wave_result)
                unitary_wave.append(current_qubit_wave)
            wave_pulse_library[key] = unitary_wave

        matrix_index = 0
        single_gate_library = {}

        for matrix in all_matrix:
            # circuit = circuits[matrix_index]
            iterations = 3
            print(f"The current circuit processing is {circuit_name[matrix_index]}")
            # baseline_choice = 1

            time_start = time.time()

            # experiment for xixi
            length = 0
            for _ in range(iterations):
                long_wave_xixi = []
                matrix_base = process_matrix(matrix)
                for i in range(len(matrix_base)):
                    qubit_wave = []
                    for j in range(len(matrix_base[0])):
                        gate_type = matrix_base[i][j]
                        current_wave_result = All_pulse_generation.wave_construction(gate_type)
                        qubit_wave.append(current_wave_result)
                    long_wave_xixi.append(qubit_wave)
                length = len(long_wave_xixi)
                # print(f"wave is {len(long_wave_xixi)}")

            time_middle1 = time.time()

            # experiment for xixi_2
            length = 0
            for _ in range(iterations):
                long_wave_xixi = []
                matrix_base = process_matrix(matrix)
                for i in range(len(matrix_base)):
                    qubit_wave = []
                    for j in range(len(matrix_base[0])):
                        gate_type = matrix_base[i][j]
                        if gate_type not in single_gate_library:
                            current_wave_result = All_pulse_generation.wave_construction(gate_type)
                            single_gate_library[gate_type] = current_wave_result
                        else:
                            current_wave_result = single_gate_library[gate_type]
                        qubit_wave.append(current_wave_result)
                    long_wave_xixi.append(qubit_wave)
                length = len(long_wave_xixi)
                # print(f"wave is {len(long_wave_xixi)}")

            time_middle2 = time.time()

            # experiment for kechuang

            for _ in range(iterations):
                circuit_matrix, gates_library, frequency, pulse_index, match_percent = match_this_conv_1(
                    matrix=matrix,
                    gate_library=gates_library,
                    frequency=frequency,
                    pulse_index=pulse_index,
                    threshold=threshold)

                long_wave_kechuang = []
                j = 0
                for i in range(len(matrix)):
                    qubit_wave = []
                    while j < len(matrix[0]):
                        if matrix[i][j] == 'done':
                            continue
                        gate_type = matrix[i][j]
                        if isinstance(gate_type, int):
                            gate_type = 'U' + str(matrix[i][j])
                        if gate_type in pulse_library.keys():
                            row, col = len(pulse_library[gate_type][0]), len(pulse_library[gate_type][0][0])
                            for _i in range(row):
                                for _j in range(col):
                                    matrix[i + _i][j + _j] = 'done'
                            long_wave_kechuang.append(wave_pulse_library[gate_type])
                            j += col
                        else:
                            if gate_type not in single_gate_library:
                                current_wave_result = All_pulse_generation.wave_construction(gate_type)
                                single_gate_library[gate_type] = current_wave_result
                            else:
                                current_wave_result = single_gate_library[gate_type]
                            qubit_wave.append(current_wave_result)
                            j += 1
                    long_wave_kechuang.append(qubit_wave)
            time_end = time.time()
            if is_add_pre_process:
                time_for_xixi_1[matrix_index] = time_middle1 - time_start + pre_process_time[matrix_index]
                time_for_xixi_2[matrix_index] = time_middle2 - time_middle1 + pre_process_time[matrix_index]
                time_for_kechuang[matrix_index] = time_end - time_middle2 + pre_process_time[matrix_index]
            else:
                time_for_xixi_1[matrix_index] = time_middle1 - time_start
                time_for_xixi_2[matrix_index] = time_middle2 - time_middle1
                time_for_kechuang[matrix_index] = time_end - time_middle2
            matrix_index += 1
            print(f"The time for xixi (without cache) is \n{time_for_xixi_1}\n the time for base (without cache) \
                                                    is \n{time_for_xixi_2}\n and the time for kechuang is \n{time_for_kechuang}")
        print(f"The time for xixi (without cache) is \n{time_for_xixi_1}\n the time for base (without cache) \
                                    is \n{time_for_xixi_2}\n and the time for kechuang is \n{time_for_kechuang}")

        output = []
        output.append(circuit_name)
        output.append(time_for_xixi_1)
        output.append(time_for_xixi_2)
        output.append(time_for_kechuang)

        output = pd.DataFrame(list(output))
        if is_big_circuit:
            writer = pd.ExcelWriter(f'output_{lower}_{300}_time_{threshold}.xlsx')
        else:
            writer = pd.ExcelWriter(f'output_{lower}_{upper - 1}_time_{threshold}.xlsx')
        output.to_excel(writer, startcol=0, index=False)
        writer.save()


    def exp(lower, upper, step, is_count_gates=False, is_count_percent=False, output_to_excel=False,
            is_count_kernel=False, is_big_circuit=False, is_rb=False, nums_rb=5, count_time=False,
            is_add_pre_process=False, threshold=5):
        all_matrix, circuit_name, pre_process_time = load_matrix(is_rb, nums_rb, is_big_circuit, lower, upper, step)

        # 统计信息
        circuits_in_matrix, name_index = {}, 0
        gates_count = []
        circuit_percent = []
        kernel_frequency = {}
        kernel_count = {}

        gates_library, pulse_index, frequency = {}, 1, {}
        for current_matrix in all_matrix:
            if circuit_name[name_index] not in kernel_count.keys():
                kernel_count[circuit_name[name_index]] = {}
            if is_count_gates:
                gates_count.append(list(count_gates(current_matrix)))
            circuit_matrix, gates_library, frequency, pulse_index, match_percent, kernel_count = match_this_conv_1(
                matrix=current_matrix,
                gate_library=gates_library,
                frequency=frequency,
                pulse_index=pulse_index,
                threshold=threshold, name=circuit_name[name_index], kernel_count=kernel_count)
            if is_count_percent:
                circuit_percent.append(match_percent)
            circuits_in_matrix[circuit_name[name_index]] = circuit_matrix
            name_index += 1
        transfer = convert_value_in_dic(gates_library)

        if count_time:
            count_matrix_time(all_matrix, frequency, transfer, circuit_name, gates_library, pulse_index,
                              is_add_pre_process, pre_process_time, lower, upper, threshold, is_big_circuit)

        if output_to_excel:
            output = [circuit_name]
            if is_count_percent:
                output.append(circuit_percent)
            if is_count_gates:
                output.append([g[0] for g in gates_count])
                output.append([g[1] for g in gates_count])
            output_df = pd.DataFrame(output)
            if not is_big_circuit:
                writer = pd.ExcelWriter(f'output_{lower}_{upper - 1}_info_{threshold}.xlsx')
            else:
                writer = pd.ExcelWriter(f'output_{lower}_{300}_info_{threshold}.xlsx')
            output_df.to_excel(writer, startcol=0, index=False)
            writer.save()
        if is_count_kernel:
            for kernel in kernels:
                kernel_frequency[kernel] = [0, 0]
                for k in transfer:
                    s = (len(transfer[k]), len(transfer[k][0]))
                    if s == kernel and kernel_frequency[kernel][1] < frequency[k]:
                        if s == (1, 2) and 'I' in transfer[k][0]:
                            continue
                        kernel_frequency[kernel] = [transfer[k], frequency[k]]
            with open(f'kernel_{threshold}.res', mode='wb') as f:
                pickle.dump(kernel_frequency, f)
        # print(kernel_count)
        return {
            'gates_library': transfer,
            'frequency': frequency,
            'circuit_name': circuit_name,
            'gates_count': gates_count,
            'circuit_percent': circuit_percent,
            'kernel_frequency': kernel_frequency
        }


    exp(5, 36, 5, is_count_percent=True, is_count_gates=True, output_to_excel=True, is_count_kernel=True,
        is_big_circuit=False, threshold=5, count_time=True, is_rb=True)

    # 统计带三个次元的multiple的时间
    # 统计info，不带multiple
    # 统计kernel_count，不带multiple
    # kernel_res
    # 不同threshold的时间
