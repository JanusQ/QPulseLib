import sys
sys.path.append('..')
from dataset.random_circuit import su4_circuit, random_circuit
import numpy as np
from multiprocessing import Pool

qubits_list = [5, 10, 15, 20, 25, 30, 35, 50, 100, 150, 200, 250, 300]

pool = Pool(10)
circuit_name = []
circuits = []
for qubits in qubits_list:
    circuit_name.append(f'rb_{qubits}_{50}')
    circuits.append(random_circuit(qubits, qubits * 50, two_qubit_prob=0.35, reverse=True))
    circuit_name.append(f'su4_{qubits}_{50}')
    circuits.append(su4_circuit(qubits, 50))

def parallel_transpile(circuits):
    # 编译转换为复数矩阵
    from common.circuit_preprocessing import circuit_preprocessing_matrix
    from coupling_map.coupling_map import one_dim_couping
    tqcs = []
    futures = []
    for idx, circuit in enumerate(circuits):
        print(circuit_name[idx])
        future = pool.apply_async(circuit_preprocessing_matrix, (circuit, one_dim_couping(circuits[0].num_qubits), ['rx', 'ry', 'cz', 'i'], False, True))
        futures.append(future)
    for future in futures:
        tqcs.append(future.get()[0])
    circuits = tqcs
    import pickle
    with open('random_circuits.pkl', mode='wb') as f:
        pickle.dump(circuits, f)
    return circuits

circuits = parallel_transpile(circuits)

# 接下来就是计算匹配时间
wx = np.pad(wave_construction('x').real, (0, 140), 'constant', constant_values=(0, 0))
wcz = np.pad(wave_construction('cz').real, (0, 80), 'constant', constant_values=(0, 0))
def gen_pulse_for_kernel(kernel):
    kernel_pulse = np.zeros((kernel.shape[0], kernel.shape[1] * 200), dtype=float)
    for col in range(kernel.shape[1]):
        for row in range(kernel.shape[0]):
            e = kernel[row, col]
            cs = 200 * col
            cd = cs + 200
            if e == 10:
                kernel_pulse[row, cs: cd] = wcz
            elif e != 10 and e != 0:
                kernel_pulse[row, cs: cd] = wx
    return kernel_pulse

# 准备那个kernel
import pickle
with open(f'../result/search/search_2_11_6_6_threshold8_cz', mode='rb') as f:
    all_components = pickle.load(f)
print(all_components.keys())
print(all_components[(3, 3)][0])
for kernel_size in all_components:
    for component in all_components[kernel_size]:
        array = component[0]
        array[np.where(array == 10 + 0j)] = -10-10j
        component[2] = np.sum(array)
        component.append(gen_pulse_for_kernel(component[0]))
print(all_components[(3, 3)][0])
# 生成kernel对应的pulse
from common.All_pulse_generation import wave_construction

# pluse_library
with open('single_gate_weights.result', mode='rb') as f:
    single_library = pickle.load(f)
single_pulse_library = {k: np.pad(wave_construction('x').real, (0, 140), 'constant', constant_values=(0, 0)) for k in single_library}
single_pulse_library[-10-10j] = np.pad(wave_construction('cz').real, (0, 80), 'constant', constant_values=(0, 0))

