import numpy as np

########################################
# 这个是卷积用的kernel ##################
########################################

# 接下来就是计算匹配时间
# 生成kernel对应的pulse
from common.All_pulse_generation import wave_construction


def load_pulse_library():
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
    with open(f'../result/search/search_2_11_6_6_threshold2_cz', mode='rb') as f:
        all_components = pickle.load(f)
    print(all_components[(1, 2)])

    for kernel_size in all_components:
        for component in all_components[kernel_size]:
            array = component[0]
            array[np.where(array == 10 + 0j)] = -10-10j
            component[2] = np.sum(array)
            component.append(gen_pulse_for_kernel(component[0]))

    # pluse_library
    with open('single_gate_weights.result', mode='rb') as f:
        single_library = pickle.load(f)
    single_pulse_library = {k: np.pad(wave_construction('x').real, (0, 140), 'constant', constant_values=(0, 0)) for k in single_library}
    single_pulse_library[-10-10j] = np.pad(wave_construction('cz').real, (0, 80), 'constant', constant_values=(0, 0))

    return wx, wcz, all_components, single_pulse_library