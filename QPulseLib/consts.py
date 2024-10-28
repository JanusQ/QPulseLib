import numpy as np

np.random.seed(30)


def gen_filter(kernel_size):
    return np.ones(shape=kernel_size)


MAX_KERNEL = 8
MAX_ROW_KERNEL = 6
MAX_COL_KERNEL = 6

KERNEL_SIZE_LIST = [(i, j) for i in range(1, MAX_ROW_KERNEL + 1) for j in range(1, MAX_COL_KERNEL + 1) if (i, j) != (1, 1)]
KERNEL_SIZE_LIST_NUM = [(1, i) for i in range(2, 1 + MAX_KERNEL)]
KERNEL_SIZE_LIST_NUM_REVERSE = list(reversed(KERNEL_SIZE_LIST_NUM))
KERNEL_CONV_LIST = [(3, 3), (4, 2), (2, 4), (3, 2), (2, 3), (4, 1), (1, 4), (2, 2), (3, 1), (1, 3), (2, 1), (1, 2)]
CONV_BASIS_GATES = ['rx', 'ry', 'cz', 'id']
BASIS_GATES = ['rx', 'cz', 'rz', 'id']
FILTERS = {kernel_size: gen_filter(kernel_size) for kernel_size in KERNEL_SIZE_LIST}