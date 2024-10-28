import sys
sys.path.append('..')
import numpy as np
import networkx as nx

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
            s = set()
            for row in window.shape[0]:
                for col in window.shape[1]:
                    s.add(window[row, col])
            # r中，0为子矩阵，1为计数，2为卷积值，3为=看类型
            # 特征值捏
            conv_sum = np.sum(np.multiply(window, np.ones(kernel_size)))
            for r in res:
                if np.array_equal(r[0], window):
                    if r[2] == s:
                        r[1] += 1
                        r[3].append(window)
                    break
            else:
                res.append([conv_sum, 1, s, [window]])
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

def min_swap_to_sort(arr):
    # Create an array of pairs where first 
    # element is array element and second element 
    # is position of first element
    n = arr.shape[0]
    arr_pos = []

    for i in range(n):
        arr_pos.append((arr[i], i))
    
    # Sort the array by array element values to 
    # get right position of every element as second 
    # element of pair. 
    arr_pos = sorted(arr_pos)

    # To keep track of visited elements. Initialize 
    # all elements as not visited or false.
    vis = [False for _ in range(n)]

    ans = 0
    for i in range(n):
        if vis[i] or arr_pos[i][1] == i:
            continue

        cycle_size = 0
        j = i
        while not vis[j]:
            vis[j] = True
            j = arr_pos[j][1]
            cycle_size += 1

        ans += cycle_size - 1
    return ans 


def min_swap(a, b):
    # c-style code
    a = a.flatten()
    b = b.flatten()
    n = a.shape[0]
    # map to store position of elements in array B 
    # we basically store element to index mapping. 
    mp = {}
    for i in range(n):
        mp[b[i]] = i
    
    # now we're storing position of array A elements 
    # in array B.
    for i in range(n):
        b[i] = mp[a[i]]

    # returing minimum swap for sorting in modified 
    # array B as final answer 
    return min_swap_to_sort(b) 


def build_MST(matrice):
    m_len = len(matrice)
    v_ids = list(range(len(matrice)))
    edges = []
    for i in range(m_len):
        for j in range(i + 1, m_len):
            edges.append((i, j, min_swap(matrice[i], matrice[j])))
    
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    mst = nx.minimum_spanning_edges(G, algorithm='kruskal')
    return list(mst)


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
                            if pattern[2] == matrix_pattern[2]:
                                pattern[1] += matrix_pattern[1]
                                pattern[3].extend(matrix_pattern[3])
                    else:
                        # not found. add to all.
                        not_exist_pattern.append(matrix_pattern)
                kernel_res.extend(not_exist_pattern)
        for pattern in kernel_res:
            pattern.append(build_MST(pattern[3]))
        res[kernel_size] = kernel_res
    import pickle
    with open(f'../result/search/search_conv_{left_range}_{right_range}_threshold{threshold}', mode='wb') as f:
        pickle.dump(res, f)
    print('finish search pattern.')



