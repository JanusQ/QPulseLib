from consts import *


class Node:
    def __init__(self, value, depth, array=None):
        self.value = value
        self.depth = depth
        self.array = array
        self.children = {}


def build_match_tree(all_array, threshold=1000, need_max=True):

    root_min = Node(None, 0)
    root = Node(None, 0)
    single = []

    def build_tree(i_array, depth, r, c, root):
        array = i_array[1]
        if r < array.shape[0] and c < array.shape[1]:
            if array[r, c] not in single:
                single.append(array[r, c])
            if array[r, c] in root.children:
                node = root.children[array[r, c]]
            else:
                node = Node(array[r, c], depth + 1)
                root.children[array[r, c]] = node
        else:
            if '*' in root.children:
                node = root.children['*']
            else:
                node = Node('*', depth + 1)
                root.children['*'] = node

        if r + 1 < MAX_ROW_KERNEL:
            build_tree(i_array, depth + 1, r + 1, c, node)
        else:
            if c + 1 < array.shape[1]:
                build_tree(i_array, depth + 1, 0, c + 1, node)
            else:
                node.array = i_array

    for a in [arr for arr in all_array if arr[2] >= threshold]:
        build_tree(a, 1, 0, 0, root_min)
    if need_max:
        for a in [arr for arr in all_array if arr[2]]:
            build_tree(a, 1, 0, 0, root)

    return root_min, root, single


def search_from_tree(sub_m, m_tree):
    all_match_m = []

    def search(sub_m, r, c, root, depth):
        if root is None:
            return

        if root.array is not None:
            all_match_m.append(root.array)
        if sub_m[r, c] >= 10:
            nodes = [root.children.get(10.0), root.children.get('*')]
        else:
            nodes = [root.children.get(sub_m[r, c]), root.children.get('*')]

        for n in nodes:
            if r + 1 < MAX_ROW_KERNEL:
                search(sub_m, r + 1, c, n, depth + 1)
            else:
                if c + 1 < MAX_COL_KERNEL:
                    search(sub_m, 0, c + 1, n, depth + 1)

    search(sub_m, 0, 0, m_tree, 0)
    return all_match_m
