def encode(int_num):
    if int_num == 0 or int_num == 10:
        return '0000000000000000'
    res = bin(int_num)
    if res[0] == '-':
        res = ''.join(['1', res[3:].zfill(15)])
    else:
        res = ''.join(['0', res[2:].zfill(15)])
    return res


def encode_int(current_num, int_num):
    if int_num < 0:
        return (current_num << 16) | 32768 | (-int_num)
    else:
        return (current_num << 16) | int_num


def encode_matrix(float_m):
    if float_m.shape[0] == 1:
        res = []
        for i, it in enumerate(float_m[0]):
            res.append(encode(int(it * 10000)))
        return int(''.join(res), base=2)
    elif float_m.ndim == 1:
        res = []
        for i, it in enumerate(float_m):
            res.append(encode(int(it * 10000)))
        return int(''.join(res), base=2)
    else:
        raise Exception('不是一维数组或者行数为1的二维数组')


def encode_matrix_int(float_m):
    current_num = 0
    if float_m.shape[0] == 1:
        for it in float_m[0]:
            current_num = encode_int(current_num, int(it * 10000))
        return current_num
    elif float_m.ndim == 1:
        for it in float_m:
            current_num = encode_int(current_num, int(it * 10000))
        return current_num
    else:
        raise Exception('不是一维数组或者行数为1的二维数组')