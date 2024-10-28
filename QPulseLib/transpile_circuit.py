from multiprocessing import Pool
from common.circuit_preprocessing import circuit_preprocessing_matrix
from dataset.high import get_dataset
import pickle
from consts import *

if __name__ == '__main__':
    pool = Pool(10)
    dataset = []
    # for qubits in ['10', '15', '20', '25', '30','35', '50', '100','150', '200', '250', '300']:
    for qubits in ['200']:
    # for qubits in ['10']:
        dataset.extend(get_dataset(int(qubits), int(qubits) + 1))
    cir_name = []
    future_list = []
    for data in dataset:
        cir_name.append(data['id'])
        # circuit_preprocessing(data['circuit'])
        future = pool.apply_async(circuit_preprocessing_matrix, args=(data['circuit'], None, CONV_BASIS_GATES, False, True))
        future_list.append(future)
    
    for i, future in enumerate(future_list):
        print(f'wait {cir_name[i]}')

        with open(f'transpiled_circuit/{cir_name[i]}', mode='wb') as f:
            a = future.get()
            pickle.dump(a, f)
        print(f'finish {cir_name[i]}')