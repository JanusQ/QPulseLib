import random
from qiskit import QuantumCircuit
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

'''
The code is to generate random benchmark as RB
The circuit serves as one layer of single-qubit gate and one or two layers of two-qubit gate
No limitations on number of qubits and length of quantum circuit
'''


# n_qubits = random.randint(2, 10)
# total_layer = random.randint(5, 30)


# the main function to generate random circuit
def generate_random_circuit(n_qubits, total_layer):

    layer = ['single', 'double']
    single_gate = ['I', 'X', 'X/2', '-X/2', 'Y', 'Y/2', '-Y/2']
    double_gate = ['cz']

    # circuit initialization with 'I' gate everywhere
    circuit = [['I' for i in range(total_layer)] for j in range(n_qubits)]

    # to get location of two-qubits gate(cz gate) at the very layer
    def get_cz_clocation(n_qubits):
            qubit_group = [i for i in range(n_qubits)]
            cz_location, used_qubit = [], []
            two_qubits_sample = random.sample(qubit_group, k = 2)             # to randomly decide where to set up the first two-qubits gate
            original_lower, original_upper = min(two_qubits_sample), max(two_qubits_sample)
            taken_qubits = list(range(original_lower, original_upper + 1))    # taken qubits contains idle qubits between [lower, upper], including used qubits at this time
            used_qubit += taken_qubits                                        # used qubits contains qubits actually employed for two-qubits
            cz_location.append(two_qubits_sample)
            lower, upper = original_lower, original_upper

            while lower > 1:
                choice_seed = random.sample(qubit_group, k=1)
                if choice_seed[0] // 2 != 0:
                    two_qubits_sample = random.sample(qubit_group[:lower], k = 2)           # to randomly decide whether to take two-qubits gate at idle space left
                    cz_location.append(two_qubits_sample)
                    lower, upper = min(two_qubits_sample), max(two_qubits_sample)
                    taken_qubits = list(range(lower, upper + 1))
                    used_qubit += taken_qubits
            lower, upper = original_lower, original_upper
            while upper < n_qubits - 2:
                choice_seed = random.sample(qubit_group, k=1)
                if choice_seed[0] // 2 != 0:
                    two_qubits_sample = random.sample(qubit_group[upper+1:], k = 2)
                    cz_location.append(two_qubits_sample)
                    lower, upper = min(two_qubits_sample), max(two_qubits_sample)
                    taken_qubits = list(range(lower, upper + 1))
                    used_qubit += taken_qubits

            rest_qubits = list(set(qubit_group).difference(set(used_qubit)))               # rest qubits contains true idle qubits at this layer
            return cz_location, rest_qubits, used_qubit


    # To generate corresponding quantum circuit using qiskit
    # to generate circuit in each layer of designated total layers
    def circuit_construction(total_layer, circuit):
        former_double_layers, former_single_layers = 0, 0
        layer_types = []
        for current_layer in range(total_layer):
            if former_double_layers < 2 and former_single_layers == 0:
                current_layer_type = random.choice(layer)         # randomly choose layer type of current layer
                if current_layer_type == 'single':
                    former_single_layers, former_double_layers = 1, 0
                else:
                    former_single_layers = 0
                    former_double_layers += 1
            elif former_double_layers > 1 and former_single_layers == 0:
                current_layer_type = 'single'
                former_single_layers, former_double_layers = 1, 0
            else:
                current_layer_type = 'double'
                former_single_layers, former_double_layers = 0, 1
            if current_layer_type == 'single':
                for qubit in range(n_qubits):
                    gate_weight = [0.28, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12]         # weight of single-qubit gate, is able to change according to actual occurences of the very gate
                    current_gate = random.choices(single_gate, weights = gate_weight, k = 1)
                    circuit[qubit][current_layer] = current_gate[0]
                    # addqc_single_gate(qubit, current_gate[0])
            elif current_layer_type == 'double':
                double_gates_location, rest_qubits, used_qubits = get_cz_clocation(n_qubits)

                for used_qubit in used_qubits:
                    circuit[used_qubit][current_layer] = None
                for each_cz_location in double_gates_location:
                    circuit[each_cz_location[0]][current_layer], circuit[each_cz_location[1]][current_layer] = 'cz', 'cz'
                    # addqc_double_gate(each_cz_location[0], each_cz_location[1])
                # for rest_qubit in rest_qubits:
                #     addqc_single_gate(rest_qubit, 'I')
            layer_types.append(current_layer_type)
        return circuit, layer_types

    circuit, layer_types = circuit_construction(total_layer, circuit)

    return circuit, layer_types

from qiskit import QuantumCircuit

# def rb2qiskitCircuit(circuit):
#     qc = QuantumCircuit(len(circuit[0]))
#     for layer in circuit:
#         for qubit, gate in enumerate(layer):
#             if gate == "I":
#                 qc.i(qubit)
#             elif gate == 
            

print(generate_random_circuit(5, 5)[0])

# draw and print the circuit
# circ = pd.DataFrame(list(circuit))
# writer = pd.ExcelWriter('circuit_layer.xlsx')
# circ.to_excel(writer, startcol=0, index=False)
# writer.save()
#
# qc.draw('mpl')
# plt.show()







