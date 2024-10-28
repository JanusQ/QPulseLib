#include <ap_int.h>

/*

Assume the input data has the width of 128 bit, and the type of data_in gets 16 bit

*/

#define QUBITS 10
#define C_WIDTH 10
#define W_WIDTH 1200

typedef ap_int<16> data_in;
typedef struct {
	ap_int<16> real;
	ap_int<16> img;
} complex;

void input_circuit(data_in* input, data_in buf_input[QUBITS][C_WIDTH]) {
#pragma HLS INLINE

	data_in temp;
	for (int qubit; qubit < QUBITS; ++qubit) {
		for (int layer; layer < C_WIDTH; ++layer) {
			temp = *(input + (qubit * C_WIDTH + layer) / 8);
			buf_input[qubit][layer * 8] = temp(15, 0);
			buf_input[qubit][layer * 8 + 1] = temp(31, 16);
			buf_input[qubit][layer * 8 + 2] = temp(47, 32);
			buf_input[qubit][layer * 8 + 3] = temp(63, 48);
			buf_input[qubit][layer * 8 + 4] = temp(79, 64);
			buf_input[qubit][layer * 8 + 5] = temp(95, 80);
			buf_input[qubit][layer * 8 + 6] = temp(111, 96);
			buf_input[qubit][layer * 8 + 7] = temp(127, 112);
		}
	}
}

