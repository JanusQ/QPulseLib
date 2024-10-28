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

void init_circuit(data_in buf_input[QUBITS][C_WIDTH]{
#pragma HLS INLINE
	
#pragma HLS UNROLL
	for (int qubit = 0; qubit < QUBITS; ++qubit) {
		for (int layer = 0; layer < C_WIDTH; ++layer) {
			buf_input[qubit][layer] = 0;
		}
	}
}

void init_match_matrix(bool match_matrix[QUBITS][C_WIDTH]) {
	// init match_matrix
#pragma HLS PIPELINE
	for (int i = 0; i < QUBITS; ++i) {
		for (int j = 0; j < C_WIDTH; ++j) {
			match_matrix[i][j] = false;
		}
	}
}