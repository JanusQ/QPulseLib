#include <ap_int.h>
#include "complex.cpp"

#define QUBITS 10
#define C_WIDTH 10
#define W_WIDTH 1200
#define PATTERNS 100
#define PERIOD 120 // period for all gates
#define TABLESIZE 400

typedef ap_int<16> data_in;
typedef ap_int<32> data_out;
typedef ap_int<16> data_f;
typedef ap_uint<16> PID;
typedef struct {
	ap_int<16> real;
	ap_int<16> img;
} complex;

typedef struct {
    ap_int<16> key;
    ap_int<16> value;
} hash_in;

void load_buf(data_in input_line[C_WIDTH], data_in buf_line[C_WIDTH]) {
#pragma HLS INLINE off
#pragma HLS PIPELINE
	for (int i = 0; i < C_WIDTH; ++i) {
		buf_line[i] = input_line[i];
	}
}


void load_wave_lib(data_out wave_lib[PATTERNS][9 * PERIOD], data_out *wlib_input) {
	// assume data_out has the type of 32 bits
#pragma INLINE off
	data_out temp;
	for (int pattern = 0; pattern < PATTERNS; ++pattern) {
		for (int layer = 0; layer < 9 * PERIOD / 4; ++layer) {
			temp = *(wlib_input + (pattern * 9 * PERIOD) / 4 + layer);
			wave_lib[pattern][layer * 4] = temp(31, 0);
			wave_lib[pattern][layer * 4 + 1] = temp(63, 32);
			wave_lib[pattern][layer * 4 + 2] = temp(95, 64);
			wave_lib[pattern][layer * 4 + 3] = temp(127, 96);
		}
	}
}


void load_pattern_lib(PID pattern_lib[PATTERNS][16], PID *plib_input) {
	// assume data of pattern has the type of 16 bits, along with data_in
#pragma HLS INLINE off
	PID temp;
	for (int pattern = 0; pattern < PATTERNS; ++pattern) {
		for (int layer = 0; layer < 2; ++layer) {
			temp = *(plib_input + pattern * 2 + layer);
			pattern_lib[pattern][layer * 8] = temp(15, 0);
			pattern_lib[pattern][layer * 8 + 1] = temp(31, 16);
			pattern_lib[pattern][layer * 8 + 2] = temp(47, 32);
			pattern_lib[pattern][layer * 8 + 3] = temp(63, 48);
			pattern_lib[pattern][layer * 8 + 4] = temp(79, 64);
			pattern_lib[pattern][layer * 8 + 5] = temp(95, 80);
			pattern_lib[pattern][layer * 8 + 6] = temp(111, 96);
			pattern_lib[pattern][layer * 8 + 7] = temp(127, 112);
		}
	}
}


void load_conv_rsts(PID conv_rsts[PATTERNS], *rsts_input){
// assume conv_rsts has the type of 16 bits
#pragma HLS INLINE off
	PID temp;
	for (int pattern = 0; pattern < PATTERNS; ++pattern){
		temp = *(rsts_input + pattern / 8);
		conv_rsts[pattern * 8] = temp(15, 0);
		conv_rsts[pattern * 8 + 1] = temp(31, 16);
		conv_rsts[pattern * 8 + 2] = temp(47, 32);
		conv_rsts[pattern * 8 + 3] = temp(63, 48);
		conv_rsts[pattern * 8 + 4] = temp(79, 64);
		conv_rsts[pattern * 8 + 5] = temp(95, 80);
		conv_rsts[pattern * 8 + 6] = temp(111, 96);
		conv_rsts[pattern * 8 + 7] = temp(127, 112);

	}
}

data_out pulse(data_in *input, PID *plib_input, data_out *wlib_input, data_out *output) {


	data_in buf_input[QUBITS][C_WIDTH], buf_line_1[C_WIDTH], buf_line_2[C_WIDTH], buf_line_3[C_WIDTH], buf_line_4[C_WIDTH], *input;
	data_f buf_filter[4][3]; // simply set buf_filter
	data_out buf_output[QUBITS][W_WIDTH], *output;
	bool match_matrix[QUBITS][C_WIDTH];
	PID conv_rsts[PATTERNS], *rsts_input;
	data_out wave_lib[PATTERNS][9 * PERIOD]; // the max pattern 3 * 3 only contains 9 elements
	PID pattern_lib[PATTERNS][16]; // but 9 cannot be divided by 8 (128 bit / 16 bit), so assume the width to be 16
	hash_in hashtable[TABLESIZE]

	init_circuit(buf_input);
	init_match_matrix(match_matrix);
	
	load_wave_lib(wave_lib, wlib_input);
	load_pattern_lib(pattern_lib, plib_input);
	load_conv_rsts(conv_rsts, rsts_input)
	init_hash(hashtable, conv_rsts);

	input_circuit(input, buf_input);
	if (QUBITS < 4) {
	// match under 4 qubits
	}
	else {
		// init buf_line * 4
		load_buf(buf_input[0], buf_line_1);
		load_buf(buf_input[1], buf_line_2);
		load_buf(buf_input[2], buf_line_3);
		load_buf(buf_input[3], buf_line_4);
		match_4(buf_line_1, buf_line_2, buf_line_3, buf_line_4, match_matrix, buf_output, 0);
		// calculate unmatched gates
		calculate_wave(buf_line_1, match_matrix[0], buf_output[0]);
		store_wave(buf_output[0], output);
		for (int qubit = 4; qubit < QUBITS; ++qubit) {
			switch (qubit % 4) {
			case 0:
				load_buf(buf_input[qubit], buf_line_1);
				add_match(buf_line_2, buf_line_3, buf_line_4, buf_line_1, match_matrix, buf_output, qubit - 3);
				calculate_wave(buf_line_2, match_matrix[qubit - 3], buf_output[qubit - 3]);
				store_wave(buf_output[qubit - 3], output + ((qubit - 3) * W_WIDTH) / 8);
				break;
			case 1:
				load_buf(buf_input[qubit], buf_line_2);
				add_match(buf_line_3, buf_line_4, buf_line_1, buf_line_2, match_matrix, buf_output, qubit - 3);
				calculate_wave(buf_line_3, match_matrix[qubit - 3], buf_output[qubit - 3]);
				store_wave(buf_output[qubit - 3], output + ((qubit - 3) * W_WIDTH) / 8);
				break;
			case 2:
				load_buf(buf_input[qubit], buf_line_3);
				add_match(buf_line_4, buf_line_1, buf_line_2, buf_line_3, match_matrix, buf_output, qubit - 3);
				calculate_wave(buf_line_4, match_matrix[qubit - 3], buf_output[qubit - 3]);
				store_wave(buf_output[qubit - 3], output + ((qubit - 3) * W_WIDTH) / 8);
				break;
			case 3:
				load_buf(buf_input[qubit], buf_line_4);
				add_match(buf_line_1, buf_line_2, buf_line_3, buf_line_4, match_matrix, buf_output, qubit - 3);
				calculate_wave(buf_line_1, match_matrix[qubit - 3], buf_output[qubit - 3]);
				store_wave(buf_output[qubit - 3], output + ((qubit - 3) * W_WIDTH) / 8);
				break;
			}
			
		}
		for (int qubit = QUBITS - 3; qubit < QUBITS; ++qubit) {
			calculate_wave(buf_input[qubit], match_matrix[qubit], buf_output[qubit]);
			store_wave(buf_output[qubit], output + (qubit * W_WIDTH) / 8);
		}
	}
}
