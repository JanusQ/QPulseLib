#include <ap_int.h>

#define QUBITS 10
#define C_WIDTH 10
#define W_WIDTH 1200

typedef ap_int<16> data_in;
typedef ap_int<32> data_out;
typedef ap_int<16> data_f;
typedef struct {
	ap_int<16> real;
	ap_int<16> img;
} complex;

void store_wave(out_buf[W_WIDTH], data_in* output, int qubit_offset) {
#pragma HLS INLINE off
	data_out temp;
	for (int i = 0; i < W_WIDTH / 4; ++i) {
		temp(31, 0) = out_buf[i * 4];
		temp(63, 32) = out_buf[i * 4 + 1];
		temp(95, 64) = out_buf[i * 4 + 2];
		temp(127, 96) = out_buf[i * 4 + 3];
		*(output + (qubit_offset * W_WIDTH + i) / 4) = temp;
	}
}