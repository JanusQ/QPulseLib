#include <ap_int.h>

#define N_ROWS 4
#define N_COLS 4

typedef ap_int<128> data_in;
typedef ap_int<32> data_buf;
// typedef ap_int<128> data_out;
int row, col, width;

void load_input(data_in* input, buf[row][col][width]);
void measure_bandwidth(data_in* input1, data_in * input2, data_in * input3, data_in * input4);
