#include <ap_int.h>
// #include <ap_fixed.h>

#define C_WIDTH 10  // length of circuit
#define QUBITS 10
#define W_WIDTH 1200 // length of wave
#define PERIOD 120 // period for all gates

typedef ap_int<16> data_in;
typedef ap_int<16> data_out;
typedef ap_int<16> data_f; // filter type
typedef ap_uint<16> PID;
typedef struct {
    ap_int<16> real;
    ap_int<16> img;
} complex;

// create matrix only containing 1
data_f buf_filter[4][3] = { {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1} };

// to calculate wave of only 1 buf_line
void calculate_wave(data_in buf_line[C_WIDTH], bool match_line[C_WIDTH], complex wave_line[W_WIDTH]) {
#pragma HLS INLINE off

    data_in gate;

#pragma HLS PIPELINE
    for (int layer = 0; layer < C_WIDTH; ++layer) {
        gate = buf_line[layer];
        // to ensure the wave has not been calculated
        if (match_line[layer] == false && gate != 0) {
            pulse_gen(gate, wave_line, layer);
        }
    }
   
}


void get_wave_from_cache(complex wave_out[4][W_WIDTH], int x_offset, int layer, PID pattern_id, int filter_x, int filter_y, int qubit_offset) {
#pragma HLS INLINE

#pragma HLS PIPELINE
    for (int i = 0; i < filter_x; ++i) {
        for (int j = 0; j < filter_y; ++j) {
            for (int dt = 0; dt < PERIOD; ++dt) {
                wave_out[qubit_offset + x_offset + i][(layer + j) * PERIOD + dt] = wave_lib[pattern_id][(i * filter_y + j) * PERIOD + dt];
            }
        }
    }
}


void kernel(data_in buf_qubit[4][C_WIDTH], data_f buf_filter[4][3], data_out out_buf[C_WIDTH - 2], int x_offset, int filter_x, int filter_y) {
#pragma HLS INLINE off
    data_in sum, add_rst, mul_rst, data_temp;
    data_f f_temp;
    for (int off = 0; off < filter_x; ++off) {
#pragma HLS PIPELINE 
        for (int i = 0; i < C_WIDTH; i += filter_x) {
            sum = 0;
            for (int j = 0; j < filter_x; ++j) {
                for (int m = 0; m < filter_y; ++m) {
                    data_temp = buf_qubit[x_offset + m][off + i + j];
                    f_temp = buf_filter[x_offset + m][j];
                    mul_rst = data_temp * f_temp;
                    add_rst = mul_rst + sum;
                    sum = add_rst;
                }
            }
            out_buf[i] = sum;
        }
    }
}



void convolver(data_in buf_qubit[4][C_WIDTH], data_f buf_filter[4][3], complex wave_out[QUBITS][W_WIDTH], bool match_matrix[QUBITS][C_WIDTH], int x_offset, int filter_x, int filter_y, int qubit_offset) {
#pragma HLS INLINE off

    data_in gate;
    PID pattern_id;
    data_out out_buf[C_WIDTH - 2];


    for (int loop_times = 0; loop_times <= 4 - filter_y; ++loop_times) {
        
        kernel(buf_qubit, buf_filter, out_buf, x_offset, filter_x, filter_y);

        for (int i = 0; i < C_WIDTH - 2; ++i) {
            pattern_id = hash_search(out_buf[i]);
            if (pattern_id == -1) {
                continue;

            }
            else {
                // accurate comparison of the pattern
                int flag = 0;
                for (int x = 0; x < filter_x; ++x) {
                    if (flag) {
                        break;
                    }
                    for (int y = 0; y < filter_y; ++y) {
                        // the data in pattern_lib lists in only one row
                        if (buf_qubit[x + x_offset][y + i] != pattern_lib[pattern_id][x * filter_y + y]) {
                            flag = 1;
                            break;
                        }
                    }
                }

                if (flag == 0){
                    // generate wave from cache first
                    get_wave_from_cache(wave_out, x_offset, i, pattern_id, filter_x, filter_y, qubit_offset);

                    // update match_matrix with TRUE
                #pragma HLS PIPELINE
                    for (int m = 0; m < filter_x; ++m) {
                        for (int n = i; n < i + filter_y; ++n) {
                            match_matrix[qubit_offset + x_offset + m][n] = true;
                        }
                    }
                }
            }
        }

    }
}



void match_4(data_in buf_line_1[C_WIDTH], data_in buf_line_2[C_WIDTH], data_in buf_line_3[C_WIDTH], data_in buf_line_4[C_WIDTH], bool match_matrix[QUBITS][C_WIDTH], data_in wave_out[QUBITS][W_WIDTH], int qubit_offset) {
#pragma HLS INLINE off


    data_in buf_qubit[4][C_WIDTH];


    // init buf_qubit by 4 * buf_line
    #pragma HLS PIPELINE
    for (int i = 0; i < C_WIDTH; ++i) {
        #pragma HLS PIPELINE
        buf_qubit[0][i] = buf_line_1[i];
        buf_qubit[1][i] = buf_line_2[i];
        buf_qubit[2][i] = buf_line_3[i];
        buf_qubit[3][i] = buf_line_4[i];
    }

// convolution 3 * 3
#pragma HLS PIPELINE
    for (int i = 0; i < 2; ++i) {
        convolver(buf_qubit, buf_filter, wave_out, match_matrix, i, 3, 3, qubit_offset);
    }

// convolution 4 * 2
    convolver(buf_qubit, buf_filter, wave_out, match_matrix, 0, 4, 2, qubit_offset);


// convolution 3 * 2
#pragma HLS PIPELINE
    for (int i = 0; i < 2; ++i) {
        convolver(buf_qubit, buf_filter, wave_out, match_matrix, i, 3, 2, qubit_offset);
    }

// convolution 2 * 4
#pragma HLS PIPELINE
    for (int i = 0; i < 3; ++i) {
        convolver(buf_qubit, buf_filter, wave_out, match_matrix, i, 2, 4, qubit_offset);
    }

// convolution 2 * 3
#pragma HLS PIPELINE
    for (int i = 0; i < 3; ++i) {
        convolver(buf_qubit, buf_filter, wave_out, match_matrix, i, 2, 3, qubit_offset);
    }

// convolution 2 * 2
#pragma HLS PIPELINE
    for (int i = 0; i < 3; ++i) {
        convolver(buf_qubit, buf_filter, wave_out, match_matrix, i, 2, 2, qubit_offset);
    }

// convolution 1 * 2
#pragma HLS PIPELINE
    for (int i = 0; i < 4; ++i) {
        convolver(buf_qubit, buf_filter, wave_out, match_matrix, i, 1, 2, qubit_offset);
    }


}


void add_match(data_in buf_line_1[C_WIDTH], data_in buf_line_2[C_WIDTH], data_in buf_line_3[C_WIDTH], data_in buf_line_4[C_WIDTH], bool match_matrix[QUBITS][C_WIDTH], data_in wave_out[QUBITS][W_WIDTH], int qubit_offset) {
#pragma HLS INLINE off


    data_in buf_qubit[4][C_WIDTH];

    // init buf_qubit by 4 * buf_line
#pragma HLS PIPELINE
    for (int i = 0; i < C_WIDTH; ++i) {
#pragma HLS PIPELINE
        buf_qubit[0][i] = buf_line_1[i];
        buf_qubit[1][i] = buf_line_2[i];
        buf_qubit[2][i] = buf_line_3[i];
        buf_qubit[3][i] = buf_line_4[i];
    }

    // convolution 3 * 3
    convolver(buf_qubit, buf_filter, wave_out, match_matrix, 1, 3, 3, qubit_offset);
    

    // convolution 4 * 2
    convolver(buf_qubit, buf_filter, wave_out, match_matrix, 0, 4, 2, qubit_offset);


    // convolution 3 * 2
    convolver(buf_qubit, buf_filter, wave_out, match_matrix, 1, 3, 2, qubit_offset);


    // convolution 2 * 4
    convolver(buf_qubit, buf_filter, wave_out, match_matrix, 2, 2, 4, qubit_offset);
   

    // convolution 2 * 3
    convolver(buf_qubit, buf_filter, wave_out, match_matrix, 2, 2, 3, qubit_offset);
    

    // convolution 2 * 2
    convolver(buf_qubit, buf_filter, wave_out, match_matrix, 2, 2, 2, qubit_offset);
    

    // convolution 1 * 2
    convolver(buf_qubit, buf_filter, wave_out, match_matrix, 3, 1, 2, qubit_offset);
    
}


/*

void match_3(data_in buf_line_1[C_WIDTH], data_in buf_line_2[C_WIDTH], data_in buf_line_3[C_WIDTH]) {
    bool match_matrix[3][C_WIDTH];
    complex wave_out[3][W_WIDTH];
    data_in buf_qubit[3][C_WIDTH];

    // init match_matrix
#pragma HLS PIPELINE
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < C_WIDTH; ++j) {
            match_matrix[i][j] = false;
        }
    }

    // init buf_qubit by 3 * buf_line
#pragma HLS PIPELINE
    for (int i = 0; i < C_WIDTH; ++i) {
#pragma HLS PIPELINE
        buf_qubit[0][i] = buf_line_1[i];
        buf_qubit[1][i] = buf_line_2[i];
        buf_qubit[2][i] = buf_line_3[i];
    }

    // convolution 3 * 3
#pragma HLS PIPELINE
    for (int i = 0; i < 2; ++i) {
        convolver(buf_qubit, buf_filter, wave_out, match_matrix, i, 3, 3);
    }

    // convolution 3 * 2
#pragma HLS PIPELINE
    for (int i = 0; i < 2; ++i) {
        convolver(buf_qubit, buf_filter, wave_out, match_matrix, i, 3, 2);
    }

    // convolution 2 * 4
#pragma HLS PIPELINE
    for (int i = 0; i < 3; ++i) {
        convolver(buf_qubit, buf_filter, wave_out, match_matrix, i, 2, 4);
    }

    // convolution 2 * 3
#pragma HLS PIPELINE
    for (int i = 0; i < 3; ++i) {
        convolver(buf_qubit, buf_filter, wave_out, match_matrix, i, 2, 3);
    }

    // convolution 2 * 2
#pragma HLS PIPELINE
    for (int i = 0; i < 3; ++i) {
        convolver(buf_qubit, buf_filter, wave_out, match_matrix, i, 2, 2);
    }

    // convolution 1 * 2
#pragma HLS PIPELINE
    for (int i = 0; i < 4; ++i) {
        convolver(buf_qubit, buf_filter, wave_out, match_matrix, i, 1, 2);
    }
}

void match_2(data_in buf_line_1[C_WIDTH], data_in buf_line_2[C_WIDTH]) {
    bool match_matrix[2][C_WIDTH];
    complex wave_out[2][W_WIDTH];
    data_in buf_qubit[2][C_WIDTH];

    // init match_matrix
#pragma HLS PIPELINE
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < C_WIDTH; ++j) {
            match_matrix[i][j] = false;
        }
    }

    // init buf_qubit by 2 * buf_line
#pragma HLS PIPELINE
    for (int i = 0; i < C_WIDTH; ++i) {
#pragma HLS PIPELINE
        buf_qubit[0][i] = buf_line_1[i];
        buf_qubit[1][i] = buf_line_2[i];
    }


    // convolution 2 * 4
#pragma HLS PIPELINE
    for (int i = 0; i < 3; ++i) {
        convolver(buf_qubit, buf_filter, wave_out, match_matrix, i, 2, 4);
    }

    // convolution 2 * 3
#pragma HLS PIPELINE
    for (int i = 0; i < 3; ++i) {
        convolver(buf_qubit, buf_filter, wave_out, match_matrix, i, 2, 3);
    }

    // convolution 2 * 2
#pragma HLS PIPELINE
    for (int i = 0; i < 3; ++i) {
        convolver(buf_qubit, buf_filter, wave_out, match_matrix, i, 2, 2);
    }

    // convolution 1 * 2
#pragma HLS PIPELINE
    for (int i = 0; i < 4; ++i) {
        convolver(buf_qubit, buf_filter, wave_out, match_matrix, i, 1, 2);
    }
}

void match_1(data_in buf_line_1[C_WIDTH]) {
    bool match_matrix[C_WIDTH];
    complex wave_out[W_WIDTH];
    data_in buf_qubit[C_WIDTH];

    // init match_matrix
#pragma HLS PIPELINE
    for (int j = 0; j < C_WIDTH; ++j) {
        match_matrix[i][j] = false;
    }
   

    // init buf_qubit by 2 * buf_line
#pragma HLS PIPELINE
    for (int i = 0; i < C_WIDTH; ++i) {
#pragma HLS PIPELINE
        buf_qubit[0][i] = buf_line_1[i];
    }

    // convolution 1 * 2
#pragma HLS PIPELINE
    for (int i = 0; i < 4; ++i) {
        convolver(buf_qubit, buf_filter, wave_out, match_matrix, i, 1, 2);
    }
}

*/