  #include <ap_int.h>
// #include <ap_fixed.h>

#define C_WIDTH 10  // length of circuit
#define QUBITS 10
#define W_WIDTH 10 // length of wave
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

// data_f buf_filter_3_3[3][3], buf_filter_4_2[4][2], buf_filter_2_4[2][4], buf_filter_2_3[2][3], buf_filter_3_2[3][2];
// data_f buf_filter_2_2[2][2], buf_filter_1_4[4], buf_filter_1_2[2];


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


void kernel_3_3(data_in buf_qubit[3][C_WIDTH], data_f buf_filter_3_3[3][3], data_out out_buf[C_WIDTH-2]){
    #pragma HLS INLINE off
    data_in sum, add_rst, mul_rst, data_temp;
    data_f f_temp;
    for (int off = 0; off < 3; ++off){
        #pragma HLS PIPELINE 
        for (int i = 0; i < C_WIDTH; i += 3){
            sum = 0;
            for (int j = 0; j < 3; ++j){
                for (int m = 0; m < 3; ++m){
                    data_temp = buf_qubit[m][off+i+j];
                    f_temp = buf_filter_3_3[m][j];
                    mul_rst = data_temp * f_temp;
                    add_rst = mul_rst + sum;
                    sum = add_rst;
                }
            }
            out_buf[i] = sum;
        }
    }
}

void kernel_4_2(data_in buf_qubit[4][C_WIDTH], data_f buf_filter_4_2[4][2], data_out out_buf[C_WIDTH-1]){
    #pragma HLS INLINE off
    data_in sum, add_rst, mul_rst, data_temp;
    data_f f_temp;
    for (int off = 0; off < 2; ++off){
        #pragma HLS PIPELINE 
        for (int i = 0; i < C_WIDTH; i += 2){
            sum = 0;
            for (int j = 0; j < 2; ++j){
                for (int m = 0; m < 4; ++m){
                    data_temp = buf_qubit[m][off+i+j];
                    f_temp = buf_filter_4_2[m][j];
                    mul_rst = data_temp * f_temp;
                    add_rst = mul_rst + sum;
                    sum = add_rst;
                }
            }
            out_buf[i] = sum;
        }
    }
}

void kernel_2_4(data_in buf_qubit[2][C_WIDTH], data_f buf_filter_2_4[2][4], data_out out_buf[C_WIDTH-3]){
    #pragma HLS INLINE off
    data_in sum, add_rst, mul_rst, data_temp;
    data_f f_temp;
    for (int off = 0; off < 4; ++off){
        #pragma HLS PIPELINE 
        for (int i = 0; i < C_WIDTH; i += 4){
            sum = 0;
            for (int j = 0; j < 4; ++j){
                for (int m = 0; m < 2; ++m){
                    data_temp = buf_qubit[m][off+i+j];
                    f_temp = buf_filter_2_4[m][j];
                    mul_rst = data_temp * f_temp;
                    add_rst = mul_rst + sum;
                    sum = add_rst;
                }
            }
            out_buf[i] = sum;
        }
    }
}

void kernel_3_2(data_in buf_qubit[3][C_WIDTH], data_f buf_filter_3_2[3][2], data_out out_buf[C_WIDTH-1]){
    #pragma HLS INLINE off
    data_in sum, add_rst, mul_rst, data_temp;
    data_f f_temp;
    for (int off = 0; off < 2; ++off){
        #pragma HLS PIPELINE 
        for (int i = 0; i < C_WIDTH; i += 2){
            sum = 0;
            for (int j = 0; j < 2; ++j){
                for (int m = 0; m < 3; ++m){
                    data_temp = buf_qubit[m][off+i+j];
                    f_temp = buf_filter_3_2[m][j];
                    mul_rst = data_temp * f_temp;
                    add_rst = mul_rst + sum;
                    sum = add_rst;
                }
            }
            out_buf[i] = sum;
        }
    }
}

void kernel_2_3(data_in buf_qubit[2][C_WIDTH], data_f buf_filter_2_3[2][3], data_out out_buf[C_WIDTH-2]){
    #pragma HLS INLINE off
    data_in sum, add_rst, mul_rst, data_temp;
    data_f f_temp;
    for (int off = 0; off < 3; ++off){
        #pragma HLS PIPELINE 
        for (int i = 0; i < C_WIDTH; i += 3){
            sum = 0;
            for (int j = 0; j < 3; ++j){
                for (int m = 0; m < 2; ++m){
                    data_temp = buf_qubit[m][off+i+j];
                    f_temp = buf_filter_2_3[m][j];
                    mul_rst = data_temp * f_temp;
                    add_rst = mul_rst + sum;
                    sum = add_rst;
                }
            }
            out_buf[i] = sum;
        }
    }
}

void kernel_2_2(data_in buf_qubit[2][C_WIDTH], data_f buf_filter_2_2[2][2], data_out out_buf[C_WIDTH-1]){
    #pragma HLS INLINE off
    data_in sum, add_rst, mul_rst, data_temp;
    data_f f_temp;
    for (int off = 0; off < 2; ++off){
        #pragma HLS PIPELINE 
        for (int i = 0; i < C_WIDTH; i += 2){
            sum = 0;
            for (int j = 0; j < 2; ++j){
                for (int m = 0; m < 2; ++m){
                    data_temp = buf_qubit[m][off+i+j];
                    f_temp = buf_filter_2_2[m][j];
                    mul_rst = data_temp * f_temp;
                    add_rst = mul_rst + sum;
                    sum = add_rst;
                }
            }
            out_buf[i] = sum;
        }
    }
}

void kernel_1_4(data_in buf_qubit[C_WIDTH], data_f buf_filter_1_4[4], data_out out_buf[C_WIDTH]){
    #pragma HLS INLINE off
    data_in sum, add_rst, mul_rst, data_temp;
    data_f f_temp;
    for (int off = 0; off < 4; ++off){
        #pragma HLS PIPELINE 
        for (int i = 0; i < C_WIDTH; i += 4){
            sum = 0;
            for (int j = 0; j < 4; ++j){
                for (int m = 0; m < 1; ++m){
                    data_temp = buf_qubit[m][off+i+j];
                    f_temp = buf_filter_1_4[j];
                    mul_rst = data_temp * f_temp;
                    add_rst = mul_rst + sum;
                    sum = add_rst;
                }
            }
            out_buf[i] = sum;
        }
    }
}

void kernel_1_2(data_in buf_qubit[C_WIDTH], data_f buf_filter_1_2[2], data_out out_buf[C_WIDTH]){
    #pragma HLS INLINE off
    data_in sum, add_rst, mul_rst, data_temp;
    data_f f_temp;
    for (int off = 0; off < 2; ++off){
        #pragma HLS PIPELINE 
        for (int i = 0; i < C_WIDTH; i += 2){
            sum = 0;
            for (int j = 0; j < 2; ++j){
                for (int m = 0; m < 1; ++m){
                    data_temp = buf_qubit[m][off+i+j];
                    f_temp = buf_filter_1_4[j];
                    mul_rst = data_temp * f_temp;
                    add_rst = mul_rst + sum;
                    sum = add_rst;
                }
            }
            out_buf[i] = sum;
        }
    }
}





void convolver_3_3(data_in buf_qubit, complex wave_out, bool match_matrix, x_offset){
    #pragma HLS INLINE off

    data_f buf_filter_3_3[3][3];
//    data_in buf_qubit[3][C_WIDTH];
    data_in gate;
    data_out out_buf[C_WIDTH - 2];
    PID pattern_id;
//    complex wave_out[3][W_WIDTH], wave_temp[3][PERIOD];


    /*
    #pragma HLS PIPELINE
    for (int i = 0; i < C_WIDTH; ++i){
        #pragma HLS PIPELINE
        buf_qubit[0][i] = buf_line_1[i];
        buf_qubit[1][i] = buf_line_2[i];
        buf_qubit[2][i] = buf_line_3[i];
    }
    */


    kernel_3_3(buf_qubit, buf_filter_3_3, out_buf);

    for (int i = 0; i < C_WIDTH - 2; ++i) {
        pattern_id = hashtable_3_3[hash_search(out_buf[i])];
        if (pattern_id == -1) {
            continue;

        // we should apply convolution first
        /*
            // generate pulse using calculation
            #pragma HLS PIPELINE
            for (int qubit = 0; qubit < 3; ++qubit){
                for (int layer = 0; layer < 3; ++layer) {
                    gate = buf_qubit[qubit][layer];
                    if (gate != 0) {
                        // to ensure the wave has not been calculated
                        if wave_out[qubit][(i + layer) * PERIOD + PERIOD / 2] + wave_out[qubit][(i + layer) * PERIOD + PERIOD / 4] == 0) {
                            pulse_gen(gate, wave_out, i + layer, qubit); 
                        }
                    }
                }
            }
        */
        
        }
        else {
            // generate wave from cache first
            get_wave_from_cache_3_3(wave_out, i, pattern_id);

            // update match_matrix with TRUE
            #pragma HLS PIPELINE
            for (int m = 0; m < 3; ++m) {
                for (int n = i; n < i + 3; ++n) {
                    match_matrix[x_offset + m][n] = true;
                }
            }
         }
    }


}




void convolver_4_2(data_in buf_line_1[C_WIDTH], data_in buf_line_2[C_WIDTH], data_in buf_line_3[C_WIDTH]， data_in buf_line_4[C_WIDTH]){
    #pragma HLS INLINE off

    data_in buf_qubit[4][C_WIDTH];
    #pragma HLS PIPELINE
    for (int i = 0; i < C_WIDTH; ++i){
        #pragma HLS PIPELINE
        buf_qubit[0][i] = buf_line_1[i];
        buf_qubit[1][i] = buf_line_2[i];
        buf_qubit[2][i] = buf_line_3[i];
    }
}



void match_4(data_in buf_line_1[C_WIDTH], data_in buf_line_2[C_WIDTH], data_in buf_line_3[C_WIDTH]， data_in buf_line_4[C_WIDTH]) {
    bool match_matrix[4][C_WIDTH];
    complex wave_out[4][W_WIDTH];
    data_in buf_qubit[4][C_WIDTH];

    #pragma HLS PIPELINE
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < C_WIDTH; ++j) {
            match_matrix = false;
        }
    }

    #pragma HLS PIPELINE
    for (int i = 0; i < C_WIDTH; ++i) {
        #pragma HLS PIPELINE
        buf_qubit[0][i] = buf_line_1[i];
        buf_qubit[1][i] = buf_line_2[i];
        buf_qubit[2][i] = buf_line_3[i];
        buf_qubit[3][i] = buf_line_4[i];
    }

    #pragma HLS PIPELINE
    for (int i = 0; i < 2; ++i) {
        convolver_3_3(buf_qubit, wave_out, match_matrix, i);
    }

    convolver_4_2(buf_line_1, buf_line_2, buf_line_3, buf_line_4, wave_out, match_matrix);

    #pragma HLS PIPELINE
    for (int i = 0; i < 2; ++i) {
        convolver_3_2(buf_line_1, buf_line_2, buf_line_3, wave_out, match_matrix, i);
    }

    #pragma HLS PIPELINE
    for (int i = 0; i < 3; ++i) {
        convolver_2_4(buf_line_1, buf_line_2, wave_out, match_matrix, i);
    }
}