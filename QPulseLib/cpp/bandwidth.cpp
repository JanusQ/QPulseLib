#include "bandwidth.h"

// data_in buf[64][64];


void load_input(data_in* input, data buf buf[row][col][width]){
    data_in temp;

    #pragma HLS PIPELINE
    buf[i][j][0] = temp(31,0);
    buf[i][j][1] = temp(63,32);
    buf[i][j][2] = temp(95,64);
    buf[i][j][3] = temp(127,96);
}

void measure_bandwidth(data_in* input1, data_in * input2, data_in * input3, data_in * input4){
#pragma HLS INTERFACE mode=m_axi depth=9999999 port=input1 bundle=IN offset=slave
#pragma HLS INTERFACE mode=m_axi depth=9999999 port=input2 bundle=IN offset=slave
#pragma HLS INTERFACE mode=m_axi depth=9999999 port=input3 bundle=IN offset=slave
#pragma HLS INTERFACE mode=m_axi depth=9999999 port=input4 bundle=IN offset=slave
// #pragma HLS INTERFACE mode=m_axi depth=9999999 port=output bundle=OUT offset=slave
#pragma HLS INTERFACE mode=s_axilite port=return

	data_buf buf[N_ROWS][N_COLS][4];
    // data_in temp1, temp2, temp3, temp4;
	// *output = 0;

#pragma HLS ARRAY_PARTITION variable=buf type=block factor=4 dim=2



#pragma HLS INLINE off
    for (int i = 0; i < N_ROWS; ++i){
        for (int j = 0; j < N_COLS / 4; ++j){
            #pragma HLS PIPELINE
            load_input(input1, buf[i][j][4]);
            load_input(input2, buf[i][j+1][4]);
            load_input(input3, buf[i][j+2][4]);
            load_input(input4, buf[i][j+3][4]);
        }
    }
}
