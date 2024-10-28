#include <ap_int.h>
#include <iostream>

#define QUBITS 10
#define WIDTH 16

typedef ap_int<128> data_in;
typedef ap_int<16> data_buf;

data_in temp;
data_in* input;
data_buf buf_input[QUBITS][8*WIDTH];

for (int k = 0; k < QUBITS; ++k){
    for (int i = 0; i < WIDTH; ++i){
    temp=*((data_in *)input + k * WIDTH + i);
        buf_input[k][i*8]  =temp(15,0);
        buf_input[k][i*8+1]=temp(31,16);
        buf_input[k][i*8+2]=temp(47,32);
        buf_input[k][i*8+3]=temp(63,48);
        buf_input[k][i*8+4]=temp(79,64);
        buf_input[k][i*8+5]=temp(95,80);
        buf_input[k][i*8+6]=temp(111,96);
        buf_input[k][i*8+7]=temp(127,112);
    }
}
