#include <stdio.h>
#include <iostream>
#include "bandwidth.h"

using namespace std;

//data_in input[64][64];
//data_in buf[64][64];

data_in input1[16] = {0};
data_in input2[16] = {0};
data_in input3[16] = {0};
data_in input4[16] = {0};


int main(){
    for (int i = 0; i < 16; ++i){
            input1[i] = 1;
    }

    for (int i = 0; i < 16; ++i){
            input2[i] = 1;
    }

    for (int i = 0; i < 16; ++i){
            input3[i] = 1;
    }

    for (int i = 0; i < 16; ++i){
            input4[i] = 1;
    }

    measure_bandwidth((data_in *) input1, (data_in *) input2, (data_in *) input3, (data_in *) input4);

    printf("measure done");

    return 0;
}
