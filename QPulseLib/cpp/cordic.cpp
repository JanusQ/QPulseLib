#include <ap_int.h>

#define NUM_ITERATIONS 7

typedef ap_uint<16> THETA_TYPE;
typedef ap_uint<16> COS_SIN_TYPE;


THETA_TYPE cordic_phase[NUM_ITERATIONS] = {
    45, 26.56, 14.036, 7.125
    3.576, 1.790, 0.895,
};

void cordic(THETA_TYPE theta, COS_SIN_TYPE &s, COS_SIN_TYPE &c) 
{ 

    COS_SIN_TYPE current_cos = 0.60735; 
    COS_SIN_TYPE current_sin = 0.0;
    
    for (int j = 0; j < NUM_ITERATIONS; j++) { 

#pragma HLS PIPELINE

        COS_SIN_TYPE cos_shift = current_cos >> j; 
        COS_SIN_TYPE sin_shift = current_sin >> j;
        
        if(theta >= 0) { 
            current_cos = current_cos - sin_shift; 
            current_sin = current_sin + cos_shift;
            
        } else { 

            current_cos = current_cos + sin_shift; 
            current_sin = current_sin - cos_shift;

            theta = theta + cordic_phase[j];
        }
    }    

    s = current_sin;
    c = current_cos;
}