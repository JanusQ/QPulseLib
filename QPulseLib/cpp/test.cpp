#include <iostream>
#include "pulse_gen.cpp"
// #include "complex.cpp"

int main() {
    float t_list[120];

    for (int i = 0; i < 120; i++) {
        t_list[i] = (float)i;
    } 

    float res[120];

    // env_gaussian(t_list, res, LEN_XY, piLen / 2, piFWHM, piAmp, dphase);
    z_construction(t_list, res);


    for (int i = 0; i < 120; i++) {
        std::cout << res[i] << " " << res[i] << std::endl;
    }
}