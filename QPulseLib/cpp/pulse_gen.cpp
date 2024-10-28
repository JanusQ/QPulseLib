#include <iostream>
#include "complex.cpp"
#include <cmath>
// #include "cordic.cpp"
// #include <ap_int.h>

// using namespace std;

#define EXITERS 10
#define PERIOD 120
#define LEN_XY 60
#define LEN_CZ 120

typedef float PARA;


// RX, RY wave
static const float piAmp = 0.5;
static const float piLen = 60.0; // ns
static const float piFWHM = 30.0; // ns
static const float f10 = 5.0; // ghz
static const float fc = 5.1; //GHz
static const float alpha = 0.5;
static const float delta = -200; //mhz
static const float pi = 3.1415926;
static const float dphase = 0;

// CZ wave

static const float w = 2.0; // ns
static const float gateTime = 100; // ns
static float ripples[4] = {0.0, 0.1, 0.0, 0.0};

float mypow(float x, int n) {
    if (n == 0)
        return 1;
    float res = 1;
    for (int i = 0; i < n; ++i)
        res *= x;
    return res;
}

int factorial(int x) {
    if (x == 0)
        return 1;
    int res = 1;
    for (int i = 1; i <= x; ++i)
        res *= i;
    return res;
}

float my_exp(float x) {
    float res = 0;
    for (int i = 0; i < EXITERS; ++i) {
        res += mypow(x, i) / float(factorial(i));
    }
    return res;
}

void env_gaussian(const float* tlist, float * res, int len, float t0, float w, float amp = 1.0, float phase = 2.0, float df = 0.0) {
    // 这里怕是要改
    float sigma = w / float(sqrt(8 * log(2)));

    for (int i = 0; i < len; ++i) {
        float a = amp * exp(pow(tlist[i] - t0, 2) / (-2 * sigma * sigma));
        res[i] = a * cos(phase);
    }
}

void env_deriv(const float* tlist, complex res[LEN_XY], float dt = 0.1) {
    float tlist1[LEN_XY], tlist2[LEN_XY];

    for (int i = 0; i < LEN_XY; ++i) {
        tlist1[i] = tlist[i] + dt;
        tlist2[i] = tlist[i] - dt;
    }

    float res1_gau[LEN_XY], res2_gau[LEN_XY];

    env_gaussian(tlist1, res1_gau, LEN_XY, piLen / 2, piFWHM, piAmp, dphase);
    env_gaussian(tlist2, res2_gau, LEN_XY, piLen / 2, piFWHM, piAmp, dphase);

    complex res1[LEN_XY], res2[LEN_XY];

    for (int i = 0; i < LEN_XY; ++i) {
        res1[i].real = res1_gau[i];
        res2[i].real = res2_gau[i];
    }

    for (int i = 0; i < LEN_XY; ++i) {
        res[i] = complex_times(complex_sub(res1[i], res2[i]), 1 / (2 * dt));
    }
}

void rotPulseHD_xy(const float* tlist, complex res[LEN_XY]) {
    complex res_x[LEN_XY], res_y[LEN_XY];
    float xy_delta = 2 * pi * delta / 1000;
    env_deriv(tlist, res_x);
    for (int i = 0; i < LEN_XY; ++i) {
        res_y[i] = complex_times(res_x[i], -alpha / xy_delta);
    }

    complex const_1;
    const_1.real = 0;
    const_1.image = 1;

    for (int i = 0; i < LEN_XY; ++i) {
        res[i] = complex_add(res_x[i], complex_mul(res_y[i], const_1));
    }
}

void env_mix(const float* tlist, complex res[LEN_XY], float df) {
    rotPulseHD_xy(tlist, res);

    for (int i = 0; i < LEN_XY; ++i) {
        float const_1 = -2 * pi * df * tlist[i] - dphase;
        complex tmp;
        tmp.real = cos(const_1);
        tmp.image = sin(const_1);
        res[i] = complex_mul(res[i], tmp);
    }
}


// res生成X波
void xy_construction(float tlist_xy[LEN_XY], complex res[LEN_XY]) {
    float df = f10 - fc;
    env_mix(tlist_xy, res, df);
}


void env_diabaticCZ(const float* tlist, float * res, float t0, float t_len, float amp, float w, float* ripples) {
    for (int i = 0; i < 4; ++i) {
        ripples[i] = amp * ripples[i];
    }

    float t_min = t0 < t0 + t_len ? t0 : t0 + t_len;
    float t_max = t0 > t0 + t_len ? t0 : t0 + t_len;
    float t_mid = (t_min + t_max) / 2;

    float window_start = -3 * w;
    float t_step = 0.05;
    int window_size = int(6 * w / t_step);
    float window_tlist[window_size];
    // range(window_tlist, window_size, window_start, t_step);

    float window[window_size];
    for (int i = 0; i < window_size; ++i) {
        window_tlist[i] = t_step * i - 3 * w;
    }

    env_gaussian(window_tlist, window, window_size, 0.0, w, 2 * sqrt(log(2) / pi) / w, 0);

    for (int i = 0; i < LEN_CZ; ++i) {

        float current_tlist[window_size];
        for (int j = 0; j < window_size; ++j) {
            current_tlist[j] = tlist[i] - 3 * w + t_step * j;
        }

        // 此处 timeFuncRect
        float amp_r[window_size] = {0.0};
        if (t_len > 0) {
            for (int j = 0; j < 4; ++j) {
                float idx_r = pow(2, j / 2);
                if (j % 2 == 0) {
                    for (int k = 0; k < window_size; ++k) {
                        amp_r[k] += ripples[j] * sin(idx_r * pi * (current_tlist[k] - t_mid) / t_len);
                    }
                }
                else {
                    for (int k = 0; k < window_size; ++k) {
                        amp_r[k] += ripples[j] * cos(idx_r * pi * (current_tlist[k] - t_mid) / t_len);
                    }
                }
            }
        }
        float amps[window_size] = {0.0};
        for (int j = 0; j < window_size; ++j) {
            amps[j] = (amp + amp_r[j]) * (current_tlist[j] >= t_min) * (current_tlist[j] < t_max);
        }

        // 卷积
        res[i] = 0;
        for (int j = 0; j < window_size; ++j) {
            res[i] += window[j] * amps[j];
        }
    }

}

void z_construction(float tlist[LEN_CZ], float res[LEN_CZ]) {
    for (int i = 0; i < LEN_CZ; ++i) {
        tlist[i] = i;
    }
    env_diabaticCZ(tlist, res, 0, gateTime, 0.2, w, ripples);
}
