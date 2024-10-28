typedef struct {
    float real;
    float image;
} complex;

complex complex_add(complex a, complex b) {
    complex res;
    res.real = a.real + b.real;
    res.image = a.image + b.image;
    return res;
}

complex complex_sub(complex a, complex b) {
    complex res;
    res.real = a.real - b.real;
    res.image = a.image - b.image;
    return res;
}

complex complex_mul(complex a, complex b) {
    complex res;
    res.real = a.real * b.real - a.image * b.image;
    res.image = a.real * b.image + a.image * b.real;
    return res;
}

complex complex_div(complex a, complex b) {
    complex res;
    float abs_square =  a.real * a.real + a.image  * a.image;
    res.real = (a.real * b.real + a.image * b.image) / abs_square;
    res.image = (a.image * b.real - a.real * b.image) / abs_square;
    return res;
}

complex complex_times(complex a, float b) {
    complex res;
    res.real = b * a.real;
    res.image = b * a.image;
    return res;
}

complex transform_complex(float a) {
    complex res;
    res.real = a;
    res.image = 0;
    return res;
}
