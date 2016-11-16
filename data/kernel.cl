
__kernel void foobar(__global float *a, __global float *b, const int n) {
    unsigned int i = get_global_id(0);
    if (i < n)
        b[i] += a[i] * 2; 
}

