#include <math.h>
#include <stdio.h>

typedef long long int64;
typedef unsigned long long uint64;
typedef double float64;

const int64 HEX_00080s = 1LL << 51;
const int64 HEX_7FF00s = ((1LL << 11) - 1) << 52;
const int64 HEX_80000s = 1LL << 63;
const int64 HEX_FFF00s = ((1LL << 12) - 1) << 52;

float64 bitround(float64 r, float64 d){
    uint64* p_r = (uint64*) &r;
    uint64* p_d = (uint64*) &d;

    int64 E_r = *p_r & HEX_7FF00s;
    int64 E_d = *p_d & HEX_7FF00s;
    int64 dE = (E_r - E_d) >> 52;

    uint64 output = \
        (dE > -1) * ((*p_r + (HEX_00080s >> dE)) & (HEX_FFF00s >> dE))\
                  + \
        (dE == -1) * ((*p_r & HEX_80000s) | E_d);

    return *((float64*) &output);
}

float64 test_bitround(float64 r, float64 d){
    if (isnan(r) || isinf(r) || (r == 0.0)){return r;}
    uint64 p = *((uint64*) &d) & HEX_FFF00s;
    float64 sgn_r = ((r > 0) - 0.5) * 2;
    float64 abs_r = r / sgn_r;
    for (int i=0; 1; i++){
        if (((i + 1) * *((float64*) &p)) > (abs_r + *((float64*) &p) / 2)){
            return sgn_r * i * *((float64*) &p);
        }
    }
}

int test(){
    float64 r_list[4] = {1.875, 1.125, -1.875, -1.125};
    float64 d_list[7] = {0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0};
    for (int i=0; i<4; i++)
    {
        for (int j=0; j<7; j++)
        {
            float64 output = bitround(r_list[i], d_list[j]);
            float64 correct_output = test_bitround(r_list[i], d_list[j]);
            printf("%d", (output == correct_output));
        }
        printf("\n");
    }
    return 0;
}

int main(){
    test();
    return 0;
}
