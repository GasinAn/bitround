#include <math.h>
#include <stdio.h>

#define HEX_00080s 2251799813685248LL
#define HEX_7FF00s 9218868437227405312LL
#define HEX_80000s -9223372036854775808LL
#define HEX_FFF00s -4503599627370496LL

typedef long long int64;
typedef unsigned long long uint64;
typedef double float64;

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

int test_const(){
    uint64 c;
    c = 0x0008000000000000;
    printf("%f\n", (float64) *((int64*) &c));
    c = 0x7FF0000000000000;
    printf("%f\n", (float64) *((int64*) &c));
    c = 0x8000000000000000;
    printf("%f\n", (float64) *((int64*) &c));
    c = 0xFFF0000000000000;
    printf("%f\n", (float64) *((int64*) &c));
    return 0;
}

int test(){
    test_const();
    printf("\n");
    float64 r_list[4] = {1.875, 1.125, -1.875, -1.125};
    float64 d_list[7] = {0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0};
    for (int i=0; i<4; i++)
    {
        for (int j=0; j<7; j++)
        {
            printf("%f\n", bitround(r_list[i], d_list[j]));
        }
        printf("\n");
    }
    return 0;
}

int main(){
    test();
    return 0;
}
