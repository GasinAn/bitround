#include <math.h>
#include <stdio.h>

#define HEX_00400000 4194304
#define HEX_7F800000 2139095040
#define HEX_80000000 -2147483648
#define HEX_FF800000 -8388608

typedef int int32;
typedef unsigned int uint32;
typedef float float32;

float32 bitround(float32 r, float32 d){
    uint32* p_r = (uint32*) &r;
    uint32* p_d = (uint32*) &d;

    int32 E_r = *p_r & HEX_7F800000;
    int32 E_d = *p_d & HEX_7F800000;
    int32 dE = (E_r - E_d) >> 23;

    uint32 output = \
        (dE > -1) * ((*p_r + (HEX_00400000 >> dE)) & (HEX_FF800000 >> dE))\
                  + \
        (dE == -1) * ((*p_r & HEX_80000000) | E_d);

    return *((float32*) &output);
}

int test_const(){
    uint32 c;
    c = 0x00400000;
    printf("%d\n", *((int32*) &c));
    c = 0x7F800000;
    printf("%d\n", *((int32*) &c));
    c = 0x80000000;
    printf("%d\n", *((int32*) &c));
    c = 0xFF800000;
    printf("%d\n", *((int32*) &c));
    return 0;
}

int main(){
    test_const();
    printf("\n");
    printf("%11.5f\n", bitround(1.875, 0.0625));
    printf("%11.5f\n", bitround(1.875, 0.125));
    printf("%11.5f\n", bitround(1.875, 0.25));
    printf("%11.5f\n", bitround(1.875, 0.5));
    printf("%11.5f\n", bitround(1.875, 1.0));
    printf("%11.5f\n", bitround(1.875, 2.0));
    printf("%11.5f\n", bitround(1.875, 4.0));
    printf("\n");
    printf("%11.5f\n", bitround(1.125, 0.0625));
    printf("%11.5f\n", bitround(1.125, 0.125));
    printf("%11.5f\n", bitround(1.125, 0.25));
    printf("%11.5f\n", bitround(1.125, 0.5));
    printf("%11.5f\n", bitround(1.125, 1.0));
    printf("%11.5f\n", bitround(1.125, 2.0));
    printf("%11.5f\n", bitround(1.125, 4.0));
    printf("\n");
    printf("%11.5f\n", bitround(-1.875, 0.0625));
    printf("%11.5f\n", bitround(-1.875, 0.125));
    printf("%11.5f\n", bitround(-1.875, 0.25));
    printf("%11.5f\n", bitround(-1.875, 0.5));
    printf("%11.5f\n", bitround(-1.875, 1.0));
    printf("%11.5f\n", bitround(-1.875, 2.0));
    printf("%11.5f\n", bitround(-1.875, 4.0));
    printf("\n");
    printf("%11.5f\n", bitround(-1.125, 0.0625));
    printf("%11.5f\n", bitround(-1.125, 0.125));
    printf("%11.5f\n", bitround(-1.125, 0.25));
    printf("%11.5f\n", bitround(-1.125, 0.5));
    printf("%11.5f\n", bitround(-1.125, 1.0));
    printf("%11.5f\n", bitround(-1.125, 2.0));
    printf("%11.5f\n", bitround(-1.125, 4.0));
    printf("\n");
    return 0;
}
