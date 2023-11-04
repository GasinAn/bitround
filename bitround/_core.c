#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"


/*** npy_float64_bitround ***/

const npy_int64 ONE = 1;
const npy_int64 HEX_00100s = ONE << 52;
const npy_int64 HEX_7FF00s = ((ONE << 11) - 1) << 52;
const npy_int64 HEX_80000s = ONE << (64 - 1);
const npy_int64 HEX_FFE00s = ((ONE << 11) - 1) << (52 + 1);
const npy_int64 HEX_FFF00s = ((ONE << (1 + 11)) - 1) << 52;

inline npy_float64 npy_float64_bitround(npy_float64 a, npy_float64 d){
    npy_uint64* p_a = (npy_uint64*) &a;
    npy_uint64* p_d = (npy_uint64*) &d;

    npy_int64 E_a = *p_a & HEX_7FF00s;
    npy_int64 E_d = *p_d & HEX_7FF00s;
    npy_int64 dE = (E_a - E_d) >> 52;

    npy_uint64 output;
    output = (dE > 0) * ((*p_a + (HEX_00100s >> dE)) & (HEX_FFE00s >> dE))
           + (dE == 0) * ((*p_a + HEX_00100s) & HEX_FFF00s)
           + (dE < 0) * (*p_a & HEX_80000s);
    output = ((E_a != HEX_7FF00s) && (dE < 64))  * output
           + ((E_a == HEX_7FF00s) || (dE >= 64)) * *p_a;

    return *((npy_float64*) &output);
}


/*** Make ufunc bitround. ***/

/* The loop definition must precede the PyMODINIT_FUNC. */

static void u_npy_float64_bitround(char **args,
                                   const npy_intp *dimensions,
                                   const npy_intp *steps,
                                   void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0], *in2 = args[1];
    char *out1 = args[2];
    npy_intp in1_step = steps[0], in2_step = steps[1];
    npy_intp out1_step = steps[2];

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/
        *((npy_float64 *)out1) = npy_float64_bitround(
            *(npy_float64 *)in1, *(npy_float64 *)in2
        );
        /*END main ufunc computation*/

        in1 += in1_step;
        in2 += in2_step;
        out1 += out1_step;
    }
}

/* This a pointer to the above function. */

PyUFuncGenericFunction funcs_bitround[1] = {&u_npy_float64_bitround};

/* These are the input and output dtypes of bitround. */

static char types_bitround[3] = {NPY_FLOAT64, NPY_FLOAT64,
                                 NPY_FLOAT64};

static void *data_bitround[1] = {NULL};


/*** Init module bitround._core. ***/

static PyMethodDef _CoreMethods[] = {
        {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_core",
    "_core_docstring",
    -1,
    _CoreMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit__core(void)
{
    PyObject *m, *bitround, *d;

    m = PyModule_Create(&moduledef);
    if (m == NULL) {
        return NULL;
    }

    import_array();
    import_umath();

    bitround = PyUFunc_FromFuncAndData(
        funcs_bitround, data_bitround, types_bitround,
        1, 2, 1, PyUFunc_None, "bitround", "bitround_docstring", 0
    );

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "bitround", bitround);
    Py_DECREF(bitround);

    return m;
}
