#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <math.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"


/*** npy_float64_bitwise_round ***/

static const npy_uint64 UINT64_1 = 1u;
static const npy_int64 INT64_3FE0_0 = ((UINT64_1 << 9) - 1u) << 53;
static const npy_int64 INT64_7FF0_0 = ((UINT64_1 << 11) - 1u) << 52;
static const npy_uint64 UINT64_7FFF_F = (UINT64_1 << 63) - 1u;

static inline void npy_float64_bitwise_round(char* a, char* d){
    npy_int64 n = (*((npy_int64*) d) & INT64_7FF0_0) - INT64_3FE0_0;
    *((npy_uint64*) a) -= n;
    *((npy_float64*) a) = roundevenf64(*((npy_float64*) a));
    *((npy_uint64*) a) += n * ((*((npy_uint64*) a) & UINT64_7FFF_F) != 0u);
}


/*** Make ufunc bitwise_round. ***/

/* The loop definition must precede the PyMODINIT_FUNC. */

static const char bitwise_round_docstring[] =
"float64 ndarray bitwise_round(float64 ndarray a, float64 ndarray d)\r\n"
"\r\n"
"Modifies `a` to ``sign(a) * round(abs(a)/2**n) * 2**n``, where ``n ==\r\n"
"floor(log2(d)) + 1``.\r\n"
"\r\n"
"Warning\r\n"
"\r\n"
"At this moment, function `bitwise_round` works only for float64s.\r\n"
"\r\n"
"The behavior of function `bitwise_round` is UNDEFINED when one of the\r\n"
"following situations holds\r\n"
"\r\n"
" * `d` is not a positive normal number;\r\n"
" * `a` is a subnormal number;\r\n"
" * ``floor(log2(a)) - floor(log2(d)) <= 1022``.\r\n";

static void u_npy_float64_bitwise_round(char **args,
                                        const npy_intp *dimensions,
                                        const npy_intp *steps,
                                        void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0], *in2 = args[1];
    npy_intp in1_step = steps[0], in2_step = steps[1];

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/
        npy_float64_bitwise_round(in1, in2);
        /*END main ufunc computation*/

        in1 += in1_step;
        in2 += in2_step;
    }
}

/* This gives a pointer to the above function. */

PyUFuncGenericFunction funcs_bitwise_round[1] = {&u_npy_float64_bitwise_round};

/* These are the input dtypes of bitwise_round. */

static char types_bitwise_round[2] = {NPY_FLOAT64, NPY_FLOAT64};


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
    PyObject *m, *bitwise_round, *d;

    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;

    import_array();
    import_umath();

    bitwise_round = PyUFunc_FromFuncAndData(
        funcs_bitwise_round, NULL, types_bitwise_round, 1, 2, 0, PyUFunc_None,
        "bitwise_round", bitwise_round_docstring, 0
    );

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "bitwise_round", bitwise_round);
    Py_DECREF(bitwise_round);

    return m;
}
