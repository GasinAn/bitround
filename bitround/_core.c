#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <math.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"


/*** npy_float64_bitround ***/

inline npy_float64 npy_float64_bitround(npy_float64 a, npy_float64 d){
    npy_float64 exp2_n = exp2f64(ceilf64(log2f64(d)));
    return ((a > 0) - (a < 0)) * roundf64(fabsf64(a) / exp2_n) * exp2_n;
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
    npy_intp in1_step = steps[0], in2_step = steps[1];

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/
        *((npy_float64 *)in1) = npy_float64_bitround(
            *((npy_float64 *)in1), *((npy_float64 *)in2)
        );
        /*END main ufunc computation*/

        in1 += in1_step;
        in2 += in2_step;
    }
}

/* This gives a pointer to the above function. */

PyUFuncGenericFunction funcs_bitround[1] = {&u_npy_float64_bitround};

/* These are the input and output dtypes of bitround. */

static char types_bitround[2] = {NPY_FLOAT64, NPY_FLOAT64};


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
    if (m == NULL)
        return NULL;

    import_array();
    import_umath();

    bitround = PyUFunc_FromFuncAndData(
        funcs_bitround, NULL, types_bitround, 1, 2, 0, PyUFunc_None,
        "bitround", "bitround_docstring", 0
    );

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "bitround", bitround);
    Py_DECREF(bitround);

    return m;
}
