#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_math.h"


/*** npy_double_bitround ***/

const npy_int64 HEX_00080s = 1LL << 51;
const npy_int64 HEX_7FF00s = ((1LL << 11) - 1) << 52;
const npy_int64 HEX_80000s = 1LL << 63;
const npy_int64 HEX_FFF00s = ((1LL << 12) - 1) << 52;

npy_double npy_double_bitround(npy_double r, npy_double d){
    npy_uint64* p_r = (npy_uint64*) &r;
    npy_uint64* p_d = (npy_uint64*) &d;

    npy_int64 E_r = *p_r & HEX_7FF00s;
    npy_int64 E_d = *p_d & HEX_7FF00s;
    npy_int64 dE = (E_r - E_d) >> 52;

    npy_uint64 output;
    output = \
        (dE > -1) * ((*p_r + (HEX_00080s >> dE)) & (HEX_FFF00s >> dE))\
        + \
        (dE <= -1) * ((*p_r & HEX_80000s) | ((dE == -1) * E_d));
    output = \
        (E_r != HEX_7FF00s) * output\
        + \
        (E_r == HEX_7FF00s) * *p_r;

    return *((npy_double*) &output);
}


/*** Make ufunc bitround. ***/

/* The loop definition must precede the PyMODINIT_FUNC. */

static void u_npy_double_bitround(char **args, npy_intp *dimensions,
                                  npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0], *in2 = args[1];
    char *out1 = args[2];
    npy_intp in1_step = steps[0], in2_step = steps[1];
    npy_intp out1_step = steps[2];

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/
        *((npy_double *)out1) = npy_double_bitround(
            *(npy_double *)in1, *(npy_double *)in2
        );
        /*END main ufunc computation*/

        in1 += in1_step;
        in2 += in2_step;
        out1 += out1_step;
    }
}

/* This a pointer to the above function. */

PyUFuncGenericFunction funcs_bitround[1] = {&u_npy_double_bitround};

/* These are the input and output dtypes of bitround. */

static char types_bitround[3] = {NPY_FLOAT64, NPY_FLOAT64,
                                 NPY_FLOAT64};

static void *data_bitround[1] = {NULL};


/*** Init module bitround.core. ***/

static PyMethodDef CoreMethods[] = {
        {NULL, NULL, 0, NULL}
};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "core",
    NULL,
    -1,
    CoreMethods,
    NULL,
    NULL,
    NULL,
    NULL
};
PyMODINIT_FUNC PyInit_core(void)
{
    PyObject *m, *bitround, *d;

    m = PyModule_Create(&moduledef);
    if (!m) {
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
#else
PyMODINIT_FUNC initcore(void)
{
    PyObject *m, *bitround, *d;

    m = Py_InitModule("core", CoreMethods);
    if (m == NULL) {
        return;
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
}
#endif
