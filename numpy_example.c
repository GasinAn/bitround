#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_math.h"

static PyMethodDef BitroundMethods[] = {
        {NULL, NULL, 0, NULL}
};

/* The loop definition must precede the PyMODINIT_FUNC. */

static void npy_double_bitround(char **args, npy_intp *dimensions,
                                npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0], *in2 = args[1];
    char *out1 = args[2];
    npy_intp in1_step = steps[0], in2_step = steps[1];
    npy_intp out1_step = steps[2];

    npy_double tmp;

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/
        tmp = *(npy_double *)in1;
        tmp *= *(npy_double *)in2;
        *((npy_double *)out1) = log(tmp/(1-tmp));
        /*END main ufunc computation*/

        in1 += in1_step;
        in2 += in2_step;
        out1 += out1_step;
    }
}

/*This a pointer to the above function*/

PyUFuncGenericFunction funcs_bitround[1] = {&npy_double_bitround};

/* These are the input and return dtypes of logit.*/

static char types_bitround[3] = {NPY_FLOAT64, NPY_FLOAT64,
                                 NPY_FLOAT64};

static void *data_bitround[1] = {NULL};

/* The loop definition must precede the PyMODINIT_FUNC. */

static void npy_double_test_bitround(char **args, npy_intp *dimensions,
                                     npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0], *in2 = args[1];
    char *out1 = args[2];
    npy_intp in1_step = steps[0], in2_step = steps[1];
    npy_intp out1_step = steps[2];

    npy_double tmp;

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/
        tmp = *(npy_double *)in1;
        tmp *= *(npy_double *)in2;
        *((npy_double *)out1) = log(tmp/(1-tmp));
        /*END main ufunc computation*/

        in1 += in1_step;
        in2 += in2_step;
        out1 += out1_step;
    }
}

/*This a pointer to the above function*/

PyUFuncGenericFunction funcs_test_bitround[1] = {&npy_double_test_bitround};

/* These are the input and return dtypes of logit.*/

static char types_test_bitround[3] = {NPY_FLOAT64, NPY_FLOAT64,
                                      NPY_FLOAT64};

static void *data_test_bitround[1] = {NULL};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "bitround",
    NULL,
    -1,
    BitroundMethods,
    NULL,
    NULL,
    NULL,
    NULL
};
PyMODINIT_FUNC PyInit_bitround(void)
{
    PyObject *m, *bitround, *test_bitround, *d;

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
    test_bitround = PyUFunc_FromFuncAndData(
        funcs_test_bitround, data_test_bitround, types_test_bitround,
        1, 2, 1, PyUFunc_None, "test_bitround", "test_bitround_docstring", 0
    );

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "bitround", bitround);
    Py_DECREF(bitround);
    PyDict_SetItemString(d, "test_bitround", test_bitround);
    Py_DECREF(test_bitround);

    return m;
}
#else
PyMODINIT_FUNC initbitround(void)
{
    PyObject *m, *bitround, *test_bitround, *d;

    m = Py_InitModule("bitround", BitroundMethods);
    if (m == NULL) {
        return;
    }

    import_array();
    import_umath();

    bitround = PyUFunc_FromFuncAndData(
        funcs_bitround, data_bitround, types_bitround,
        1, 2, 1, PyUFunc_None, "bitround", "bitround_docstring", 0
    );
    test_bitround = PyUFunc_FromFuncAndData(
        funcs_test_bitround, data_test_bitround, types_test_bitround,
        1, 2, 1, PyUFunc_None, "test_bitround", "test_bitround_docstring", 0
    );

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "bitround", bitround);
    Py_DECREF(bitround);
    PyDict_SetItemString(d, "test_bitround", test_bitround);
    Py_DECREF(test_bitround);
}
#endif
