#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>
#include <Python.h>

struct linterp
{
    int dim;
    float fill[2];
    float *x;
    float *y;
};

void linterp_log(struct linterp *interp, float * __restrict x, float * __restrict y, int npts)
{
    float lbound = logf(interp->x[0]), ubound = logf(interp->x[interp->dim - 1]);
    float range = ubound - lbound;
    for(int i = 0; i < npts; i++)
    {
        float xval = x[i];
        float lxval = logf(xval);
        float guess = (interp->dim * ((lxval - lbound) / range));
        int j = (int)guess;
        if (j <= 0)
        {
            y[i] = interp->fill[0];
            continue;
        }
        if (j >= (interp->dim - 1))
        {
            y[i] = interp->fill[1];
            continue;
        }

        for(;;)
        {
            j++;
            if (interp->x[j] >= xval) break;
        }
        const float slope = (interp->y[j] - interp->y[j-1]) / (interp->x[j] - interp->x[j-1]);
        y[i] = interp->y[j-1] + (xval - interp->x[j-1]) * slope;
    }
}

static PyObject*
np_log_interp(PyObject* self, PyObject* args)
{
    // Inputs
    PyArrayObject *xs, *ys, *pts;

    // Read the arguments from Python into the C variables.
    if (!PyArg_ParseTuple(args, "OOO", &pts, &xs, &ys)) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    if (!xs || !ys || !pts) {
        PyErr_SetString(PyExc_RuntimeError, "Could not parse array inputs.");
        Py_INCREF(Py_None);
        return Py_None;
    }

    // We can operate only on one-dimensional arrays (sequences).
    int xs_dim = PyArray_NDIM(xs);
    int ys_dim = PyArray_NDIM(ys);
    int pts_dim = PyArray_NDIM(pts);
    if(xs_dim != 1 || ys_dim != 1 || pts_dim != 1) {
        PyErr_SetString(PyExc_RuntimeError, "All input arrays must be one-dimensonal.");
        Py_INCREF(Py_None);
        return Py_None;
    }
    if(PyArray_DIM(xs, 0) != PyArray_DIM(ys, 0)) {
        PyErr_SetString(PyExc_RuntimeError, "xs.shape[0] must equal ys.shape[0]");
        Py_INCREF(Py_None);
        return Py_None;
    }

    int xdim = PyArray_DIM(xs, 0);
    int resdim = PyArray_DIM(pts, 0);
    npy_intp dims[1];
    dims[0] = resdim;
    PyObject *results = PyArray_SimpleNew(1, dims, NPY_FLOAT);
    if (results == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Could not create result array.");
        Py_DECREF(results);
        Py_INCREF(Py_None);
        return Py_None;
    }

    struct linterp interp_struct;
    float *pts_data = (float *)PyArray_DATA(pts);
    float *res_data = (float *)PyArray_DATA(results);
    interp_struct.dim = xdim;
    interp_struct.x = (float *)PyArray_DATA(xs);
    interp_struct.y = (float *)PyArray_DATA(ys);
    interp_struct.fill[0] = interp_struct.y[0];
    interp_struct.fill[1] = interp_struct.y[xdim - 1];
    linterp_log(&interp_struct, pts_data, res_data, resdim);
    return results;
}


static PyMethodDef _linterp_methods[] = {
    {"log_interp", np_log_interp, METH_VARARGS,
        "Interpolate log spaced data into a new set of pts"
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef _linterp_mod = 
{
    PyModuleDef_HEAD_INIT,
    "_linterp", "module docstring",
    -1,
    _linterp_methods
};

PyMODINIT_FUNC
PyInit__linterp(void) 
{
    import_array();
    return PyModule_Create(&_linterp_mod);
}
