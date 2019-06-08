from libc.math cimport sqrt
import numpy as np
cimport numpy as np

def cosine_similarity(double[:] x, double[:] y, int shrink):
    cdef double xx=0.0
    cdef double yy=0.0
    cdef double xy=0.0
    cdef Py_ssize_t i
    for i in range(len(x)):
        xx+=x[i]*x[i]
        yy+=y[i]*y[i]
        xy+=x[i]*y[i]
    if xx != 0.0 and yy !=0.0:
      return 1-xy/(sqrt(xx*yy)+shrink)
    else:
      return 0
