cimport cython 

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
cpdef double mrr(int[:] l, float[:] p, int[:] g, int n_groups):
    cdef int index = 0
    cdef int idx_one = 0
    cdef int gr = 0
    cdef int ggr = 0
    cdef int i = 0
    cdef int times_worse = 1
    cdef float time = 0.0
    cdef float mrr = 0.0
    cdef float rr = 0.0
    cdef float our_guess = 0.0
    cdef int[:] lgr
    cdef int[:] pgr

    for gr in range(n_groups):
        idx_one = 0
        ggr = g[gr]
        for i in range(ggr):
            if l[index+i]==1:
                idx_one = i+index
        our_guess = p[idx_one]
        times_worse = 0
        for i in range(ggr):
            if p[index+i] >= our_guess:
                times_worse += 1
        rr = 1.0/times_worse
        time = gr+1.0
        mrr = ((time-1)/(time))*mrr + (1/time)*rr
        index += ggr
        
    return mrr
