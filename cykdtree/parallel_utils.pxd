cimport numpy as np
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool
from libc.stdint cimport uint32_t, uint64_t, int64_t, int32_t

cdef extern from "c_parallel_utils.hpp":
    uint64_t parallel_distribute(double **pts, uint64_t **idx,
                                 uint32_t ndim, uint64_t npts)
    double parallel_pivot_value(vector[int] pool,
                                double *pts, uint64_t *idx,
                                uint32_t ndim, uint32_t d,
                                int64_t l, int64_t r)
    int64_t parallel_select(vector[int] pool,
                            double *pts, uint64_t *idx,
                            uint32_t ndim, uint32_t d,
                            int64_t l, int64_t r, int64_t n)
