cimport numpy as np
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool
from libc.stdint cimport uint32_t, uint64_t, int64_t, int32_t

cdef extern from "c_parallel_kdtree.hpp":
    cdef cppclass ParallelKDTree nogil:
        ParallelKDTree(double *pts, uint64_t *idx, uint64_t n, uint32_t m,
                       uint32_t leafsize, double *left_edge0,
                       double *right_edge0, bool *periodic0, bool include_self)
        ParallelKDTree(double *pts, uint64_t *idx, uint64_t n, uint32_t m,
                       uint32_t leafsize, double *left_edge0,
                       double *right_edge0, bool *periodic0)
        void build(bool include_self)
        void build()
