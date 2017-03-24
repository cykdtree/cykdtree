cimport numpy as np
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool
from libc.stdint cimport uint32_t, uint64_t, int64_t, int32_t

from cykdtree.kdtree cimport Node as Node
from cykdtree.kdtree cimport PyNode

cdef extern from "c_parallel_kdtree.hpp":
    cdef cppclass ParallelKDTree nogil:
        uint64_t npts
        uint32_t ndim
        uint32_t leafsize
        double* domain_left_edge
        double* domain_right_edge
        double* domain_width
        bool* periodic
        vector[Node*] leaves
        ParallelKDTree(double *pts, uint64_t *idx, uint64_t n, uint32_t m,
                       uint32_t leafsize, double *left_edge0,
                       double *right_edge0, bool *periodic0, bool include_self)
        ParallelKDTree(double *pts, uint64_t *idx, uint64_t n, uint32_t m,
                       uint32_t leafsize, double *left_edge0,
                       double *right_edge0, bool *periodic0)
        void build(bool include_self)
        void build()

cdef class PyParallelKDTree:
    cdef int rank
    cdef int size
    cdef ParallelKDTree *_tree
    cdef readonly uint64_t npts
    cdef readonly uint32_t ndim
    cdef readonly uint32_t num_leaves
    cdef readonly uint32_t leafsize
    cdef double *_left_edge
    cdef double *_right_edge
    cdef bool *_periodic
    cdef readonly object leaves
    cdef readonly object idx
    cdef void _make_tree(self, double *pts)
    # cdef np.ndarray[np.uint32_t, ndim=1] _get_neighbor_ids(self, np.ndarray[double, ndim=1] pos)
    # cdef np.ndarray[np.uint32_t, ndim=1] _get_neighbor_ids_3(self, np.float64_t pos[3])
    # cdef kdtree.PyNode _get(self, np.ndarray[double, ndim=1] pos)
