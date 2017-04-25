cimport numpy as np
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool
from libc.stdint cimport uint32_t, uint64_t, int64_t, int32_t

from cykdtree.kdtree cimport Node, KDTree
from cykdtree.kdtree cimport PyNode, PyKDTree

cdef extern from "c_parallel_kdtree.hpp":
    cdef cppclass ParallelKDTree nogil:
        uint32_t ndim
        uint64_t inter_npts
        uint64_t local_npts
        uint32_t total_num_leaves
        uint64_t *all_idx
        double *all_pts
        double *total_domain_left_edge
        double *total_domain_right_edge
        double *total_domain_width
        bool *total_periodic
        double *local_domain_left_edge
        double *local_domain_right_edge
        double *local_domain_width
        bool *local_periodic_left
        bool *local_periodic_right
        KDTree *tree
        ParallelKDTree(double *pts, uint64_t *idx, uint64_t n, uint32_t m,
                       uint32_t leafsize, double *left_edge0,
                       double *right_edge0, bool *periodic0, bool include_self)
        ParallelKDTree(double *pts, uint64_t *idx, uint64_t n, uint32_t m,
                       uint32_t leafsize, double *left_edge0,
                       double *right_edge0, bool *periodic0)
        vector[uint32_t] get_neighbor_ids(double* pos)
        Node* search(double* pos)
        KDTree* consolidate_tree()
        void consolidate_edges(double *leaves_le, double *leaves_re)

cdef class PyParallelKDTree:
    cdef int rank
    cdef int size
    cdef ParallelKDTree *_ptree
    cdef readonly uint64_t npts
    cdef readonly uint32_t ndim
    cdef readonly uint32_t num_leaves
    cdef readonly uint32_t total_num_leaves
    cdef readonly uint32_t local_num_leaves
    cdef readonly uint32_t leafsize
    cdef double *_left_edge
    cdef double *_right_edge
    cdef bool *_periodic
    cdef readonly object leaves
    cdef readonly object _idx
    cdef void _make_tree(self, double *pts)
    cdef object _get_neighbor_ids(self, np.ndarray[double, ndim=1] pos)
    # cdef np.ndarray[np.uint32_t, ndim=1] _get_neighbor_ids_3(self, np.float64_t pos[3])
    cdef object _get(self, np.ndarray[double, ndim=1] pos)
    cdef object _consolidate(self)
