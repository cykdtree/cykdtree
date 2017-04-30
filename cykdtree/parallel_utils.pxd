cimport numpy as np
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool
from libc.stdint cimport uint32_t, uint64_t, int64_t, int32_t
from mpi4py.libmpi cimport MPI_Comm

cdef extern from "c_parallel_utils.hpp":
    uint64_t parallel_distribute(double **pts, uint64_t **idx,
                                 uint32_t ndim, uint64_t npts)
    uint64_t parallel_distribute(double **pts, uint64_t **idx,
                                 uint32_t ndim, uint64_t npts,
                                 MPI_Comm comm)
    double parallel_pivot_value(double *pts, uint64_t *idx,
                                uint32_t ndim, uint32_t d,
                                int64_t l, int64_t r)
    double parallel_pivot_value(double *pts, uint64_t *idx,
                                uint32_t ndim, uint32_t d,
                                int64_t l, int64_t r,
                                MPI_Comm comm)
    int64_t parallel_select(double *pts, uint64_t *idx,
                            uint32_t ndim, uint32_t d,
                            int64_t l, int64_t r, int64_t n,
                            double &pivot_val)
    int64_t parallel_select(double *pts, uint64_t *idx,
                            uint32_t ndim, uint32_t d,
                            int64_t l, int64_t r, int64_t n,
                            double &pivot_val,
                            MPI_Comm comm)
