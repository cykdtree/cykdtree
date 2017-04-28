import numpy as np
cimport numpy as np
cimport cython
from mpi4py import MPI
from libc.stdlib cimport malloc, free
from libcpp cimport bool as cbool
from cpython cimport bool as pybool
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libc.stdint cimport uint32_t, uint64_t, int64_t, int32_t


def py_parallel_partition(np.ndarray[np.float64_t, ndim=2] pts0 = None):
    r"""Split points between all processes in the world.

    Args:
        pts0 (np.ndarray of np.float64): Array of points that should be split
            among the processes. This should only be passed to process 0.

    Returns:
        tuple(np.ndarray of float64, np.ndarray of uint64): The positions and
            original indices of those positions assigned to this processes.

    """
    cdef object comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    cdef uint64_t npts = 0
    cdef uint32_t ndim = 0
    cdef double *ptr_pts = NULL
    cdef uint64_t *ptr_idx = NULL
    cdef np.uint64_t[:] idx0
    if rank == 0:
        assert(pts0 is not None)
        npts = pts0.shape[0]
        ndim = pts0.shape[1]
        idx0 = np.arange(npts).astype('uint64')
        ptr_idx = &idx0[0]
        ptr_pts = &pts0[0,0]
    else:
        assert(pts0 is None)
    ndim = comm.bcast(ndim, root=0)
    cdef uint64_t nout;
    nout = parallel_partition(&ptr_pts, &ptr_idx, ndim, npts)
    assert(ptr_pts != NULL)
    assert(ptr_idx != NULL)
    # Memory view on pointers (memory may not be freed)
    # cdef np.float64_t[:,:] pts
    # cdef np.uint64_t[:] idx
    # pts = <np.float64_t[:nout, :ndim]> ptr_pts
    # idx = <np.uint64_t[:nout]> ptr_idx
    # Direct construction (ensures memory freed)
    cdef np.ndarray[np.float64_t, ndim=2] pts = np.empty((nout, ndim), 'float64')
    cdef np.ndarray[np.uint64_t, ndim=1] idx = np.empty((nout,), 'uint64')
    cdef uint64_t i
    cdef uint32_t d
    for i in range(nout):
        idx[i] = ptr_idx[i]
        for d in range(ndim):
            pts[i,d] = ptr_pts[i*ndim+d]
    if rank != 0:
        free(ptr_pts)
        free(ptr_idx)
    return (pts, idx)

#def py_parallel_pivot_value(
