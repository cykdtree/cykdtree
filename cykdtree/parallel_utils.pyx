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


def py_parallel_distribute(np.ndarray[np.float64_t, ndim=2] pts0 = None):
    r"""Split points between all processes in the world.

    Args:
        pts0 (np.ndarray of np.float64): Array of points that should be split
            among the processes. This should only be passed to process 0.

    Returns:
        tuple(np.ndarray of float64, np.ndarray of uint64): The positions and
            original indices of those positions assigned to this processes.

    """
    cdef object comm = MPI.COMM_WORLD
    cdef int size = comm.Get_size()
    cdef int rank = comm.Get_rank()
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
    nout = parallel_distribute(&ptr_pts, &ptr_idx, ndim, npts)
    if nout > 0:
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
        if ptr_pts != NULL:
            free(ptr_pts)
        if ptr_idx != NULL:
            free(ptr_idx)
    return (pts, idx)


def py_parallel_pivot_value(np.ndarray[np.float64_t, ndim=2] pts,
                            np.uint32_t pivot_dim):
    r"""Determine the pivot using median of medians across a pool of processes
    along a specified dimension.

    Args:
        pts (np.ndarray of float64): Positions on this process.
        pivot_dim (uint32): Dimension that median of medians should be performed
            along.

    Returns:
        float64: Median of medians across pool of processes.
    
    """
    cdef object comm = MPI.COMM_WORLD
    cdef int size = comm.Get_size()
    cdef int rank = comm.Get_rank()
    cdef uint64_t npts = pts.shape[0]
    cdef uint32_t ndim = pts.shape[1]
    cdef int64_t l = 0
    cdef int64_t r = npts-1
    cdef int i
    # Get pivot
    cdef np.float64_t pivot
    cdef uint64_t[:] idx = np.arange(npts).astype('uint64')
    cdef double *ptr_pts = NULL
    cdef uint64_t *ptr_idx = NULL
    if npts != 0:
        ptr_pts = &pts[0,0]
        ptr_idx = &idx[0]
    pivot = parallel_pivot_value(ptr_pts, ptr_idx,
                                 ndim, pivot_dim, l, r);
    return pivot


def py_parallel_select(np.ndarray[np.float64_t, ndim=2] pts,
                       np.uint32_t pivot_dim, np.int64_t t):
    r"""Get the indices required to partition coordiantes such that the first
    q elements in pos[:,d] on each process cummulativly contain the smallest
    t elements in pos[:,d] across all processes. 

    Args:
        pts (np.ndarray of float64): Positions on this process.
        pivot_dim (uint32): Dimension that median of medians should be performed
            along.
        t (int64): Number of smallest elements in positions across all
            processes that should be partitioned.

    Returns:
        tuple(int64, float64, np.ndarray of uint64): Max index (q)  of points
            on this process that fall in the smallest t points overall, the
            value of element t (whether its on this process or not), and the
            index required to order the points to put the smallest ones first.

    """
    cdef object comm = MPI.COMM_WORLD
    cdef int size = comm.Get_size()
    cdef int rank = comm.Get_rank()
    cdef uint64_t npts = pts.shape[0]
    cdef uint32_t ndim = pts.shape[1]
    cdef int64_t l = 0
    cdef int64_t r = npts-1
    cdef int i
    cdef double pivot_val = 0.0
    # Get pivot
    cdef uint64_t[:] idx = np.arange(npts).astype('uint64')
    cdef double *ptr_pts = NULL
    cdef uint64_t *ptr_idx = NULL
    if npts != 0:
        ptr_pts = &pts[0,0]
        ptr_idx = &idx[0]
    cdef int64_t q = parallel_select(ptr_pts, ptr_idx,
                                     ndim, pivot_dim, l, r, t, pivot_val);
    return q, pivot_val, idx


def py_parallel_split(np.ndarray[np.float64_t, ndim=2] pts, np.int64_t t,
                      np.ndarray[np.float64_t, ndim=1] mins = None,
                      np.ndarray[np.float64_t, ndim=1] maxs = None):
    r"""Get the indices required to partition coordinates such that the first
    q elements in pos on each process cummulativly contain the smallest
    t elements in pos along the largest dimension.

    Args:
        pts (np.ndarray of float64): Positions on this process.
        t (int64): Number of smallest elements in positions across all
            processes that should be partitioned.
        mins (np.ndarray of float64, optional): (m,) array of mins for this
            process. Defaults to None and is set to mins of pos along each
            dimension.
        maxs (np.ndarray of float64, optional): (m,) array of maxs for this
            process. Defaults to None and is set to maxs of pos along each
            dimension.

    Returns:
        tuple(int64, float64, np.ndarray of uint64): Max index (q)  of points
            on this process that fall in the smallest t points overall, the
            dimension the split was performed over, the value of element t
            (whether its on this process or not), and the index required to
            order the points to put the smallest ones first.

    """
    cdef object comm = MPI.COMM_WORLD
    cdef int size = comm.Get_size()
    cdef int rank = comm.Get_rank()
    cdef uint64_t npts = pts.shape[0]
    cdef uint32_t ndim = pts.shape[1]
    cdef uint64_t Lidx = 0
    cdef int i
    # Get pivot
    cdef uint64_t[:] idx = np.arange(npts).astype('uint64')
    cdef double *ptr_pts = NULL
    cdef uint64_t *ptr_idx = NULL
    cdef double *ptr_mins = NULL
    cdef double *ptr_maxs = NULL
    if (npts != 0) and (ndim != 0):
        if mins is None:
            mins = np.min(pts, axis=0)
        if maxs is None:
            maxs = np.max(pts, axis=0)
        ptr_pts = &pts[0,0]
        ptr_idx = &idx[0]
        ptr_mins = &mins[0]
        ptr_maxs = &maxs[0]
    cdef int64_t q = 0
    cdef double split_val = 0.0
    cdef uint32_t dsplit = parallel_split(ptr_pts, ptr_idx, Lidx, npts, ndim,
                                          ptr_mins, ptr_maxs, q, split_val)
    return q, dsplit, split_val, idx


