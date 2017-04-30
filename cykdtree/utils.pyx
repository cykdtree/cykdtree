import numpy as np
cimport numpy as np
cimport cython
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool
from libc.stdint cimport uint32_t, uint64_t, int64_t, int32_t

import copy

import scipy
from scipy.sparse import csr_matrix

def py_max_pts(np.ndarray[np.float64_t, ndim=2] pos):
    r"""Get the maximum of points along each coordinate. 

    Args: 
        pos (np.ndarray of float64): (n,m) array of n m-D coordinates. 

    Returns: 
        np.ndarray of float64: Maximum of pos along each coordinate. 

    """
    cdef uint64_t n = <uint64_t>pos.shape[0]
    cdef uint32_t m = <uint32_t>pos.shape[1]
    cdef np.float64_t* cout = max_pts(&pos[0,0], n, m)
    cdef uint32_t i = 0
    cdef np.ndarray[np.float64_t] out = np.zeros(m, 'float64')
    for i in range(m):
        out[i] = cout[i]
    return out

def py_min_pts(np.ndarray[np.float64_t, ndim=2] pos):
    r"""Get the minimum of points along each coordinate. 

    Args: 
        pos (np.ndarray of float64): (n,m) array of n m-D coordinates. 

    Returns: 
        np.ndarray of float64: Minimum of pos along each coordinate. 

    """
    cdef uint64_t n = <uint64_t>pos.shape[0]
    cdef uint32_t m = <uint32_t>pos.shape[1]
    cdef np.float64_t* cout = min_pts(&pos[0,0], n, m)
    cdef uint32_t i = 0
    cdef np.ndarray[np.float64_t] out = np.zeros(m, 'float64')
    for i in range(m):
        out[i] = cout[i]
    return out

def py_quickSort(np.ndarray[np.float64_t, ndim=2] pos, np.uint32_t d):
    r"""Get the indices required to sort coordinates along one dimension.

    Args:
        pos (np.ndarray of float64): (n,m) array of n m-D coordinates. 
        d (np.uint32_t): Dimension that pos should be sorted along.

    Returns:
        np.ndarray of uint64: Indices that sort pos along dimension d.

    """
    cdef uint32_t ndim = pos.shape[1]
    cdef int64_t l = 0
    cdef int64_t r = pos.shape[0]-1
    cdef uint64_t[:] idx
    idx = np.arange(pos.shape[0]).astype('uint64')
    cdef double *ptr_pos = NULL
    cdef uint64_t *ptr_idx = NULL
    if pos.shape[0] != 0:
        ptr_pos = &pos[0,0]
        ptr_idx = &idx[0]
    quickSort(ptr_pos, ptr_idx, ndim, d, l, r)
    return idx

def py_insertSort(np.ndarray[np.float64_t, ndim=2] pos, np.uint32_t d):
    r"""Get the indices required to sort coordinates along one dimension.

    Args:
        pos (np.ndarray of float64): (n,m) array of n m-D coordinates. 
        d (np.uint32_t): Dimension that pos should be sorted along.

    Returns:
        np.ndarray of uint64: Indices that sort pos along dimension d.

    """
    cdef uint32_t ndim = pos.shape[1]
    cdef int64_t l = 0
    cdef int64_t r = pos.shape[0]-1
    cdef uint64_t[:] idx
    idx = np.arange(pos.shape[0]).astype('uint64')
    cdef double *ptr_pos = NULL
    cdef uint64_t *ptr_idx = NULL
    if pos.shape[0] != 0:
        ptr_pos = &pos[0,0]
        ptr_idx = &idx[0]
    insertSort(ptr_pos, ptr_idx, ndim, d, l, r)
    return idx

def py_pivot(np.ndarray[np.float64_t, ndim=2] pos, np.uint32_t d):
    r"""Get the index of the median of medians along one dimension and indices 
    that partition pos according to the median of medians.

    Args:
        pos (np.ndarray of float64): (n,m) array of n m-D coordinates. 
        d (np.uint32_t): Dimension that pos should be partitioned along.

    Returns:
        tuple of int64 and np.ndarray of uint64: Index q of idx that is the 
            pivot. All elements of idx before the pivot will be less than
            the pivot. If there is an odd number of points, the pivot will
            be the median.

    """
    cdef uint32_t ndim = pos.shape[1]
    cdef int64_t l = 0
    cdef int64_t r = pos.shape[0]-1
    cdef uint64_t[:] idx
    idx = np.arange(pos.shape[0]).astype('uint64')
    cdef double *ptr_pos = NULL
    cdef uint64_t *ptr_idx = NULL
    if pos.shape[0] != 0:
        ptr_pos = &pos[0,0]
        ptr_idx = &idx[0]
    cdef int64_t q = pivot(ptr_pos, ptr_idx, ndim, d, l, r)
    return q, idx

def py_partition(np.ndarray[np.float64_t, ndim=2] pos, np.uint32_t d,
                 np.int64_t p):
    r"""Get the indices required to partition coordinates along one dimension.

    Args:
        pos (np.ndarray of float64): (n,m) array of n m-D coordinates. 
        d (np.uint32_t): Dimension that pos should be partitioned along.
        p (np.int64_t): Element of pos[:,d] that should be used as the pivot
            to partition pos.

    Returns:
        tuple of int64 and np.ndarray of uint64: Location of the pivot in the
            partitioned array and the indices required to partition the array
            such that elements before the pivot are smaller and elements after
            the pivot are larger.

    """
    cdef uint32_t ndim = pos.shape[1]
    cdef int64_t l = 0
    cdef int64_t r = pos.shape[0]-1
    cdef uint64_t[:] idx
    idx = np.arange(pos.shape[0]).astype('uint64')
    cdef double *ptr_pos = NULL
    cdef uint64_t *ptr_idx = NULL
    if pos.shape[0] != 0:
        ptr_pos = &pos[0,0]
        ptr_idx = &idx[0]
    cdef int64_t q = partition(ptr_pos, ptr_idx, ndim, d, l, r, p)
    return q, idx

def py_partition_given_pivot(np.ndarray[np.float64_t, ndim=2] pos,
                             np.uint32_t d, np.float64_t pval):
    r"""Get the indices required to partition coordinates along one dimension.

    Args:
        pos (np.ndarray of float64): (n,m) array of n m-D coordinates. 
        d (np.uint32_t): Dimension that pos should be partitioned along.
        pval (np.float64_t): Value that should be used to partition pos.

    Returns:
        tuple of int64 and np.ndarray of uint64: Location of the largest value
            that is smaller than pval in partitioned array and the indices 
            required to partition the array such that elements before the pivot
            are smaller and elements after the pivot are larger.

    """
    cdef uint32_t ndim = pos.shape[1]
    cdef int64_t l = 0
    cdef int64_t r = pos.shape[0]-1
    cdef uint64_t[:] idx
    idx = np.arange(pos.shape[0]).astype('uint64')
    cdef double *ptr_pos = NULL
    cdef uint64_t *ptr_idx = NULL
    if pos.shape[0] != 0:
        ptr_pos = &pos[0,0]
        ptr_idx = &idx[0]
    cdef int64_t q = partition_given_pivot(ptr_pos, ptr_idx, ndim, d, l, r,
                                           pval)
    return q, idx

def py_select(np.ndarray[np.float64_t, ndim=2] pos, np.uint32_t d,
              np.int64_t t):
    r"""Get the indices required to partition coordiantes such that the first 
    t elements in pos[:,d] are the smallest t elements in pos[:,d].

    Args:
        pos (np.ndarray of float64): (n,m) array of n m-D coordinates.
        d (np.uint32_t): Dimension that pos should be partitioned along. 
        t (np.int64_t): Number of smallest elements in pos[:,d] that should be 
            partitioned.

    Returns:
        tuple of int64 and np.ndarray of uint64: Location of element t in the
            partitioned array and the indices required to partition the array
            such that elements before element t are smaller and elements after
            the pivot are larger.

    """
    cdef uint32_t ndim = pos.shape[1]
    cdef int64_t l = 0
    cdef int64_t r = pos.shape[0]-1
    cdef uint64_t[:] idx
    idx = np.arange(pos.shape[0]).astype('uint64')
    cdef double *ptr_pos = NULL
    cdef uint64_t *ptr_idx = NULL
    if pos.shape[0] != 0:
        ptr_pos = &pos[0,0]
        ptr_idx = &idx[0]
    cdef int64_t q = select(ptr_pos, ptr_idx, ndim, d, l, r, t)
    return q, idx


def py_split(np.ndarray[np.float64_t, ndim=2] pos, 
             np.ndarray[np.float64_t, ndim=1] mins = None,
             np.ndarray[np.float64_t, ndim=1] maxs = None):
    r"""Get the indices required to split the positions equally along the
    largest dimension.

    Args:
        pos (np.ndarray of float64): (n,m) array of n m-D coordinates. 
        mins (np.ndarray of float64, optional): (m,) array of mins. Defaults
            to None and is set to mins of pos along each dimension.
        maxs (np.ndarray of float64, optional): (m,) array of maxs. Defaults
            to None and is set to maxs of pos along each dimension.

    Returns:
        tuple(int64, uint32, np.ndarray of uint64): The index of the split in
            the partitioned array, the dimension of the split, and the indices
            required to partition the array.

    """
    cdef uint64_t npts = pos.shape[0]
    cdef uint32_t ndim = pos.shape[1]
    cdef uint64_t Lidx = 0
    cdef uint64_t[:] idx
    idx = np.arange(pos.shape[0]).astype('uint64')
    cdef double *ptr_pos = NULL
    cdef uint64_t *ptr_idx = NULL
    cdef double *ptr_mins = NULL
    cdef double *ptr_maxs = NULL
    if (npts != 0) and (ndim != 0):
        if mins is None:
            mins = np.min(pos, axis=0)
        if maxs is None:
            maxs = np.max(pos, axis=0)
        ptr_pos = &pos[0,0]
        ptr_idx = &idx[0]
        ptr_mins = &mins[0]
        ptr_maxs = &maxs[0]
    cdef int64_t q = 0
    cdef double split_val = 0.0
    cdef uint32_t dsplit = split(ptr_pos, ptr_idx, Lidx, npts, ndim,
                                 ptr_mins, ptr_maxs, q, split_val)
    return q, dsplit, idx
