import numpy as np
import time
from nose.tools import istest, nottest, assert_raises, assert_equal
from mpi4py import MPI
from cykdtree.tests import MPITest
from cykdtree import utils
from cykdtree import parallel_utils

Nproc = 4

@MPITest(Nproc, ndim=(2,3))
def test_parallel_distribute(ndim=2):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    npts = 50
    if rank == 0:
        pts = np.random.rand(npts, ndim).astype('float64')
    else:
        pts = None
    total_pts = comm.bcast(pts, root=0)
    local_pts, local_idx = parallel_utils.py_parallel_distribute(pts)
    npts_local = npts/size
    if rank < (npts%size):
        npts_local += 1
    assert_equal(local_pts.shape, (npts_local, ndim))
    assert_equal(local_idx.shape, (npts_local, ))
    np.testing.assert_array_equal(total_pts[local_idx], local_pts)


@MPITest(Nproc, ndim=(2,3))
def test_parallel_pivot_value(ndim=2):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    npts = 50
    if rank == 0:
        pts = np.random.rand(npts, ndim).astype('float64')
    else:
        pts = None
    total_pts = comm.bcast(pts, root=0)
    local_pts, local_idx = parallel_utils.py_parallel_distribute(pts)
    pivot_dim = ndim-1

    piv = parallel_utils.py_parallel_pivot_value(local_pts, pivot_dim)

    nmax = (7*npts/10 + 6)
    assert(np.sum(total_pts[:, pivot_dim] < piv) <= nmax)
    assert(np.sum(total_pts[:, pivot_dim] > piv) <= nmax)

    # if rank == 0:
    #     pp, idx = utils.py_pivot(total_pts, pivot_dim)
    #     np.testing.assert_approx_equal(piv, total_pts[idx[pp], pivot_dim])


@MPITest(Nproc, ndim=(2,3))
def test_parallel_select(ndim=2):
    total_npts = 50
    pivot_dim = ndim-1
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        total_pts = np.random.rand(total_npts, ndim).astype('float64')
    else:
        total_pts = None
    pts, orig_idx = parallel_utils.py_parallel_distribute(total_pts)
    npts = pts.shape[0]

    p = int(total_npts)/2 + int(total_npts)%2
    q, piv, idx = parallel_utils.py_parallel_select(pts, pivot_dim, p)
    assert_equal(idx.size, npts)

    total_pts = comm.bcast(total_pts, root=0)
    if npts == 0:
        assert_equal(q, -1)
    else:
        med = np.median(total_pts[:, pivot_dim])
        if (total_npts%2):
            np.testing.assert_approx_equal(piv, med)
        else:
            np.testing.assert_array_less(piv, med)
        np.testing.assert_array_less(pts[idx[:q], pivot_dim], med)
        np.testing.assert_array_less(med, pts[idx[(q+1):], pivot_dim])
        assert(pts[idx[q], pivot_dim] <= med)


@MPITest(Nproc, ndim=(2,3))
def test_parallel_split(ndim=2):
    total_npts = 50
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        total_pts = np.random.rand(total_npts, ndim).astype('float64')
    else:
        total_pts = None
    pts, orig_idx = parallel_utils.py_parallel_distribute(total_pts)
    npts = pts.shape[0]

    p = int(total_npts)/2 + int(total_npts)%2
    q, pivot_dim, piv, idx = parallel_utils.py_parallel_split(pts, p)
    assert_equal(idx.size, npts)

    total_pts = comm.bcast(total_pts, root=0)
    if npts == 0:
        assert_equal(q, -1)
    else:
        med = np.median(total_pts[:, pivot_dim])
        if (total_npts%2):
            np.testing.assert_approx_equal(piv, med)
        else:
            np.testing.assert_array_less(piv, med)
        np.testing.assert_array_less(pts[idx[:q], pivot_dim], med)
        np.testing.assert_array_less(med, pts[idx[(q+1):], pivot_dim])
        assert(pts[idx[q], pivot_dim] <= med)

    if rank == 0:
        sq, sd, sidx = utils.py_split(total_pts)
        assert_equal(pivot_dim, sd)
        assert_equal(piv, total_pts[sidx[sq], sd])


@MPITest(Nproc, ndim=(2,3))
def test_redistribute_split(ndim=2):
    total_npts = 50
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        total_pts = np.random.rand(total_npts, ndim).astype('float64')
    else:
        total_pts = None
    pts, orig_idx = parallel_utils.py_parallel_distribute(total_pts)
    npts = pts.shape[0]
    total_pts = comm.bcast(total_pts, root=0)

    new_pts, new_idx, sidx, sdim, sval = parallel_utils.py_redistribute_split(
        pts, orig_idx)

    assert_equal(new_pts.shape[0], new_idx.size)
    assert_equal(new_pts.shape[1], ndim)

    np.testing.assert_array_equal(new_pts, total_pts[new_idx, :])

    if rank < size/2:
        assert((new_pts[:, sdim] <= sval).all())
    else:
        np.testing.assert_array_less(sval, new_pts[:, sdim])

    med = np.median(total_pts[:, sdim])
    if (total_npts%2):
        np.testing.assert_approx_equal(sval, med)
    else:
        np.testing.assert_array_less(sval, med)

