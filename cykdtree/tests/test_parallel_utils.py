import numpy as np
import time
from nose.tools import istest, nottest, assert_raises, assert_equal
from mpi4py import MPI
from cykdtree.tests import MPITest, assert_less_equal
from cykdtree import utils
from cykdtree import parallel_utils

Nproc = (3,4,5)

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


@MPITest(Nproc, ndim=(2,3), npts=(10, 11, 50, 51))
def test_parallel_pivot_value(ndim=2, npts=50):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
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

    # Not equivalent because each processes does not have multiple of 5 points
    # if rank == 0:
    #     pp, idx = utils.py_pivot(total_pts, pivot_dim)
    #     np.testing.assert_approx_equal(piv, total_pts[idx[pp], pivot_dim])


@MPITest(Nproc, ndim=(2,3), npts=(10, 11, 50, 51))
def test_parallel_select(ndim=2, npts=50):
    total_npts = npts
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
        if q >= 0:
            assert_less_equal(pts[idx[:(q+1)], pivot_dim], piv)
            np.testing.assert_array_less(piv, pts[idx[(q+1):], pivot_dim])


@MPITest(Nproc, ndim=(2,3), npts=(10, 11, 50, 51))
def test_parallel_split(ndim=2, npts=50):
    total_npts = npts
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
        if q >= 0:
            assert_less_equal(pts[idx[:(q+1)], pivot_dim], piv)
            np.testing.assert_array_less(piv, pts[idx[(q+1):], pivot_dim])

    if rank == 0:
        sq, sd, sidx = utils.py_split(total_pts)
        assert_equal(pivot_dim, sd)
        assert_equal(piv, total_pts[sidx[sq], sd])


@MPITest(Nproc, ndim=(2,3), npts=(10, 11, 50, 51), split_left=(None, False, True))
def test_redistribute_split(ndim=2, npts=50, split_left=None):
    total_npts = npts
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if split_left is None:
        split_rank = -1
    else:
        split_rank = size/2
        if split_left:
            split_rank += size%2
    if rank == 0:
        total_pts = np.random.rand(total_npts, ndim).astype('float64')
    else:
        total_pts = None
    pts, orig_idx = parallel_utils.py_parallel_distribute(total_pts)
    npts = pts.shape[0]
    total_pts = comm.bcast(total_pts, root=0)

    new_pts, new_idx, sidx, sdim, sval = parallel_utils.py_redistribute_split(
        pts, orig_idx, split_rank=split_rank)
    # Assume split_left is default for split_rank == -1
    if split_rank < 0:
        split_rank = size/2 + size%2

    assert_equal(new_pts.shape[0], new_idx.size)
    assert_equal(new_pts.shape[1], ndim)

    np.testing.assert_array_equal(new_pts, total_pts[new_idx, :])

    if rank < split_rank:
        assert_less_equal(new_pts[:, sdim], sval)
    else:
        np.testing.assert_array_less(sval, new_pts[:, sdim])

    med = np.median(total_pts[:, sdim])
    if (total_npts%2):
        np.testing.assert_approx_equal(sval, med)
    else:
        np.testing.assert_array_less(sval, med)


@MPITest(Nproc, ndim=(2,3), npts=(10, 11, 50, 51))
def test_redistribute_split_errors(ndim=2, npts=50):
    total_npts = npts
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        total_pts = np.random.rand(total_npts, ndim).astype('float64')
    else:
        total_pts = None
    pts, orig_idx = parallel_utils.py_parallel_distribute(total_pts)
    assert_raises(ValueError, parallel_utils.py_redistribute_split,
                  pts, orig_idx, split_rank=size)


def test_calc_split_rank():
    # Default split (currently left)
    assert_equal(parallel_utils.py_calc_split_rank(4), 2)
    assert_equal(parallel_utils.py_calc_split_rank(5), 3)
    # Left split
    assert_equal(parallel_utils.py_calc_split_rank(4, split_left=True), 2)
    assert_equal(parallel_utils.py_calc_split_rank(5, split_left=True), 3)
    # Right split
    assert_equal(parallel_utils.py_calc_split_rank(4, split_left=False), 2)
    assert_equal(parallel_utils.py_calc_split_rank(5, split_left=False), 2)


@MPITest(Nproc)
def test_calc_rounds():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # Get answers
    ans_nrounds = int(np.ceil(np.log2(size)))
    ans_src_round = 0
    curr_rank = rank
    curr_size = size
    while curr_rank != 0:
        split_rank = parallel_utils.py_calc_split_rank(curr_size)
        if curr_rank < split_rank:
            curr_size = split_rank
            curr_rank = curr_rank
        else:
            curr_size = curr_size - split_rank
            curr_rank = curr_rank - split_rank
        ans_src_round += 1
    # Test
    nrounds, src_round = parallel_utils.py_calc_rounds()
    assert_equal(nrounds, ans_nrounds)
    assert_equal(src_round, ans_src_round)


@MPITest(Nproc, ndim=(2,3), npts=(10, 11, 50, 51))
def test_kdtree_parallel_distribute(ndim=2, npts=50):
    total_npts = npts
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        total_pts = np.random.rand(total_npts, ndim).astype('float64')
    else:
        total_pts = None
    pts, idx, le, re, ple, pre = parallel_utils.py_kdtree_parallel_distribute(total_pts)
    total_pts = comm.bcast(total_pts, root=0)
    assert_equal(pts.shape[0], idx.size)
    np.testing.assert_array_equal(pts, total_pts[idx,:])
    for d in range(ndim):
        assert_less_equal(pts[:,d], re[d])
        assert_less_equal(le[d], pts[:,d])
