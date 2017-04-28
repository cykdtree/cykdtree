import numpy as np
import time
from nose.tools import istest, nottest, assert_raises, assert_equal
from mpi4py import MPI
from cykdtree.tests import MPITest
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
