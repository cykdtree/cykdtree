import numpy as np
import time
from nose.tools import istest, nottest, assert_raises, assert_equal
from mpi4py import MPI
import cykdtree
from cykdtree.tests import MPITest
from cykdtree.tests.test_kdtree import left_neighbors_x, left_neighbors_y, left_neighbors_x_periodic, left_neighbors_y_periodic
np.random.seed(100)
Nproc = 4

N = 100
leafsize = 10
pts2 = np.random.rand(N, 2).astype('float64')
left_edge2 = np.zeros(2, 'float64')
right_edge2 = np.ones(2, 'float64')
pts3 = np.random.rand(N, 3).astype('float64')
left_edge3 = np.zeros(3, 'float64')
right_edge3 = np.ones(3, 'float64')
rand_state = np.random.get_state()


def fake_input(ndim, N=100, leafsize=10):
    ndim = int(ndim)
    N = int(N)
    leafsize=int(leafsize)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    np.random.seed(100)
    if rank == 0:
        if ndim == -2:
            np.random.set_state(rand_state)
            pts = np.random.rand(50, 2).astype('float64')
            left_edge = left_edge2
            right_edge = right_edge2
        else:
            pts = np.random.rand(N, ndim).astype('float64')
            left_edge = np.zeros(ndim, 'float64')
            right_edge = np.ones(ndim, 'float64')
    else:
        pts = None
        left_edge = None
        right_edge = None
    return pts, left_edge, right_edge, leafsize


@MPITest(Nproc, periodic=(False, True), ndim=(2,3))
def test_PyParallelKDTree(periodic=False, ndim=2):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    pts, le, re, ls = fake_input(ndim, N=20, leafsize=3)
    Tpara = cykdtree.PyParallelKDTree(pts, le, re, leafsize=ls,
                                      periodic=periodic)


@MPITest(Nproc, periodic=(False, True), ndim=(2,3))
def test_PyParallelKDTree_errors(periodic=False, ndim=2):
    pts, le, re, ls = fake_input(ndim, N=20, leafsize=3)
    assert_raises(ValueError, cykdtree.PyParallelKDTree, pts,
                  le, re, leafsize=1)


@MPITest(Nproc, periodic=(False, True), ndim=(2,3))
def test_consolidate(periodic=False, ndim=2):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    pts, le, re, ls = fake_input(ndim, N=20, leafsize=3)
    Tpara0 = cykdtree.PyParallelKDTree(pts, le, re, leafsize=ls,
                                       periodic=periodic)
    Tpara = Tpara0.consolidate()
    if rank == 0:
        if False:
            from cykdtree.plot import plot2D_serial
            plot2D_serial(Tpara, label_boxes=True,
                          plotfile='test_consolidate.png')
        Tseri = cykdtree.PyKDTree(pts, le, re, leafsize=ls,
                                  periodic=periodic)
        np.testing.assert_array_equal(Tpara.idx, Tseri.idx)
        Tpara.assert_equal(Tseri)
    else:
        assert(Tpara is None)


@MPITest(Nproc, periodic=(False, True), ndim=(2,3))
def test_search(periodic=False, ndim=2):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    pts, le, re, ls = fake_input(ndim)
    tree = cykdtree.PyParallelKDTree(pts, le, re, leafsize=ls,
                                     periodic=periodic)
    if periodic:
        vals = [0, 0.5, 1.0]
    else:
        vals = [0, 0.5, 0.9]
    for v in vals:
        pos = v*np.ones(ndim, 'double')
        out = tree.get(pos)
        if out is not None:
            out.neighbors


@MPITest(Nproc, periodic=(False, True), ndim=(2,3))
def test_search_errors(periodic=False, ndim=2):
    pts, le, re, ls = fake_input(ndim)
    tree = cykdtree.PyParallelKDTree(pts, le, re, leafsize=ls,
                                     periodic=periodic)
    if not periodic:
        assert_raises(ValueError, tree.get, np.ones(ndim, 'double'))
    assert_raises(AssertionError, tree.get, np.zeros(ndim+1, 'double'))


@MPITest(Nproc, periodic=(False, True))
def test_neighbors(periodic = True):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    pts, le, re, ls = fake_input(-2, leafsize=10)
    tree = cykdtree.PyParallelKDTree(pts, le, re, leafsize=ls,
                                     periodic=periodic)
    if False:
        from cykdtree.plot import plot2D_parallel
        plot2D_parallel(tree, label_boxes=True, label_procs=True,
                        plotfile='test_neighbors.png')
    # Prepare neighbors
    if periodic:
        left_neighbors = [left_neighbors_x_periodic, left_neighbors_y_periodic]
        right_neighbors = [[[] for i in range(tree.total_num_leaves)] for
                           _ in range(tree.ndim)]
    else:
        left_neighbors = [left_neighbors_x, left_neighbors_y]
        right_neighbors = [[[] for i in range(tree.total_num_leaves)] for _
                           in range(tree.ndim)]
    for d in range(tree.ndim):
        for i in range(tree.total_num_leaves):
            for j in left_neighbors[d][i]:
                right_neighbors[d][j].append(i)
        for i in range(tree.total_num_leaves):
            right_neighbors[d][i] = list(set(right_neighbors[d][i]))
    # Check left/right neighbors
    for leaf in tree.leaves.values():
        out_str = str(leaf.id)
        try:
            for d in range(tree.ndim):
                out_str += '\nleft:  {} {} {}'.format(d, leaf.left_neighbors[d],
                                               left_neighbors[d][leaf.id])
                out_str += ' {}'.format(leaf.periodic_left)
                assert(len(left_neighbors[d][leaf.id]) ==
                       len(leaf.left_neighbors[d]))
                for i in range(len(leaf.left_neighbors[d])):
                    assert(left_neighbors[d][leaf.id][i] ==
                           leaf.left_neighbors[d][i])
                out_str += '\nright: {} {} {}'.format(d, leaf.right_neighbors[d],
                                                right_neighbors[d][leaf.id])
                out_str += ' {}'.format(leaf.periodic_right)
                assert(len(right_neighbors[d][leaf.id]) ==
                       len(leaf.right_neighbors[d]))
                for i in range(len(leaf.right_neighbors[d])):
                    assert(right_neighbors[d][leaf.id][i] ==
                           leaf.right_neighbors[d][i])
        except:
            time.sleep(rank)
            print(out_str)
            raise


@MPITest(Nproc, periodic=(False, True))
def test_get_neighbor_ids(periodic=False, ndim=2):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    pts, le, re, ls = fake_input(ndim)
    tree = cykdtree.PyParallelKDTree(pts, le, re, leafsize=ls,
                                     periodic=periodic)
    if periodic:
        vals = [0, 0.5, 1.0]
    else:
        vals = [0, 0.5, 0.9]
    for v in vals:
        pos = v*np.ones(ndim, 'float')
        tree.get_neighbor_ids(pos)


@MPITest(Nproc, periodic=(False, True), ndim=(2,3))
def test_consolidate_edges(periodic=False, ndim=2):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    pts, le, re, ls = fake_input(ndim, N=20, leafsize=3)
    Tpara = cykdtree.PyParallelKDTree(pts, le, re, leafsize=ls,
                                      periodic=periodic)
    LEpara, REpara = Tpara.consolidate_edges()
    if rank == 0:
        Tseri = cykdtree.PyKDTree(pts, le, re, leafsize=ls,
                                  periodic=periodic)
        LEseri, REseri = Tseri.consolidate_edges()
    else:
        LEseri, REseri = None, None
    LEseri, REseri = comm.bcast((LEseri, REseri), root=0)
    np.testing.assert_allclose(LEpara, LEseri)
    np.testing.assert_allclose(REpara, REseri)


@MPITest(Nproc, periodic=(False, True), ndim=(2,3))
def test_consolidate_process_bounds(periodic=False, ndim=2):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    pts, le, re, ls = fake_input(ndim, N=20, leafsize=3)
    Tpara = cykdtree.PyParallelKDTree(pts, le, re, leafsize=ls,
                                      periodic=periodic)
    LEpara, REpara = Tpara.consolidate_process_bounds()
    assert_equal(LEpara.shape, (size, ndim))
    assert_equal(REpara.shape, (size, ndim))
    np.testing.assert_allclose(LEpara[rank,:], Tpara.left_edge)
    np.testing.assert_allclose(REpara[rank,:], Tpara.right_edge)


def time_tree_construction(Ntime, LStime, ndim=2):
    pts, le, re, ls = fake_input(ndim, N=Ntime, leafsize=LStime)
    t0 = time.time()
    cykdtree.PyParallelKDTree(pts, le, re, leafsize=LStime)
    t1 = time.time()
    print("{} {}D points, leafsize {}: took {} s".format(Ntime, ndim, LStime, t1-t0))


def time_neighbor_search(Ntime, LStime, ndim=2):
    pts, le, re, ls = fake_input(ndim, N=Ntime, leafsize=LStime)
    tree = cykdtree.PyParallelKDTree(pts, le, re, leafsize=LStime)
    t0 = time.time()
    tree.get_neighbor_ids(0.5*np.ones(tree.ndim, 'double'))
    t1 = time.time()
    print("{} {}D points, leafsize {}: took {} s".format(Ntime, ndim, LStime, t1-t0))

