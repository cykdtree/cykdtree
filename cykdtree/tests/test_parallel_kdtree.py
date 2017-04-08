# mpirun -n 4 python -c 'from cykdtree.tests.test_parallel_kdtree import *; test_neighbors()'
# mpirun -n 4 python -c 'from cykdtree.tests.test_parallel_kdtree import *; test_neighbors()'
import numpy as np
import time
# from nose.tools import assert_equal
from nose.tools import istest, nottest, assert_raises
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
        pts = np.random.rand(N, ndim).astype('float64')
        left_edge = np.zeros(ndim, 'float64')
        right_edge = np.ones(ndim, 'float64')
    else:
        pts = None
        left_edge = None
        right_edge = None
    return pts, left_edge, right_edge, leafsize


@MPITest(Nproc)
def test_PyParallelKDTree():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    for periodic in (False, True):
        for ndim in (2, 3):
            pts, le, re, ls = fake_input(ndim, N=20, leafsize=3)
            Tpara = cykdtree.PyParallelKDTree(pts, le, re, leafsize=ls,
                                              periodic=periodic)
            # time.sleep(rank*1)
            # print(rank, [i for i in Tpara.all_idx])
            # for leaf in Tpara.leaves.values():
            #     print (leaf.id, #leaf.left_edge, leaf.right_edge, 
            #            Tpara.all_pts[Tpara.idx[leaf.slice],:])
            if rank == 0:
                # time.sleep(size*1)
                Tseri = cykdtree.PyKDTree(pts, le, re, leafsize=ls,
                                          periodic=periodic)
                # print(rank, 'result', [i for i in Tseri.idx])
                # for leaf in Tseri.leaves:
                #     print (leaf.id, #leaf.left_edge, leaf.right_edge, 
                #            pts[Tseri.idx[leaf.slice],:])
                np.testing.assert_array_equal(Tpara.idx, Tseri.idx)
            assert_raises(ValueError, cykdtree.PyParallelKDTree, pts,
                          le, re, leafsize=1)


@MPITest(Nproc)
def test_search():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    for periodic in (False,):
        for ndim in (2, 3):
            pts, le, re, ls = fake_input(ndim)
            tree = cykdtree.PyParallelKDTree(pts, le, re, leafsize=ls,
                                             periodic=periodic)
            for v in [0, 0.5, 0.9]:
                pos = v*np.ones(ndim, 'double')
                out = tree.get(pos)
            assert_raises(ValueError, tree.get, np.ones(ndim, 'double'))
            assert_raises(AssertionError, tree.get, np.zeros(ndim+1, 'double'))


@MPITest(Nproc)
def test_search_periodic():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    for periodic in (True,):
        for ndim in (2, 3):
            pts, le, re, ls = fake_input(ndim)
            tree = cykdtree.PyParallelKDTree(pts, le, re, leafsize=ls,
                                             periodic=periodic)
            for v in [0, 0.5, 1.0]:
                pos = v*np.ones(ndim, 'double')
                out = tree.get(pos)
                if out is not None:
                    out.neighbors
            assert_raises(AssertionError, tree.get, np.zeros(ndim+1, 'double'))


@MPITest(Nproc)
def test_neighbors():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    if rank == 0:
        np.random.set_state(rand_state)
        pts = np.random.rand(50, 2).astype('float64')
    else:
        pts = None
    tree = cykdtree.PyParallelKDTree(pts, left_edge2, right_edge2, leafsize=10)

    # from cykdtree.plot import plot2D_parallel
    # plot2D_parallel(tree, label_boxes=True, label_procs=True,
    #                 plotfile='test_neighbors.png')

    # 2D
    left_neighbors = [left_neighbors_x, left_neighbors_y]
    right_neighbors = [[[] for i in range(tree.total_num_leaves)] for _
                       in range(tree.ndim)]
    time.sleep(rank)
    for d in range(tree.ndim):
        for i in range(tree.total_num_leaves):
            for j in left_neighbors[d][i]:
                right_neighbors[d][j].append(i)
        for i in range(tree.total_num_leaves):
            right_neighbors[d][i] = list(set(right_neighbors[d][i]))
    for leaf in tree.leaves.values():
        out_str = str(leaf.id)
        try:
            for d in range(tree.ndim):
                out_str += '\nleft:  {} {} {}'.format(d, leaf.left_neighbors[d],
                                               left_neighbors[d][leaf.id])
                assert(len(left_neighbors[d][leaf.id]) ==
                       len(leaf.left_neighbors[d]))
                for i in range(len(leaf.left_neighbors[d])):
                    assert(left_neighbors[d][leaf.id][i] ==
                           leaf.left_neighbors[d][i])
                out_str += '\nright: {} {} {}'.format(d, leaf.right_neighbors[d],
                                                right_neighbors[d][leaf.id])
                assert(len(right_neighbors[d][leaf.id]) ==
                       len(leaf.right_neighbors[d]))
                for i in range(len(leaf.right_neighbors[d])):
                    assert(right_neighbors[d][leaf.id][i] ==
                           leaf.right_neighbors[d][i])
        except:
            print(out_str)
            raise


@MPITest(Nproc)
def test_neighbors_periodic():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    if rank == 0:
        np.random.set_state(rand_state)
        pts = np.random.rand(50, 2).astype('float64')
    else:
        pts = None
    tree = cykdtree.PyParallelKDTree(pts, left_edge2, right_edge2, leafsize=10,
                                     periodic=True)

    from cykdtree.plot import plot2D_parallel
    plot2D_parallel(tree, label_boxes=True, label_procs=True,
                    plotfile='test_neighbors_parallel.png')
    # 2D
    left_neighbors = [left_neighbors_x_periodic, left_neighbors_y_periodic]
    right_neighbors = [[[] for i in range(tree.total_num_leaves)] for
                       _ in range(tree.ndim)]
    for d in range(tree.ndim):
        for i in range(tree.total_num_leaves):
            for j in left_neighbors[d][i]:
                right_neighbors[d][j].append(i)
        for i in range(tree.total_num_leaves):
            right_neighbors[d][i] = list(set(right_neighbors[d][i]))
    for leaf in tree.leaves.values():
        out_str = str(leaf.id)
        try:
            for d in range(tree.ndim):
                out_str += '\nleft:  {} {} {}'.format(d, leaf.left_neighbors[d],
                                               left_neighbors[d][leaf.id])
                assert(len(left_neighbors[d][leaf.id]) ==
                       len(leaf.left_neighbors[d]))
                for i in range(len(leaf.left_neighbors[d])):
                    assert(left_neighbors[d][leaf.id][i] ==
                           leaf.left_neighbors[d][i])
                out_str += '\nright: {} {} {}'.format(d, leaf.right_neighbors[d],
                                                right_neighbors[d][leaf.id])
                assert(len(right_neighbors[d][leaf.id]) ==
                       len(leaf.right_neighbors[d]))
                for i in range(len(leaf.right_neighbors[d])):
                    assert(right_neighbors[d][leaf.id][i] ==
                           leaf.right_neighbors[d][i])
        except:
            print(out_str)
            raise

@MPITest(Nproc)
def test_get_neighbor_ids():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    for periodic in (True,):
        for ndim in (2, 3):
            pts, le, re, ls = fake_input(ndim)
            tree = cykdtree.PyParallelKDTree(pts, le, re, leafsize=ls,
                                             periodic=periodic)
            for v in [0., 0.5, 1.0]:
                pos = v*np.ones(ndim, 'float')
                tree.get_neighbor_ids(pos)


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
