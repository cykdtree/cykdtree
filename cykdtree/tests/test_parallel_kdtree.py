# mpirun -n 4 python -c 'from cykdtree.tests.test_parallel_kdtree import *; test_neighbors()'
# mpirun -n 4 python -c 'from cykdtree.tests.test_parallel_kdtree import *; test_neighbors()'
import cykdtree
import numpy as np
import time
from cykdtree.tests.test_kdtree import left_neighbors_x, left_neighbors_y
# from nose.tools import assert_equal
from nose.tools import assert_raises
from mpi4py import MPI
np.random.seed(100)

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


def test_PyParallelKDTree():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    for periodic in (False, True):
        for ndim in (2, 3):
            pts, le, re, ls = fake_input(ndim)
            Tpara = cykdtree.PyParallelKDTree(pts, le, re, leafsize=ls,
                                              periodic=periodic)
            if rank == 0:
                Tseri = cykdtree.PyKDTree(pts, le, re, leafsize=ls,
                                          periodic=periodic)
                np.testing.assert_array_equal(Tpara.idx, Tseri.idx)
            assert_raises(ValueError, cykdtree.PyParallelKDTree, pts,
                          le, re, leafsize=1)


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


# def test_search_periodic():
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     for periodic in (True,):
#         for ndim in (2, 3):
#             pts, le, re, ls = fake_input(ndim)
#             tree = cykdtree.PyParallelKDTree(pts, le, re, leafsize=ls,
#                                              periodic=periodic)
#             for v in [0, 0.5, 1.0]:
#                 pos = v*np.ones(ndim, 'double')
#                 out = tree.get(pos)
#                 if out is not None:
#                     out.neighbors
#             assert_raises(AssertionError, tree.get, np.zeros(ndim+1, 'double'))


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
    # left_neighbors_x = [[],  # None
    #                     [0],
    #                     [1],
    #                     [2],
    #                     [],  # None
    #                     [],  # None
    #                     [4, 5],
    #                     [5]]
    # left_neighbors_y = [[],  # None
    #                     [],  # None
    #                     [],  # None
    #                     [],  # None
    #                     [0, 1],
    #                     [4],
    #                     [1, 2, 3],
    #                     [6]]
    left_neighbors = [left_neighbors_x, left_neighbors_y]
    right_neighbors = [[[] for i in range(tree.tot_num_leaves)] for _
                       in range(tree.ndim)]
    time.sleep(rank)
    for d in range(tree.ndim):
        for i in range(tree.tot_num_leaves):
            for j in left_neighbors[d][i]:
                right_neighbors[d][j].append(i)
        for i in range(tree.tot_num_leaves):
            right_neighbors[d][i] = list(set(right_neighbors[d][i]))
    for leaf in tree.leaves.values():
        print(leaf.id, leaf.left_edge, leaf.right_edge)
    for leaf in tree.leaves.values():
        print(leaf.id)
        for d in range(tree.ndim):
            print('    ', d, leaf.left_neighbors[d],
                  left_neighbors[d][leaf.id])
            assert(len(left_neighbors[d][leaf.id]) ==
                   len(leaf.left_neighbors[d]))
            for i in range(len(leaf.left_neighbors[d])):
                assert(left_neighbors[d][leaf.id][i] ==
                       leaf.left_neighbors[d][i])
            print('    ', d, leaf.right_neighbors[d],
                  right_neighbors[d][leaf.id])
            assert(len(right_neighbors[d][leaf.id]) ==
                   len(leaf.right_neighbors[d]))
            for i in range(len(leaf.right_neighbors[d])):
                assert(right_neighbors[d][leaf.id][i] ==
                       leaf.right_neighbors[d][i])



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
    # left_neighbors_x = [[3, 6, 7],
    #                     [0], 
    #                     [1],
    #                     [2],
    #                     [6], # [3, 6],  # Added corner, was [6],
    #                     [6, 7], #[3, 6, 7],  # Added corner, was [6, 7],
    #                     [4, 5],
    #                     [5]]
    # left_neighbors_y = [[5, 7],
    #                     [5, 7],
    #                     [7],
    #                     [5, 7],
    #                     [0, 1],
    #                     [4],
    #                     [1, 2, 3],
    #                     [6]]
    left_neighbors = [left_neighbors_x, left_neighbors_y]
    right_neighbors = [[[] for i in range(tree.tot_num_leaves)] for
                       _ in range(tree.ndim)]
    for d in range(tree.ndim):
        for i in range(tree.tot_num_leaves):
            for j in left_neighbors[d][i]:
                right_neighbors[d][j].append(i)
        for i in range(tree.tot_num_leaves):
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

# def test_get_neighbor_ids():
#     # 2D
#     tree2 = cykdtree.PyKDTree(pts2, left_edge2, right_edge2,
#                               leafsize=leafsize, periodic=True)
#     for pos in [left_edge2, (left_edge2+right_edge2)/2., right_edge2]:
#         tree2.get_neighbor_ids(pos)
#     # 3D
#     tree3 = cykdtree.PyKDTree(pts3, left_edge3, right_edge3,
#                               leafsize=leafsize, periodic=True)
#     for pos in [left_edge3, (left_edge3+right_edge3)/2., right_edge3]:
#         tree3.get_neighbor_ids(pos)


# def time_tree_construction(Ntime, LStime):
#     pts = np.random.rand(Ntime, 2).astype('float64')
#     t0 = time.time()
#     cykdtree.PyKDTree(pts, left_edge2, right_edge2, leafsize=LStime)
#     t1 = time.time()
#     print("{} points, leafsize {}: took {} s".format(Ntime, LStime, t1-t0))


# def time_neighbor_search(Ntime, LStime):
#     pts = np.random.rand(Ntime, 2).astype('float64')
#     tree = cykdtree.PyKDTree(pts, left_edge2, right_edge2, leafsize=LStime)
#     t0 = time.time()
#     tree.get_neighbor_ids(0.5*np.ones(tree.ndim, 'double'))
#     t1 = time.time()
#     print("{} points, leafsize {}: took {} s".format(Ntime, LStime, t1-t0))
