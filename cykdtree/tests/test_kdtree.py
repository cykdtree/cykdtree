import cykdtree
import numpy as np
import time
# from nose.tools import assert_equal
from nose.tools import assert_raises
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


left_neighbors_x = [[],  # None
                    [0],
                    [1],
                    [2],
                    [],  # None
                    [],  # None
                    [4, 5],
                    [5]]
left_neighbors_y = [[],  # None
                    [],  # None
                    [],  # None
                    [],  # None
                    [0, 1],
                    [4],
                    [1, 2, 3],
                    [6]]
left_neighbors_x_periodic = [[3],
                             [0],
                             [1],
                             [2],
                             [6],
                             [6, 7],
                             [4, 5],
                             [5]]
left_neighbors_y_periodic = [[5],
                             [5, 7],
                             [7],
                             [7],
                             [0, 1],
                             [4],
                             [1, 2, 3],
                             [6]]
# Add corners
# left_neighbors_x_periodic[0].append(6)
# left_neighbors_x_periodic[0].append(7)
# left_neighbors_x_periodic[4].append(3) # not fully periodic
# left_neighbors_x_periodic[5].append(3)
# left_neighbors_y_periodic[3].append(5)
# left_neighbors_y_periodic[4].append(3) # not fully periodic
# left_neighbors_y_periodic[6].append(0) # not fully periodic

def fake_input(ndim, N=100, leafsize=10):
    np.random.seed(100)
    pts = np.random.rand(N, ndim).astype('float64')
    left_edge = np.zeros(ndim, 'float64')
    right_edge = np.ones(ndim, 'float64')
    return pts, left_edge, right_edge, leafsize


def test_PyKDTree():
    cykdtree.PyKDTree(pts2, left_edge2, right_edge2, leafsize=leafsize)
    cykdtree.PyKDTree(pts3, left_edge3, right_edge3, leafsize=leafsize)
    cykdtree.PyKDTree(pts2, left_edge2, right_edge2,
                      leafsize=leafsize, periodic=True)
    cykdtree.PyKDTree(pts3, left_edge3, right_edge3,
                      leafsize=leafsize, periodic=True)
    assert_raises(ValueError, cykdtree.PyKDTree, pts2,
                  left_edge2, right_edge2, leafsize=1)


def test_search():
    # 2D
    tree2 = cykdtree.PyKDTree(pts2, left_edge2, right_edge2, leafsize=leafsize)
    for pos in [left_edge2, (left_edge2+right_edge2)/2.]:
        tree2.get(pos)
    assert_raises(ValueError, tree2.get, right_edge2)
    # 3D
    tree3 = cykdtree.PyKDTree(pts3, left_edge3, right_edge3, leafsize=leafsize)
    for pos in [left_edge3, (left_edge3+right_edge3)/2.]:
        tree3.get(pos)
    assert_raises(ValueError, tree3.get, right_edge3)


def test_search_periodic():
    # 2D
    tree2 = cykdtree.PyKDTree(pts2, left_edge2, right_edge2,
                              leafsize=leafsize, periodic=True)
    for pos in [left_edge2, (left_edge2+right_edge2)/2., right_edge2]:
        leaf2 = tree2.get(pos)
        leaf2.neighbors
    # 3D
    tree3 = cykdtree.PyKDTree(pts3, left_edge3, right_edge3,
                              leafsize=leafsize, periodic=True)
    for pos in [left_edge3, (left_edge3+right_edge3)/2., right_edge3]:
        leaf3 = tree3.get(pos)
        leaf3.neighbors


def test_neighbors():
    np.random.set_state(rand_state)
    pts = np.random.rand(50, 2).astype('float64')
    tree = cykdtree.PyKDTree(pts, left_edge2, right_edge2, leafsize=10)
    # 2D
    left_neighbors = [left_neighbors_x, left_neighbors_y]
    right_neighbors = [[[] for i in range(tree.num_leaves)] for _
                       in range(tree.ndim)]
    for d in range(tree.ndim):
        for i in range(tree.num_leaves):
            for j in left_neighbors[d][i]:
                right_neighbors[d][j].append(i)
        for i in range(tree.num_leaves):
            right_neighbors[d][i] = list(set(right_neighbors[d][i]))
    for leaf in tree.leaves:
        print(leaf.id, leaf.left_edge, leaf.right_edge)
    for leaf in tree.leaves:
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
    np.random.set_state(rand_state)
    pts = np.random.rand(50, 2).astype('float64')
    tree = cykdtree.PyKDTree(pts, left_edge2, right_edge2,
                             leafsize=10, periodic=True)

    # from cykdtree.plot import plot2D_serial
    # plot2D_serial(tree, label_boxes=True,
    #               plotfile='test_neighbors_serial.png')

    # 2D
    left_neighbors = [left_neighbors_x_periodic, left_neighbors_y_periodic]
    right_neighbors = [[[] for i in range(tree.num_leaves)] for
                       _ in range(tree.ndim)]
    for d in range(tree.ndim):
        for i in range(tree.num_leaves):
            for j in left_neighbors[d][i]:
                right_neighbors[d][j].append(i)
        for i in range(tree.num_leaves):
            right_neighbors[d][i] = list(set(right_neighbors[d][i]))
    for leaf in tree.leaves:
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


def test_get_neighbor_ids():
    # 2D
    tree2 = cykdtree.PyKDTree(pts2, left_edge2, right_edge2,
                              leafsize=leafsize, periodic=True)
    for pos in [left_edge2, (left_edge2+right_edge2)/2., right_edge2]:
        tree2.get_neighbor_ids(pos)
    # 3D
    tree3 = cykdtree.PyKDTree(pts3, left_edge3, right_edge3,
                              leafsize=leafsize, periodic=True)
    for pos in [left_edge3, (left_edge3+right_edge3)/2., right_edge3]:
        tree3.get_neighbor_ids(pos)


def time_tree_construction(Ntime, LStime):
    pts = np.random.rand(Ntime, 2).astype('float64')
    t0 = time.time()
    cykdtree.PyKDTree(pts, left_edge2, right_edge2, leafsize=LStime)
    t1 = time.time()
    print("{} points, leafsize {}: took {} s".format(Ntime, LStime, t1-t0))


def time_neighbor_search(Ntime, LStime):
    pts = np.random.rand(Ntime, 2).astype('float64')
    tree = cykdtree.PyKDTree(pts, left_edge2, right_edge2, leafsize=LStime)
    t0 = time.time()
    tree.get_neighbor_ids(0.5*np.ones(tree.ndim, 'double'))
    t1 = time.time()
    print("{} points, leafsize {}: took {} s".format(Ntime, LStime, t1-t0))
