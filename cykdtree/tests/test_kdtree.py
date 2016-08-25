import numpy as np
from nose.tools import assert_equal
from nose.tools import assert_raises

import cykdtree

N = 100 ; leafsize = 10
pts2 = np.random.rand(N,2).astype('float64')
left_edge2 = np.zeros(2, 'float64')
right_edge2 = np.ones(2, 'float64')
pts3 = np.random.rand(N,3).astype('float64')
left_edge3 = np.zeros(3, 'float64')
right_edge3 = np.ones(3, 'float64')

def test_Leaf():
    leaf2 = cykdtree.Leaf(0, np.arange(N), left_edge2, right_edge2)
    leaf3 = cykdtree.Leaf(0, np.arange(N), left_edge3, right_edge3)
    assert(len(leaf2.all_neighbors) == 0)
    assert(len(leaf3.all_neighbors) == 0)

def test_PyKDTree():
    tree2 = cykdtree.PyKDTree(pts2, left_edge2, right_edge2, leafsize=leafsize)
    tree3 = cykdtree.PyKDTree(pts3, left_edge3, right_edge3, leafsize=leafsize)
    tree2 = cykdtree.PyKDTree(pts2, left_edge2, right_edge2, 
                              leafsize=leafsize, periodic=True)
    tree3 = cykdtree.PyKDTree(pts3, left_edge3, right_edge3, 
                              leafsize=leafsize, periodic=True)
    assert_raises(ValueError, cykdtree.PyKDTree, pts2, left_edge2, right_edge2, leafsize=1)
    
def test_search():
    # 2D
    tree2 = cykdtree.PyKDTree(pts2, left_edge2, right_edge2, leafsize=leafsize)
    for pos in [left_edge2, (left_edge2+right_edge2)/2.]:
        leaf2 = tree2.get(pos)
    assert_raises(ValueError, tree2.get, right_edge2)
    # 3D
    tree3 = cykdtree.PyKDTree(pts3, left_edge3, right_edge3, leafsize=leafsize)
    for pos in [left_edge3, (left_edge3+right_edge3)/2.]:
        leaf3 = tree3.get(pos)
    assert_raises(ValueError, tree3.get, right_edge3)

def test_search_periodic():
    # 2D
    tree2 = cykdtree.PyKDTree(pts2, left_edge2, right_edge2, leafsize=leafsize, periodic=True)
    for pos in [left_edge2, (left_edge2+right_edge2)/2., right_edge2]:
        leaf2 = tree2.get(pos)
        neigh2 = leaf2.all_neighbors
    # 3D
    tree3 = cykdtree.PyKDTree(pts3, left_edge3, right_edge3, leafsize=leafsize, periodic=True)
    for pos in [left_edge3, (left_edge3+right_edge3)/2., right_edge3]:
        leaf3 = tree3.get(pos)
        neigh3 = leaf3.all_neighbors

def test_neighbors():
    # TODO
    pass
