from cykdtree import Leaf

import cython
import numpy as np
cimport numpy as np

from libcpp cimport bool as cbool
from cpython cimport bool as pybool

from libc.stdint cimport uint32_t, uint64_t, int32_t, int64_t

cdef class PyKDTree:
    r"""Construct a KDTree for a set of points.

    Args:
        pts (np.ndarray of double): (n,m) array of n coordinates in a 
            m-dimensional domain.
        left_edge (np.ndarray of double): (m,) domain minimum in each dimension.
        right_edge (np.ndarray of double): (m,) domain maximum in each dimension.
        periodic (bool, optional): True if the domain is periodic. Defaults to
            `False`.
        leafsize (int, optional): The maximum number of points that should be in 
            a leaf. Defaults to 10000.
        
    Returns:
        list of :class:`domain_decomp.Leaf`s: Leaves in the KDTree.

    Raises:
        ValueError: If `leafsize < 2`. This currectly segfaults.

    """

    def __cinit__(self, np.ndarray[double, ndim=2] pts, 
                  np.ndarray[double, ndim=1] left_edge, 
                  np.ndarray[double, ndim=1] right_edge,
                  pybool periodic = False, int leafsize = 10000):
        if (leafsize < 2):
            # This is here to prevent segfault. The cpp code needs modified to 
            # support leafsize = 1
            raise ValueError("'leafsize' cannot be smaller than 2.")
        cdef uint32_t k,i
        self.npts = <uint64_t>pts.shape[0]
        self.ndim = <uint32_t>pts.shape[1]
        self.leafsize = leafsize
        cdef np.ndarray[np.float64_t] domain_width = right_edge - left_edge
        self.left_edge = &left_edge[0]
        self.right_edge = &right_edge[0]
        self.domain_width = &domain_width[0]
        cdef np.ndarray[np.uint64_t] idx = np.arange(self.npts).astype('uint64')
        self._tree = new KDTree(&pts[0,0], &idx[0], self.npts, self.ndim, 
                                    <uint32_t>leafsize, 
                                    &left_edge[0], &right_edge[0], periodic)
        # Create list of leaves
        self.leaves = [None for _ in xrange(self._tree.leaves.size())]
        cdef Node* leafnode
        cdef np.ndarray[np.float64_t] leaf_left_edge = np.zeros(self.ndim, 'float64')
        cdef np.ndarray[np.float64_t] leaf_right_edge = np.zeros(self.ndim, 'float64')
        print(self._tree.leaves.size())
        for k in xrange(<uint32_t>self._tree.leaves.size()):
            leafnode = self._tree.leaves[k]
            leaf_idx = idx[leafnode.left_idx:(leafnode.left_idx + leafnode.children)] 
            assert(len(leaf_idx) == <int>leafnode.children)
            for i in range(self.ndim):
                leaf_left_edge[i] = leafnode.left_edge[i]
                leaf_right_edge[i] = leafnode.right_edge[i]
            print(leafnode.leafid, k)
            # self.leaves[leafnode.leafid] = Leaf(k, leaf_idx, leaf_left_edge, leaf_right_edge)
            assert(leafnode.leafid == k)
            self.leaves.append(Leaf(k, leaf_idx, leaf_left_edge, leaf_right_edge))

    def get(self, np.ndarray[double, ndim=1] pos):
        r"""Return the leaf containing a given position.

        Args:
            pos (np.ndarray of double): Coordinates.
            
        Returns:
            :class:`cykdtree.Leaf`: Leaf containing `pos`.

        Raises:
            ValueError: If pos is not contained withing the KDTree.

        """
        cdef Node *leafnode = self._tree.search(&pos[0])
        if leafnode == NULL:
            raise ValueError("Position is not within the kdtree root node.")
        return self.leaves[leafnode.leafid]

def kdtree(np.ndarray[double, ndim=2] pts,
           np.ndarray[double, ndim=1] left_edge, 
           np.ndarray[double, ndim=1] right_edge, 
           int leafsize = 10000):
    r"""Get the leaves in a KDTree constructed for a set of points.

    Args:
        pts (np.ndarray of double): (n,m) array of n coordinates in a 
            m-dimensional domain.
        left_edge (np.ndarray of double): (m,) domain minimum in each dimension.
        right_edge (np.ndarray of double): (m,) domain maximum in each dimension.
        leafsize (int, optional): The maximum number of points that should be in 
            a leaf. Defaults to 10000.
        
    Returns:
        list of :class:`domain_decomp.Leaf`s: Leaves in the KDTree.

    Raises:
        ValueError: If `leafsize < 2`. This currectly segfaults.

    """
    if (leafsize < 2):
        # This is here to prevent segfault. The cpp code needs modified to 
        # support leafsize = 1
        raise ValueError("'leafsize' cannot be smaller than 2.")
    cdef uint32_t k,i
    cdef uint64_t npts = <uint64_t>pts.shape[0]
    cdef uint32_t ndim = <uint32_t>pts.shape[1]
    cdef np.ndarray[np.uint64_t] idx = np.arange(npts).astype('uint64')
    cdef KDTree* tree = new KDTree(&pts[0,0], &idx[0], npts, ndim, 
                                   <uint32_t>leafsize, 
                                   &left_edge[0], &right_edge[0])
    cdef object leaves = []
    cdef np.ndarray[np.float64_t] leaf_left_edge = np.zeros(ndim, 'float64')
    cdef np.ndarray[np.float64_t] leaf_right_edge = np.zeros(ndim, 'float64')
    for k in xrange(tree.leaves.size()):
        leafnode = tree.leaves[k]
        leaf_idx = idx[leafnode.left_idx:(leafnode.left_idx + leafnode.children)] 
        assert(len(leaf_idx) == <int>leafnode.children)
        for i in range(ndim):
            leaf_left_edge[i] = leafnode.left_edge[i]
            leaf_right_edge[i] = leafnode.right_edge[i]
        leaves.append(Leaf(k, leaf_idx, leaf_left_edge, leaf_right_edge))
    return leaves
