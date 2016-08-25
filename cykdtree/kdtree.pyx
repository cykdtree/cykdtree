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
        cdef uint32_t k,i,j
        self.npts = <uint64_t>pts.shape[0]
        self.ndim = <uint32_t>pts.shape[1]
        self.leafsize = leafsize
        self.left_edge = left_edge
        self.right_edge = right_edge
        self.domain_width = right_edge - left_edge
        self.periodic = periodic
        cdef np.ndarray[np.uint64_t] idx = np.arange(self.npts).astype('uint64')
        self._tree = new KDTree(&pts[0,0], &idx[0], self.npts, self.ndim, <uint32_t>leafsize, 
                                &left_edge[0], &right_edge[0], periodic)
        # Create list of Python leaves
        self.num_leaves = <uint32_t>self._tree.leaves.size()
        self.leaves = []
        cdef Node* leafnode
        cdef np.ndarray[np.float64_t] leaf_left_edge = np.zeros(self.ndim, 'float64')
        cdef np.ndarray[np.float64_t] leaf_right_edge = np.zeros(self.ndim, 'float64')
        cdef np.ndarray[np.uint8_t] leaf_periodic_left = np.zeros(self.ndim, 'uint8')
        cdef np.ndarray[np.uint8_t] leaf_periodic_right = np.zeros(self.ndim, 'uint8')
        cdef object leaf_neighbors = None
        for k in xrange(self.num_leaves):
            leafnode = self._tree.leaves[k]
            assert(leafnode.leafid == k)
            # Index
            leaf_idx = idx[leafnode.left_idx:(leafnode.left_idx + leafnode.children)] 
            assert(len(leaf_idx) == <int>leafnode.children)
            # Bounds & periodicity
            for i in range(self.ndim):
                leaf_left_edge[i] = leafnode.left_edge[i]
                leaf_right_edge[i] = leafnode.right_edge[i]
                leaf_periodic_left[i] = <np.uint8_t>leafnode.periodic_left[i]
                leaf_periodic_right[i] = <np.uint8_t>leafnode.periodic_right[i]
            # Neighbors
            leaf_neighbors = [
                {'left':[],'left_periodic':[],
                 'right':[],'right_periodic':[]} for i in range(self.ndim)]
            for i in range(self.ndim):
                if leaf_periodic_left[i]:
                    leaf_neighbors[i]['left_periodic'] = \
                      [leafnode.left_neighbors[i][j] for j in range(leafnode.left_neighbors[i].size())]
                else:
                    leaf_neighbors[i]['left'] = \
                      [leafnode.left_neighbors[i][j] for j in range(leafnode.left_neighbors[i].size())]
                if leaf_periodic_right[i]:
                    leaf_neighbors[i]['right_periodic'] = \
                      [leafnode.right_neighbors[i][j] for j in range(leafnode.right_neighbors[i].size())]
                else:
                    leaf_neighbors[i]['right'] = \
                      [leafnode.right_neighbors[i][j] for j in range(leafnode.right_neighbors[i].size())]
            # Add leaf
            self.leaves.append(Leaf(k, leaf_idx, leaf_left_edge, leaf_right_edge,
                                    periodic_left = leaf_periodic_left.astype('bool'),
                                    periodic_right = leaf_periodic_right.astype('bool'),
                                    neighbors = leaf_neighbors, num_leaves = self.num_leaves))

    def get(self, np.ndarray[double, ndim=1] pos):
        r"""Return the leaf containing a given position.

        Args:
            pos (np.ndarray of double): Coordinates.
            
        Returns:
            :class:`cykdtree.Leaf`: Leaf containing `pos`.

        Raises:
            ValueError: If pos is not contained withing the KDTree.

        """
        assert(<uint32_t>len(pos) == self.ndim)
        cdef np.ndarray[double, ndim=1] wrapped_pos = pos
        cdef np.uint32_t i
        # Wrap positions for periodic domains to make search easier
        if self.periodic:
            wrapped_pos = self.left_edge + ((pos - self.left_edge) % self.domain_width)
        # Search
        cdef Node *leafnode = self._tree.search(&wrapped_pos[0])
        if leafnode == NULL:
            raise ValueError("Position is not within the kdtree root node.")
        return self.leaves[leafnode.leafid]

