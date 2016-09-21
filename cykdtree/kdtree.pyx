import cython
import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free
from libcpp cimport bool as cbool
from cpython cimport bool as pybool

from libc.stdint cimport uint32_t, uint64_t, int32_t, int64_t

cdef class PyNode:
    r"""A container for leaf info.

    Attributes:
        npts (np.uint64_t): Number of points in this node.
        ndim (np.uint32_t): Number of dimensions in domain.
        num_leaves (np.uint32_t): Number of leaves in the tree containing this 
            node.
        start_idx (np.uint64_t): Index where indices for this node begin.
        stop_idx (np.uint64_t): One passed the end of indices for this node.
        left_edge (np.ndarray of float64): Minimum bounds of this node in each 
            dimension.
        right_edge (np.ndarray of float64): Maximum bounds of this node in each 
            dimension.
        periodic_left (np.ndarray of bool): Periodicity of minimum bounds.
        periodic_right (np.ndarray of bool): Periodicity of maximum bounds.
        domain_width (np.ndarray of float64): Width of the total domain in each
            dimension.
        left_neighbors (list of lists): Indices of neighbor leaves at the 
            minimum bounds in each dimension.
        right_neighbors (list of lists): Indices of neighbor leaves at the 
            maximum bounds in each dimension.

    """

    def __cinit__(self, uint32_t ndim):
        self.ndim = ndim
        self._left_edge = <double *>malloc(self.ndim*sizeof(double))
        self._right_edge = <double *>malloc(self.ndim*sizeof(double))
        self._periodic_left = <cbool *>malloc(self.ndim*sizeof(cbool))
        self._periodic_right = <cbool *>malloc(self.ndim*sizeof(cbool))

    cdef void _init_node(self, Node* node, uint32_t num_leaves,
                         double *domain_width):
                         # np.ndarray[np.float64_t, ndim=1] domain_width):
        cdef np.uint32_t i, j
        self._node = node
        self.id = node.leafid
        self.npts = node.children
        assert(self.ndim == node.ndim)
        # self.ndim = node.ndim
        self.num_leaves = num_leaves
        self.start_idx = node.left_idx
        self.stop_idx = (node.left_idx + node.children)
        self._domain_width = domain_width
        # self.domain_width = np.zeros(self.ndim, 'float64')
        # self.left_edge = np.zeros(self.ndim, 'float64')
        # self.right_edge = np.zeros(self.ndim, 'float64')
        # self.periodic_left = np.zeros(self.ndim, 'bool')
        # self.periodic_right = np.zeros(self.ndim, 'bool')
        self.left_neighbors = []
        self.right_neighbors = []
        for i in range(self.ndim):
            self._left_edge[i] = node.left_edge[i]
            self._right_edge[i] = node.right_edge[i]
            self._periodic_left[i] = node.periodic_left[i]
            self._periodic_right[i] = node.periodic_right[i]
            # self.domain_width[i] = domain_width[i]
            self.left_neighbors.append(list(set([node.left_neighbors[i][j] for j in range(node.left_neighbors[i].size())])))
            self.right_neighbors.append(list(set([node.right_neighbors[i][j] for j in range(node.right_neighbors[i].size())])))

    def __dealloc__(self):
        free(self._left_edge)
        free(self._right_edge)
        free(self._periodic_left)
        free(self._periodic_right)

    @property
    def periodic_left(self):
        cdef cbool[:] view = <cbool[:self.ndim]> self._periodic_left
        return np.asarray(view)
    @property
    def periodic_right(self):
        cdef cbool[:] view = <cbool[:self.ndim]> self._periodic_right
        return np.asarray(view)
    @property
    def left_edge(self):
        cdef np.float64_t[:] view = <np.float64_t[:self.ndim]> self._left_edge
        return np.asarray(view)
    @property
    def right_edge(self):
        cdef np.float64_t[:] view = <np.float64_t[:self.ndim]> self._right_edge
        return np.asarray(view)
    @property
    def domain_width(self):
        cdef np.float64_t[:] view = <np.float64_t[:self.ndim]> self._domain_width
        return np.asarray(view)

    @property
    def slice(self):
        """slice: Slice of kdtree indices contained by this node."""
        return slice(self.start_idx, self.stop_idx)

    @property
    def neighbors(self):
        """list of int: Indices of all neighboring leaves."""
        cdef np.uint32_t i
        cdef object out = []
        for i in range(self.ndim):
            out += self.left_neighbors[i] + self.right_neighbors[i]
        return set(out)        

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

    Attributes:

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
        self._left_edge = <double *>malloc(self.ndim*sizeof(double))
        self._right_edge = <double *>malloc(self.ndim*sizeof(double))
        self._domain_width = <double *>malloc(self.ndim*sizeof(double))
        for i in range(self.ndim):
            self._left_edge[i] = left_edge[i]
            self._right_edge[i] = right_edge[i]
            self._domain_width[i] = right_edge[i] - left_edge[i]
        self.periodic = periodic
        cdef np.ndarray[np.uint64_t] idx = np.arange(self.npts).astype('uint64')
        self._tree = new KDTree(&pts[0,0], &idx[0], self.npts, self.ndim, <uint32_t>leafsize, 
                                self._left_edge, self._right_edge, periodic)
        self.idx = idx
        # Create list of Python leaves
        self.num_leaves = <uint32_t>self._tree.leaves.size()
        self.leaves = []
        cdef Node* leafnode
        cdef PyNode leafnode_py
        cdef object leaf_neighbors = None
        for k in xrange(self.num_leaves):
            leafnode = self._tree.leaves[k]
            assert(leafnode.leafid == k)
            leafnode_py = PyNode(self.ndim)
            leafnode_py._init_node(leafnode, self.num_leaves, self._domain_width)
            self.leaves.append(leafnode_py)

    @property
    def left_edge(self):
        cdef np.float64_t[:] view = <np.float64_t[:self.ndim]> self._left_edge
        return np.asarray(view)
    @property
    def right_edge(self):
        cdef np.float64_t[:] view = <np.float64_t[:self.ndim]> self._right_edge
        return np.asarray(view)
    @property
    def domain_width(self):
        cdef np.float64_t[:] view = <np.float64_t[:self.ndim]> self._domain_width
        return np.asarray(view)

    def leaf_idx(self, np.uint32_t leafid):
        r"""Get array of indices for points in a leaf.

        Args:
            leafid (np.uint32_t): Unique index of the leaf in question.

        Returns:
            np.ndarray of np.uint64_t: Indices of points belonging to leaf.

        """
        cdef np.ndarray[np.uint64_t] out = self.idx[self.leaves[leafid].slice]
        return out

    cdef void _wrap_pos(self, double *pos):
        cdef np.uint32_t i
        for i in range(self.ndim):
            pos[i] = self._left_edge[i] + \
              ((pos[i] - self._left_edge[i]) % self._domain_width[i])

    cdef void _wrap_pos_3(self, double pos[3]):
        cdef np.uint32_t i
        for i in range(self.ndim):
            pos[i] = self._left_edge[i] + \
              ((pos[i] - self._left_edge[i]) % self._domain_width[i])

    cdef np.ndarray[np.uint32_t, ndim=1] _get_neighbor_ids(self, np.ndarray[double, ndim=1] pos):
        assert(<uint32_t>len(pos) == self.ndim)
        cdef np.ndarray[double, ndim=1] wrapped_pos = np.empty(self.ndim, 'double')
        cdef np.uint32_t i, j
        for i in range(self.ndim):
            wrapped_pos[i] = pos[i]
        # Wrap positions for periodic domains to make search easier
        if self.periodic:
            self._wrap_pos(&wrapped_pos[0])
        # Search
        cdef Node *leafnode = self._tree.search(&wrapped_pos[0])
        if leafnode == NULL:
            raise ValueError("Position is not within the kdtree root node.")
        # Transfer neighbor ids to total array
        cdef np.uint32_t ntot = 1
        for i in range(self.ndim):
            ntot += leafnode.left_neighbors[i].size()
            ntot += leafnode.right_neighbors[i].size()
        cdef np.ndarray[np.uint32_t, ndim=1] out = np.empty(ntot, 'uint32')
        cdef np.uint32_t c = 0
        for i in range(self.ndim):
            for j in range(leafnode.left_neighbors[i].size()):
                out[c] = leafnode.left_neighbors[i][j]
                c += 1
            for j in range(leafnode.right_neighbors[i].size()):
                out[c] = leafnode.right_neighbors[i][j]
                c += 1
        out[c] = leafnode.leafid
        return np.unique(out)

    def get_neighbor_ids(self, np.ndarray[double, ndim=1] pos):
        r"""Return the IDs of leaves containing & neighboring a given position.

        Args:
            pos (np.ndarray of double): Coordinates.
            
        Returns:
            np.ndarray of uint32: Leaves containing/neighboring `pos`.

        Raises:
            ValueError: If pos is not contained withing the KDTree.

        """
        return self._get_neighbor_ids(pos)

    cdef np.ndarray[np.uint32_t, ndim=1] _get_neighbor_ids_3(self, np.float64_t pos[3]):
        assert(self.ndim == 3)
        cdef np.uint32_t i, j
        cdef np.float64_t wrapped_pos[3]
        for i in range(3):
            wrapped_pos[i] = pos[i]
        # Wrap positions for periodic domains to make search easier
        if self.periodic:
            self._wrap_pos(&wrapped_pos[0])
            # self._wrap_pos_3(wrapped_pos)
        # Search
        cdef Node *leafnode = self._tree.search(&wrapped_pos[0])
        if leafnode == NULL:
            raise ValueError("Position is not within the kdtree root node.")
        # Transfer neighbor ids to total array
        cdef np.uint32_t ntot = 1
        for i in range(self.ndim):
            ntot += leafnode.left_neighbors[i].size()
            ntot += leafnode.right_neighbors[i].size()
        cdef np.ndarray[np.uint32_t, ndim=1] out = np.empty(ntot, 'uint32')
        cdef np.uint32_t c = 0
        for i in range(self.ndim):
            for j in range(leafnode.left_neighbors[i].size()):
                out[c] = leafnode.left_neighbors[i][j]
                c += 1
            for j in range(leafnode.right_neighbors[i].size()):
                out[c] = leafnode.right_neighbors[i][j]
                c += 1
        out[c] = leafnode.leafid
        return np.unique(out)

    cdef PyNode _get(self, np.ndarray[double, ndim=1] pos):
        assert(<uint32_t>len(pos) == self.ndim)
        cdef np.ndarray[double, ndim=1] wrapped_pos = np.empty(self.ndim, 'double')
        cdef np.uint32_t i
        for i in range(self.ndim):
            wrapped_pos[i] = pos[i]
        # Wrap positions for periodic domains to make search easier
        if self.periodic:
            self._wrap_pos(&wrapped_pos[0])
        # Search
        cdef Node *leafnode = self._tree.search(&wrapped_pos[0])
        if leafnode == NULL:
            raise ValueError("Position is not within the kdtree root node.")
        cdef PyNode out = self.leaves[leafnode.leafid]
        return out

    def get(self, np.ndarray[double, ndim=1] pos):
        r"""Return the leaf containing a given position.

        Args:
            pos (np.ndarray of double): Coordinates.
            
        Returns:
            :class:`cykdtree.PyNode`: Leaf containing `pos`.

        Raises:
            ValueError: If pos is not contained withing the KDTree.

        """
        return self._get(pos)
