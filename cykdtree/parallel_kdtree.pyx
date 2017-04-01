import cython
import numpy as np
cimport numpy as np
import traceback
from mpi4py import MPI

from libc.stdlib cimport malloc, free
from libcpp cimport bool as cbool
from cpython cimport bool as pybool

from libc.stdint cimport uint32_t, uint64_t, int32_t, int64_t


cdef class PyParallelKDTree:
    r"""Object for constructing a KDTree in parallel.

    """

    def __cinit__(self, np.ndarray[double, ndim=2] pts = None,
                  np.ndarray[double, ndim=1] left_edge = None,
                  np.ndarray[double, ndim=1] right_edge = None,
                  object periodic = False, int leafsize = 10000,
                  int nleaves = 0):
        cdef object comm = MPI.COMM_WORLD
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()
        self.leafsize = leafsize
        self._left_edge = NULL
        self._right_edge = NULL
        self._periodic = NULL
        cdef uint32_t i
        cdef object error = None
        cdef object error_flags = None
        cdef int error_flag = 0
        # Init & broadcast basic properties to all processes
        if self.rank == 0:
            self.ndim = pts.shape[1]
            self.npts = pts.shape[0]
            if nleaves > 0:
                nleaves = <int>(2**np.ceil(np.log2(<float>nleaves)))
                self.leafsize = self.npts/nleaves + 1
        else:
            self.ndim = 0
            self.npts = 0
        self.ndim = comm.bcast(self.ndim, root=0)
        self.leafsize = comm.bcast(self.leafsize, root=0)
        if (self.leafsize < 2):
            # This is here to prevent segfault. The cpp code needs modified
            # to support leafsize = 1
            raise ValueError("Process %d: 'leafsize' cannot be smaller than 2." %
                             self.rank)
        # Determine bounds of domain
        try:
            if self.rank == 0:
                if left_edge is None:
                    left_edge = np.min(pts, axis=0)
                if right_edge is None:
                    right_edge = np.max(pts, axis=0)
                assert(left_edge.size == self.ndim)
                assert(right_edge.size == self.ndim)
                self._left_edge = <double *>malloc(self.ndim*sizeof(double))
                self._right_edge = <double *>malloc(self.ndim*sizeof(double))
                self._periodic = <cbool *>malloc(self.ndim*sizeof(cbool));
                for i in range(self.ndim):
                    self._left_edge[i] = left_edge[i]
                    self._right_edge[i] = right_edge[i]
                if isinstance(periodic, pybool):
                    for i in range(self.ndim):
                        self._periodic[i] = <cbool>periodic
                else:
                    for i in range(self.ndim):
                        self._periodic[i] = <cbool>periodic[i]
            else:
                assert(pts is None)
        except Exception as error:
            error_flag = 1
        # Handle errors
        error_flags = comm.allgather(error_flag)
        if sum(error_flags) > 0:
            if error_flag:
                raise error
                # traceback.print_exception(type(error), error, error.__traceback__)
            raise Exception("Process %d: There were errors on %d processes." % 
                            (self.rank, sum(error_flags)))
        # Create c object
        if self.rank == 0:
            self._make_tree(&pts[0,0])
        else:
            self._make_tree(NULL)
        # Create list of Python leaves 
        self.num_leaves = <uint32_t>self._tree.leaves.size()
        self.tot_num_leaves = self._tree.tot_num_leaves
        self.leaves = {}
        cdef Node* leafnode
        cdef PyNode leafnode_py
        cdef object leaf_neighbors = None
        for k in xrange(self.num_leaves):
            leafnode = self._tree.leaves[k]
            leafnode_py = PyNode(self.ndim)
            leafnode_py._init_node(leafnode, self.num_leaves,
                                   self._tree.domain_width)
            self.leaves[leafnode.leafid] = leafnode_py

    def __dealloc__(self):
        if self.rank == 0:
            free(self._left_edge)
            free(self._right_edge)
            free(self._periodic)
        free(self._tree)

    cdef void _make_tree(self, double *pts):
        cdef np.ndarray[np.uint64_t] idx = np.arange(self.npts).astype('uint64')
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            self._tree = new ParallelKDTree(pts, &idx[0], self.npts, self.ndim,
                                            self.leafsize, self._left_edge,
                                            self._right_edge, self._periodic)
        self.idx = idx

    @property
    def left_edge(self):
        cdef np.float64_t[:] view = <np.float64_t[:self.ndim]> self._tree.domain_left_edge
        return np.asarray(view)
    @property
    def right_edge(self):
        cdef np.float64_t[:] view = <np.float64_t[:self.ndim]> self._tree.domain_right_edge
        return np.asarray(view)
    @property
    def domain_width(self):
        cdef np.float64_t[:] view = <np.float64_t[:self.ndim]> self._tree.domain_width
        return np.asarray(view)
    @property
    def periodic(self):
        cdef cbool[:] view = <cbool[:self.ndim]> self._tree.periodic
        return np.asarray(view)

    cdef object _get(self, np.ndarray[double, ndim=1] pos):
        cdef object out = None
        assert(<uint32_t>len(pos) == self.ndim)
        cdef object comm = MPI.COMM_WORLD
        cdef Node* leafnode = self._tree.search(&pos[0])
        # cdef PyNode out = PyNode()
        cdef pybool found = (leafnode != NULL)
        cdef object all_found = comm.allgather(found)
        if sum(all_found) != 1:
            raise ValueError("Position is not within the kdtree root node.")
        if leafnode != NULL:
            out = self.leaves[leafnode.leafid]
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

    # def build(self, pybool include_self = False):
    #     cdef cbool c_is = <cbool>include_self
    #     with nogil, cython.boundscheck(False), cython.wraparound(False):
    #         self._tree.build(c_is)
