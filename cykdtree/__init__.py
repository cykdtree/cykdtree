import numpy as np

class Leaf(object):
    def __init__(self, leafid, idx, left_edge, right_edge,
                 periodic_left=None, periodic_right=None,
                 neighbors=None, num_leaves=None):
        r"""A container for leaf info.

        Args:
            leafid (int): Unique index of this leaf.
            idx (np.ndarray of uint64): Indices of points in this leaf.
            left_edge (np.ndarray of float64): Domain minimum along each 
                dimension.
            right_edge (np.ndarray of float64): Domain maximum along each 
                dimension.
            periodic_left (np.ndarray of bool, optional): Truth of left edge 
                being periodic in each dimension. Defaults to None. Is set by
                :meth:`cgal4py.domain_decomp.leaves`.
            periodic_right (np.ndarray of bool, optional): Truth of right edge 
                being periodic in each dimension. Defaults to None. Is set by 
                :meth:`cgal4py.domain_decomp.leaves`.
            neighbors (list of dict, optional): Indices of neighboring leaves in 
                each dimension. Defaults to None. Is set by 
                :meth:`cgal4py.domain_decomp.leaves`.
            num_leaves (int, optional): Number of leaves in the domain 
                decomposition. Defaults to None. Is set by 
                :meth:`cgal4py.domain_decomp.leaves`.  

        Attributes:
            id (int): Unique index of this leaf.
            idx (np.ndarray of uint64): Indices of points in this leaf.
            ndim (int): Number of dimensions in the domain.
            left_edge (np.ndarray of float64): Domain minimum along each 
                dimension.
            right_edge (np.ndarray of float64): Domain maximum along each 
                dimension.
            periodic_left (np.ndarray of bool): Truth of left edge being 
                periodic in each dimension. 
            periodic_right (np.ndarray of bool): Truth of right edge being 
                periodic in each dimension. 
            neighbors (list of dict): Indices of neighboring leaves in each 
                dimension. 
            num_leaves (int): Number of leaves in the domain decomposition. 

        """
        self.id = leafid
        self.idx = idx
        self.ndim = len(left_edge)
        self.left_edge = left_edge
        self.right_edge = right_edge
        self.periodic_left = periodic_left
        self.periodic_right = periodic_right
        self.neighbors = neighbors
        self.num_leaves = num_leaves

    @property
    def all_neighbors(self):
        """list of int: Indices of all neighboring leaves."""
        if self.neighbors is None:
            return []
        out = []
        for i in range(self.ndim):
            out += self.neighbors[i]['left'] + self.neighbors[i]['right'] + \
              self.neighbors[i]['left_periodic'] + \
              self.neighbors[i]['right_periodic']
        return set(out)

from kdtree import PyKDTree

__all__ = ["Leaf", "PyKDTree"]

