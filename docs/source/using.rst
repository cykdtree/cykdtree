Using cykdtree
##############

From Python
***********

The ``cykdtree`` package provides a python interface to a C++ kdtree
implementation. The primary interface to ``cykdtree`` is the ``PyKDTree``
class::

  from cykdtree import PyKDTree
  import numpy as np

  # Randomly generate a million particle positions
  np.random.seed(0xdeadbeef)
  positions = np.random.random((int(1e6), 3))

  # Initialize PyKDTree object
  tree = PyKDTree(
      positions,
      left_edge=np.array([0., 0., 0.]),
      right_edge=np.array([1., 1., 1.]),
      periodic=np.array([True, True, True]),
      leafsize=30
  )

  # Working with the tree
  print("The number of tree leaves is %i" % len(tree.leaves))
  print("")
  node = tree.get(np.array([0.5, 0.5, 0.5]))
  print("The KDTree node containing the point [0.5, 0.5, 0.5] is:\n%s" % node)
  print("")
  print("The KDTree IDs of that node's neighbors:\n%s" % node.neighbors)

This should print the following output::

  The number of tree leaves is 65536

  The KDTree node containing the point [0.5, 0.5, 0.5] is:
  PyNode(id=22235, npts=15, start_idx=339280, stop_idx=339295,
         left_edge=[ 0.49900133  0.47859409  0.466449  ],
         right_edge=[ 0.52928342  0.50126165  0.50050787])

  The KDTree IDs of that node's neighbors:
  [8190, 8191, 13778, 13783, 22232, 22233, 22234, 22235, 22236, 22237, 22238, 22239, 26816, 26824, 26825, 26826, 26827, 39773, 43589, 53828, 53829]

One can think of the KDTree as a spatially sorted list of particles. These
sorted indices are stored in the ``idx`` attribute of the ``PyKDTree``. One can
use ``idx`` and the ``slice`` attribute of a ``PyNode`` instance to find the
positions of the particles contained in that node::

  node_particle_positions = positions[tree.idx[node.slice]]
  print(node_particle_positions)

This should print the positions of all of the particles contained in the KDTree
node we selected earlier.

From Cython
***********

This is a bit more cumbersome, as your C++ compiler will need to be able to
find the ``cykdtree`` headers when it builds your code. The most straightforward
way to get this working is to integrate ``cykdtree`` into your ``setup.py``
script::

  ex = Extension("my_cython_cykdtree_wrapper",
                 sources=["my_cython_cykdtree_wrapper.pyx"]
                 include_dirs=[numpy.get_include, cykdtree.get_include()])

This will allow you to write Cython code that cimports code from cykdtree.
  
