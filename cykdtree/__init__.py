import nose
import sys
import os
PY_MAJOR_VERSION = sys.version_info[0]
FLAG_MULTIPROC = True # TODO: check that mpi installed
from cykdtree.kdtree import PyKDTree, PyNode
from cykdtree.parallel_kdtree import PyParallelKDTree
from cykdtree import tests, plot

def run_nose(verbose=False):
    nose_argv = sys.argv
    nose_argv += ['--detailed-errors', '--exe']
    if verbose:
        nose_argv.append('-v')
    initial_dir = os.getcwd()
    my_package_file = os.path.abspath(__file__)
    my_package_dir = os.path.dirname(my_package_file)
    if os.path.samefile(os.path.dirname(my_package_dir), initial_dir):
        # Provide a nice error message to work around nose bug
        # see https://github.com/nose-devs/nose/issues/701
        raise RuntimeError(
            """
    The cykdtree.run_nose function does not work correctly when invoked in
    the same directory as the installed cykdtree package. Try starting
    a python session in a different directory before invoking cykdtree.run_nose
    again. Alternatively, you can also run the "nosetests" executable in
    the current directory like so:

        $ nosetests
            """
            )
    os.chdir(my_package_dir)
    try:
        nose.core.run(tests, argv=nose_argv)
    finally:
        os.chdir(initial_dir)


def get_include():
    """
    Return the directory that contains the NumPy \\*.h header files.
    Extension modules that need to compile against NumPy should use this
    function to locate the appropriate include directory.
    Notes
    -----
    When using ``distutils``, for example in ``setup.py``.
    ::
        import numpy as np
        ...
        Extension('extension_name', ...
                include_dirs=[np.get_include()])
        ...
    """
    import cykdtree
    return os.path.dirname(cykdtree.__file__)


def make_tree(pts, nproc=0, **kwargs):
    r"""Build a KD-tree for a set of points.

    Args:
        pts (np.ndarray of float64): (n,m) Array of n mD points.
        nproc (int, optional): The number of MPI processes that should be
            spawned. If <2, no processes are spawned. Defaults to 0.
        \*\*kwargs: Additional keyword arguments are passed to the appropriate
            class for constructuing the tree.

    Returns:
        T (:class:`cykdtree.PyKDTree`): KDTree object.

    Raises:
        ValueError: If `pts` is not a 2D array.

    """
    # Check input
    if (pts.ndim != 2):
        raise ValueError("pts must be a 2D array of coordinates")
    # Parallel
    if nproc > 1 and FLAG_MULTIPROC:
        T = spawn_parallel(pts, nproc, **kwargs)
    # Serial
    else:
        T = PyKDTree(pts, **kwargs)
    return T


__all__ = ["PyKDTree", "PyNode", "tests", "run_nose", "get_include",
           "PyParallelKDTree", "plot", "make_tree"]
