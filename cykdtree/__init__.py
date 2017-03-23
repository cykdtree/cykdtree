from cykdtree.kdtree import PyKDTree, PyNode
from cykdtree.parallel_kdtree import PyParallelKDTree
from cykdtree import tests
import nose
import sys
import os

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



__all__ = ["PyKDTree", "PyNode", "tests", "run_nose",
           "PyParallelKDTree"]
