from subprocess import Popen, PIPE
from nose.tools import istest, nottest
from mpi4py import MPI


def call_subprocess(args):
    args = ' '.join(args)
    p = Popen(args, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
    output, err = p.communicate()
    print(err)
    exit_code = p.returncode
    print(exit_code)
    if exit_code != 0:
        return None
    print(output)
    return output


def MPITest(Nproc):

    if not isinstance(Nproc, (tuple, list)):
        Nproc = (Nproc,)
    max_size = max(Nproc)

    def dec(func):

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        # print(size, Nproc, size in Nproc)

        # First do setup
        if (size not in Nproc):
            def spawn(s):
                def wrapped(*args, **kwargs):
                    # print(dir(func))
                    # print(func.__class__, func.__module__, func.__name__)
                    # func(*args, **kwargs)
                    # call function on size processes
                    args = ["mpirun", "-n", str(s), "python", "-c",
                            "'from %s import %s; %s()'" % (
                                func.__module__, func.__name__, func.__name__)]
                    call_subprocess(args)

                wrapped.__name__ = func.__name__ + "_%d" % s
                return wrapped
            # spawn.__name__ = func.__name__
            # return spawn
            def generator():
                for s in Nproc:
                    yield spawn(s)
            generator.__name__ = func.__name__

            return generator
        # Then just call the function
        else:
            return func

    return dec

from cykdtree.tests import test_utils
from cykdtree.tests import test_kdtree
from cykdtree.tests import test_plot
from cykdtree.tests import test_parallel_kdtree

__all__ = ["MPITest", "test_utils", "test_kdtree",
           "test_parallel_kdtree", "test_plot"]
