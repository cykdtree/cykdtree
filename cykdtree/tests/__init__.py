from subprocess import Popen, PIPE
from nose.tools import istest, nottest
from mpi4py import MPI
import numpy as np
import itertools
import sys


def assert_less_equal(x, y):
    size_match = True
    try:
        xshape = (1,)
        yshape = (1,)
        if (isinstance(x, np.ndarray) or isinstance(y, np.ndarray)):
            if isinstance(x, np.ndarray):
                xshape = x.shape
            if isinstance(y, np.ndarray):
                yshape = y.shape
            size_match = (xshape == yshape)
            assert((x <= y).all())
        else:
            assert(x <= y)
    except:
        if not size_match:
            raise AssertionError("Shape mismatch\n\n"+
                                 "x.shape: %s\ny.shape: %s\n" % 
                                 (str(x.shape), str(y.shape)))
        raise AssertionError("Variables are not less-equal ordered\n\n" +
                             "x: %s\ny: %s\n" % (str(x), str(y)))

def call_subprocess(np, func, args, kwargs):
    # Create string with arguments & kwargs
    args_str = ""
    for a in args:
        args_str += str(a)+","
    for k, v in kwargs.items():
        args_str += k+"="+str(v)+","
    if args_str.endswith(","):
        args_str = args_str[:-1]
    cmd = ["mpirun", "-n", str(np), "python", "-c",
           "'from %s import %s; %s(%s)'" % (func.__module__, func.__name__,
                                            func.__name__, args_str)] 
    cmd = ' '.join(cmd)
    print('Running the following command:\n%s' % cmd)
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
    output, err = p.communicate()
    exit_code = p.returncode
    print(output.decode('utf-8'))
    if exit_code != 0:
        print(err.decode('utf-8'))
        raise Exception("Error on spawned process. See output.")
        return None
    return output.decode('utf-8')

def iter_dict(dicts):
    try:
        return (dict(itertools.izip(dicts, x)) for x in
                itertools.product(*dicts.itervalues()))
    except AttributeError:
        # python 3
        return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

def parametrize(**pargs):
    for k in pargs.keys():
        if not isinstance(pargs[k], (tuple, list)):
            pargs[k] = (pargs[k],)

    def dec(func):

        def pfunc(kwargs0):
            # Wrapper so that name encodes parameters
            def wrapped(*args, **kwargs):
                kwargs.update(**kwargs0)
                return func(*args, **kwargs)
            wrapped.__name__ = func.__name__
            for k,v in kwargs0.items():
                wrapped.__name__ += "_{}{}".format(k,v)
            return wrapped

        def func_param(*args, **kwargs):
            out = []
            for ipargs in iter_dict(pargs):
                out.append(pfunc(ipargs)(*args, **kwargs))
            return out

        func_param.__name__ = func.__name__

        return func_param

    return dec

def MPITest(Nproc, **pargs):

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
            @parametrize(Nproc=Nproc)
            def wrapped(*args, **kwargs):
                s = kwargs.pop('Nproc', 1)
                call_subprocess(s, func, args, kwargs)

            wrapped.__name__ = func.__name__
            return wrapped

        # Then just call the function
        else:
            @parametrize(**pargs)
            def try_func(*args, **kwargs):
                error_flag = np.array([0], 'int')
                try:
                    out = func(*args, **kwargs)
                except Exception as error:
                    import traceback
                    print(traceback.format_exc())
                    error_flag[0] = 1
                flag_count = np.zeros(1, 'int')
                comm.Allreduce(error_flag, flag_count)
                if flag_count[0] > 0:
                    raise Exception("Process %d: There were errors on %d processes." %
                                    (rank, flag_count[0]))
                return out
            return try_func
    return dec

from cykdtree.tests import test_utils
from cykdtree.tests import test_kdtree
from cykdtree.tests import test_plot
from cykdtree.tests import test_parallel_kdtree

__all__ = ["MPITest", "test_utils", "test_kdtree",
           "test_parallel_kdtree", "test_plot"]
