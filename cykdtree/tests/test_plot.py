import os
from cykdtree.plot import plot2D_serial, plot2D_parallel
from cykdtree.kdtree import PyKDTree
from cykdtree.parallel_kdtree import PyParallelKDTree
from cykdtree.tests.test_kdtree import fake_input as fake_input_serial
from cykdtree.tests.test_parallel_kdtree import fake_input as fake_input_parallel


def test_plot2D_serial():
    fname_test = "test_plot2D_serial.png"
    pts, le, re, ls = fake_input_serial(2)
    tree = PyKDTree(pts, le, re, leafsize=ls)
    axs = plot2D_serial(tree, pts, title="Serial Test", plotfile=fname_test,
                        label_boxes=True)
    os.remove(fname_test)
    # plot2D_serial(tree, pts, axs=axs)
    del axs

def test_plot2D_parallel():
    fname_test = "test_plot2D_parallel.png"
    pts, le, re, ls = fake_input_parallel(2)
    tree = PyParallelKDTree(pts, le, re, leafsize=ls)
    axs = plot2D_parallel(tree, pts, title="Parallel Test", plotfile=fname_test,
                          label_boxes=True, label_procs=True)
    os.remove(fname_test)
    # plot2D_parallel(tree, pts, axs=axs)
    del axs


