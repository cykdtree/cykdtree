import numpy as np
from nose.tools import istest, nottest, assert_raises, assert_equal
from cykdtree.tests import parametrize
from cykdtree import utils

def test_max_pts():
    pts = np.arange(5*3).reshape((5, 3)).astype('float64')
    out = utils.py_max_pts(pts)
    np.testing.assert_allclose(out, np.max(pts, axis=0))


def test_min_pts():
    pts = np.arange(5*3).reshape((5, 3)).astype('float64')
    out = utils.py_min_pts(pts)
    np.testing.assert_allclose(out, np.min(pts, axis=0))


@parametrize(N=(0, 10, 11), ndim=(2, 3))
def test_quickSort(N=10, ndim=2):
    d = 1
    np.random.seed(10)
    pts = np.random.rand(N, ndim).astype('float64')
    idx = utils.py_quickSort(pts, d)
    assert_equal(idx.size, N)
    if (N != 0):
        np.testing.assert_allclose(idx, np.argsort(pts[:, d]))


@parametrize(N=(0, 10, 11), ndim=(2, 3))
def test_insertSort(N=10, ndim=2):
    d = 1
    np.random.seed(10)
    pts = np.random.rand(N, ndim).astype('float64')
    idx = utils.py_insertSort(pts, d)
    assert_equal(idx.size, N)
    if (N != 0):
        np.testing.assert_allclose(idx, np.argsort(pts[:, d]))


@parametrize(N=(0, 10, 11), ndim=(2, 3))
def test_pivot(N=10, ndim=2):
    d = 1
    np.random.seed(10)
    pts = np.random.rand(N, ndim).astype('float64')
    q, idx = utils.py_pivot(pts, d)
    if (N == 0):
        np.testing.assert_equal(q, -1)
    else:
        med = np.median(pts)
        piv = pts[idx[q], d]
        # Check in correct percentile


@parametrize(N=(0, 10, 11), ndim=(2, 3))
def test_partition(N=10, ndim=2):
    d = 1
    p = 0
    np.random.seed(10)
    pts = np.random.rand(N, ndim).astype('float64')
    q, idx = utils.py_partition(pts, d, p)
    if (N == 0):
        assert(q == -1)
    else:
        piv = pts[p, d]
        np.testing.assert_approx_equal(pts[idx[q], d], piv)
        np.testing.assert_array_less(pts[idx[:q], d], piv)
        np.testing.assert_array_less(piv, pts[idx[(q+1):], d])


def test_select():
    d = 1
    np.random.seed(10)
    # Even number
    N = 10
    p = q = int(N/2)
    p -= 1
    pts = np.random.rand(N, 2).astype('float64')
    idx = utils.py_select(pts, d, p)
    assert((pts[idx[:q], d] <= np.median(pts[:, d])).all())
    assert((pts[idx[q:], d] > np.median(pts[:, d])).all())
    pts = np.random.rand(N, 3).astype('float64')
    idx = utils.py_select(pts, d, p)
    assert((pts[idx[:q], d] <= np.median(pts[:, d])).all())
    assert((pts[idx[q:], d] > np.median(pts[:, d])).all())
    # Odd number
    N = 11
    p = q = int(N/2)
    q += 1
    pts = np.random.rand(N, 2).astype('float64')
    idx = utils.py_select(pts, d, p)
    assert((pts[idx[:q], d] <= np.median(pts[:, d])).all())
    assert((pts[idx[q:], d] > np.median(pts[:, d])).all())
    pts = np.random.rand(N, 3).astype('float64')
    idx = utils.py_select(pts, d, p)
    assert((pts[idx[:q], d] <= np.median(pts[:, d])).all())
    assert((pts[idx[q:], d] > np.median(pts[:, d])).all())
