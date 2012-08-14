import fista
from nose.tools import *
import numpy as np

def quard_with_L1(x):
    return np.dot(x-1,x-1) + np.sum(np.abs(x))

def quard(x):
    return np.dot(x-1,x-1)

def der_quard(x):
    return 2*(x-1)

def log_with_L1(x):
    return np.sum(np.log(1+np.exp(-x)) + np.abs(x))

def der_log(x):
    return -1/(np.exp(x)+1)

A = np.array([[2,3],[3,4]])
b = np.array([5,7])
def pd_with_L1(x):
    global A,b
    d = np.dot(A,x)-b
    return np.dot(d,d) + np.sum(np.abs(x))

def der_pd(x):
    global A,b
    return 2*np.dot(A.T, np.dot(A,x)-b)

def test_fista_quard_L1():
    res,step = fista.fista_solve(quard_with_L1, der_quard,
                             np.array([100,100]), L=2, with_L1_reg=True)
    assert_almost_equal(res[0],0.5)
    assert_almost_equal(res[1],0.5)

def test_fista_quard():
    res,step = fista.fista_solve(quard, der_quard, np.array([100,100]), L=2)
    assert_almost_equal(res[0], 1)
    assert_almost_equal(res[1], 1)

def test_fista_log_L1():
    res,step = fista.fista_solve(log_with_L1, der_log, np.array([100]), with_L1_reg=True)
    assert_almost_equal(res[0], 0)

def test_fista_pd_L1():
    fista.set_fista_param(tol=1e-4, eta=2)
    res,step = fista.fista_solve(pd_with_L1, der_pd, np.array([0.0971,0.8235]),
                             with_L1_reg=True) 
    assert_almost_equal(res[0], 0)
    assert_almost_equal(res[1], 1.70009, 4)
