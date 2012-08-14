import pylab
import numpy as np
import fista

class LogisticSolver:
    def __init__(self, C=1.0, L1_or_L2='L2', tol=1e-5):
        self.c = C
        self.reg = L1_or_L2 
        self.tol = tol

    def solve(self, x, y):
        self.x = x
        self.y = y
        m = len(x)
        fista.set_fista_param(tol=self.tol, Li0=1.5)
        res, steps = fista.fista_solve(self._objective, self._derivative,
                                       np.ones(m+1), 
                                       with_L1_reg=(self.reg != 'L2'))
        print 'Iterate numers:', steps
        return res

    def _objective(self, w):
        t = -(np.dot(w[:-1], self.x) + w[-1])*self.y
        sindex = (t<50)
        # when t is big, exp(t) -> infinity, ln(1+exp(t)) will not get t but
        # an infinity instead, which is wrong. here we use a (t<50) judgement to get
        # round it
        t[sindex] = np.log(1+np.exp(t[sindex]))
        loss = np.sum(t)
        if self.reg == 'L2':
            #return self.c * loss + 0.5*np.dot(w,w)
            return self.c * loss + 0.5*np.dot(w[:-1],w[:-1])
        else:
            return self.c * loss + np.sum(np.abs(w))

    def _derivative(self, w):
        # t - >infinity, 1/(1+t) -> 0, so following is safe
        t = np.exp((np.dot(w[:-1], self.x) + w[-1])*self.y)
        u = -1/(1+t)*self.y
        q = np.sum(u) * self.c
        p = np.dot(self.x,u) * self.c
        if self.reg == 'L2':
            p += w[:-1]
            # q += w[-1]
        return np.hstack((p,q))


if __name__=='__main__':
    from sklearn.datasets import load_iris
    from sklearn import linear_model

    data = load_iris()
    ls = LogisticSolver(L1_or_L2='L1', C=1.0, tol=1e-5)
    x = np.array([[-3,3],[0,1],[-1,-1],[1,3],[3,1],[1,-1]],dtype=float)
    y = np.array([1,1,1,-1,-1,-1])
    w = ls.solve(data.data[:100,:].T, (data.target[:100]-0.5)*2)
    #w = ls.solve(x.T, y)
    print(w)

    logistic = linear_model.LogisticRegression(penalty='l1', tol=1e-5)

    #print(logistic.fit(x,y))
    print(logistic.fit(data.data[:100,:], data.target[:100]))
    print(logistic.coef_, logistic.intercept_)
    print(logistic.coef_/w[:-1], logistic.intercept_/w[-1]) 
    print(ls._objective(np.r_[logistic.coef_[0],logistic.intercept_]))
    print(ls._objective(w))
