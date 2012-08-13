import numpy as np
import fista

class LogisticSolver:
    def solve(self, x, y, c):
        self.x = x
        self.y = y
        self.c = c
        m = len(x)
        res, steps = fista.fista_solve(self._objective, self._derivative, np.ones(m+1))
        print 'Iterate numers:', steps
        return res

    def _objective(self, w):
        t = (np.dot(w[:-1], self.x) + w[-1])*self.y
        return self.c * np.sum(np.log(1+np.exp(-t))) + 0.5*np.dot(w[:-1],w[:-1])
#        return self.c * np.sum(np.log(1+np.exp(-t))) + 0.5*np.dot(w,w)


    def _derivative(self, w):
        m = len(self.x)
        t = np.exp(-(np.dot(w[:-1], self.x) + w[-1])*self.y)
        u = -t/(1+t)*self.y
        p = np.dot(self.x,u) * self.c + w[:-1]
        q = np.sum(u) * self.c
#        q = np.sum(u) * self.c + w[-1]
        return np.hstack((p,q))


if __name__=='__main__':
    from sklearn.datasets import load_iris
    from sklearn import linear_model

    data = load_iris()
    ls = LogisticSolver()
    x = np.array([[-3,3],[0,1],[-1,-1],[1,3],[3,1],[1,-1]],dtype=float)
    y = np.array([1,1,1,-1,-1,-1])
    w = ls.solve(data.data[:100,:].T, (data.target[:100]-0.5)*2, 1)
    #w = ls.solve(x.T, y, 1.0)
    print(w)
    #print(1/(1+np.exp(-(np.dot(w[:-1],x.T) + w[-1])*y)))
    #print(1/(1+np.exp((np.dot(w[:-1],x.T) + w[-1])*y)))

    logistic = linear_model.LogisticRegression(tol=1e-12)

    #print(logistic.fit(x,y))
    print(logistic.fit(data.data[:100,:], data.target[:100]))
    print(logistic.coef_, logistic.intercept_)

    print(logistic.coef_/w[:-1], logistic.intercept_/w[-1]) 
