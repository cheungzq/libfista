import numpy as np

TOL = 1e-12
L0 = 1.0
ETA = 1.2
L1_reg = False

def set_fista_param(tol=1e-12, Li0=1.0, eta=1.2):
    ''' set some parameters used in FISTA algorithm

        tol -- used as the stop criterion, in fact iteration stops when the
               objective values in two consecutive steps are within tol.
        Li0 -- initial value of Lipschitz constant when the backtracking
               procedure is used.
        eta -- the probing step used in backtracking procedure
    '''
    global TOL, L0, ETA
    TOL = tol
    L0 = float(Li0)
    ETA = float(eta)

def _pl_step(y, L, der_at_y, squared_der_at_y = 0, obj_at_y = 0):
    pl = y - 1/L * der_at_y
    ql = -1/(2*L)*squared_der_at_y + obj_at_y
    if L1_reg:
        sindex = np.abs(pl)>(1/L)
        # deep copy
        dl = pl.copy()
        pl[np.logical_not(sindex)] = 0
        pl[sindex] = (np.abs(pl[sindex])-1/L) * np.sign(pl[sindex])
        dl -= pl
        ql += L/2 * np.dot(dl,dl) + np.sum(np.abs(pl))
    return pl,ql

def fista_solve(object_func, derivative_func, x0, L=None, with_L1_reg = False):
    ''' this implements the FISTA algorithm to minimize a unconstrained convex
    function having the following form:
        object_func = f(x) + g(x)
    where f(x) has a first order derivation, and g(x) is L1 norm for now.
        
        object_func -- objective function as stated above
        derivative_func -- the first order derivative of f(x)
        x0 -- initial x
        L -- Lipschitz constant of f(x), if None, backtracking procedure is
             used.
        with_L1_reg -- whether or not objective function has L1 part(g(x))
       
        return -- (optimal x, optimal objective value, iterate numbers)
    '''
    if L is None:
        backtracking = True
        L = float(L0)
    else:
        backtracking = False
        L = float(L)

    global L1_reg
    L1_reg = with_L1_reg

    cur_y = x0
    cur_x = x0
    cur_t = 1
    step = 0
    last_obj_v = object_func(x0)

    while (True):
        der_at_y = derivative_func(cur_y)
        if backtracking:
            ik = 0
            eta_ik = ETA
            obj_at_y = object_func(cur_y)
            # if L1 reg is used, we cancel the L1 part to get the value of  smooth
            # part
            if L1_reg:
                obj_at_y -= np.sum(np.abs(cur_y))
            cur_L = L
            squared_der_at_y = (der_at_y.dot(der_at_y)) 
            # use binary search to find the smallest L
            # s.t. F(pl(y)) <= G(pl(y),y)
            while True:
                pl, ql = _pl_step(cur_y, cur_L, der_at_y, squared_der_at_y, obj_at_y)
                if (object_func(pl) <= ql):
                    break
                cur_L *= eta_ik
                if (ik==0):
                    ik += 1
                else:
                    ik *= 2
                    eta_ik **= 2
            low = ik/2
            high = ik
            while (low < high):
                mid = low + (high-low)/2
                cur_L = L * (ETA**mid)
                pl, ql = _pl_step(cur_y, cur_L, der_at_y, squared_der_at_y, obj_at_y)
                if (object_func(pl) <= ql):
                    high = mid
                else:
                    low = mid+1
            # if L need not to increase, test if it can be decreased. so we that can
            # get faster convergence rate.
            if low == 0:
                pl, ql = _pl_step(cur_y, L/ETA, der_at_y, squared_der_at_y, obj_at_y)
                if object_func(pl) <= ql:
                    L /= ETA
            else:
                L *= ETA**low 
        last_x = cur_x
        cur_x = _pl_step(cur_y, L, der_at_y)[0]
        cur_obj_v = object_func(cur_x)
        if (np.abs(last_obj_v - cur_obj_v) < TOL):
            break
        last_obj_v = cur_obj_v
        last_t = cur_t
        cur_t = (1+np.sqrt(1+4*(cur_t**2)))/2
        last_y = cur_y
        cur_y = cur_x + (last_t - 1)/cur_t * (cur_x - last_x)
        step += 1

    return cur_x, cur_obj_v, step

