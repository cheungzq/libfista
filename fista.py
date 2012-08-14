import numpy as np

TOL = 1e-12
L0 = 1.0
ETA = 1.2
L1_reg = False

def set_fista_param(tol=1e-12, Li0=1.0, eta=1.2):
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

    while (True):
        der_at_y = derivative_func(cur_y)
        if backtracking:
            # find Lk
            ik = 0
            eta_ik = ETA
            obj_at_y = object_func(cur_y) - np.sum(np.abs(cur_y))
            cur_L = L
            squared_der_at_y = (der_at_y.dot(der_at_y)) 
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
            if low == 0:
                pl, ql = _pl_step(cur_y, L/ETA, der_at_y, squared_der_at_y, obj_at_y)
                if object_func(pl) <= ql:
                    L /= ETA
            else:
                L *= ETA**low 
        last_x = cur_x
        cur_x = _pl_step(cur_y, L, der_at_y)[0]
        last_t = cur_t
        cur_t = (1+np.sqrt(1+4*(cur_t**2)))/2
        last_y = cur_y
        cur_y = cur_x + (last_t - 1)/cur_t * (cur_x - last_x)
        step += 1
        if (np.abs(object_func(last_x) - object_func(cur_x)) < TOL):
            break

    return cur_x, step

