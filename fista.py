import math

ERR = 1e-12
L0 = 1.0
eta = 1.2

def fista_solver(object_func, derivative_func, x0, L=None):
    cur_y = x0
    if L is None:
        backtracking = True
        L = float(L0)
    else:
        backtracking = False
        L = float(L)
    cur_x = x0
    cur_t = 1
    step = 0

    while (True):
        # find Lk
        ik = 0
        eta_ik = eta

        der_at_y = derivative_func(cur_y)
        if backtracking:
            obj_at_y = object_func(cur_y)
            cur_L = L
            squared_norm_df = (der_at_y.dot(der_at_y)) 
            while True:
                pl = cur_y - 1/cur_L*der_at_y
                ql = -1/(2*cur_L)*squared_norm_df + obj_at_y
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
#            print low, L 
            while (low < high):
                mid = low + (high-low)/2
                cur_L = L * (eta**mid )
                pl = cur_y - 1/cur_L*der_at_y
                ql = -1/(2*cur_L)*squared_norm_df + obj_at_y
                if (object_func(pl) <= ql):
                    high = mid
                else:
                    low = mid+1
            L *= eta**low 

        last_x = cur_x
        cur_x = (cur_y - 1/L * der_at_y)
        last_t = cur_t
        cur_t = (1+math.sqrt(1+4*(cur_t**2)))/2
        last_y = cur_y
        cur_y = cur_x + (last_t - 1)/cur_t * (cur_x - last_x)

        step += 1
        if (object_func(last_x) - object_func(cur_x) < ERR):
            break

    return cur_x, step

