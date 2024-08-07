##FFN having 4 inputs which is passed into single node which is also the output.
import numpy as np
def neural_network(w, b, x):
    z = np.dot(w, x) + b
    a = 1 / (1 + np.exp(-z))
    return a
def main():
    ###Randomly defining weights, biases and inputs
    w = np.array([10, 20, -30, 15])
    b = np.array([2])
    x = np.array([1, 2, 3, 4])
    result = neural_network(w, b, x)
    #print(result)
if __name__=="__main__":
    main()



###FFN having 4 inputs which is passed into 2 hidden layers having 3 and 2 nodes respectively which gets passed into 1 output node.

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def neural_network(weights, biases, X):
    a = np.reshape(X, (len(X), 1))
    for w, b in zip(weights, biases):
        z = np.dot(w, a) + b
        a = sigmoid(z)
    return a
def main():
    ###Defining weights, biases and inputs using np.random.uniform
    weights = [np.random.uniform(size=(3, 4)), np.random.uniform(size=(2, 3)), np.random.uniform(size=(1, 2))]  ###w1,w2andw3
    biases = [np.random.uniform(size=(3, 1)), np.random.uniform(size=(2, 1)), np.random.uniform(size=(1, 1))]   ###b1,b2andb3
    inputs = [np.random.uniform(size=(4, 1))]
    for X in inputs:
        output = neural_network(weights, biases, X)
        #print(output)
if __name__ == "__main__":
    main()




##Back Propagation 1 (Hard code using symbol concept of sympy)
import sympy as sp
def back_prop(x, y, z):
    ###Defining symbolic variables
    x_sym, y_sym, z_sym = sp.symbols('x y z')
    ###Defining q and f symbolically
    q = x_sym + y_sym
    f = q * z_sym
    ###Computing the derivatives with respect to x, y, and z
    df_by_dx = sp.diff(f, x_sym)
    df_by_dy = sp.diff(f, y_sym)
    df_by_dz = sp.diff(f, z_sym)
    ###Substituting the values of x, y, z
    df_by_dx_val = df_by_dx.subs({x_sym: x, y_sym: y, z_sym: z})
    df_by_dy_val = df_by_dy.subs({x_sym: x, y_sym: y, z_sym: z})
    df_by_dz_val = df_by_dz.subs({x_sym: x, y_sym: y, z_sym: z})
    print(f"df/dx: {df_by_dx_val}")
    print(f"df/dy: {df_by_dy_val}")
    print(f"df/dz: {df_by_dz_val}")
def main():
    x = -2
    y = 5
    z = -4
    back_prop(x, y, z)
if __name__ == "__main__":
    main()



##Backpropagation 1 (calculating the gradient for each variable wrt f)----->HARD CODE
def dfbydz(df_by_dq, x, y, z, f, q):
    df_by_dz = q
    print(f"df_by_dz: {df_by_dz}")
    return df_by_dz
def dfbydy(df_by_dq, x, y, z, f, q):
    dq_by_dy = 1
    df_by_dy = df_by_dq * dq_by_dy
    print(f"df_by_dy: {df_by_dy}")
    result_dz = dfbydz(df_by_dq, x, y, z, f, q)
    return df_by_dy, result_dz
def dfbydx(df_by_dq, x, y, z, f, q):
    dq_by_dx = 1
    df_by_dx = df_by_dq * dq_by_dx
    print(f"df_by_dx: {df_by_dx}")
    result_dy, result_dz = dfbydy(df_by_dq, x, y, z, f, q)
    return df_by_dx, result_dy, result_dz
def dfbydq(df_by_df, x, y, z, f, q):
    df_by_dq = z * df_by_df
    print(f"df_by_dq: {df_by_dq}")
    result_dx, result_dy, result_dz = dfbydx(df_by_dq, x, y, z, f, q)
    return df_by_dq, result_dx, result_dy, result_dz
def dfbydf(x, y, z, f, q):
    df_by_df = 1
    print(f"df_by_df: {df_by_df}")
    result_dq, result_dx, result_dy, result_dz = dfbydq(df_by_df, x, y, z, f, q)
    return df_by_df, result_dq, result_dx, result_dy, result_dz
def main():
    x = -2
    y = 5
    z = -4
    q = x + y
    f = q * z
    dfbydf(x, y, z, f, q)
if __name__ == "__main__":
    main()


###Back propagation 2 (HARD CODE)
def dfbydw(df_by_dr, x, y, z, w, q, r, s, f):
    dr_dw = 0
    df_by_dw = df_by_dr * dr_dw
    print(f"df_by_dw: {df_by_dw}")
    return df_by_dw
def dfbydz(df_by_dr,x, y, z, w, q, r, s, f):
    dr_dz = 1
    df_by_dz = df_by_dr * dr_dz
    print(f"df_by_dz: {df_by_dz}")
    result_dw = dfbydw(df_by_dr, x, y, z, w, q, r, s, f)
    return df_by_dz, result_dw
def dfbydy(df_by_dq, df_by_dr,x, y, z, w, q, r, s, f):
    dq_dy = x
    df_by_dy = df_by_dq * dq_dy
    print(f"df_by_dy: {df_by_dy}")
    result_dz, result_dw = dfbydz(df_by_dr,x, y, z, w, q, r, s, f)
    return df_by_dy, result_dz, result_dw
def dfbydx(df_by_dq, df_by_dr ,x, y, z, w, q, r, s, f):
    dq_dx = y
    df_by_dx = df_by_dq * dq_dx
    print(f"df_by_dx: {df_by_dx}")
    result_dy, result_dz, result_dw = dfbydy(df_by_dq, df_by_dr,x, y, z, w, q, r, s, f)
    return df_by_dx, result_dy, result_dz, result_dw
def dfbydr(df_by_ds,df_by_dq, x, y, z, w, q, r, s, f):
    ds_dr = 1
    df_by_dr = df_by_ds * ds_dr
    print(f"df_by_dr: {df_by_dr}")
    result_dx, result_dy, result_dz, result_dw = dfbydx(df_by_dq,df_by_dr ,x, y, z, w, q, r, s, f)
    return df_by_dr, result_dx, result_dy, result_dz, result_dw
def dfbydq(df_by_ds, x, y, z, w, q, r, s, f):
    ds_dq = 1
    df_by_dq = df_by_ds * ds_dq
    print(f"df_by_dq: {df_by_dq}")
    result_dr, result_dx, result_dy, result_dz, result_dw = dfbydr(df_by_ds,df_by_dq, x, y, z, w, q, r, s, f)
    return df_by_dq, result_dr, result_dx, result_dy, result_dz, result_dw
def dfbyds(df_by_df,x, y, z, w, q, r, s, f):
    df_ds = 2
    df_by_ds = df_ds * df_by_df
    print(f"df_by_ds: {df_by_ds}")
    result_dq, result_dr, result_dx, result_dy, result_dz, result_dw = dfbydq(df_by_ds, x, y, z, w, q, r, s, f)
    return df_by_ds, result_dq, result_dr, result_dx, result_dy, result_dz, result_dw
def dfbydf(x, y, z, w, q, r, s, f):
    df_by_df = 1
    print(f"df_by_df: {df_by_df}")
    result_ds, result_dq, result_dr,result_dx, result_dy, result_dz, result_dw = dfbyds(df_by_df,x, y, z, w, q, r, s, f)
    return df_by_df, result_ds, result_dq, result_dr,result_dx, result_dy, result_dz, result_dw
def main():
    x = 3
    y = -4
    z = 2
    w = -1
    q = x*y
    r = max(z, w)
    s = q + r
    f = s * 2
    dfbydf(x, y, z, w, q, r, s, f)
if __name__=="__main__":
    main()

