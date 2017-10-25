from svmcmpl import softmargin
import numpy as np
import cvxopt as co

train = np.genfromtxt('data/MNIST-13.csv', delimiter=",")
data = train[:, 1:]
labels = train[:, 0:1]
labels[labels == 1] = -1
labels[labels == 3] = 1
print data.shape

def solve_svm(x, y, C):
    N, D = x.shape
    Y = y
    Y = np.matrix(Y)
    outer_prod = co.matrix(Y.dot(Y.T))
    x = co.matrix(x.copy())
    y = co.matrix(y.copy())

    # Q is the kernel matrix where each element i,j is k(i,j) or x_i' * x_j
    Q = co.matrix(0.0, size=(N,N))
    dotX = co.matrix(0.0, size=(N,N))
    # X * X_T = D
    # Y * Y_T = S
    # Q = S * D
    co.blas.syrk(x, dotX)
    Q = co.mul(outer_prod, dotX)
    q = co.matrix(-1.0, (N, 1))

    # G for inequalities
    # First half: alpha_i <= C
    # Second half: alpha_i >= 0
    G = co.spmatrix([], [], [], size=(2*N,N))
    G[::2*N+1], G[N::2*N+1] = co.matrix(1.0, (N,1)), co.matrix(-1.0, (N,1)) # set diagonal to -1 to flip equality sign
    h = co.matrix(0.0, (2*N,1))
    h[:N] = C

    # Sum of alpha times y = 0
    A = co.matrix(y, (1, N))

    sol = co.solvers.qp(Q, q, G, h, A, co.matrix(0.0))
    alpha = sol['x']
    b = sol['y'][0]
    max_alpha = max(abs(alpha))
    print max_alpha
    svec = [ k for k in range(N) if abs(alpha[k]) > 1e-5 * max_alpha ]
    print "Support vectors: %d" % len(svec)
    # w = X_t * (alpha .* y)
    w = co.matrix(0.0, (D, 1))
    co.blas.gemv(x, co.mul(alpha, y), w, trans='T')
    def predict(new_x):
        projected = co.blas.dotu(w, new_x)
        return np.sign(projected + b)

    return w, predict, alpha

#sol_comp = softmargin(co.matrix(data), co.matrix(labels), 0.1)
w, predict, alpha = solve_svm(data, labels, 0.1)
diff = np.array([ predict(co.matrix(data[i])) == labels[i] for i in range(data.shape[0]) ])
print diff.sum()
