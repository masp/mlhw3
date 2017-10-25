import numpy as np
import time

NUM_GRADS = 100

# Credit to answer here: https://codereview.stackexchange.com/questions/77927/generate-random-unit-vectors-around-circle
def gen_rand_vec(dims):
    vecs = np.random.normal(size=(1,dims))
    mags = np.linalg.norm(vecs, axis=-1)

    return vecs / mags[..., np.newaxis]

def subsample(x, y, k):
    idx = np.random.choice(x.shape[0], k)
    return x[idx], y[idx]

def myPegasos(filename, k, numruns):
    train = np.genfromtxt('data/MNIST-13.csv', delimiter=",")
    data = train[:, 1:]
    labels = train[:, 0:1]
    labels[labels == 1] = -1
    labels[labels == 3] = 1
    print len(labels), len(labels[labels==1]), len(labels[labels==-1])
    x = np.matrix(data)
    y = np.matrix(labels)
    print x.shape, y.shape
    N = data.shape[0]
    D = data.shape[1]

    l = 1 # lambda
    start = time.time()
    w_est = np.matrix(gen_rand_vec(D)).T
    for t in range(NUM_GRADS * N):
        A_sub, y_sub = subsample(x, y, k)
        A_test = np.multiply(A_sub.dot(w_est), y_sub)
        pos_idx = (np.array(A_test).flatten() < 1)
        A_pos, y_pos = A_sub[pos_idx], y_sub[pos_idx]
        eta = 1.0 / ((t+1) * l)
        # Descend
        w_est = (1 - 1/(t+1)) * w_est + (eta/k) * (A_pos.T.dot(y_pos))
        # Project
        w_est = w_est / min(1, np.linalg.norm(w_est))

    end = time.time()
    print w_est
    print "duration:", (end-start)

    out = np.sign(x.dot(w_est))
    err = y[out != y]
    print "error rate: %d" % (len(err) / N)

myPegasos("", 1, 1)
