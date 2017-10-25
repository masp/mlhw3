import numpy
from numpy import linalg, newaxis, random
from matplotlib import collections, pyplot
def gen_rand_vecs(dims, num):
    vecs = random.normal(size=(num,dims))
    mags = linalg.norm(vecs, axis=-1)

    return vecs / mags[..., newaxis]

def main():
    ends = gen_rand_vecs(2, 1000)

    # Add 0 vector to start
    vectors = numpy.insert(ends[:, newaxis], 0, 0, axis=1)

    figure, axis = pyplot.subplots()
    axis.add_collection(collections.LineCollection(vectors))
    axis.axis((-1, 1, -1, 1))

    pyplot.show()

main()