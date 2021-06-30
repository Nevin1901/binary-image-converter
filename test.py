import numpy as np
from numba import jit, cuda

@cuda.jit
def test(an_array):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x

    pos = tx + ty * bw
    if pos < an_array.size:
        an_array[pos] += 1


my_array = np.random.rand(200)

print(my_array)

test[200 + 31, 32](my_array)

print(my_array)


