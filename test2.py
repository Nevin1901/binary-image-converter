import cv2 as cv
import numpy as np
from numba import jit, cuda


img = cv.imread('gta.jpg')

scale = 25;


width = int(img.shape[1] * scale / 100)
height = int(img.shape[0] * scale / 100)
dim = (width, height)


convertedImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

resized = cv.resize(convertedImg, dim, interpolation = cv.INTER_AREA)

@cuda.jit
def checkImage(image):
    # i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    # k = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    x = cuda.threadIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.threadIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    print(x)
    # image[x][y] += 100
    # if (image[y][x] > 72):
    #     image[y][x] = 255
    # else:
    #     image[y][x] = 0
    #for i in range(len(image)):
    #    for k in range(len(image[i])):
    #        if (image[i][k] > maxColor):
    #            image[i][k] = 255;
    #        else:
    #            image[i][k] = 0;

maxColor = 72 
for b in range(256):
    newImg = resized.copy()
    checkImage[1, (32, 32)](newImg)
    maxColor += 1

    cuda.synchronize()
    # cv.imwrite('gta-' + str(maxColor) + '.png', hi)

    cv.destroyAllWindows()
    cv.imshow('frame', newImg)
    cv.waitKey()

