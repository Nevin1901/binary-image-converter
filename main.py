import cv2 as cv
from numba import cuda, jit 
import numpy as np
import sys 

img = cv.imread('gta.jpg')

scale = 100; 

width = int(img.shape[1] * scale / 100)
height = int(img.shape[0] * scale / 100)
dim = (width, height)

numBlocks = (height, width)
threadsPerBlock = (1, 1)

resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)

greyscale = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)

print(resized.shape)

amount = 70

@cuda.jit
def binaryify(image, amt):
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x <= height and y <= width:
        if image[x][y] > amt:
            image[x][y] = 255 
        else:
            image[x][y] = 0 

while(1):
    binaryImage = greyscale.copy()
    binaryify[numBlocks, threadsPerBlock ](binaryImage, amount)
    cuda.synchronize()
    cv.imshow('frame', binaryImage)
    print(amount)
    k = cv.waitKey()
    if k == 81 and amount >= 5: 
        amount -= 5
    if k == 83 and amount <= 250: 
        amount += 5
    if k == ord('q'):
        quit()
