# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 11:57:42 2021

@author: lefin
"""

import numpy as np
import skimage.measure
from timeit import timeit

array = np.array([[-0.00153485,  0.00150026],
                  [-0.00292359,  0.00326345]])

array = np.array([[1,2,3,4,6,7],
                  [3,4,1,2,5,1],
                  [1,2,4,5,2,3],
                  [3,4,5,6,8,9]])
''''''

def f(x):
    # return math.sqrt(x)
    return np.sqrt(x)


vf = np.vectorize(f)


def step_nondestructive(array, thresh0 = 0, thresh1 = 1, **args):
    return (array > thresh0).astype(float) * thresh1

def relu_nondesructive(array, thresh0 = 0, **args):
    return (array > thresh0).astype(float) * array

def step(array, thresh0 = 0, thresh1 = 1, **args):
    array[array <= thresh0] = thresh0
    array[array > thresh0] = thresh1
    return array
    

def relu(array, thresh0 = 0, **args):
    array[array <= thresh0] = thresh0
    return array


def leakrelu(array, thresh0 = 0, leak = 0.001, **args):
    product_array = (array > thresh0).astype(float)
    product_array[product_array == 0] = leak
    return array * product_array

def sigmoid(array, c = 1, **args):
    return np.reciprocal(1+np.exp(-c*array))

def swish(array,c = 1,**args):
    return array/(1+np.exp(-c*array))

def softmax(array, **args):
    return(np.exp(array)/np.sum(np.exp(array),axis = 1)[:,None])

def maxpoolOriginal(array, stride = 2, **args): #full original method
    rel0,rel1 = int(array.shape[0]/stride),int(array.shape[1]/stride)
    #print(np.concatenate(np.hsplit(np.array(np.hsplit(array,rel1)),rel0)))
    return np.reshape(np.amax(
                        np.amax(
                            np.concatenate(
                                np.hsplit(
                                    np.array(
                                        np.hsplit(array,rel1)
                                        ),rel0)
                                ),axis = 1),axis=1),(rel0,rel1))

def blockshaped(arr, nrows, ncols):
    """
    by unutbu on StackOverflow
    
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def maxpool(array, stride = 2, **args): #the most efficient method (reshape)
    rel0,rel1 = int(array.shape[0]/stride),int(array.shape[1]/stride)
    h, w = array.shape
    #print(np.concatenate(np.hsplit(np.array(np.hsplit(array,rel1)),rel0)))
    return np.reshape(np.amax(
                        np.amax(
                            array.reshape(h//stride, stride, -1, stride)
                               .swapaxes(1,2)
                               .reshape(-1, stride, stride)
                                ,axis = 1),axis=1),(rel0,rel1))


def maxpoolsk(array, stride = 2, **args): #skimage method (slowest)
    return skimage.measure.block_reduce(array, (stride,stride), np.max)

def tanh(array, c = 1, **args):
    pos,neg = np.exp(c*array),np.exp(-c*array)
    return (pos-neg)/(pos+neg)


print(swish(array))
'''
def test1():
    relu(array)

def test2():
    leakrelu(array)

def test3():
    swish(array)

#(maxpool(array))

print(timeit(test1,number=10000))
print(timeit(test2,number=10000))
print(timeit(test3,number=10000))
'''