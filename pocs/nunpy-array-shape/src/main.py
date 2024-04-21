import numpy as np

array = np.array([[[0, 1, 2, 3],
                   [4, 5, 6, 7]],

                   [[0, 1, 2, 3],
                   [4, 5, 6, 7]],

                   [[0 ,1 ,2, 3],
                   [4, 5, 6, 7]]])

def shape(array):
    print(array.shape)
    print("dimensions: ", array.ndim)
    print("size: ", array.size)

shape(array)