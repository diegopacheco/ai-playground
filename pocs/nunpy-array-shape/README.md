### Result

*ndarray.ndim* will tell you the number of axes, or dimensions, of the array.

*ndarray.size* will tell you the total number of elements of the array. This is the product of the elements of the array’s shape.

*ndarray.shape* will display a tuple of integers that indicate the 
number of elements stored along each dimension of the array. If, for example, you have a 2-D array with 2 rows and 3 columns, the shape of your array is (2, 3).

```
array = np.array([[[0, 1, 2, 3],
                   [4, 5, 6, 7]],

                   [[0, 1, 2, 3],
                   [4, 5, 6, 7]],

                   [[0 ,1 ,2, 3],
                   [4, 5, 6, 7]]])
```
```
(3, 2, 4)
dimensions:  3
size:  24
```