import numpy as np

arr1 = np.array([])
arr2 = np.array([4, 5, 6])
arr3 = np.array([7, 8, 9])

result = np.concatenate((arr1, arr2, arr3))

print(result)