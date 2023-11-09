import numpy as np
a = np.array([1, 2, 3, 4])
b = np.array([1, 2, 4, 5])
c = np.concatenate((a.reshape(1, -1), b.reshape(1, -1)), axis= 0)
print(c)
