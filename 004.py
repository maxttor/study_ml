import numpy as np

X = np.array([[1,60], [1, 50], [1, 75]])
y = np.array([10, 7, 12])

step1 = X.T.dot(X)
# print(step1)
step2 = np.linalg.inv(step1)
# print(step2)
step3 = step2.dot(X.T)
# print(step3)
b = step3.dot(y)
print(b)