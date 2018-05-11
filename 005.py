import urllib
from urllib import request
import numpy as np

fname = input()  # read file name from stdin
# fname = 'https://stepic.org/media/attachments/lesson/16462/boston_houses.csv'
f = urllib.request.urlopen(fname)  # open file from URL
data = np.loadtxt(f, delimiter=',', skiprows=1)  # load data to work with

# here goes your solution
x = np.hstack((np.ones_like(data[:,0:1]), data[:,1:]))
y = data[:,0:1]

step1 = x.T.dot(x)
step2 = np.linalg.inv(step1)
step3 = step2.dot(x.T)
b = step3.dot(y)
print(" ".join(map(str, b[:,0])))