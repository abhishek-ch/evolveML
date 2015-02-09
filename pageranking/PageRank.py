__author__ = 'achoudhary'
import numpy as np
x = np.arange(9).reshape((3,3))
y = np.arange(3)

x = np.array([[0.3,0.3,0.3],[0.5,0,0.5],[0,0.5,0.5]])
# x = np.array([[0.3,0.5,0],[0.3,0,0.5],[0.3,0.5,0.5]])
y = np.array([0.33,0.33,0.33])


x = np.array([[0.47,0.47,0.06],[0.47,0.06,0.06],[0.06,0.47,0.87]])
x = np.array([[0.3,.3,.3],[.5,0,.5],[0,.5,.5]])

print(x)
print('==============OTHER===========')
x = np.array([[0.84,0.84,.84],[1,0.6,1],[0.6,1,1]]) #with beta added in above 0.8
x = np.array([[.1,.45,.45],[.1,.1,0.8],[.1,.1,.8]])
x1 = np.array([[0,0,0],[0.5,0,0],[0.5,1,1]])
rt = np.array([[.1,.1,.1],[.45,.1,.1],[.45,.8,.8]])
rt2 = np.array([[.045,.045,.895],[.47,.045,.045],[.47,.895,.045]])
rt3 = np.array([[0,0,1],[0.5,0,0],[.5,1,0]])
y = np.array([1,1,1])
print(rt2)
print("**********************\n")

#eigenvalues, eigenvectors = np.linalg.eig(x)
i = 0
#https://class.coursera.org/mmds-002/forum/thread?thread_id=49
print("\n\n\n")
while(i <20):
    y = np.around(np.dot(rt3,y),decimals=3)
    print(y)
    i += 1


