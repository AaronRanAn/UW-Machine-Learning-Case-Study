# Note:

# The follow function is not a real neural network
# it is just a helper function to help you answer the quiz 6
# in UW's Coursera Machine Learning Specialization 
# But it does illustrate how a nnnet works

import numpy as np

# helper function to convert nparray into binary

def gen_l(dot):
    y = dot >0 
    return 1*y
    
def nnet(w10, w20, w30, w11 = 1, w22 = -1):
    
    l0 = np.array([ [1,0,1], [1,1,0], [1,1,1], [1,0,0] ])

    w1 = np.array([w10,w11,1])
    w2 = np.array([w20, -1, w22])
    w3 = np.array([w30, 1, 1])
    
    a0 = np.array([1,1,1,1])
    a1 = gen_l(np.dot(l0, w1))
    a2 = gen_l(np.dot(l0, w2))

    l1 = np.vstack((a0, a1, a2)).T
    l2_out = np.dot(l1, w3)

    return l2_out
    
print 'option A', nnet(-0.5,1.5,-0.5,-1,1) # option A
print 'option B', nnet(-1.5,0.5,-1.5) # option B
print 'option C', nnet(-1.5,1.5,-0.5) # option C
print 'option D', nnet(-1.5,0.5,-0.5) # option D
print 'option E', nnet(-1.5,1.5,-1.5) # option E
