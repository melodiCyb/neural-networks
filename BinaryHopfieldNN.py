zero="""
XXXXXXXX
X______X
X______X
X______X
X______X
X______X
X______X
XXXXXXXX
"""
one="""
____X___
____X___
____X___
____X___
____X___
____X___
____X___
____X___
"""

two="""
XXXXXXXX
_______X
_______X
XXXXXXXX
X_______
X_______
XXXXXXXX
________
"""

three="""
________
XXXXXXXX
_______X
_______X
XXXXXXXX
_______X
_______X
XXXXXXXX
"""
def bi_vec_pattern(numeral):
    from numpy import array
    return array([+1 if i=='X' else -1 for i in numeral.replace('\n','')])

bipolar_0=bi_vec_pattern(zero)
bipolar_1=bi_vec_pattern(one)
bipolar_2=bi_vec_pattern(two)
bipolar_3=bi_vec_pattern(three)

def visualize(pattern):
    from pylab import imshow, cm, show
    imshow(pattern.reshape((8,8)),cmap=cm.binary, interpolation='nearest')
    show()
    
visualize(bipolar_0)
visualize(bipolar_1)
visualize(bipolar_2)
visualize(bipolar_3)
    
import numpy as np
#store sample patterns as rows of the matrix sample_patterns
sample_patterns = np.array([bipolar_0,bipolar_1,bipolar_2,bipolar_3])

def network_train(sample_patterns):
    from numpy import zeros, outer, diag_indices 
    i=sample_patterns.shape[1]
    j=sample_patterns.shape[0]
    T = zeros((i,i)) #create 8*8 zero matrix
    
    #weight matrix as summations of outer products of sample vectors.
    for s in sample_patterns:
        T = T + outer(s,s)
        
    #set diagonal entries to zero to avoid self inputs.    
    T[diag_indices(i)] = 0
    
    return T/j #normalization   

T=network_train(sample_patterns)

from numpy import vectorize, dot
sgn = vectorize(lambda x: -1 if x<0 else +1)

def classify(T, test_patterns, iteration_steps):
    for i in xrange(iteration_steps):        
        test_patterns = sgn(dot(test_patterns,T)) 
    return test_patterns

zero_test_1="""
XXXXXXX_
X______X
X___X___
X_______
X______X
_______X
X______X
XXXXXX_X
"""

one_test_1="""
_____XXX
_______X
_______X
________
_______X
_______X
_______X
______X_
"""

two_test_1="""
XXXXXXXX
______X_
_______X
X__XXXXX
X_______
X_______
XXXXXXXX
________
"""

three_test_1="""
________
X___XXXX
_______X
_______X
X_XXXXXX
_______X
__X____X
XX_XXXXX
"""

#Stores test patterns as rows in test_patterns matrix.

test_patterns_1= np.array([bi_vec_pattern(zero_test_1),
                           bi_vec_pattern(one_test_1),
                           bi_vec_pattern(two_test_1),
                           bi_vec_pattern(three_test_1)])
    
for i in range(4):
    visualize(test_patterns_1[i])

result_1=classify(T, test_patterns_1, 1)
result_2=classify(T, test_patterns_1, 2)
result_3=classify(T, test_patterns_1, 3)

for i in range(4):
    visualize(result_1[i])
    
for i in range(4):
    visualize(result_2[i])

for i in range(4):
    visualize(result_3[i])

from sklearn.datasets import load_digits
#8*8 images of digits data set from sklearn 
digits = load_digits()
print(digits.data.shape)

def to_bipolar(img, lower, upper): #convert images to bipolar arrays
    img=(lower < img) & (img < upper)
    img=img.astype(int)
    img=np.asarray([j for i in img for j in i])
    img[img==0]=-1
    return img

zero_test_3=to_bipolar(digits.images[0],10,100)
one_test_3=to_bipolar(digits.images[1],10,100)
two_test_3=to_bipolar(digits.images[2],10,100)
three_test_3=to_bipolar(digits.images[3],10,100)

visualize(zero_test_3)
visualize(one_test_3)
visualize(two_test_3)
visualize(three_test_3)

test_patterns_3 = [zero_test_3, one_test_3,two_test_3,three_test_3]

result1_=classify(T, test_patterns_3, 1)
result2_=classify(T, test_patterns_3, 2)

for i in range(4):
    visualize(result_1[i])


import random
for i in range(64):
    while random.random()<0.4:
        bipolar_0[i]=-bipolar_0[i]
        bipolar_1[i]=-bipolar_1[i]
        bipolar_2[i]=-bipolar_2[i]
        bipolar_3[i]=-bipolar_3[i]
        
a,b,c,d=bipolar_0,bipolar_1,bipolar_2,bipolar_3
test_patternsw= np.array([a,b,c,d])
for i in [a,b,c,d]:
    visualize(i)
r1=classify(T, test_patternsw, 1)
r2=classify(T, test_patternsw, 2)
r3=classify(T, test_patternsw, 3)
for i in range(4):
    visualize(r1[i])

for i in range(4):
    visualize(r2[i])
for i in range(4):
    visualize(r3[i])
