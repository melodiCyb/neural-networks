
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(123)

mu, sigma = 0, 0.1
#zero-mean Gaussian noise with standard deviation 0.1
noise= np.random.normal(mu, sigma, 10)

#10 random data points
data=np.random.rand(10)*4 

#original parameter values
b=np.array([1.1, 0.45, 0.1])

y=[]
for u in data:
    y.append(b[0]+b[1]*u+b[2]*(u**2)) 

for i in range(len(y)):
    y[i]+=noise[i]


data


y #original outputs


import numpy as np
plt.scatter(data, y)
plt.show()


modelA=[0]
modelB=[0,1]
modelC=[0,1,2]
modelD=[0,1,2,3]


# **Model a**


resultA = np.array([x**p for x in data for p in modelA])
resultA=resultA.reshape(10,1)


X_A=resultA
X_A1=np.linalg.inv(np.dot(X_A.T,X_A))
X_A2=np.dot(X_A.T,y)
X_A3=np.dot(X_A1,X_A2)
a0=X_A3



print "a0 :" , a0[0]


def model_a(coefficient):
    y_a=[]
    for i in range(10):
        a_=coefficient*(data[i]**0)
        y_a.append(a_)
    return y_a



y_a=model_a(a0[0])


Error_a=(np.array(y)-np.array(y_a)).T


Cost_a=0.5*Error_a.T.dot(Error_a)


plt.plot(data, y, 'o', label='Original data', markersize=10)
plt.plot(data, y_a, 'r', label='Fitted line')
plt.legend()
plt.title('Model A')
plt.show()


# **Model b**


resultB = np.array([x**p for x in data for p in modelB])
resultB=resultB.reshape(10,2)


X_B=resultB
X_B1=np.linalg.inv(np.dot(X_B.T,X_B))
X_B2=np.dot(X_B.T,y)
X_B3=np.dot(X_B1,X_B2)
[b0,b1]=X_B3



print "b0 :" , b0
print "b1 :" , b1



def model_b(coefficient):
    y_b=[]
    for i in range(10):
        b_=coefficient[0]+coefficient[1]*data[i]
        y_b.append(b_)
    return y_b


y_b=model_b([b0,b1])



Error_b=(np.array(y)-np.array(y_b)).T

Cost_b=0.5*Error_b.T.dot(Error_b)


plt.plot(data, y, 'o', label='Original data', markersize=10)
plt.plot(data, y_b , 'r', label='Fitted line')
plt.legend()
plt.title('Model B')
plt.show()


# **Model c**


resultC = np.array([x**p for x in data for p in modelC])
resultC=resultC.reshape(10,3)


X_C=resultC
X_C1=np.linalg.inv(np.dot(X_C.T,X_C))
X_C2=np.dot(X_C.T,y)
X_C3=np.dot(X_C1,X_C2)
[c0,c1,c2]=X_C3



print "c0 :" ,c0
print "c1 :" ,c1
print "c2 :" ,c2


def model_c(coefficient):
    y_c=[]
    for i in range(10):
        c_=coefficient[0]+coefficient[1]*data[i]+coefficient[2]*(data[i]**2)
        y_c.append(c_)
    return y_c


y_c=model_c([c0,c1,c2])

Error_c=(np.array(y)-np.array(y_c)).T

Cost_c=0.5*Error_c.T.dot(Error_c)


plt.plot(data, y, 'o', label='Original data', markersize=10)
x = np.linspace(0, 3.5, 1000)
plt.plot(x, c0+x*c1+(x**2)*c2 , 'r', label='Fitted line')
plt.legend()
plt.title('Model C')
plt.show()


# **Model d**

resultD = np.array([x**p for x in data for p in modelD])
resultD=resultD.reshape(10,4)


X_D=resultD
X_D1=np.linalg.inv(np.dot(X_D.T,X_D))
X_D2=np.dot(X_D.T,y)
X_D3=np.dot(X_D1,X_D2)
[d0,d1,d2,d3]=X_D3


print "d0 :" ,d0
print "d1 :" ,d1
print "d2 :" ,d2
print "d3 :" ,d3


def model_d(coefficient):
    y_d=[]
    for i in range(10):
        d_=coefficient[0]+coefficient[1]*data[i]+coefficient[2]*(data[i]**2)+coefficient[3]*(data[i]**3)
        y_d.append(d_)
    return y_d


y_d=model_d([d0,d1,d2,d3])


Error_d=(np.array(y)-np.array(y_d)).T



Cost_d=0.5*Error_d.T.dot(Error_d)


plt.plot(data, y, 'o', label='Original data', markersize=10)
x = np.linspace(0, 3.5, 1000)
plt.plot(x, d0+x*d1+(x**2)*d2+(x**3)*d3  , 'r', label='Fitted line')
plt.legend()
plt.title('Model D')
plt.show()



