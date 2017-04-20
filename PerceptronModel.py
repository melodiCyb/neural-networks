import numpy as np

#Class A 0< x_{1},x_{2},x_{3} <5 
#Class B -5 < x_{1},x_{2},x_{3} <0 
#Pick  25  random  samples  for each class
A=np.random.rand(25,3)*5
B=-np.random.rand(25,3)*5
                 


# Plot these 50 data points in a 3-D plot with different representation points.
import matplotlib.pyplot as plt
plt3d=plt.subplot(projection='3d')
plt3d.scatter(A[:, 0], A[:, 1], A[:, 2], c='r')
plt3d.scatter(B[:, 0], B[:, 1], B[:, 2], c='b')
plt3d.set_xlabel('X axis')
plt3d.set_ylabel('Y axis')
plt3d.set_zlabel('Z axis')
plt.show()



#We store the desired values at the end of every point as: [x_{1},x_{2},x_{3}, d]
#d_{i}=1 for Class A , d_{i}=-1  for  Class  B

A_with_d = [np.append(i,1) for i in A]
B_with_d = [np.append(i,-1) for i in B]

train_data=np.vstack((A_with_d,B_with_d))



sgn = lambda x: -1.0 if x<0 else 1.0
def Output(x, weights):
    y = weights[0] # -bias
    for i in range(len(x)-1):
        y += weights[i + 1] * x[i]
    return sgn(y)



#Adopt the weights
def adopt_weights(sample_data, eta, epochs):
    import pylab
    #weights = np.random.rand(4) #initialize weights to small numbers
    
    #initialize weights to zero 
    weights = [0.0 for i in range(len(sample_data[0]))] 
    #-bias is placed to 0th index of weights
    c=[]
    epoch=[]
    for n in range(epochs):
        cost = 0.0
        for x in sample_data:
            output = Output(x, weights)
            error = x[-1] - output #desired value - output
            cost += error**2
            weights[0] = weights[0] + eta * error #update bias
            for i in range(len(x)-1):
                weights[i + 1] = weights[i + 1] + eta * error * x[i]
        c.append(cost)
        epoch.append(n)
        print '>n=%d, cost=%f' % (n, cost)
    #Plot the cost function value vs the number of epochs
    pylab.plot(epoch, c) 
    pylab.xlabel('Number of Epochs')
    pylab.ylabel('Cost')
    pylab.show()
    return weights


epochs=5
eta=0.1 #learning constant
W=adopt_weights(train_data, eta, epochs)


print "-Bias: ", W[0]
print "w1: ",W[1]
print "w2: " ,W[2]
print "w3:" ,W[3]


#Plot the hyperplane in a 3-D plot showing the separation of two classes

theta=W[0] #-bias
xx,yy=np.meshgrid(range(-5,5),range(-5,5))


zz=(-theta-W[1]*xx-W[2]*yy)*1.0/W[3]



plt3d=plt.subplot(projection='3d')
plt3d.plot_wireframe(xx,yy,zz,rstride=1,cstride=1,color="purple")
plt3d.scatter(A[:, 0], A[:, 1], A[:, 2], c='r')
plt3d.scatter(B[:, 0], B[:, 1], B[:, 2], c='b')
plt3d.set_xlabel('X axis')
plt3d.set_ylabel('Y axis')
plt3d.set_zlabel('Z axis')
plt.show()


xx1,yy1=np.meshgrid(range(-5,5),range(-5,5))
zz1=(-theta-W[1]*xx1-W[2]*yy1)*1.0/W[3]

fig=plt.figure()
ax=fig.add_subplot(111, projection='3d')

ax.plot_surface(xx1,yy1,zz1,rstride=1,cstride=1,color="purple");
ax.scatter(A[:, 0], A[:, 1], A[:, 2], c='r');
ax.scatter(B[:, 0], B[:, 1], B[:, 2], c='b');
ax.view_init(30,-90)
plt.show()
#rotated the cube to see separation clearly.


#classify data points with trained model.
def classify(test_data):
    for x in test_data:
        output = Output(x, W)
        print 'Test Point:', x[:3], " Desired=%d, Output=%d" % ( x[-1], output)

test_data=np.array([[2,3,4,1],[1,3,2,1],[ -2,-3,-4,-1],[-1,-3,-2,-1]])




classify(test_data) #test the model

