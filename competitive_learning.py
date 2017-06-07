

import numpy as np
import pandas as pd


def unit_sphere_data_generator(M):
    
    np.random.seed(56)
    u = np.random.uniform(0,1,M)
    v = np.random.uniform(0,1,M)
    theta = 2 * np.pi * u
    phi= np.arccos((2*v - 1.0))
    
   
    c=np.zeros((M,3))
   
    
    c[:,0] = np.cos(theta) * np.sin(phi)
    c[:,1] = np.sin(theta) * np.sin(phi)
    c[:,2] =np.cos(phi)
    
 
   
    sphere_points = pd.DataFrame(c)
    
    ##filter if there are any duplicates
    sphere_points_no_duplicates = sphere_points.drop_duplicates()
    
    generators = sphere_points_no_duplicates.as_matrix()
    
    return generators


# In our case, we need 3 distinct data groups to make clustering easier.

# With the above generator function, I decided to generate uniformly distributed data points on unit sphere and then apply 3 different restrictions and choose 50 points for each different group.


points_on_sphere=unit_sphere_data_generator(2000)



distinct_data1=[]
distinct_data2=[]
distinct_data3=[]
for i in points_on_sphere:
    #If all coordinates are positive and above 0.05
    if np.all(i>0.05):
        distinct_data1.append(i)
    #If all coordinates are negative and belove -0.05
    elif np.all(i<-0.05):
        distinct_data2.append(i)
    #If x-axis negative and the others positive 
    #and between provided values
    elif i[0]<0 and i[1]>0 and i[1]<0.8 and i[2]>0 and i[2]<0.5:
        distinct_data3.append(i)
    



#take 50 points from first area
distinct_data1=distinct_data1[:50]

#take 40 points for train
data1_train=distinct_data1[:40]

#take remaining 10 points for test
data1_test=distinct_data1[40:]

#similar steps for the second and the third area's points.

distinct_data2=distinct_data2[:50]
data2_train=distinct_data2[:40]
data2_test=distinct_data2[40:]


#np.random.shuffle(distinct_data3)
distinct_data3=distinct_data3[:50]
data3_train=distinct_data3[:40]
data3_test=distinct_data3[40:]




#collect training data points
training_datas=np.vstack((data1_train,data2_train,data3_train))

#shuffle training data
np.random.seed(91)
np.random.shuffle(training_datas)



#collect testing data points
testing_datas=np.vstack((data1_test,data2_test,data3_test))

#shuffle testing data
np.random.seed(101)
np.random.shuffle(testing_datas)


#for ploting all data
all_data_points=np.vstack((training_datas,testing_datas))


import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from matplotlib import cm

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt3d=plt.subplot(projection='3d')

#plot unit sphere
u = np.linspace(0,2*np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u),np.cos(v))
y = np.outer(np.cos(u),np.sin(v))
z = np.outer(np.sin(u),np.ones(np.size(v)))

plt3d.plot_surface(x, y, z, color='w', alpha=0.3)


#plot data points on the unit sphere
plt3d.scatter(all_data_points[:,0],all_data_points[:,1],
              all_data_points[:,2], c='k')



plt3d.set_xlabel('X axis')
plt3d.set_ylabel('Y axis')
plt3d.set_zlabel('Z axis')
plt3d.view_init(0,-90)
plt.title('Before Clustering')
plt.show()


# **Train Phase**

from sklearn.preprocessing import normalize

def CompetitiveNetwork(X,epochs,eta=0.01):
    '''
    X:Input data points
    epochs: number of epochs
    eta: learning rate
    '''

    #number of data points
    N=X.shape[0]
    
    #it's convenient to choose
    #initial 3 weights randomly from data points 
    W=X[np.random.choice(np.arange(len(X)), 3), :]
    
    #store weights to plot weights trajectories
    trj=[W] 
    
    
    X2=(X**2).sum(axis=1)[:,np.newaxis]
    

    for epoch in range(epochs):
        
        #to store datas with assigned labels 
        assigned_labels=[]
        for i in range(N):
            distance=X2[i:i+1].T-2*np.dot(W,X[i:i+1,:].T)
            +(W**2).sum(axis=1)[:,np.newaxis]
            output=(distance==distance.min(axis=0)[np.newaxis,:]).T 
            output=output.astype("int")
            assigned_labels.append([X[i:i+1,:],output])
            #output=[1,0,0] if first class is winner and so on.
            
            #So multiplication with "output" in below, 
            #provides update for the winner weight only.
            
            #weight update
            W+= eta*(np.dot(output.T,X[i:i+1,:])
                     -output.sum(axis=0)[:,np.newaxis]*W)
            
            #normalize the weights
            W=normalize(W, norm='l2', axis=1)
          
        
        trj.append(W)
        #if weights doesn't change then break
        if (trj[epoch]==trj[epoch-1]).all():
            break
            
    return W,trj,assigned_labels



XX=training_datas
#epochs: 100



centers,trajectories,labelled_data=CompetitiveNetwork(XX,100,eta=0.01)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt3d=plt.subplot(projection='3d')

#plot unit sphere
plt3d.plot_surface(x, y, z, color='w', alpha=0.3)

#plot weight trajectories
for i in trajectories:
    plt3d.scatter(i[:,0],i[:,1],i[:,2], c='k', marker='x')
    plt3d.view_init(0,-90)
plt.title('Weight Trajectories')
plt.show()



#labelled_data has a form as given: [data_point, class]
#for example: 
#[array([[-0.89138016,  0.31313103,  0.3277047 ]]), array([[ 1.,  0.,  0.]])]
#I will assign classes as follows:
#pink class --> [1,0,0]
#blue class --> [0,1,0]
#green class --> [0,0,1]


pink_class=[]
blue_class=[]
green_class=[]
for i in labelled_data:
    if np.all(i[1][0]==[1,0,0]):
        pink_class.append(i[0][0])
    elif np.all(i[1][0]==[0,1,0]):
        blue_class.append(i[0][0])
    elif np.all(i[1][0]==[0,0,1]):
        green_class.append(i[0][0])



pink_class=np.array(pink_class)
blue_class=np.array(blue_class)
green_class=np.array(green_class)



#plot resulted clusters with the last weights

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt3d=plt.subplot(projection='3d')

#plot unit sphere
plt3d.plot_surface(x, y, z, color='w', alpha=0.3)

#plot data points
plt3d.scatter(pink_class[:,0],pink_class[:,1],pink_class[:,2], c='m')
plt3d.scatter(green_class[:,0],green_class[:,1],green_class[:,2], c='g')
plt3d.scatter(blue_class[:,0],blue_class[:,1],blue_class[:,2], c='b')


#plot centers
plt3d.scatter(centers[:,0],centers[:,1],centers[:,2], c='k', marker='x')



plt3d.set_xlabel('X axis')
plt3d.set_ylabel('Y axis')
plt3d.set_zlabel('Z axis')
plt3d.view_init(0,-90)
plt.title('Clustered data with their centers')
plt.show()


# **Test phase for unseen datas**


#In the test step we classify datas with provided centers 
#from training step.
#In the algorithm, determined centers should be given 
#and we no longer update weights.
def Classify(testX,c):
    '''
    testX: testing datas
    c: centers
    '''
    classes=[]
    X2=(testX**2).sum(1)[:,np.newaxis]
    for i in range(0,testX.shape[0],1):
        distance=X2[i:i+1].T-2*np.dot(c,testX[i:i+1,:].T)
        +(c**2).sum(1)[:,np.newaxis]
        output=(distance==distance.min(0)[np.newaxis,:]).T
        output=output.astype("int")
        classes.append([testX[i:i+1,:],output])
    return classes


test_classes=Classify(testing_datas,centers)


test_pink_class=[]
test_blue_class=[]
test_green_class=[]
for i in test_classes:
    if np.all(i[1][0]==[1,0,0]):
        test_pink_class.append(i[0][0])
    elif np.all(i[1][0]==[0,1,0]):
        test_blue_class.append(i[0][0])
    elif np.all(i[1][0]==[0,0,1]):
        test_green_class.append(i[0][0])



test_pink_class=np.array(test_pink_class)
test_blue_class=np.array(test_blue_class)
test_green_class=np.array(test_green_class)



#plot resulted clusters with the last weights

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt3d=plt.subplot(projection='3d')

#plot unit sphere
plt3d.plot_surface(x, y, z, color='w', alpha=0.3)

#plot data points
plt3d.scatter(test_pink_class[:,0],test_pink_class[:,1],
              test_pink_class[:,2], c='m',marker='^')
plt3d.scatter(test_green_class[:,0],test_green_class[:,1],
              test_green_class[:,2], c='g',marker='^')
plt3d.scatter(test_blue_class[:,0],test_blue_class[:,1],
              test_blue_class[:,2], c='b',marker='^')


#plot centers
plt3d.scatter(centers[:,0],centers[:,1],centers[:,2], c='k', marker='x')



plt3d.set_xlabel('X axis')
plt3d.set_ylabel('Y axis')
plt3d.set_zlabel('Z axis')
plt3d.view_init(0,-90)
plt.title('Classified test datas with given centers')
plt.show()


# We can observe that test data points are classified as expected color classes.

test_classes

