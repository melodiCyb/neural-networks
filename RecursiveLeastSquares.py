# ### Estimation with Recursive Least Squares



def recursive_leastsquares(N,inputs,outputs):
    import numpy as np
    #Initial values of coefficients are set to zero
    Theta=np.zeros((N,1)) 
    
    #Gain matrix initialization
    #It requires large values
    P=np.eye(N)*1000

    #Regressor initialization
    X = np.ones((10,N))
    
    #Estimated output generations
    #Here only our 4 models are considered.
    if N==1:
        X=X
    
    elif N==2:
         for i in range(10):
            X[i,1] = inputs[i]
    
    elif N==3:
        for i in range(10):
            X[i,1] = inputs[i]
            X[i,2] = inputs[i]**2
    
    else:
        for i in range(10):
            X[i,1] = inputs[i]
            X[i,2] = inputs[i]**2
            X[i,3] = inputs[i]**3
            
    #Recursion part
    for n in range(10):
        
        R=np.array([X[n,:]]) 
        K=P.dot(R.T)/(1+np.dot(R,P).dot(R.T)) #Equation (3)
        P=P-K*R.dot(P) #Equation (4)
        E=outputs[n]-R.dot(Theta) #Equation (5)
        Theta=Theta+K*E #Equation (1)
                      
    #Returns estimated coefficients
    return  Theta



inputs=data

outputs=np.array([y]).transpose()


# **Recursive Model a**


Ra_b0=recursive_leastsquares(1,data,outputs)


print "Ra_b0", Ra_b0.flatten()


Ra_y=model_a(Ra_b0[0][0])


Error_Ra=(np.array(y)-np.array(Ra_y)).T


Rcost_a=0.5*Error_Ra.T.dot(Error_Ra)


plt.plot(data, y, 'o', label='Original data', markersize=10)
plt.plot(data, Ra_y, 'r', label='Fitted line')
plt.legend()
plt.title('Recursive Model A')
plt.show()


# **Recursive Model b**


Rb_=recursive_leastsquares(2,data,outputs).flatten()



Rb_y=model_b(Rb_)


Error_Rb=(np.array(y)-np.array(Rb_y)).T

Rcost_b=0.5*Error_Rb.T.dot(Error_Rb)


plt.plot(data, y, 'o', label='Original data', markersize=10)
plt.plot(data, Rb_y, 'r', label='Fitted line')
plt.legend()
plt.title('Recursive Model B')
plt.show()


# **Recursive Model c**

Rc_=recursive_leastsquares(3,data,outputs).flatten()


Rc_y=model_c(Rc_)


Error_Rc=(np.array(y)-np.array(Rc_y)).T


Rcost_c=0.5*Error_Rc.T.dot(Error_Rc)


plt.plot(data, y, 'o', label='Original data', markersize=10)
x = np.linspace(0, 3.5, 1000)
plt.plot(x, 
         Rc_[0]+x*Rc_[1]+(x**2)*Rc_[2] ,
         'r', label='Fitted line')
plt.legend()
plt.title('Recursive Model C')
plt.show()


# ** Recursive Model d**


Rd_=recursive_leastsquares(4,data,outputs).flatten()



Rd_y=model_d(Rd_)



Error_Rd=(np.array(y)-np.array(Rd_y)).T



Rcost_d=0.5*Error_Rd.T.dot(Error_Rd)


plt.plot(data, y, 'o', label='Original data', markersize=10)
x = np.linspace(0, 3.5, 1000)
plt.plot(x, 
         Rd_[0]+x*Rd_[1]+(x**2)*Rd_[2] ,
         'r', label='Fitted line')
plt.legend()
plt.title('Recursive Model D')
plt.show()


# ### Generate a table showing each model's parameters along with the value of the cost function.

rows2=[('Recursive a',Ra_b0[0][0], 0,0,0),('Recursive b', Rb_[0], Rb_[1],0,0), ('Recursive c', Rc_[0], Rc_[1],Rc_[2],0),('Recursive d', Rd_[0],Rd_[1],Rd_[2],Rd_[3])]

t2 = Table(rows=rows2, names=('Model', 'b0*', 'b1*', 'b2*','b3*'))
print(t2)



cost_rows2=[('Recursive a',Rcost_a),('Recursive b',Rcost_b), ('Recursive c',Rcost_c),
       ('Recursive d', Rcost_d)]

cost_t2 = Table(rows=cost_rows2, names=('model', 'cost'))
print(cost_t2)


