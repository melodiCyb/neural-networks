
import numpy as np

class MultiLayerPerceptron:
       
    def __init__(self, network_size):
        
        """Initialize the network
        
        network_size=(n_input,n_hidden1,...,n_hiddenk, n_output)
        
        n_input: number of neurons in input layer
        
        n_hiddenj: number of hidden neurons in hidden layer j 
                   where j=1,2,..k
        
        n_output:number of output neurons
        """
        
        self.indices=0
        self.shape=None
        self.weights=[]
 
      
        
        #set layer values
        self.indices = len(network_size) - 1
        self.shape = network_size
        
        
        #to store inputs and outputs after forward propagation
        self._S = []
        self._O = []
        
        #to store previous weight changes for momentum term
        self.prev_weight_change = []
        
        
        #Initialize weights
        
        layer_array=np.array([network_size[:-1], network_size[1:]]).T
        for (layerpair_1,layerpair_2) in layer_array:
            
            self.wi=np.zeros((layerpair_2,layerpair_1+1))
            
            for i in range(layerpair_2):
                for j in range(layerpair_1+1):
                    self.wi[i][j]=np.random.uniform(-1, 1)
                    
            self.weights.append(self.wi)
            self.prev_weight_change.append(np.zeros((layerpair_2
                                                     ,layerpair_1+1)))
    
    
    #Forward Propagation
    def FeedForward(self, input):
        """Feed the network with inputs"""
        
        #Reset values
        self._S = []
        self._O = []
        
        #Feedforward
        for k in range(self.indices):
            
            # Determine layer inputs
            
            #if we are at the input layer
            if k == 0:
                #we also add bias
                input_with_bias=np.array([np.append(i,1) for i in input])
                S = self.weights[0].dot(input_with_bias.T)
                
            #else we are the hidden layer
            else:
                #we take the data from previous layer
                #hidden_input_with_bias
                b=np.ones([1, input.shape[0]])
                S = self.weights[k].dot(np.vstack([self._O[-1],b]))
                
            #layer inputs
            self._S.append(S)
            
            #layer outputs
            self._O.append(self.sigmoid(S))
            
        #return output from the last layer
        return self._O[-1].T
    
  
    # Sigmoid Activation Function 
    def sigmoid(self,x):
            return 1 / (1+ np.exp(-x))
        
    #Derivative of Sigmoid
    def sigmoid_derivative(self,x):
            output = self.sigmoid(x)
            return output * (1 - output)
        
           
    #Backpropagation
    def BackPropagation(self, input, target, eta,momentum_coef):
        """
        Backpropagate the network for one epoch
        
        eta:learning rate
        momentum_coef: momentum coefficient
        
        """
        #to store deltas in Equation (1) and (2)
        delta = []
        
        # FeedForward the network
        self.FeedForward(input)
        
        #Compute deltas
        #start from Output Layer and move backwards
        for k in range(self.indices)[::-1]:
            
            #if we are at Output Layer
            if k== self.indices - 1:
                e= self._O[k]-target.T
                #Equation (1)
                output_delta=e*self.sigmoid_derivative(self._S[k]) 
                error = 0.5*np.sum(e**2)
                delta.append(output_delta)
                
            #else we are at hidden layer      
            else:
                
                # delta_h--> following layer's delta
                delta_h = self.weights[k + 1].T.dot(delta[-1])
                f_deriv_S=self.sigmoid_derivative(self._S[k])
                #Equation (2)
                #takes all the but last rows that correspond to biases
                hidden_delta=delta_h[:-1, :]*f_deriv_S 
                delta.append(hidden_delta)
            
        #Compute weight changes
        for k in range(self.indices):
            
            '''
            *get outputs of the layers
            
            *multiply all the outputs from previous layer 
            by all of the deltas from the current layer 
            
            *update the weights that connect 
            previous layer to the current layer
            
            *return error
            '''
    
            if k == 0:
                # if we are in input layer
                #add biases also
                input_with_bias=np.array([np.append(i,1) for i in input])
                O= input_with_bias.T
                
            else:
                #output for previous layer
                #add biases also
                b=np.ones([1, self._O[k - 1].shape[1]])
                O = np.vstack([self._O[k - 1],b])
                
            
            
            #adapt index of delta for reverse order
            k_delta = self.indices - 1 - k
            
            
            #Equation (6)
            
            #take current deltas and multiply it
            #with previous layers' outputs
            delta_x_O=delta[k_delta][np.newaxis,:,:].transpose(2, 1, 0)                                       * O[np.newaxis,:,:].transpose(2, 0 ,1)
            Delta_w_current=eta*np.sum(delta_x_O, axis = 0)
            
            
            momentum_effect= momentum_coef * self.prev_weight_change[k]
            
            #Equation(3)
            #update the weights
            Delta_w = Delta_w_current + momentum_effect
            
            self.weights[k] -= eta*Delta_w
            
            self.prev_weight_change[k] = Delta_w
        
            
        #returns error
        return error
   
  
    
    def train(self, patterns, epochs, eta, mu):
        
        #eta: learning rate
        #mu: momentum coefficient
        
        
        import pylab
        E=np.zeros(epochs)
        etas = []
        etas.append(eta)

        c=[]
        epoch=[]
        
        for n in range(1,epochs):
            cost = 0.0
            inputs=[]
            targets=[]
            for p in patterns:
                inputs.append(p[0])
                targets.append(p[1])
                
            inputs=np.array(inputs)
            targets=np.array(targets)
            
            #cost=self.BackPropagation(inputs,targets, eta, mu)
            
            #Update rule for Learning Rate "eta"
            E[0]=self.BackPropagation(inputs,targets, eta, mu)
            E_new=self.BackPropagation(inputs,targets, eta, mu)
            epsilon=0.0001
            if not abs(E_new-E[n-1])<epsilon: #Equation (4)
                #Equation (5)
                if E_new > E[n-1]:
                    # Decrease learning rate
                    eta = eta * 0.5
                    E_new=self.BackPropagation(inputs,targets, eta, mu)
                                       
                elif E_new < E[n-1]:
                    #Increase learning rate
                    eta = eta * 1.05
            
            etas.append(eta)    
            E[n] = E_new
            
    
        
            cost =cost +self.BackPropagation(inputs,targets, eta, mu)
            c.append(cost)
            epoch.append(n)
            threshold=0.01
    
            #terminate if cost is less then the threshold=0.01
            if cost<threshold:
                break
                
        #print learning rate list        
        #print etas
        
        #Plot the cost function value vs the number of epochs
        pylab.plot(epoch, c)
        pylab.xlabel('Number of Epochs')
        pylab.ylabel('Cost')
        pylab.show()
        
    def test(self, patterns,plot=False,input_index=False):
        inputs=[]
        targets=[]
        outputs=[]
        
        #get inputs and targets in given patterns
        for p in patterns:
            inputs.append(p[0])
            targets.append(p[1])
            
        inputs=np.array(inputs)
        targets=np.array(targets)
        #print inputs
        if not input_index:
            for i in range(len(inputs)):
            
                print "Input:",inputs[i],'->',                "Desired:",targets[i],',',                "Output:",self.FeedForward(inputs)[i]
            
                outputs.append(self.FeedForward(inputs)[i])
        else:
            for i in range(len(inputs)):
            
                print "Input:",i,'->',                "Desired:",targets[i],',',                "Output:",self.FeedForward(inputs)[i]
            
                outputs.append(self.FeedForward(inputs)[i])
            
            
        #plotting
        if plot:
            import matplotlib.pyplot as plt
            y=np.amax(inputs)
            x=np.linspace(0, y, len(targets))
            plt.scatter(x,targets, c='b')
            plt.scatter(x,outputs, c='r')
            plt.title('Comparison of Results')
            plt.show()
    


# **Remark:** In the following examples the number of layers and the number of neurons in each layer can be changed.

# ## **Test the model with XOR function**

# ### **Plot cost function vs epochs.**

# ### **Terminate the update rule if the cost function reaches a certain threshold (i.e 0.01)**

# ### **Check the model with 4 inputs and write the output for each input.**



def XOR():
    XOR_= [
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [0]]
    ]

    
    # Multilayer Perceptron model with:
    #2 input neurons
    #2 hidden layers with 3 neurons
    #1 output neuron
    
    network_form2=(2,3,3,1)
    MLP2=MultiLayerPerceptron(network_form2)
    print "Performance of the MLP with 2 hidden layers with 3 neurons"
    eta=0.2
    mu=0.7
    epochs=100000
    #train
    MLP2.train(XOR_,epochs,eta,mu)
    #test
    MLP2.test(XOR_)


# **Remark** In the results we expect not exactly the target values but close to the target values.



if __name__ == "__main__":
    XOR()


# ## Use the model to approximate a non-linear function. 

# 

# ### Implement the algorithm


def SinApproximation():
    import matplotlib.pyplot as plt
    import numpy as np
    
    
    np.random.seed(578)
    
    mu, sigma = 0, 0.1
    
    #zero-mean Gaussian noise with standard deviation 0.1
    noise1= np.random.normal(mu, sigma,100)
    
    #inputs for train
    trainSinus = np.linspace(0, 2*np.pi, 100)
    
    #desired outputs for train
    targetTrainSinus=np.sin(trainSinus+noise1)

    #plot tranining noisy data
    plt.plot(trainSinus,targetTrainSinus)
    plt.title('Sinus Train Data Plot')
    plt.show()
    
    #inputs for test
    testSinus = np.linspace(0, 2*np.pi, 25)
    
    #desired outputs for test
    targetTestSinus=np.sin(testSinus)
    
    #plot test data points
    plt.plot(testSinus,targetTestSinus)
    plt.title('Sinus Test Data Plot')
    plt.show()
    
    #make patterns with inputs and targets
    #for training
    SinusTrainPatterns=[[[j]] for j in trainSinus]
  
    for i in range(len(trainSinus)):
        SinusTrainPatterns[i].append([targetTrainSinus[i]])
        
    #make patterns with inputs and targets
    #for testing
    SinusTestPatterns=[[[j]] for j in testSinus]
    
    for i in range(len(testSinus)):
        SinusTestPatterns[i].append([targetTestSinus[i]])        
        
    
    
    #Multilayer Perceptron model with:
    #1 input neuron
    #1 hiddden layer with 6 neurons
    #and 1 output neuron

    network_form=(1,6,1)
    MLP = MultiLayerPerceptron(network_form)
    print "Performance of the MLP with 1 hidden layer with 6 neurons"
    eta=0.002
    mu=0.7
    epochs=10000
    
    # train
    MLP.train(SinusTrainPatterns,epochs,eta,mu)
    
    # test 
    MLP.test(SinusTestPatterns,True)





if __name__ == "__main__":
    SinApproximation()


# ##  **Apply the algorithm to Iris Data Set.**

# ### **There 150 sample patterns. Pick 125 for training (randomly). Test the system with the rest 25 sample patterns. **

# ### **Pick 3 outputs, one for each class of flowers

# To implement the algorithm we convert target flowers names to numeric values.


def Classify_IrisFlowers():
    import csv
    import random
    
    random.seed(123)

    #Load iris dataset
    with open('data/iris.csv') as csvfile:
        csvreader = csv.reader(csvfile)
        #skips the header
        next(csvreader, None) 
        dataset = list(csvreader)

    #Change string targets to numeric as: 
    #Setosa=[1,0,0]
    #Versicolor=[0,1,0]
    #Virginica=[0,0,1]
    for row in dataset:
        row[4] = ["setosa", "versicolor", "virginica"].index(row[4])
        row[:4] = [float(row[j]) for j in xrange(len(row))]
        if row[4]==0:
            row[4]=[1,0,0]
        elif row[4]==1:
            row[4]=[0,1,0]
        else:
            row[4]=[0,0,1]

    #Split data to features and targets
    #X is input
    #y is target output
    
    #shuffle data set
    random.shuffle(dataset)
    
    #data for training
    datatrain = dataset[:125]
    
    #data for testing
    datatest = dataset[25:]
    
    #inputs for training
    train_X = [data[:4] for data in datatrain]
    
    #targets for training
    train_y = [data[4] for data in datatrain]
    
    #inputs for testing
    test_X = [data[:4] for data in datatest]
    
    #targets for testing
    test_y = [data[4] for data in datatest]
    
    #rearrange training data form 
    datas=[]
    for i in range(len(train_X)):
        datas.append([train_X[i]])
        datas[i].append(train_y[i])
    
    
    # Multilayer Perceptron model with:
    #4 input neurons
    #1 hidden layer with 4 neurons
    #and 3 output neurons
    
    network_form=(4, 4,3)
    MLP = MultiLayerPerceptron(network_form)
    print "Performance of the MLP with 1 hidden layer with 4 neurons"
    eta=0.02
    mu=0.7
    epochs=10000

    # train 
    MLP.train(datas,epochs,eta,mu)
    
    #rearrange testing data form
    testdatas=[]
    for i in range(len(test_X)):
        testdatas.append([test_X[i]])
        testdatas[i].append([test_y[i]])

    # test
    MLP.test(testdatas,input_index=True)


    

if __name__ == "__main__":
    Classify_IrisFlowers()

