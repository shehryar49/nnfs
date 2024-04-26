import numpy as np
import random
# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x*(1-x)

# Error functions
def MSE(aL,y):
    l = aL - y
    return np.sum(np.square(l)) / len(y)
def MSE_derivative(aL,y):
    return 2 * (aL - y)
def CEL_derivative(aL,y):
    return aL - y
class Layer:
    def __init__(self,num_neurons,num_inputs,fn = None,fnd = None,err_fun_deriv=MSE_derivative):
        self.n = num_neurons
        self.weights = np.random.randn(num_inputs,num_neurons)
        self.fn = fn
        self.fnd = fnd
        self.err_fnd = err_fun_deriv
    def forward(self,inputs):
        res = np.matmul(inputs,self.weights)
        if self.fn != None:
            res = self.fn(res)
        self.aL = res
        return res
    # deltas at output layer
    def deltas_output(self,y):
        if self.fnd == None:
            ld = self.err_fnd(self.aL,y) 
        else:
            ld = self.fnd(self.aL) * self.err_fnd(self.aL,y)
        return ld
    # deltas at hidden layers
    def deltas(self,dout,dw): # dw are weights of next layer
        a = np.matmul(dout,dw.T)
        a = self.fnd(self.aL) * a
        return a
        
    def backpropagate(self,a,deltas,lr = 1):
        # stochastic gradient descent
        # pick a random input and try to fit model on that
        idx = random.randint(0,len(deltas)-1)
        inputs = np.array([a[idx]])
        deltas = deltas[idx]
        inputs = np.repeat(inputs.T,len(deltas),axis=1)
        delta_weight = inputs*deltas
        self.weights = self.weights - lr*delta_weight


# XOR 
# 0 0 0
# 0 1 1
# 1 0 1
# 1 1 0


inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])
w1 = np.array([[1,0],[0,1]])

# hidden layer 1
hidden = Layer(2,2,sigmoid,sigmoid_derivative,MSE_derivative)
output = Layer(1,2,sigmoid,sigmoid_derivative,MSE_derivative)
for i in range(1000000):
    a1 = hidden.forward(inputs)
    a2 = output.forward(a1)
    print('Loss = ',MSE(output.aL,y))
    dout = output.deltas_output(y)
    dh = hidden.deltas(dout,output.weights)
    output.backpropagate(a1,dout)
    hidden.backpropagate(inputs,dh)
print(a2)