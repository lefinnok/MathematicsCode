# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 13:03:55 2021

@author: lefin
"""

import numpy as np
from timeit import timeit
from math import exp, ceil
from time import sleep
from random import randint
#Inefficient but easy to read class model





class simple_network_model():
    class layer_types():
        class dense():
            def connect(layer, target_layer):
                for self_node in layer.nodes:
                    for target_layer_node in target_layer.nodes:
                        self_node.link(target_layer_node)
                    
            
            def disconnect(layer, target_layer):
                for self_node in layer.nodes:
                    for target_layer_node in target_layer.nodes:
                        self_node.unlink(target_layer_node)
                    
    
    class activations():
        def linear(out_value, *args):
            return out_value
        
        def step(out_value, limit0 = 0, limit1 = 1, *args):
            if out_value > limit0:
                return limit1
            else:
                return limit0
        
        def relu(out_value, limit0 = 0, *args):
            return max(limit0,out_value)
        
        def leakrelu(out_value, limit0 = 0, leak = 0.001, *args):
            if out_value > limit0:
                return out_value
            else:
                return out_value * leak
        
        def sigmoid(out_value, c = 1, *args):
            return 1/(1+exp(-c*out_value))
        
        def swish(out_value, c = 1, *args):
            return out_value/(1+exp(-c*out_value))
        
        def tanh(out_value, c = 1, *args):
            pos,neg = exp(c*out_value),exp(-c*out_value)
            return (pos-neg)/(pos+neg)

        
    
    class link():
            def __init__(self,originNode,targetNode, weight):
                self.targetNode = targetNode
                self.weight = weight
                self.originNode = originNode 

    
    class node():
        def __init__(self,name = '0', bias=0, activation = None, inputs = None):
            self.links = {}
            self.links_targeted_to_this = {}
            self.bias = bias
            self.name = name
            self.input = inputs
            self.output_value = None
            if activation == None:
                activation = simple_network_model.activations.relu
            
            self.activation = activation
            self.called = False
            
        def link(self,node):
            weight = 1.0
            self.links[node.name] = simple_network_model.link(self,node,weight)
            node.links_targeted_to_this[self.name] = self.links[node.name]
            
        def unlink(self,node):
            self.links.pop(node.name)
            node.links_linked_to_this.pop(self.name)
        
        def set_coordinates(self,x,y,r,root): #for GUI
            self.x = x
            self.y = y
            self.r = r
            self.root = root
        
        def output(self): #basically, this is kind of a class based recursion function
            # where if it has a manual input(so in the input layer), it returns its output as the input
            # and the input of other nodes will be the sum of output*linkweight for all linked nodes
            # and to trigger the network, one will just have to call this output() function for each node at the end
            # output layer of the network
            
            # ATTENTION: activation still needs to be implemented
            self.called = True
            if self.input != None:
                output = self.input
            else:
                output = sum([self.links_targeted_to_this[link].originNode.activation(self.links_targeted_to_this[link].originNode.output()*self.links_targeted_to_this[link].weight) for link in self.links_targeted_to_this]) + self.bias
            self.output_value = output
            self.called = False
            return output
        
        def manual_input(self,inpt):
            self.input = inpt
        
            
    
    class layer():
        def __init__(self,dimention,l_type = None, name='0'):
            self.dimention = dimention
            self.nodes = [simple_network_model.node(name+str(n)) for n in range(dimention)]
            
            if l_type == None:
                l_type = simple_network_model.layer_types.dense
            
            self.l_type = l_type
            self.name = name

        def connect(self,layer):
            self.l_type.connect(self,layer)
            layer.n_input_nodes = len(self.nodes)
        
        def disconnect(self,layer):
            self.l_type.disconnect(self,layer)
        
            
            
    
    class network():
        def __init__(self,layers, inpt = 1): 
            if type(layers[0]) == int:
                layers = [simple_network_model.layer(n,name=str(n)) for n in layers]
            self.layers = layers
            for layern,next_layer in zip(self.layers,self.layers[1:]):
                layern.connect(next_layer)
            
            if type(inpt) == int:
                for node in self.layers[0].nodes:
                    node.manual_input(inpt)
        
        def foward_pass(self):
            for node in self.layers[-1].nodes:
                node.output()
        

#print(simple_network_model.activations.leakrelu(0.1))


## Operation in a single node demonstration (Dense Layer)

def dense_node_demo():
    inputs = [10, 2.5, 3] #These are outputs from the previous layer with 3 nodes
    weights = [2,30,1] #These are the weights given by each node from the previous layer with 3 nodes
    bias = 10 #This is the bias of the current single node
    
    #This is the output of the current single node, where each input is multiplied
    #by the corrisponding weight and summed along with the node bias
    output = sum([inpt*weight for inpt,weight in zip(inputs,weights)]) + bias
    
    print(output)



## Operation in a single layer demonstration (Dense Layer)

def dense_layer_demo():
    inputs = [10, 2.5, 3] #These are outputs from the previous layer with 3 nodes
    #The inputs will be the same for each node, and will be passed to each node in the same fasion
    #modified by only the weights
    
    weights = [[2,30,1],
               [3,20,4],
               [2,6,7]] 
    
    #These are weights for each node in the dense layer given by the node from the previous layer
    
    #each row corresponse to the node of the current layer and the input given
    #each column corresponse to the value given by the node from the previous layer

    #e.g. row 1 is the weights used by node 1 of the current layer to modify the
    #input values by.
    
    #e.g. column 2 is the weights given by node 2 of the previous layer to node
    #1, 2 and 3 on the current layer correspondingly 
    
    biases = [10,20,10] #These are biases for the nodes of the current layer correspondingly
    
    layer_output = [ sum(inpt*weight for inpt,weight in zip(inputs,n_weights))+bias for bias, n_weights in zip(biases,weights)]
    
    print(layer_output)
    

## Operation in a single layer demonstration (Dense Layer, numpy)

def numpy_dense_layer_demo():
    inputs = np.array([[10,2.5,3]])
    
    weights = np.array([[2,30,1],
                        [3,20,4],
                        [2,6,7],
                        [2,6,7]] )
    
    biases = np.array([10,20,10,30])
    
    print(np.dot(inputs,weights.T)+biases) # The sum of multiples of two vectors/lists can be done
    #with matrix dot products
    
    #after which the resultant matrix can be summed with the biases matrix to get the final layer output
    

## Layer pass demonstration (Dense Layer, numpy)

def numpy_2_dense_layer_demo():
    inputs = np.array([[10,2.5,3],
                       [20,5.2,4]]) #This is the Initial inputs/sources (e.g. images)
    
    weights = np.array([[2,30,1],
                        [3,20,4],
                        [2,6,7],
                        [2,6,7]] )
    
    biases = np.array([10,20,10,30])
    
    output1 = np.dot(inputs,weights.T)+biases #This is the output of the first layer
    #which will be passed onto the second layer as input
    #note: it will be 4 columns wide
    print(output1)
    
    weights2 = np.array([[2,30,1,2],
                        [3,20,4,5]])
    
    #These are the weights from the previous layers, therefore must be 4 columns wide
    #This layer has 2 nodes, and therefore 2 rows
    
    biases2 = np.array([10,5])

    output2 = np.dot(output1,weights2.T)+biases2
    #The output of layer 2 gets the output of layer1 as its input
    
    print(output2)
    
    
#numpy classes

class activations():
    def linear(result):
        return result
    
    def step(array, thresh0 = 0, thresh1 = 1):
        array[array <= thresh0] = thresh0
        array[array > thresh0] = thresh1
        return array
    

    def relu(array, thresh0 = 0):
        array[array <= thresh0] = thresh0
        return array
    
    
    def leakrelu(array, thresh0 = 0, leak = 0.001):
        product_array = (array > thresh0).astype(float)
        product_array[product_array == 0] = leak
        return array * product_array
    
    def sigmoid(array, c = 1):
        return np.reciprocal(1+np.exp(-c*array))
    
    def swish(array,c = 1):
        return array/(1+np.exp(-c*array))
    
    def softmax(array):
        return(np.exp(array)/np.sum(np.exp(array),axis = 1)[:,None])
    
    def maxpool(array, stride = 2): #the most efficient method (reshape)
        rel0,rel1 = int(array.shape[0]/stride),int(array.shape[1]/stride)
        h, w = array.shape
        #print(np.concatenate(np.hsplit(np.array(np.hsplit(array,rel1)),rel0)))
        return np.reshape(np.amax(
                            np.amax(
                                array.reshape(h//stride, stride, -1, stride)
                                   .swapaxes(1,2)
                                   .reshape(-1, stride, stride)
                                    ,axis = 1),axis=1),(rel0,rel1))
    
    def tanh(array, c = 1):
        pos,neg = np.exp(c*array),np.exp(-c*array)
        return (pos-neg)/(pos+neg)

class Layer():
    def __init__(self,n_input_nodes: int,n_nodes: int, activation = activations.linear, **kwargs):
        #self.weights = np.ones((n_input_nodes,n_nodes)) #initiallize weights as ones
        self.weights = np.random.randn(n_input_nodes,n_nodes) * 0.01 #initiallize weights as randoms
        self.activation = activation
        #with the number of rows by the number of nodes
        #with the number of columns by the number of input nodes
                    #Transpose manually#
        self.biases = np.zeros((n_nodes)) #initiallize biases as zeros
    
    def foward(self,inputs):
        #return the output of the layer by the inputs
        result = np.dot(inputs,self.weights)+self.biases
        return self.activation(result)
    
    
    
    

    
def numpy_2_dense_layer_class_demo():
    inputs = np.array([[10,2.5,3],
                       [20,5.2,4]]) #This is the Initial inputs/sources (e.g. images)
    
    d1 = Layer(3, 4)
    
    output1 = d1.foward(inputs) 
    (output1)
    
    
    d2  = Layer(4,2,activations.leakrelu)
    
    output2 = d2.foward(output1)
    
    print(output2)


class DenseNetwork():
    def __init__(self, layer_shapes, activations, input_data, expected_data, batch_size):
        '''
        self.input_data = input_data
        self.expected_data = expected_data
        '''
        print('Network Initializing...')
        print('Checking Expected Datatype...')
        print(f'\nDatatype: {expected_data.dtype}\n')
        if expected_data.dtype != np.float64:
            print('One_hot Encoding...')
            self.one_hot = True
            self.one_hot_dict = {}
            self.one_hot_unique = np.unique(expected_data)
            print(f'\nPossible Outputs: {self.one_hot_unique}\n')
            for option in self.one_hot_unique:
                self.one_hot_dict[option] = (self.one_hot_unique == option).astype(int)
            print(f'\nEncodings: ')
            for option in self.one_hot_dict:
                print(f'          {option} | {self.one_hot_dict[option]}')
            print('')
            expected_data = [self.one_hot_dict[output] for output in expected_data]
        else:
            self.one_hot = False
        self.batched = zip(np.array_split(input_data,ceil(len(input_data)/batch_size)),np.array_split(expected_data,ceil(len(input_data)/batch_size)))
        print('\nConstructing Layers...')
        self.layers = [Layer(*shape,activation=activation) for shape,activation in zip(layer_shapes,activations)]
        self.n_layers = len(self.layers)
        self.n_layers_m1 = len(self.layers) - 1
        print('Complete Initialization.')
    
    #Demo functions for a single batch of data
    '''
    def foward_pass(self):
        def pas(layer,n):
            if n < 0:
                return self.input_data
            else:
                return layer.foward(pas(self.layers[n-1],n-1))
        
        return pas(self.layers[-1],self.n_layers_m1)
    
    def loss(self):
        return -np.einsum('ij,ij->i', np.log(self.foward_pass()), self.expected_data)
    '''
    
    def foward_pass(self):
        def pas(layer,n,data):
            if n < 0:
                return data
            else:
                return layer.foward(pas(self.layers[n-1],n-1,data))
        
        if self.one_hot:
            def loss(data,expectation):
                return np.mean(-np.log(np.clip(data[expectation.astype(bool)],1e-7, 1 - 1e-7)))
        else:
            def loss(data,expectation):
                #print(data,np.log(data),expectation)
                return np.mean(-np.einsum('ij,ij->i', np.log(np.clip(data,1e-7, 1 - 1e-7)), expectation))
        
        def accuracy(data,expectation):
            return np.mean(np.mean(np.argmax(data,axis=1) == np.argmax(expectation,axis=1)))
        
        def output(data,expectation):
            return data, loss(data,expectation), accuracy(data,expectation)
        
        return [output(pas(self.layers[-1],self.n_layers_m1,inpt),expect) for inpt,expect in self.batched]
    
    
d = DenseNetwork(((10,8),(8,5),(5,3)),(activations.swish,activations.swish,activations.softmax),np.array([[4,2,6,2,4,5,4,1,4,5],[1,1,1,1,1,1,1,1,1,1],[4,2,6,2,4,5,4,1,4,5],[4,2,6,2,4,5,4,1,4,5],[4,2,6,2,4,5,4,1,4,5]]),np.array(['aa','ba','ca','ca','aa']),3)

print(d.foward_pass())
#print(d.loss(0))

#print(list(d.batched))

#numpy_2_dense_layer_class_demo()

#print(activations.softmax([[4.8, 1.21, 2.385]]))

'''
def loss(data,expectation):
    #print(data,np.log(data),expectation)
    return data,np.mean(-np.einsum('ij,ij->i', np.log(np.clip(data,1e-7, 1 - 1e-7)), expectation))
def lossOH(data,expectation):
    return data, np.mean(-np.log(data[expectation.astype(bool)]))
print(loss(np.array([[0.7,0.1,0.2]]),np.array([[1,0,0]])))
print(lossOH(np.array([[0.7,0.1,0.2]]),np.array([[1,0,0]])))
'''
