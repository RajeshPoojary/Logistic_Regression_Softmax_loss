#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import math
data = []
labels = []
regc=0.1
learning_rate=0.1
s = np.zeros(3)
data.append([0.5, 0.4])
data.append([0.8, 0.3])
data.append([0.3, 0.8])
data.append([-0.4, 0.3])
data.append([-0.3, 0.7])
data.append([-0.7, 0.2])
data.append([0.7, -0.4])
data.append([0.5, -0.6])
data.append([-0.4, -0.5])
labels.append(0)
labels.append(0)
labels.append(0)
labels.append(1)
labels.append(1)
labels.append(1)
labels.append(2)
labels.append(2)
labels.append(2)

Classes = 3

# initialize Weights and bias with ramdom values

w = np.random.random((3, 2))
dw = np.zeros((3, 2))
b = np.random.random(3)
db = np.zeros(3)
p = np.zeros(3)
score = np.zeros(3)


for i in range(10000):
    costloss = 0
    for item in range(len(data)):
        esum = 0
        loss = 0
        for Class in range(Classes):
            
            #Calculate score for each class
            
            s[Class] = w[Class][0] * data[item][0] + w[Class][1] * data[item][1] + b[Class]
            
            #Softmax loss
            
            p[Class] = math.exp(s[Class])
            esum += p[Class]
        p = p / esum
        for Class in range(Classes):
            
            #Calculate the gradient
            if labels[item] == Class:
                loss += -math.log(p[Class])

                #Add up all the gradients

                dw[Class][0] += (p[Class] - 1) * data[item][0]
                dw[Class][1] += (p[Class] - 1) * data[item][1]
                db[Class] += p[Class] - 1
            else:
                dw[Class][0] += p[Class] * data[item][0]
                dw[Class][1] += p[Class] * data[item][1]
                db[Class] += p[Class]


        #Adding loss to get total loss

        costloss += loss

    #Calculate mean from the costloss

    costloss /= len(data)
    regloss = 0
    
    #Calculate mean from the gradients

    dw = dw / len(data)
    db = db / len(data)

    #L2 Regularization 
    for i in range(w.shape[0]):
    	for j in range(w.shape[1]):
    		regloss +=regc*w[i][j]+w[i][j]

    

    #Regularization gradient 
    dw += 0.5 * regc * w

   

    totalloss=costloss+regloss


    #Update the Weights and bias

    w -= learning_rate * dw
    b -= learning_rate * db

    print(totalloss)


			