#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt



# In[2]:


df1 = pd.read_csv('http://web.iitd.ac.in/~sumeet/A1/2016EE10459/NonGaussian_noise.csv', header=None, nrows=100)
data = np.array(df1.values)
n = int(len(data)*9/10)
#print(n)
train_x = data[:n,0:1]
train_y = data[:n,1:2]
#print (train_x.shape)
test_x = data[n:,0:1]
test_y = data[n:,1:2]


# In[3]:


def generate(x,m):
    n = len(x)
    dm = np.ones((n,m+1))
    
    x = x.T
    dm = dm.T
    for i in range(m+1) :
        dm[i] = x**i
    x = x.T
    dm = dm.T
    return dm


# In[4]:


def getResult(m, weight, test_x):
    test_dm = generate(test_x,m)
    test_r = np.dot(test_dm, weight)
    
    return test_r
    


# In[5]:


def getWeight(m , lam, train_x, train_y) :
    train_dm = generate(train_x,m)

    inverse = inv(np.dot(train_dm.T,train_dm) + lam*np.identity(m+1))

    projection = np.dot(inverse,train_dm.T)

    weight = np.dot(projection,train_y)
    return weight


# In[6]:


def getError(m, weight, test_x, test_y) :
    test_r = getResult(m, weight, test_x)
    
    test_error = (test_y - test_r)**2
    error = (np.sum(test_error)/len(test_y))**0.5
    return error


# In[13]:


order = 13
lam = 0

er = []
m = 7
graph = np.zeros((order,3))
fig = plt.figure(1)
for m in range(4,order) :
    weight =  getWeight(m,lam, train_x, train_y)
    trn = getError(m, weight, train_x, train_y)
    tst = getError(m, weight, test_x, test_y)
    
    graph[m][0] = m
    graph[m][1] = trn
    graph[m][2] = tst
    
plt.plot(graph[5:,0:1],graph[5:,1:2], label = 'training loss')
plt.plot(graph[5:,0:1],graph[5:,2:3], label = 'testing loss')
plt.title('Loss vs Order')
plt.ylabel('E_RMS')
plt.xlabel('Order')
plt.legend()
plt.show()

fig = plt.figure(2)
m = 10
weight =  getWeight(m,lam, train_x, train_y)
print (weight)
print (getError(m, weight, train_x, train_y))
new_x, new_y = zip(*sorted(zip(train_x,getResult(m, weight, train_x))))
plt.plot(new_x,new_y)
plt.scatter(train_x,train_y)
plt.title('10th order Polynomial')
plt.ylabel('y')
plt.xlabel('x')
plt.legend()
plt.show()


# In[ ]:




