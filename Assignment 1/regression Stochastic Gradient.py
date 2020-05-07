#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[35]:


df1 = pd.read_csv('http://web.iitd.ac.in/~sumeet/A1/2016EE10459/NonGaussian_noise.csv', header=None, nrows=100)
data = np.array(df1.values)
n = int(len(data)*9/10)
#print(n)
train_x = data[:n,0:1]
train_y = data[:n,1:2]
#print (train_x.shape)
test_x = data[n:,0:1]
test_y = data[n:,1:2]


# In[16]:


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


# In[17]:


def getResult(m, test_x):
    test_dm = generate(test_x,m)
    test_r = np.dot(test_dm, weight)
    
    return test_r
    


# In[28]:


def getError(m, test_x, test_y) :
    test_r = getResult(m, test_x)
    
    test_error = (test_y - test_r)**2
    error = (np.sum(test_error)/len(test_y))**0.5
    return error


# In[38]:


def gradDescent(lr, lam, err, x, y):
    #print (train_dm)
    global weight
    r = np.dot(x,weight) ;
    err = (y-r)*x + err + lam*weight
    del_w = lr*err 
    weight = weight+del_w
    
def backProp(lr, lam, m, b_s, epoch, train_x, train_y):
    global weight
    train_dm = generate(train_x,m)
    weight = np.random.randn(m+1)

    for i in range(epoch):
        err = 0 ;
        for j in range(n) :
            y = train_y[j]
            x = train_dm[j]

            if j>0 and j%b_s==0 :
                gradDescent(lr, lam, err, x, y)
                err = 0 
            else :
                r = np.dot(x,weight) ;
                err += (y-r)*x


# In[42]:


lr = 0.0000001
lam = 10**-200
b_s = 1
epoch = 100
m = 10
train_dm = generate(train_x,m)
weight = np.random.randn(m+1)

graph = np.zeros((epoch,2))

fig = plt.figure(1)

for i in range(epoch):
    err = 0 ;
    for j in range(n) :
        y = train_y[j]
        x = train_dm[j]
        
        if j>0 and j%b_s==0 :
            gradDescent(lr, lam, err, x, y)
            err = 0 
        else :
            r = np.dot(x,weight) ;
            err += (y-r)*x
    trn = getError(m, train_x, train_y)
    graph[i][0] = i
    graph[i][1] = trn
        
print (weight)
print (getError(m,train_x,train_y))

plt.plot(graph[:,0:1],graph[:,1:2], label = 'training loss')
plt.title('Loss vs iteration @ batch_size = 1')
plt.ylabel('E_RMS')
plt.xlabel('iteration')
plt.legend()
plt.show()

fig = plt.figure(2)
#plt.scatter(train_x,train_y)
new_x, new_y = zip(*sorted(zip(train_x,getResult(m, train_x))))
plt.plot(new_x,new_y,color = 'b')
plt.scatter(train_x,train_y,color = 'r')
plt.title('7th order Polynomial')
plt.ylabel('y')
plt.xlabel('x')
plt.legend()
plt.show()


# In[ ]:
#lr = 0.00005
#lam = 0
#b_s = 1 
#epoch = 1000
#order = 10
#m = 7
##train_dm = generate(train_x,m)
#
#graph = np.zeros((n+1,3))
#weight = np.random.randn(1)
#fig = plt.figure(1)
#
#for i in range(1,n+1,1) :
#    backProp(lr, lam, m, i, epoch, train_x, train_y)
#    trn = getError(m, train_x, train_y)
#    #tst = getError(m, test_x, test_y)
#    
#    
#    graph[i][0] = i
#    graph[i][1] = trn
#    #graph[i][2] = tst
#    
#plt.plot(graph[1:,0:1],graph[1:,1:2], label = 'training loss')
##plt.plot(graph[:b_s,0:1],graph[:b_s,2:3], label = 'testing loss')
#plt.title('Loss vs batch_size')
#plt.ylabel('E_RMS')
#plt.xlabel('batch_size')
#plt.legend()
#plt.show()
##                
#                
#lr = 0.0005
#lam = 10
#b_s = 1 
#epoch = 1000
#order = 10
#m = 7
##train_dm = generate(train_x,m)
#
#graph = np.zeros((10,3))
#weight = np.random.randn(1)
#fig = plt.figure(1)
#
#for i in range(0,1000,100) :
#    weight = np.random.randn(1)
#    backProp(lr, lam**(-i), m, b_s, epoch, train_x, train_y)
#    trn = getError(m, train_x, train_y)
#    tst = getError(m, test_x, test_y)
#    
#    p = int((i)/100)
#    graph[p][0] = lam**(-i)
#    graph[p][1] = trn
#    graph[p][2] = tst
#    
#plt.plot(graph[:,0:1],graph[:,1:2], label = 'training loss')
#plt.plot(graph[:,0:1],graph[:,2:3], label = 'testing loss')
#plt.title('Loss vs lam')
#plt.ylabel('E_RMS')
#plt.xlabel('lam')
#plt.xscale('log')
#plt.legend()
#plt.show()

#fig = plt.figure(2)
#lam  = 0
#weight = np.random.randn(1)
#backProp(lr, lam, m, b_s, epoch, train_x, train_y)
#new_x, new_y = zip(*sorted(zip(train_x,getResult(m, train_x))))
#plt.plot(new_x,new_y)
#plt.scatter(train_x,train_y)
#plt.title('7th order Polynomial')
#plt.ylabel('y')
#plt.xlabel('x')
#plt.legend()
#plt.show()
