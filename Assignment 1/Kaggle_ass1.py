#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt



# In[140]:


df1 = pd.read_csv('http://web.iitd.ac.in/~sumeet/A1/train.csv', header=None, nrows=111)
data = np.array(df1.values)[1:,:]
np.random.shuffle(data)
l = len(data)
print (l)
ndata = np.zeros((l,3))
for i in range(l) :
    ndata[i][2] = float(data[i][1])
    p = str(data[i][0]).split("/")
    ndata[i][0] = int(p[0])
    ndata[i][1] = (int(p[1])-2004)/10

n = int(len(ndata)*10/11)
train_x1 = ndata[:n,0:1]
train_x2 = ndata[:n,1:2]
train_y = ndata[:n,2:3]
#print (train_x.shape)
test_x1 = ndata[n:,0:1]
test_x2 = ndata[n:,1:2]
test_y = ndata[n:,2:3]

df2 = pd.read_csv('http://web.iitd.ac.in/~sumeet/A1/test.csv', header=None, nrows=11)
sdata = np.array(df2.values)[1:,:]
s = len(sdata)
print (s)
tdata = np.zeros((s,2))
for i in range(s) :
    
    p = str(sdata[i][0]).split("/")
    tdata[i][0] = int(p[0])
    tdata[i][1] = (int(p[1])-2004)/10

test1 = tdata[:,0:1]
test2 = tdata[:,1:2]
#test_y = ndata[:,2:3]
# In[89]:


def generate(x1,m1,x2,m2):
    n = len(x1)
    dm = np.ones((n,m1+m2+2))
    
    x1 = x1.T
    x2 = x2.T
    dm = dm.T
    for i in range(m1+1) :
        dm[i] = x1**i
    for i in range(m2+1) :
        dm[m1+1+i] = x2**i
    x1 = x1.T
    x2 = x2.T
    dm = dm.T
    return dm


# In[90]:


def getResult(m1,m2, weight, test_x1, test_x2):
    test_dm = generate(test_x1,m1,test_x2,m2)
    test_r = np.dot(test_dm, weight)
    
    return test_r


# In[113]:


def getWeight(m1 ,m2 , lam, train_x1,train_x2, train_y) :
    train_dm = generate(train_x1,m1,train_x2,m2)
    #print (train_dm.shape)
    inverse = inv(np.dot(train_dm.T,train_dm) + lam*np.identity(m1+m2+2))

    projection = np.dot(inverse,train_dm.T)

    weight = np.dot(projection,train_y)
    return weight


# In[114]:


def getError(m1,m2, weight, test_x1, test_x2, test_y) :
    test_r = getResult(m1,m2, weight, test_x1, test_x2)
    
    test_error = (test_y - test_r)**2
    error = (np.sum(test_error)/len(test_y))**0.5
    return error


# In[141]:


order = 12
lam = 10**-2

er = []
#m = 7
graph = np.zeros((order,3))
fig = plt.figure(1)
m1 = 6
m2 = 3
for m1 in range(2,order) :
#for m2 in range (-1,order) :
    weight =  getWeight(m1 ,m2 , lam, train_x1,train_x2, train_y)
    trn = getError(m1,m2, weight, train_x1, train_x2, train_y)
    tst = getError(m1,m2, weight, test_x1, test_x2, test_y)
    print(m1," ",trn," ",tst)
    graph[m1][0] = m1
    graph[m1][1] = trn
    graph[m1][2] = tst
    
plt.plot(graph[3:,0:1],graph[3:,1:2], label = 'training loss')
plt.plot(graph[3:,0:1],graph[3:,2:3], label = 'testing loss')
plt.title('Loss vs order[Month] @ order[year] = 3')
plt.ylabel('E_RMS')
plt.xlabel('Order')
plt.legend()
plt.show()

fig = plt.figure(2)
m1 = 6
m2 = 3
weight =  getWeight(m1 ,m2 , lam, train_x1,train_x2, train_y)
print (weight)
print (" result is ",getResult(m1,m2, weight, test1, test2))
print("error ",getError(m1,m2, weight, train_x1, train_x2, train_y))
#new_x, new_y = zip(*sorted(zip(test_x1,getResult(m1,m2, weight, test_x1, test_x2))))
#plt.plot(new_x,new_y)
#plt.scatter(train_x1,train_y)
#plt.title('3rd order [year] Polynomial')
#plt.ylabel('y')
#plt.xlabel('x')
#plt.legend()
#plt.show()
#

# In[ ]:





# In[ ]:




