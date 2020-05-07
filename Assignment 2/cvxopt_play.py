#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from sklearn import svm


# In[3]:


df1 = pd.read_csv('http://web.iitd.ac.in/~sumeet/A2/2016EE10459.csv', header=None, nrows=3000)
df1 = df1.sort_values(df1.columns[25])
train_input = df1.values
data = np.array(train_input)
#print (data)


# In[34]:


a = 0
b = 1
n1 = a*300 
n2 = b*300 
t = 25
train_x = np.append(data[n1:n1+250,:t],data[n2:n2+250,:t],axis = 0)
train_y = np.append(data[n1:n1+250,25],data[n2:n2+250,25],axis = 0)
for i in range(train_y.shape[0]):
    if train_y[i] == a :
        train_y[i] = 1
    else :
        train_y[i] = -1

test_x = np.append(data[n1+250:n1+300,:t],data[n2+250:n2+300,:t],axis = 0)
test_y = np.append(data[n1+250:n1+300,25],data[n2+250:n2+300,25],axis = 0)
for i in range(test_y.shape[0]):
    if test_y[i] == a :
        test_y[i] = 1
    else :
        test_y[i] = -1
        
print (train_x.shape)
print (train_y.shape)
print (test_x.shape)
print (test_y.shape)


# In[58]:


def poly_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p


# In[72]:


def poly_fit(C,gamma,X, y):
    m,n = X.shape
    y = y.reshape(-1,1) * 1.
    X_dash = y * X
    K = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            K[i,j] = poly_kernel(X[i], X[j],gamma)*1.

    P = matrix(np.outer(y,y) * K)
    q = matrix(-np.ones((m, 1)))
    G = matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
    h = matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = matrix(y.reshape(1, -1))
    b = matrix(np.zeros(1))

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    
    return alphas


# In[73]:


def cvx_fit(C,X,y) : 
    
    m,n = X.shape
    y = y.reshape(-1,1) * 1.
    X_dash = y * X
    H = np.dot(X_dash , X_dash.T) * 1.

    
    P = matrix(H)
    q = matrix(-np.ones((m, 1)))
    G = matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
    h = matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = matrix(y.reshape(1, -1))
    b = matrix(np.zeros(1))

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    
    return alphas


# In[79]:


x = train_x
y = train_y
#alphas = cvx_fit(10,x, y)
alphas = poly_fit(10,1,x, y)
#print(alphas.shape)
w = np.sum(alphas * y[:, None] * x, axis = 0)
cond = (alphas > 1e-4).reshape(-1)
b = y[cond] - np.dot(x[cond], w)
#bias = b[0]


#Display results
print('Alphas = ',alphas[alphas > 1e-4])
print('w = ', w.flatten())
print('b = ', b[0])


# In[42]:


#svm = svm.SVC(C = 10, kernel = 'linear')
svc = svm.SVC(kernel='poly', C=10,gamma= 1)
svc.fit(x, y) 

#print('w = ',svc.coef_)
#print('b = ',svc.intercept_)
print(svc.score(train_x, train_y))
print(svc.score(test_x, test_y))

# In[80]:


def get_result(X, y, w, b):
    
    n = y.shape[0]
    res = np.dot(X,w)+b
    for i in range(n):
        if res[i] >= 0:
            res[i] = 1
        else :
            res[i] = -1
     
    acc = 0
    for i in range(n):
        if res[i]*y[i] > 0 :
            acc += 1
    acc = acc/n
    return acc, res


# In[81]:


acc, res =  get_result(test_x,test_y,w,b[0])
print (acc)


# In[ ]:




