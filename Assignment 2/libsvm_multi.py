#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[2]:


df1 = pd.read_csv('http://web.iitd.ac.in/~sumeet/A2/2016EE10459.csv', header=None, nrows=3000)
train_input = df1.values
data = np.array(train_input)


# In[3]:


x_tr = data[:,:11]
y_tr = data[:,25]
train_x, test_x, train_y, test_y = train_test_split(x_tr,y_tr,test_size=0.2, random_state=30, stratify=y_tr)


# In[4]:


#steps = [('scaler', StandardScaler()), ('SVM', svm.SVC(kernel='rbf'))]
#pipeline = Pipeline(steps) # define Pipeline object
#
#parameters = {'SVM__C':[ 0.1, 1, 10], 'SVM__gamma':[10,1,0.1]}
#
#grid = GridSearchCV(pipeline, param_grid=parameters, cv=5)

steps = [('scaler', StandardScaler()), ('SVM', svm.SVC(kernel='linear'))]
pipeline = Pipeline(steps) # define Pipeline object

parameters = {'SVM__C':[ 0.1, 1, 10]}

grid = GridSearchCV(pipeline, param_grid=parameters, cv=5)
# In[5]:


grid.fit(train_x, train_y)
print(grid.score(train_x, train_y))
print (grid.score(test_x, test_y))
print (grid.best_params_)


# In[6]:


#svc = svm.SVC(kernel='rbf', C=1,gamma= 0.1)
#svc.fit(train_x, train_y)
#print(svc.score(train_x, train_y))
#print(svc.score(test_x, test_y))


# In[7]:


#fig = plt.figure(1)
# 
#for i in range(-3,4,1):
#    c = 0
#    graph = np.zeros((7,3))
#    for j in range(-3,4,1):
#        svc = svm.SVC(kernel='poly', C=10**i,gamma= 10**j)
#        svc.fit(train_x, train_y)
#        #print ("for i = ",i," and j = ",j," :")
#        #print(" train acc-   ",svc.score(train_x, train_y))
#        #print(" test acc-   ",svc.score(test_x, test_y))
#        trn = svc.score(train_x, train_y)
#        tst = svc.score(test_x, test_y)
#        graph[c][0] = 10**j
#        graph[c][1] = trn
#        graph[c][2] = tst
#        c += 1
#    lbl1 =  'training accuracy for C = ',10**i    
#    lbl2 =  'test accuracy for C = ',10**i    
#    plt.plot(graph[0:,0],graph[0:,1],label = lbl1)
#    plt.plot(graph[0:,0],graph[0:,2],label = lbl2)
#
#    plt.title('accuracy vs gamma for poly kernel')
#    plt.ylabel('accuracy')
#    plt.xlabel('gamma')
#    plt.xscale('log')
#    plt.legend()
#    plt.show()


# In[ ]:

#a = 7 
#b = 5
#X = np.zeros((a,b))
#Y = np.zeros((a,b))
#Z1 = np.zeros((a,b))
#Z2 = np.zeros((a,b))
#for i in range(a):
#    for j in range(b):
#        c  = 10**(i-a/2) 
#        g = 10**(j-b/2)
#        svc = svm.SVC(kernel='rbf', C = c,gamma = g)
#        svc.fit(train_x, train_y)
#        trn = svc.score(train_x, train_y)
#        tst = svc.score(test_x, test_y)
#        X[i][j] = i-a/2
#        Y[i][j] = j-b/2
#        Z1[i][j] = trn
#        Z2[i][j] = tst
##
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_wireframe(X, Y, Z1,rstride=1, cstride=1,label = 'train',color = 'red' )
#ax.plot_wireframe(X, Y, Z2,rstride=1, cstride=1,label = 'test', color = 'blue')
#plt.title('accuracy vs C and gamma for rbf kernel')
#
#plt.ylabel('log(gamma)')
#plt.xlabel('log(C)')
##plt.zlabel('accuracy')
##ax.yaxis.set_scale('log')
##ax.xaxis.set_scale('log')
#plt.legend()
#plt.show()


c = 0
graph = np.zeros((7,3))
for i in range(-3,4,1):
    svc = svm.SVC(kernel='linear', C=10**i)
    svc.fit(train_x, train_y)
    #print ("for i = ",i," and j = ",j," :")
    #print(" train acc-   ",svc.score(train_x, train_y))
    #print(" test acc-   ",svc.score(test_x, test_y))
    trn = svc.score(train_x, train_y)
    tst = svc.score(test_x, test_y)
    graph[c][0] = 10**i
    graph[c][1] = trn
    graph[c][2] = tst
    c += 1
lbl1 =  'training accuracy'   
lbl2 =  'test accuracy'    
plt.plot(graph[0:,0],graph[0:,1],label = lbl1)
plt.plot(graph[0:,0],graph[0:,2],label = lbl2)

plt.title('accuracy vs C for linear kernel')
plt.ylabel('accuracy')
plt.xlabel('C')
plt.xscale('log')
plt.legend()
plt.show()
