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


df1 = pd.read_csv('http://web.iitd.ac.in/~sumeet/A2/train_set.csv', header=None, nrows=10000)
train_input = df1.values
data = np.array(train_input)


# In[3]:


x_tr = data[:,:25]/1000
y_tr = data[:,25]
train_x, test_x, train_y, test_y = train_test_split(x_tr,y_tr,test_size=0.2, random_state=30, stratify=y_tr)


# In[7]:


#steps = [('scaler', StandardScaler()), ('SVM', svm.SVC(kernel='poly'))]
#pipeline = Pipeline(steps) # define Pipeline object
#
#parameters = {'SVM__C':[0.001, 0.1, 100], 'SVM__gamma':[10,1,0.1,0.01]}
#
#grid = GridSearchCV(pipeline, param_grid=parameters, cv=5)


# In[8]:

#
#grid.fit(train_x, train_y)
#print (grid.score(test_x, test_y))
#print (grid.best_params_)


# In[55]:


#svc = svm.SVC(kernel='poly', C=0.0008,gamma= 1.23)
#svc.fit(train_x, train_y)
#print(svc.score(train_x, train_y))
#print(svc.score(test_x, test_y))
#print(svc.predict(test_x))


svc = svm.SVC(kernel='rbf', C=1.2,gamma= 6.825)
svc.fit(train_x, train_y)
print(svc.score(train_x, train_y))
print(svc.score(test_x, test_y))
print(svc.predict(test_x))


# In[56]:


svc.fit(x_tr,y_tr)
print(svc.score(x_tr,y_tr))
df2 = pd.read_csv('http://web.iitd.ac.in/~sumeet/A2/test_set.csv', header=None, nrows=2000)
test_input = df2.values
test_data = np.array(test_input)/1000
prediction = svc.predict(test_data)
prediction = prediction.astype(int)
print (prediction)
submission = np.zeros((2000,2))
for i in range(2000):
    submission[i][0] = i
    submission[i][1] = int(prediction[i])
print (submission)
np.savetxt("submission.csv", submission, delimiter=",")


# In[ ]:




