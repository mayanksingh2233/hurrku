#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


dataset=pd.read_csv('LR.csv')
dataset.head()


# In[3]:


x=dataset[['Age','Salary']]
y=dataset['Purchased']


# In[4]:


x


# In[5]:


y


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[7]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[11]:


lr=LogisticRegression()


# In[12]:


lr.fit(x_train,y_train)


# In[13]:


import pickle


# In[20]:


pickle.dump(lr,open('model.pkl','wb'))


# In[21]:


model=pickle.load(open('model.pkl','rb'))


# In[22]:


print(model.predict([[19,20000]]))


# In[ ]:




