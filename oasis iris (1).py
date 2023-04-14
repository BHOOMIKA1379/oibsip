#!/usr/bin/env python
# coding: utf-8

# In[59]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


# In[60]:


df=pd.read_csv(r"C:\Users\bhoomika tadala\Downloads\iris real.csv")


# In[61]:


data.head()


# In[62]:


df['Species'].unique()


# In[63]:


df.info()


# In[64]:


df.head()


# In[65]:


df.tail()


# In[66]:


df.shape


# In[67]:


df.isnull().sum()


# In[68]:


df.dtypes


# In[69]:


data=df.groupby('Species')


# In[70]:


data.head()


# In[77]:


plt.boxplot(df['SepalLengthCm'])


# In[72]:


plt.boxplot(df['SepalWidthCm'])


# In[73]:


plt.boxplot(df['PetalLengthCm'])


# In[74]:


plt.boxplot(df['PetalWidthCm'])


# In[76]:


sns.heatmap(df.corr())


# In[78]:


df.drop('Id',axis=1,inplace=True)


# In[79]:


sp={'Iris-setosa':1,'Iris-versicolor':2,'Iris-virginica':3}


# In[81]:


df.Species=[sp[i] for i in df.Species]


# In[82]:


df


# In[83]:


X=df.iloc[:,0:4]


# In[84]:


X


# In[85]:


y=df.iloc[:,4]


# In[86]:


y


# In[87]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)


# In[88]:


model=LinearRegression()


# In[89]:


model.fit(X,y)


# In[90]:


model.coef_


# In[91]:


model.intercept_


# In[30]:


y_pred=model.predict(X_test)


# In[31]:


print("Mean squared error: %.2f" % np.mean((y_pred - y_test) ** 2))

