#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


# In[59]:


df = pd.read_csv("C:\\Users\shashikant\Desktop\polynomial_regression\polynomial.csv")
df


# In[58]:


model = RandomForestRegressor(n_estimators=800,random_state=0)


# In[60]:


x = df[['level']].values
x


# In[14]:


y = df[['salary']].values
y


# In[15]:


model.fit(x,y)


# In[17]:


model.predict([[6]])


# In[64]:


x_grid = np.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape(len(x_grid),1)
x_grid


# In[38]:


plt.title('Random Forest Algorithm')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.scatter(x,y,color='r')
plt.plot(x_grid,model.predict(x_grid),color='b')

