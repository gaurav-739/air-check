#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn import metrics
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from math import sqrt
import pickle
get_ipython().run_line_magic('matplotlib', 'inline')


# In[26]:


df=pd.read_csv(r"C:\Users\Jarvis\Downloads\Aircheck\air.csv")


# In[27]:


le=LabelEncoder()
df['State'] = le.fit_transform(df.State.values)
df['City'] = le.fit_transform(df.City.values)
df['Place'] = le.fit_transform(df.Place.values)


# In[28]:


df.dtypes


# In[29]:


df.head(2)


# In[30]:


df.isnull().any()


# In[31]:


df.isnull().sum()


# In[32]:


X=df.iloc[:,:5].values
Y=df.iloc[:,-1].values


# In[33]:


df.describe()


# In[34]:


plt.scatter(df['Max'],df['Min'],df['Avg'],df['State'])
plt.show()
 


# In[35]:


df.plot(kind='hist',figsize=(50,10))
plt.show()


# In[36]:


df.boxplot()


# In[37]:



df.plot(kind='line',figsize=(50,10))
plt.show()


# In[38]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# In[39]:


regressor = LinearRegression()  
regressor.fit(X_train,Y_train)


# In[40]:


Y_pred = regressor.predict(X_test)


# In[41]:


pickle.dump(regressor, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[0,4,101,108,42]]))


# In[42]:


print('Coefficients: \n', regressor.coef_)


# In[43]:


df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})
df.tail(5)


# In[44]:


print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))


# In[45]:


np.var(df)


# In[22]:


np.std(df)


# In[47]:


df.plot(kind='bar',figsize=(500,40))
plt.grid(which='major', linestyle='-', linewidth='1', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[46]:


score=r2_score(Y_test,Y_pred)
print(score)


# In[ ]:




