#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv('Housing.csv')


# In[2]:


df


# In[3]:


df.info()


# In[4]:


df.columns


# In[5]:


col=['area', 'bedrooms', 'bathrooms', 'stories','parking','price']
for i in col:
    print(df[i].value_counts())


# In[6]:


df.shape


# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt
col=['area']
for i in col:
    print(sns.boxplot(y=df[i]))
    plt.figure()


# In[8]:


col=['area']
for i in col:
    q1=df[i].quantile(.25)
    q3=df[i].quantile(.75)
    iqr=q3-q1
    upper_extreme=q3+ (1.5 * iqr)
    lower_extreme=q1- (1.5*iqr)
    df=df[df[i]>= lower_extreme]
    df=df[df[i]<= upper_extreme]
    


# In[9]:


df.shape


# In[10]:


df.bathrooms.value_counts()


# In[11]:


import seaborn as sns
import matplotlib.pyplot as plt
col=['area']
for i in col:
    print(sns.boxplot(y=df[i]))
    plt.figure()
plt.show()


# In[12]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[13]:


df


# In[14]:


col=['mainroad', 'guestroom','basement', 'hotwaterheating', 
     'airconditioning','prefarea','furnishingstatus']
for i in col:
    df[i]= le.fit_transform(df[i])


# In[15]:


df


# In[16]:


X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values


# In[17]:


X.shape


# In[18]:


X


# In[19]:


Y.shape


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


X_train, X_test, Y_train,Y_test= train_test_split(X,Y,test_size=.25,random_state=67)


# In[22]:


X_train.shape


# In[23]:


X_test


# In[24]:


Y_train


# In[25]:


Y_test


# In[26]:


X_train.shape


# In[27]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[28]:


X_train=sc.fit_transform(X_train)


# In[29]:


X_test=sc.transform(X_test)


# In[30]:


X_train


# In[33]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()


# In[34]:


reg.fit(X_train,Y_train)


# In[35]:


reg.coef_


# In[36]:


reg.intercept_


# In[37]:


Y_pred=reg.predict(X_test)


# In[38]:


Y_pred


# In[40]:


from sklearn import metrics


# In[41]:


metrics.mean_squared_error(Y_test,Y_pred)


# In[42]:


import numpy as np
np.sqrt(metrics.mean_squared_error(Y_test,Y_pred))


# In[43]:


r2=metrics.r2_score(Y_test,Y_pred)
n=df.shape[0] #sample size
p=df.shape[1] # columns/independent variables


# In[44]:


df.shape


# In[ ]:




