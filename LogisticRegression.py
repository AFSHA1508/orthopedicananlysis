#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
import seaborn as sns
import pickle


# In[19]:


data=pd.read_csv('A:\orthopedic\dataset.csv')


# In[3]:


data.head(2)


# In[20]:


data.isnull().any()


# In[21]:


data.isnull().sum()


# In[22]:


data.dtypes


# In[23]:


label_encoder =  preprocessing.LabelEncoder()
a=data['class'] = label_encoder.fit_transform(data['class'])

print(a)


# In[6]:


data.head(2)


# In[7]:


data.isnull()


# In[7]:


data.dtypes


# In[9]:


print(data.shape)


# In[10]:


data['class'].value_counts()


# In[11]:


sns.countplot(x = 'class',data = data, palette = 'hls')


# In[24]:


X = pd.DataFrame(data.iloc[:,:-1])
y = pd.DataFrame(data.iloc[:,-1])


# In[25]:


X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2,random_state=1)


# In[26]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)


# In[27]:


y_pred = logmodel.predict(X_test)


# In[28]:


print('Accuracy : %d',(logmodel.score(X_test, y_test)))


# In[16]:


from pandas.plotting import scatter_matrix
scatter_matrix(data,figsize=(10,10))
plt.show()


# In[17]:


data.describe()


# In[29]:


pickle.dump(logmodel, open('LogisticRegression.pkl','wb'))

model = pickle.load(open('LogisticRegression.pkl','rb'))
print(model.predict([[63.02,22.55,39.60,40.47,98.67,-0.25]]))


# In[30]:


from sklearn import metrics

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[20]:


data.plot(kind='bar',figsize=(500,40))
plt.grid(which='major', linestyle='-', linewidth='1', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[21]:


plt.hist(data['pelvic_incidence'],
        facecolor='cyan',
        edgecolor='blue',
        bins=20)
plt.show()


# In[29]:


plt.hist(data['pelvic_tilt numeric'],
        facecolor='cyan',
        edgecolor='blue',
        bins=20)
plt.show()


# In[31]:


plt.hist(data['lumbar_lordosis_angle'],
        facecolor='cyan',
        edgecolor='blue',
        bins=20)
plt.show()


# In[34]:


plt.hist(data['sacral_slope'],
        facecolor='cyan',
        edgecolor='blue',
        bins=24)
plt.show()


# In[36]:


plt.hist(data['pelvic_radius'],
        facecolor='cyan',
        edgecolor='blue',
        bins=24)
plt.show()


# In[41]:


plt.hist(data['degree_spondylolisthesis'],
        facecolor='cyan',
        edgecolor='blue',
        bins=24)
plt.show()


# In[42]:


np.var(data)


# In[43]:


np.std(data)


# In[44]:


data.describe()


# In[22]:


data.boxplot()


# In[26]:


data.plot(kind='hist',figsize=(50,10))
plt.show()


# In[31]:


data.plot(kind='line',figsize=(50,10))
plt.show()


# In[ ]:





# In[ ]:




