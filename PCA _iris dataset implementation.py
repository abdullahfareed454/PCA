#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import  matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from sklearn import datasets
from sklearn.decomposition import PCA


# In[4]:


iris =  datasets.load_iris()


# In[5]:


x = iris.data
y = iris.target


# In[6]:


x.shape


# In[7]:


y.shape


# In[8]:


pca = PCA(n_components=2)


# In[12]:


pca.fit(x)


# In[14]:


Z = pca.transform(x)


# In[16]:


Z.shape


# In[20]:


plt.scatter(Z[:,0],Z[:,1])


# In[21]:


R = np.array(iris.data)


# In[22]:


R_cov = np.cov(R, rowvar=False)


# In[23]:


import pandas as pd
iris_covmat = pd.DataFrame(data=R_cov, columns=iris.feature_names)
iris_covmat.index = iris.feature_names
iris_covmat


# In[24]:


eig_values, eig_vectors = np.linalg.eig(R_cov)


# In[25]:


eig_values
eig_vectors


# In[26]:


featureVector = eig_vectors[:,:2]
featureVector


# In[27]:


featureVector_t = np.transpose(featureVector)


# In[28]:


R_t = np.transpose(R)


# In[29]:


newDataset_t = np.matmul(featureVector_t, R_t)
newDataset = np.transpose(newDataset_t)


# In[30]:


newDataset.shape


# In[31]:


import seaborn as sns


# In[32]:


df = pd.DataFrame(data=newDataset, columns=['PC1', 'PC2'])
y = pd.Series(iris.target)
y = y.replace(0, 'setosa')
y = y.replace(1, 'versicolor')
y = y.replace(2, 'virginica')
df['Target'] = y 


# In[33]:


sns.lmplot(x='PC1', y='PC2', data=df, hue='Target', fit_reg=False, legend=True)


# In[ ]:




