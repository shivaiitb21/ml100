#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[6]:


df = pd.read_csv("https://raw.githubusercontent.com/campusx-official/100-days-of-machine-learning/main/kmeans/student_clustering.csv")
print(df.shape)
df.head()


# In[7]:


plt.scatter(df['cgpa'],df["iq"])


# In[12]:


from sklearn.cluster import KMeans


# In[15]:


#WCSS = Within Cluster Sum of Squares -- This is also known as the inertia

wcss = []

for i in range(1,11):
    km = KMeans(n_clusters=i)
    km.fit_predict(df)
    wcss.append(km.inertia_)


# In[16]:


wcss


# In[17]:


plt.plot(range(1,11),wcss)
#It gives elbow point
#no of clusters is 4


# In[18]:


X = df.iloc[:,:].values
km = KMeans(n_clusters=4)
y_means = km.fit_predict(X)


# In[19]:


y_means


# In[27]:


plt.scatter(X[y_means==0,0],X[y_means==0,1], color="blue")
plt.scatter(X[y_means==1,0],X[y_means==1,1],color="red")
plt.scatter(X[y_means==2,0],X[y_means==2,1],color="green")
plt.scatter(X[y_means==3,0],X[y_means==3,1],color="magenta")


# In[38]:


km.predict(X[:1])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




