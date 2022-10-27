#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Opening a csv file from an URL
import pandas as pd
import requests
from io import StringIO

#paste url with data here
url = "https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv"

#this is snippet
headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:66.0) Gecko/20100101 Firefox/66.0"}
req = requests.get(url, headers=headers)
data = StringIO(req.text)

pd.read_csv(data)


# In[5]:


#paste url with data here
url = "https://raw.githubusercontent.com/shivaiitb21/100-days-of-machine-learning/main/day15%20-%20working%20with%20csv%20files/movie_titles_metadata.tsv"

#this is snippet
headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:66.0) Gecko/20100101 Firefox/66.0"}
req = requests.get(url, headers=headers)
data = StringIO(req.text)

pd.read_csv(data, sep='\t',names=['sno','name','release_year','rating','votes','genres'])


# In[6]:


pd.read_csv('aug_train.csv',index_col='enrollee_id')


# In[7]:


pd.read_csv('test.csv',header=1)


# In[8]:


pd.read_csv('aug_train.csv',usecols=['enrollee_id','gender','education_level'])


# In[9]:


pd.read_csv('aug_train.csv',usecols=['gender'],squeeze=True)


# In[10]:


pd.read_csv('aug_train.csv',nrows=100)


# In[11]:


pd.read_csv('zomato.csv',encoding='latin-1')


# In[14]:


pd.read_csv('zomato.csv', encoding="latin-1", error_bad_lines=False)


# In[15]:


pd.read_csv('aug_train.csv',dtype={'target':int}).info()


# In[16]:


pd.read_csv('IPL Matches 2008-2020.csv',parse_dates=['date']).info()


# In[17]:


def rename(name):
    if name == "Royal Challengers Bangalore":
        return "RCB"
    else:
        return name


# In[18]:


pd.read_csv('IPL Matches 2008-2020.csv',converters={'team1':rename})


# In[19]:


pd.read_csv('aug_train.csv',na_values=['Male',])


# In[20]:


dfs = pd.read_csv('aug_train.csv',chunksize=5000)


# In[22]:


for chunks in dfs:
    print(chunks.shape)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




