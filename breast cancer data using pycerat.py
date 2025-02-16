#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pycaret')


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from pycaret.classification import *
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# In[4]:


df = pd.read_csv(r"C:\Users\muthy\Downloads\archive (1)\data.csv")
df.head()


# In[5]:


# delete unnecessary columns
df = df.drop(columns=['id', 'Unnamed: 32'], axis=1)


# In[6]:


# statistical info
df.describe()


# In[7]:


# datatype info
df.info()


# In[10]:


df_temp = df.drop(columns=['diagnosis'], axis=1)


# In[11]:


# create dist plot
fig, ax = plt.subplots(ncols=6, nrows=5, figsize=(20, 20))
index = 0
ax = ax.flatten()

for col in df_temp.columns:
    sns.distplot(df[col], ax=ax[index])
    index+=1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)


# In[12]:


# create box plot
fig, ax = plt.subplots(ncols=6, nrows=5, figsize=(20, 20))
index = 0
ax = ax.flatten()

for col in df_temp.columns:
    sns.boxplot(y=col, data=df, ax=ax[index])
    index+=1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)


# In[13]:


# setup the data
clf = setup(df, target='diagnosis')


# In[14]:


# train and test the models
compare_models()

