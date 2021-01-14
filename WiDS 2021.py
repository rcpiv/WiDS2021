#!/usr/bin/env python
# coding: utf-8

# In[4]:


#import the necessary libraries: 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[5]:


#calling this option to stop the rows and columns from being truncated in their display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[2]:


train = pd.read_csv('TrainingWiDS2021.csv')


# In[7]:


train.shape


# In[14]:


train.head()


# In[8]:


train.diabetes_mellitus.mean()


# 21.6% of the patients in the training dataset have diabetes_mellitus

# In[23]:


train.age.hist()


# In[12]:


train.describe()


# In[ ]:


-thangs to look at:
    -age where < 5 vs weight


# In[15]:


vars = train.columns.tolist()
vars


# In[20]:


round(train.isna().sum()/len(train)*100,2)


# In[22]:


train.albumin_apache.unique()


# In[ ]:


#variables to exclude from analysis
-hospital_admit_source - 25% missing and no vital info
-


# In[ ]:


variables with large % of missing values
-albumin_apache - 60%

