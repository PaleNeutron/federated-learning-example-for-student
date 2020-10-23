#!/usr/bin/env python
# coding: utf-8

# In[5]:


from preprocess import *
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport


# In[2]:


UserRoundData._load_data = print
ud = UserRoundData()
dfs = ud._get_raw_df()


# In[6]:


profile = ProfileReport(pd.concat(dfs), title='Pandas Profiling Report')


# In[ ]:


profile.to_file("all_data_report.html")


# In[ ]:




