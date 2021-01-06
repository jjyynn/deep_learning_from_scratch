#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##손실함수 (이 식에서 y와 t는 batch가 아님)


# In[2]:


import numpy as np
import sys


# In[7]:


# 오차제곱합 (SSE)
def sse(y, t): # y와 t는 모두 np.array이고, y는 신경망의 출력, t는 정답 
    return 0.5 * np.sum((y - t) ** 2)


# In[6]:


# 교차 엔트로피 오차 (CEE)
def cee(y, t):
    delta = -(sys.maxsize)
    return -np.sum(t * np.log(y + delta))

