#!/usr/bin/env python
# coding: utf-8

# In[1]:


def func(x):
    return 0.01*x**2 + 0.1*x


# In[5]:


import numpy as np
import matplotlib.pylab as plt

x = np.arange(0.0, 20.0, 0.1)
y = func(x)
plt.xlabel("x")
plt.plot(x,y)
plt.show


# In[7]:


def diff(f, x): #미분함수
    h = 1e-4
    return (f(x+h)-f(x-h))/ (2*h)


# In[9]:


diff(func, 5)


# In[10]:


def line_(f, x):    # f라는 함수에서 x에서의 접선식을 구하는 함수
    d = diff(f, x) # 접선의 기울기
    y = f(x) - (d*x) # 접선의 y절편
    return lambda t : d*t + y


# In[11]:


x = np.arange(0.0, 20.0, 0.1) # x범위 설정
y = func(x) 
line_1 = line_(func, 5) # func() 곡선 함수에서 x가 5일때의 접선인 1차식을 생성한다.
y2 = line_1(x) #line()함수의 return 값인 d*t + y라는 t가 변수인 함수에 x값을 넣어준다.

plt.xlabel("x")
plt.xlabel("y")
plt.plot(x,y)
plt.plot(x, y2)
plt.show

