#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[11]:


def init_network():  # 가중치와 편향값을 network에 저장하는 함수
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) # 2 * 3 행렬
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]) # 1층의 결과값이 1 * 3 행렬  --> W2는 3 *2 행렬
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]]) # 2층의 결과값이 1 * 2 행렬 --> W3는 2 * 2 행렬
    network['b3'] = np.array([0.1, 0.2])
    
    return network

def sigmoid(x) :  # 활성화 함수로 쓸 시그모이드 함수 선언
    return 1 / (1 + np.exp(-x))


# In[12]:


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3'] # forward 함수 내의 각 가중치에 network에 선언한 가중치 값을 가져온다.
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1= np.dot(x, W1) + b1 # 입력값(np.array)에 1층 가중치를 곱하고 편향을 더한다.
    z1 = sigmoid(a1) #시그모이드 활성화 함수에 a1 결과값을 넣는다. (시그모이드 함수는 내장함수가 아니라 따로 선언한 것)
    a2= np.dot(z1, W2) + b2 
    z2 = sigmoid(a2)
    a3= np.dot(z2, W3) + b3 
    y = a3  # 출력층의 활성화함수는 은닉층의 활성화 함수와 다름 (여기선 그냥 그대로 return함)
    
    return y


# In[13]:


network = init_network() # 가중치와 편향 선언
x = np.array([1.0, 0.5]) # 신경망에 넣을 입력값

y = forward(network, x) # 신경망 함수
y


# In[ ]:





# # for check

# In[6]:


print(init_network())
type(init_network())


# In[9]:


x = np.array([1,2]) #1 * 2 행렬
y = np.array([[1],[2]]) #2 * 1 행렬
np.dot(x,y)

