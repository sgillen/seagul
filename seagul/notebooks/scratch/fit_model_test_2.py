#!/usr/bin/env python
# coding: utf-8

# In[1]:
import sys
sys.path.insert(1, './../../')
from nn import MLP, fit_model, RBF
import torch
import matplotlib.pyplot as plt
import time
start_time = time.time()
# In[2]:


train_size = 10000

input_size = 1
output_size = 1
hidden_size = 64
num_layers = 2


network = "RBF"
if network == "RBF":
    net = RBF(input_size, output_size, hidden_size)
    # training of the centres and variances
    X = torch.randn(train_size/2,input_size)
    Y = 2*X
    loss_hist = fit_model(net,X,Y,5)
    # training of the weights
    X = torch.randn(train_size/2,input_size)
    Y = 2*X
    loss_hist = fit_model(net,X,Y,5)
else
    net = MLP(input_size, output_size, num_layers, hidden_size)
    X = torch.randn(train_size,input_size)
    Y = 2*X 
    loss_hist = fit_model(net,X,Y,5)
# In[3]:

# You should see loss get to effectively zero
loss_hist = fit_model(net,X,Y,5)
plt.plot(loss_hist)


# In[4]:


# test error should be close to zero, MLP got to ~.0002 after 5 epochs of training on 4000 examples
X_test = torch.randn(train_size,input_size)
Y_test = 2*X_test
Y_pred = net(X_test)

test_error = (Y_test - Y_pred)**2
print(test_error.mean())
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()

# In[ ]:




