
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

class RNNCell(nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W  = Parameter(torch.Tensor(hidden_size,input_size))
        self.Wo = Parameter(torch.Tensor(hidden_size, 1))
        self.U  = Parameter(torch.Tensor(hidden_size,hidden_size))
        self.b  = Parameter(torch.Tensor(hidden_size))
    
    def forward(self, x , hidden):
        hidden = torch.relu(x.matmul(self.W.t()) + hidden.matmul(self.U.t()) + self.b)
        out = hidden.matmul(self.Wo)

        return out, hidden



class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.W = Parameter(torch.Tensor(hidden_size))
        self.cell = RNNCell(input_size, hidden_size)
        
    
    
    def forward(self, input_):
        
        hidden = torch.randn(self.hidden_size)
        outputs = []
        
        for i in torch.unbind(input_, dim=0):#this could work or could be a terrible mistake
            #import ipdb; ipdb.set_trace()
            
            _, hidden = self.cell(i, hidden)
            outputs.append(hidden.clone())
        
        
        return outputs[-1].matmul(self.W)
        #return torch.stack(outputs,dim=0)


if __name__ == '__main__':


    #rnn = RNNCell(4,16)
    #out = rnn(torch.randn(4), torch.randn(16))

    rnn = RNN(4,16)
    out = rnn(torch.randn(10,1,4))
    pnn = nn.RNN(4,16)
    pout, _ = pnn(torch.randn(10,1,4))

    print(out.shape)
    print(pout.shape)



