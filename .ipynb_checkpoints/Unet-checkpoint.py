import torch
import torch.nn as nn
import torchvision
from netCDF4 import Dataset
import numpy as np
import horovod.torch as hvd
from data import get_data,get_X_names
from unet.unet import *

# from pytorch_model_summary import summary


# In[11]:


# ### COPY THIS

# In[5]:

print("Loading data")
x_names = get_X_names()
x_names = x_names[3:8] + x_names[25:30]
# X,Y,ymu,ystd = get_data(x_names[:10],['ccn_001','ccn_003'])
X = torch.load('data/trainX')
Y = torch.load('data/trainY')

dataset = torch.utils.data.TensorDataset(X,Y)
print("Done Loading")

# In[ ]:

# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
torch.cuda.set_device(hvd.local_rank())

# Define dataset...
train_dataset = dataset

# Partition dataset among workers using DistributedSampler
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=3, sampler=train_sampler)

# Build model...
model = UNet(10,2)
model.cuda()

optimizer = optim.SGD(model.parameters(),lr=0.001)

# Add Horovod Distributed Optimizer
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

# Broadcast parameters from rank 0 to all other processes.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
loss_fn = nn.MSELoss()
model.train()
for epoch in range(100):
    print("epoch ", epoch)
    train_loss = 0 
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print("Loss ",train_loss)