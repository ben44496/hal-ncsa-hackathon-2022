import torch
import torch.nn as nn
import torchvision
from netCDF4 import Dataset
import numpy as np
import horovod.torch as hvd
from data import get_data,get_X_names
from unet.unet import *
import argparse

parser = argparse.ArgumentParser(description="UNet Hal Hackathon 2022")
parser.add_argument('--batch-size', type=int, default=3, metavar='N',
                    help='input batch size for training (default: 3)')

args = parser.parse_args()
BATCH_SIZE = args.batch_size

print("Batch size:", BATCH_SIZE)

print("Loading data")
x_names = get_X_names()
x_names = x_names[3:8] + x_names[25:30]
# X,Y,ymu,ystd = get_data(x_names[:10],['ccn_001','ccn_003'])
X = torch.load('data/trainX')
Y = torch.load('data/trainY')
# test_X = torch.load('data/testX')
# test_Y = torch.load('data/testY')

dataset = torch.utils.data.TensorDataset(X,Y)
print("Done Loading")

# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
torch.cuda.set_device(hvd.local_rank())

# Define dataset...
train_dataset = dataset

# Partition dataset among workers using DistributedSampler
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)

# Matty's del
del X
del Y
del dataset

# Build model...
model = UNet(10,2)
model.cuda()

optimizer = optim.Adam(model.parameters())

# Add Horovod Distributed Optimizer
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

# Broadcast parameters from rank 0 to all other processes.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
loss_fn = nn.MSELoss()
model.train()
losses = []
validation_losses = []

torch.cuda.empty_cache()
for epoch in range(101):
    print("epoch ", epoch)
    train_loss = 0 
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        print("Loaded Data")
        optimizer.zero_grad()
        print("Optimizer Zero Grad")
        output = model(data)
        print("Get model output")
        loss = loss_fn(output, target)
        print("Loss calculated")
        loss.backward()
        print("loss.backward()")
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 1 == 0:
            # Horovod: use train_sampler to determine the number of examples in
            # this worker's partition.
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_sampler),
                100. * batch_idx / len(train_loader), loss.item()))
        
    # Save model every 10 epochs
    if epoch % 1 == 0:
        torch.save(model.state_dict(),'checkpoints/unet_epoch_%d.pth' % (epoch))
        
#         rand_sample = torch.randint(0, 11, (1,)) # 0-11 test samples index
#         val_x = test_X[rand_sample].cuda()
#         val_y = test_Y[rand_sample].cuda()
#         y_hat = model(val_x)
#         val_loss = loss_fn(y_hat, val_y)
#         validation_losses.append(val_loss)
#         print("Validation Loss ", val_loss)
        
#         torch.save(torch.Tensor(validation_losses), 'checkpoints/validation_loss')
        torch.save(torch.Tensor(losses), 'checkpoints/train_loss')
        
#         del val_x
#         del val_y
    
    torch.cuda.empty_cache()

    
    print("Loss ",train_loss)
    losses.append(train_loss)
    
torch.save(model.state_dict(),'checkpoints/unet_final.pth')

