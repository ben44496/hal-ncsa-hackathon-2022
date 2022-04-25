import torch
import torch.nn as nn
import torchvision
from netCDF4 import Dataset
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import tempfile
import sys
import torch.nn.functional as F

from unet import * # from unet import UNet

import horovod.torch as hvd






def train():
    pass




if __name__ == "__main__":
    hvd.init()
    
    torch.cuda.set_device(hvd.local_rank())
    
    train_dataset = ...
    
    # Partition dataset among workers using DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)

    # Build model...
    model = ...
    model.cuda()

    optimizer = optim.SGD(model.parameters())

    # Add Horovod Distributed Optimizer
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    # Broadcast parameters from rank 0 to all other processes.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    for epoch in range(100):
       for batch_idx, (data, target) in enumerate(train_loader):
           optimizer.zero_grad()
           output = model(data)
           loss = F.nll_loss(output, target)
           loss.backward()
           optimizer.step()
           if batch_idx % args.log_interval == 0:
               print('Train Epoch: {} [{}/{}]\tLoss: {}'.format(
                   epoch, batch_idx * len(data), len(train_sampler), loss.item()))