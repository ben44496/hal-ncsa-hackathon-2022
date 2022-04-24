import numpy as np
from netCDF4 import Dataset
import cartopy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch
import torchvision
import sys
import os

def get_data(num_features):
    ncf = Dataset("data/training.nc", "r", format="NETCDF4")
    aero = open('data/x_aerosols.txt', 'r')
    gas = open('data/x_gases.txt', 'r')
    aero_e = aero.readlines()
    gas_e = gas.readlines()
    feature_names = []
    for i in aero_e:
        feature_names.append(i.strip())
    for i in gas_e:
        feature_names.append(i.strip())
        
    resizer = torchvision.transforms.Resize((157, 157))
    X = torch.empty(num_features, 133, 39, 157, 157)
    for i in range(num_features):
        variable = feature_names[i]
        data = ncf.variables[variable]
        data = torch.from_numpy(np.array(data))
        # Crop boundaries
        data = data[:,:,1:-1,1:-1]
        # Resize and Standardize 
        data = resizer(data)
        data = (data - data.mean(dim=0))/data.std(dim=0)
        X[i] = data
    X = torch.permute(X, (1, 0, 2,3,4))
    
    Y = torch.from_numpy(np.array(ncf.variables['ccn_001']))
    Y = resizer(Y)
    Y = (Y-Y.mean(dim=0))/Y.std(dim=0)
    return X,Y