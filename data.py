import numpy as np
from netCDF4 import Dataset
import cartopy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch
import torchvision
import sys
import os

def get_X_names():
    aero = open('data/x_aerosols.txt', 'r')
    gas = open('data/x_gases.txt', 'r')
    aero_e = aero.readlines()
    gas_e = gas.readlines()
    feature_names = []
    for i in aero_e:
        feature_names.append(i.strip())
    for i in gas_e:
        feature_names.append(i.strip())
    aero.close()
    gas.close()
    return feature_names
    
def get_data(x_features, y_features,datapath):
    ncf = Dataset(datapath, "r", format="NETCDF4")
    resizer = torchvision.transforms.Resize((157, 157))
    
    N = ncf.variables['ccn_001'].shape[0]
    
    X = torch.empty(len(x_features), N, 39, 157, 157)
    for i,name in enumerate(x_features):
        data = ncf.variables[name]
        data = torch.from_numpy(np.array(data))
        # Crop boundaries
        data = data[:,:,1:-1,1:-1]
        # Resize and Standardize 
        data = resizer(data)
        data = (data - data.mean(dim=0))/data.std(dim=0)
        X[i] = data
    X = torch.permute(X, (1, 0, 2,3,4))
    
    Y = torch.empty(len(y_features),N,39,157,157)
    y_means = torch.empty(len(y_features),39,157,157)
    y_stds = torch.empty(len(y_features),39,157,157)
    for i,name in enumerate(y_features):
        data = torch.from_numpy(np.array(ncf.variables[name]))
        data = data[:,:,1:-1,1:-1]
        data = resizer(data)
        y_means[i]=data.mean(dim=0)
        y_stds[i]=data.std(dim=0)
        data = (data - data.mean(dim=0))/data.std(dim=0)
        Y[i] = data
        
    Y = torch.permute(Y, (1, 0, 2,3,4))
    
    return X,Y,y_means,y_stds