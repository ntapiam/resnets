#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 14:19:32 2021

Read out the ResNet as trained in resnet.py and further analyze it.

@author: bayerc
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from p_var import p_var_backbone, p_var_backbone_ref

## Check if the variations should be re-computed
nLayers = 128
nEpochs = 100

##fNameBase = "resnet" + str(nLayers) + "_relu_lin_e" + str(nEpochs)
fNameBase = f"resnet{nLayers}_relu_lin_e{nEpochs}"
fName = fNameBase + ".pth"

## To get parameters: resnet.resnet_stack[i].block[1].weight.data
## for the ith ResNet-block, i=1,.. (i=0 is a Flatten layer)

if not os.path.isfile(fNameBase + "_weights.npy"):
    resnet_dic = torch.load(fName)
    resnet_vals = list(resnet_dic.values())
    # now need to convert the parameters of Resnet blocks to numpy arrays
    dW_list = [resnet_vals[i].cpu().numpy() for i in range(1, nLayers)]
    dW = np.array(dW_list)
    W = np.cumsum(dW, axis=0)
    
    ## Save the weights
    with open(fNameBase + "_weights.npy", 'wb') as f:
        np.save(f, dW)
        np.save(f, W)
else:
    with open(fNameBase + "_weights.npy", 'rb') as f:
        dW = np.load(f)
        W = np.load(f)

iIndex = 0
jIndex = 0
pGrid = np.linspace(1.0, 3.0, 100)

if not os.path.isfile(fNameBase + "_pVar.npy"):
    ## Compute p-variations of 1-dimensional sample path
    path_dist = lambda k, l: np.abs(W[k,iIndex,jIndex] - W[l,iIndex,jIndex])
    
    pVars_i_j = np.array([p_var_backbone(nLayers-1, p, path_dist).value for p in pGrid])
    
    ## Compute the p-variation for the whole matrix, using a specific matrix norm
    mNorm = 'fro'
    path_dist = lambda k,l: np.linalg.norm(W[k,:,:] - W[l,:,:], mNorm)
    
    pVarsFro = np.array([p_var_backbone(nLayers-1, p, path_dist).value for p in pGrid])
    
    mNorm = 2
    path_dist = lambda k,l: np.linalg.norm(W[k,:,:] - W[l,:,:], mNorm)
    fName2 = fNameBase + "_pVar_2.npy"
    
    pVars2 = np.array([p_var_backbone(nLayers-1, p, path_dist).value for p in pGrid])
    
    with open(fNameBase + "_pVar.npy", "wb") as f:
        np.save(f, pVars_i_j)
        np.save(f, pVarsFro)
        np.save(f, pVars2)
else:
    with open(fNameBase + "_pVar.npy", "rb") as f:
        pVars_i_j = np.load(f)
        pVarsFro = np.load(f)
        pVars2 = np.load(f)
        
## Plot the p-variations
plt.plot(pGrid, pVars_i_j, "r-")
plt.xlabel("p")
plt.title(f"p-Var of W[{iIndex},{jIndex}]")
plt.show()

plt.plot(pGrid, pVarsFro, "b-")
plt.xlabel("p")
plt.title("p-Var of W w.r.t. Frobenius norm")
plt.show()

plt.plot(pGrid, pVars2, "g-")
plt.xlabel("p")
plt.title("p-Var of W w.r.t. spectral norm")
plt.show()