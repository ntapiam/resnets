#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 14:19:32 2021

Read out the ResNet as trained in resnet.py and further analyze it.

@author: bayerc
"""

import torch
import numpy as np
from p_var import p_var_backbone, p_var_backbone_ref

## Check if the variations should be re-computed
nLayers = 128
nEpochs = 100

##fNameBase = "resnet" + str(nLayers) + "_relu_lin_e" + str(nEpochs)
fNameBase = f"resnet{nLayers}_relu_lin_e{nEpochs}"
fName = fNameBase + ".pth"

## To get parameters: resnet.resnet_stack[i].block[1].weight.data
## for the ith ResNet-block, i=1,.. (i=0 is a Flatten layer)

resnet_dic = torch.load(fName)
resnet_vals = list(resnet_dic.values())
# now need to convert the parameters of Resnet blocks to numpy arrays
dW_list = [resnet_vals[i].cpu().numpy() for i in range(1, nLayers)]
dW = np.array(dW_list)

## Save the weights
with open(fNameBase + "_weights.npy", 'wb') as f:
    np.save(f, dW)

## Now compute the path W
W = np.cumsum(dW, axis=0)
with open(fNameBase + "_weights_cumsum.npy", 'wb') as f:
    np.save(f, W)

## Compute p-variations of 1-dimensional sample path
iIndex = 0
jIndex = 0
pGrid = np.linspace(1.0, 3.0, 100)
path_dist = lambda k, l: np.abs(W[k,iIndex,jIndex] - W[l,iIndex,jIndex])
fName_i_j = fNameBase + f"_pVar_{iIndex}_{jIndex}.npy"

pVars_i_j = np.array([p_var_backbone(nLayers-1, p, path_dist).value for p in pGrid])

with open(fName_i_j, "wb") as f:
    np.save(f, pVars_i_j)

## Compute the p-variation for the whole matrix, using a specific matrix norm
mNorm = 'fro'
pGrid = np.linspace(1.0, 3.0, 100)
path_dist = lambda k,l: np.linalg.norm(W[k,:,:] - W[l,:,:], mNorm)
fNameFro = fNameBase + "_pVar_fro.npy"

pVarsFro = np.array([p_var_backbone(nLayers-1, p, path_dist).value for p in pGrid])

with open(fNameFro, "wb") as f:
    np.save(f, pVarsFro)

mNorm = 2
pGrid = np.linspace(1.0, 3.0, 100)
path_dist = lambda k,l: np.linalg.norm(W[k,:,:] - W[l,:,:], mNorm)
fName2 = fNameBase + "_pVar_2.npy"

pVars2 = np.array([p_var_backbone(nLayers-1, p, path_dist).value for p in pGrid])
with open(fName2, "wb") as f:
    np.save(f, pVars2)