#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 14:19:32 2021

Read out the ResNet as trained in resnet.py and further analyze it.

@author: bayerc
"""

import numpy as np
import matplotlib.pyplot as plt

nLayers = 128
nEpochs = 100

## save figures?
saveP = True

##fNameBase = "resnet" + str(nLayers) + "_relu_lin_e" + str(nEpochs)
fNameBase = f"resnet{nLayers}_relu_lin_e{nEpochs}"
fName = fNameBase + ".pth"

## Load the weights
with open(fNameBase + "_weights.npy", 'rb') as f:
    dW = np.load(f)
    
with open(fNameBase + "_weights_cumsum.npy", 'rb') as f:
    W = np.load(f) 

## Plot some sample paths
plt.plot(dW[:,3,90])
if saveP:
    plt.savefig("./plots/"+fNameBase+f"increments_{3}_{90}.pdf")
plt.show()

## plot sample path of the process W
plt.plot(W[:,3,90])
if saveP:
    plt.savefig("./plots/"+fNameBase+f"path_{3}_{90}.pdf")
plt.show()


## Compute p-variations of 1-dimensional sample path
iIndex = 0
jIndex = 0
pGrid = np.linspace(1.0, 3.0, 100)
fName_i_j = fNameBase + f"_pVar_{iIndex}_{jIndex}.npy"
with open(fName_i_j, "rb") as f:
    pVars_i_j = np.load(f)
    
plt.plot(pGrid, pVars_i_j**(1.0/pGrid))
if saveP:
    plt.savefig("./plots/"+fNameBase+f"pVar_{iIndex}_{jIndex}.pdf")
plt.show()

## Compute the p-variation for the whole matrix, using a specific matrix norm
mNorm = 'fro'
pGrid = np.linspace(1.0, 3.0, 100)
fNameFro = fNameBase + "_pVar_fro.npy"

with open(fNameFro, "rb") as f:
    pVarsFro = np.load(f)
        
plt.plot(pGrid, pVarsFro**(1.0/pGrid))
if saveP:
    plt.savefig("./plots/"+fNameBase+"pVar_Fro.pdf")
plt.show()

mNorm = 2
pGrid = np.linspace(1.0, 3.0, 100)
fName2 = fNameBase + "_pVar_2.npy"

with open(fName2, "rb") as f:
    pVars2 = np.load(f)
        
plt.plot(pGrid, pVars2**(1.0/pGrid))
if saveP:
    plt.savefig("./plots/"+fNameBase+"pVar_2.pdf")
plt.show()