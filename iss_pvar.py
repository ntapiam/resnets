#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 10:05:55 2021

@author: bayerc
"""
import torch
import numpy as np
from iss2 import compute
from p_var import p_var_backbone, p_var_backbone_ref

def p_var_2(x, v_norm, m_norm, p, device = "cpu"):
    """Compute the p-variation norm of the degree 2 part of the iterated sum
    signature of a time series.
    
    Parameters
    ----------
    x : np.ndarray. Array of shape (M,d), representing a time series of M steps
            with entries in R^d.
    v_norm: function. A norm on R^d. Accepts torch.tensor.
    m_norm : function. A matrix norm on R^{dxd}. norm is supposed to accept a
            torch.tensor as argument.
    p : float. A positive scalar.
    device : string. The device to use for torch, "cuda" or "cpu"
    
    Output
    ------
    The (homogeneous) p-variation norms of the three components of the ISS of
    degree 2."""
    assert len(x.shape) == 2
    M = x.shape[0]
    X = torch.tensor(x).to(device)
    ## Construct norm(S_{n,m}) for all n < m.
    S0n_l = _get_S0n(X, range(1, M))
    S0n_il = [g.inv() for g in S0n_l]
    ## Now construct the norms of all ISS S_{i,j} for 0 <= i < j < M
    chi = {} # empty dictionary for indexing
    counter = 0
    ## ndarrays for holding the norms
    a_norms = np.full((M-1)*M/2, np.nan)
    S_norms = np.full((M-1)*M/2, np.nan)
    D_norms = np.full((M-1)*M/2, np.nan)
    for j in range(1, M):
        for i in range(j):
            chi[(i,j)] = counter
            Sij = S0n_l[j-1] if i == 0 else S0n_il[i-1] * S0n_l[j-1]
            a_norms[counter] = v_norm(Sij.a)
            S_norms[counter] = m_norm(Sij.S)
            D_norms[counter] = m_norm(Sij.D)
            counter = counter + 1
    ## Now compute the p-variation norms
    a_var = p_var_backbone_ref(M, p, lambda i,j: a_norms[chi[(i,j)]]) ** (1/p)
    S_var = p_var_backbone_ref(M, p/2, lambda i,j: S_norms[chi[(i,j)]]) ** (2/p)
    D_var = p_var_backbone_ref(M, p/2, lambda i,j: D_norms[chi[(i,j)]]) ** (2/p)
    return a_var, S_var, D_var
    
def _get_S0n(X, n_list):
    """Compute the ISS of X between step 0 and n for all n in n_list."""
    return [compute(X[0:(n+1),:]) for n in n_list]

def l2_norm(X):
    """Compute the L2 norm of a general tensor X, i.e., the square root of the
    sum of all squared entries."""
    return torch.linalg.norm(X)    