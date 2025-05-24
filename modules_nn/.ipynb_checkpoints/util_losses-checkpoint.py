import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence

class WeightedMSE:

    def __init__(self,
                 weights,
                 device,
                 hyperparam=2.0,
                 min_threshold=0,
                 max_threshold=0,
                 reduction='mean',
                 loss_area=None,
                 exclude_dim=None):
        
        self.reduction     = reduction
        self.hyperparam    = hyperparam
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.loss_area     = loss_area
        self.device        = device
        self.exclude_dim   = exclude_dim
        
        if self.loss_area is not None:

            if weights.ndim>1:
                lat_min, lat_max, lon_min, lon_max = self.loss_area
                self.weights = torch.from_numpy(weights[lat_min:lat_max+1,
                                                        lon_min:lon_max+1]).to(device)
            else:
                indices =  self.loss_area
                self.weights = torch.from_numpy(weights[indices]).to(device)
                
        else:
            self.weights = torch.from_numpy(weights).to(device)

    def __call__(self,
                 data,
                 target,
                 mask=None):

        if self.loss_area is not None:

            if self.weights.ndim>1:

                lat_min, lat_max, lon_min, lon_max = self.loss_area
                y_hat = data[...,
                             lat_min:lat_max+1,
                             lon_min:lon_max+1]
                y = target[...,
                           lat_min:lat_max+1,
                           lon_min:lon_max+1]

            else:
                
                indices = self.loss_area
                y_hat = data[...,
                             indices]
                y = target[...,
                           indices]
        else:
            y_hat = data
            y = target

        m = torch.ones_like(y)
        m[(y < self.min_threshold) & (y_hat >= 0)] *= self.hyperparam
        m[(y > self.max_threshold) & (y_hat <= 0)] *= self.hyperparam

        if mask is not None:
            weight = self.weights * mask
        else:
            weight = self.weights 
        
        dims_to_sum = tuple(d for d in range(y.dim()) if d != self.exclude_dim) if self.exclude_dim is not None else tuple(d for d in range(y.dim()))

        if self.reduction == 'mean':
            loss = ((y_hat - y)**2 * m * weight).sum(dims_to_sum) / (torch.ones_like(y) * weight).sum(dims_to_sum)
        elif self.reduction == 'sum':
            loss = torch.sum((y_hat - y)**2 * m,
                             dim = -1).mean()
        elif self.reduction == 'none':
            loss = (y_hat - y)**2 * m * (weight /weight.sum())
        
        return loss

