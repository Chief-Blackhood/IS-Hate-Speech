import os
import torch
from torch import nn
import numpy

class MultiTaskLoss(nn.Module):
  def __init__(self, n_tasks, reduction='none'):
    super(MultiTaskLoss, self).__init__()
    self.n_tasks = n_tasks
    self.log_vars = nn.Parameter(torch.zeros(self.n_tasks))
    self.reduction = reduction

  def forward(self, losses):
    dtype = losses.dtype
    device = losses.device
    stds = (torch.exp(self.log_vars)**(1/2)).to(device).to(dtype)
    multi_task_losses = (1 / (stds ** 2)) * losses + torch.log(stds)

    if self.reduction == 'sum':
      multi_task_losses = multi_task_losses.sum()
    if self.reduction == 'mean':
      multi_task_losses = multi_task_losses.mean()

    return multi_task_losses