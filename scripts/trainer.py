#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 17:48:03 2018

@author: zqwu
"""


import torch as t
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from copy import deepcopy
import numpy as np
from sklearn.metrics import r2_score
from models import CELoss
from torch.utils.data import Dataset, DataLoader
import os

class Trainer(object):
  def __init__(self, 
               net, 
               opt, 
               criterion=CELoss,
               featurize=True):
    self.net = net
    self.opt = opt
    self.criterion = criterion
    if self.opt.gpu:
      self.net = self.net.cuda()
  
  def assemble_batch(self, data, batch_size=None, sort=True):
    if batch_size is None:
      batch_size = self.opt.batch_size
    # Sort by length
    if sort:
      lengths = [x[0].shape[0] for x in data]
      order = np.argsort(lengths)
      data = [data[i] for i in order]
    
    # Assemble samples with similar lengths to a batch
    data_batches = []
    for i in range(int(np.ceil(len(data)/float(batch_size)))):
      batch = data[i * batch_size:min((i+1) * batch_size, len(data))]
      batch_length = max([sample[0].shape[0] for sample in batch])
      out_batch_X = []
      out_batch_X_metas = []
      out_batch_y = []
      for sample in batch:
        sample_length = sample[0].shape[0]
        X = deepcopy(sample[0][:])
        y = deepcopy(sample[2][:])
        out_batch_X.append(np.pad(X, ((0, batch_length - sample_length), (0, 0)), 'constant'))
        out_batch_X_metas.append(sample[1])
        out_batch_y.append(np.pad(y, ((0, batch_length - sample_length), (0, 0)), 'constant'))
        assert X.shape[0] - y.shape[0] == 1
        
      if len(batch) < batch_size:
        pad_length = batch_size - len(batch)
        out_batch_X.extend([out_batch_X[0]] * pad_length)
        out_batch_X_metas.extend([out_batch_X_metas[0]] * pad_length)
        out_batch_y.extend([out_batch_y[0]] * pad_length)
        
      data_batches.append((np.stack(out_batch_X, axis=1), 
                           np.stack(out_batch_X_metas, axis=0), 
                           np.stack(out_batch_y, axis=1)))
    return data_batches
  
  def save(self, path):
    t.save(self.net.state_dict(), path)
  
  def load(self, path):
    s_dict = t.load(path)
    self.net.load_state_dict(s_dict)
  
  def train(self, train_data, n_epochs=None, **kwargs):
    self.run_model(train_data, train=True, n_epochs=n_epochs, **kwargs)
    return
  
  def display_loss(self, train_data, **kwargs):
    self.run_model(train_data, train=False, n_epochs=1, **kwargs)
    return
    
  def run_model(self, data, train=False, n_epochs=None, **kwargs):
    if train:
      optimizer = Adam(self.net.parameters(),
                       lr=self.opt.lr,
                       betas=(.9, .999))
      self.net.zero_grad()
      epochs = self.opt.max_epoch
    else:
      epochs = 1
    if n_epochs is not None:
      epochs = n_epochs
    n_points = len(data)
    
    data_batches = self.assemble_batch(data)
    
    for epoch in range(epochs):      
      np.random.shuffle(data_batches)
      loss = 0
      print ('start epoch {epoch}'.format(epoch=epoch))
      for batch in data_batches:
        X = Variable(t.from_numpy(batch[0])).float()
        X_metas = Variable(t.from_numpy(batch[1])).float()
        y = Variable(t.from_numpy(batch[2])).float()
        if self.opt.gpu:
          X = X.cuda()
          X_metas = X_metas.cuda()
          y = y.cuda()
        output = self.net(X, X_metas, self.opt.batch_size)
        error = self.criterion(y, output, self.opt.batch_size)
        loss += error
        error.backward()
        if train:
          optimizer.step()
          self.net.zero_grad()
      print ('epoch {epoch} loss: {loss}'.format(epoch=epoch, loss=loss.data[0]/n_points))
      
  def predict(self, test_data):
    test_batches = self.assemble_batch(test_data, batch_size=1, sort=False)
    preds = []
    for sample in test_batches:
      X = Variable(t.from_numpy(sample[0])).float()
      X_metas = Variable(t.from_numpy(sample[1])).float()
      y = Variable(t.from_numpy(sample[2])).float()
      if self.opt.gpu:
        X = X.cuda()
        X_metas = X_metas.cuda()
        y = y.cuda()
      preds.append(self.net.predict(X, X_metas, 1, self.opt.gpu))
    return preds