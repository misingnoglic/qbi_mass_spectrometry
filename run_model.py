#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 20:54:51 2018

@author: zqwu
"""
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import torch as t
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np
import pandas as pd
from scripts.biLSTM import BiLSTM, MultiheadAttention
from scripts.models import TestModel
from scripts.trainer import Trainer
import os
import pickle

class Config:
    lr = 0.0002
    batch_size = 8
    max_epoch = 50
    gpu = False # use gpu or not
    
opt=Config()

n_charges = 7

df = pd.read_csv("./output.csv")
Xs = []
ys = []
X_metas = []
for line in np.array(df):
  """
  ['acetyl',
 'name',
 'charge',
 'precursor',
 'mz',
 'intensity',
 'ion',
 'position',
 'neutral_loss',
 'ion_charge',
 'delta']
  """
  Xs.append(np.array(eval(line[1])))
  X_metas.append((line[0], line[2]))
  seq_len = np.array(eval(line[1])).shape[0]
  
  intensities = np.array(eval(line[5])).astype(float)
  ions = np.array(eval(line[6])).astype(int)
  ion_charges = np.array(eval(line[9])).astype(int)
  positions = np.array(eval(line[7])).astype(int)
  assert np.min(positions) >= 1
  assert np.max(positions) <= seq_len
  y = np.zeros((seq_len - 1, 2*n_charges))
  for i in range(len(intensities)):
    y[positions[i]-1, ions[i]*n_charges + ion_charges[i] - 1] += intensities[i]
  y = y/np.sum(y)
  ys.append(y)
  

net = TestModel(input_dim=24,
                n_tasks=2 * n_charges,
                embedding_dim=256,
                hidden_dim_lstm=128,
                hidden_dim_attention=32,
                n_lstm_layers=2,
                n_attention_heads=8,
                gpu=False)
trainer = Trainer(net, opt)
all_input = [(_X, _y) for _X, _y in zip(Xs, ys)]
n_train_samples = int(len(all_input) * 0.8)
train_input = all_input[:n_train_samples]
test_input = all_input[n_train_samples:]


for i in range(20): 
  print("start fold %d" % i)
  trainer.train(train_input, n_epochs=50)
  if not os.path.exists('./saved_models'):
    os.mkdir('./saved_models')
  trainer.save(os.path.join('./saved_models', 'model-%d.pth' % i))
