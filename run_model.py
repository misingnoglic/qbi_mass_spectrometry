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
from sklearn.metrics import precision_score, recall_score
import os
import pickle

class Config:
    lr = 0.0002
    batch_size = 8
    max_epoch = 50
    gpu = True
    
opt=Config()

def evaluate(inputs, preds, thr=0.005):
  precisions = []
  recalls = []
  for inp, pred in zip(inputs, preds):
    label = inp[2]
    _label = label > thr
    _pred = pred > thr
    precisions.append(precision_score(_label.flatten(), _pred.flatten()))
    recalls.append(recall_score(_label.flatten(), _pred.flatten()))
  print("Precision %f" % np.mean(precisions))
  print("Recall %f" % np.mean(recalls))
  return precisions, recalls

n_charges = 7

df = pd.read_csv("./output.csv")
Xs = []
ys = []
X_metas = []
for line in np.array(df):
  """
0  ['acetyl',
1 'name',
2 'charge',
3 'precursor',
4 'mz',
5 'intensity',
6 'ion',
7 'position',
8 'neutral_loss',
9 'ion_charge',
10 'delta']
  """
  Xs.append(np.array(eval(line[1])))
  X_metas.append((line[0], line[2]))
  seq_len = np.array(eval(line[1])).shape[0]
  
  intensities = np.array(eval(line[5])).astype(float)
  ions = np.array(eval(line[6])).astype(int)
  ion_charges = np.array(eval(line[9])).astype(int)
  positions = np.array(eval(line[7])).astype(int)
  offsets = np.array(eval(line[10])).astype(float)
  assert np.min(positions) >= 1
  assert np.max(positions) <= seq_len
  y = np.zeros((seq_len - 1, 2*n_charges))
  for i in range(len(intensities)):
    if np.abs(offsets[i]) > 0.1:
      continue
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
                gpu=opt.gpu)
trainer = Trainer(net, opt)
all_input = [(_X, _X_meta, _y) for _X, _X_meta, _y in zip(Xs, X_metas, ys)]
n_train_samples = int(len(all_input) * 0.8)
train_input = all_input[:n_train_samples]
test_input = all_input[n_train_samples:]


for i in range(20): 
  print("start fold %d" % i)
  trainer.train(train_input, n_epochs=50)
  if not os.path.exists('./saved_models'):
    os.mkdir('./saved_models')
  trainer.save(os.path.join('./saved_models', 'model-%d.pth' % i))
  evaluate(train_input, trainer.predict(train_input))
  evaluate(test_input, trainer.predict(test_input))
  