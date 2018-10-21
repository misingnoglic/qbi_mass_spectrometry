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
    lr = 0.0001
    batch_size = 8
    max_epoch = 50
    gpu = True
    
opt=Config()


def evaluate(inputs, preds, thr=0.01):
  precisions = []
  recalls = []
  cos_sims = []
  for inp, pred in zip(inputs, preds):
    label = inp[2]
    _label = label > thr
    _pred = pred > thr
    precisions.append(precision_score(_label.flatten(), _pred.flatten()))
    recalls.append(recall_score(_label.flatten(), _pred.flatten()))
    sim = np.sum(label * pred)/np.sqrt(np.sum(label*label) * np.sum(pred*pred))
    cos_sims.append(sim)
  print("Precision %f" % np.mean(precisions))
  print("Recall %f" % np.mean(recalls))
  print("Cos similarity %f" % np.mean(cos_sims))
  return precisions, recalls, cos_sims


neutral_loss_choices = [0, 17, 18, 35, 36, 44, 46]
n_neutral_losses = len(neutral_loss_choices)
n_charges = 7

"""
df = pd.read_csv("../data/output.csv")
Xs = []
ys = []
X_metas = []

for line in np.array(df):
  '''
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
  '''
  Xs.append(np.array(eval(line[1])))
  X_metas.append((line[0], line[2]))
  seq_len = np.array(eval(line[1])).shape[0]
  
  intensities = np.array(eval(line[5])).astype(float)
  ions = np.array(eval(line[6])).astype(int)
  neutral_losses = np.array(eval(line[8])).astype(int)
  ion_charges = np.array(eval(line[9])).astype(int)
  positions = np.array(eval(line[7])).astype(int)
  offsets = np.array(eval(line[10])).astype(float)
  assert np.min(positions) >= 1
  assert np.max(positions) <= seq_len
  y = np.zeros((seq_len - 1, 2*n_charges*n_neutral_losses))
  for i in range(len(intensities)):
    if np.abs(offsets[i]) > 0.1 or \
       ion_charges[i] > n_charges or \
       neutral_losses[i] not in neutral_loss_choices:
      continue
    peak_type = ions[i] * n_charges * n_neutral_losses + \
        neutral_loss_choices.index(neutral_losses[i]) * n_charges + \
        ion_charges[i] - 1
    y[positions[i]-1, peak_type] += intensities[i]
  y = y/np.sum(y)
  y = np.concatenate([y[:, :n_neutral_losses*n_charges],
                      np.flip(y[:, n_neutral_losses*n_charges:], 0)], 1)
  ys.append(y)
all_input = [(_X, _X_meta, _y) for _X, _X_meta, _y in zip(Xs, X_metas, ys)]
"""

net = TestModel(input_dim=24,
                n_tasks=2 * n_charges,
                embedding_dim=256,
                hidden_dim_lstm=128,
                hidden_dim_attention=32,
                n_lstm_layers=2,
                n_attention_heads=8,
                gpu=opt.gpu)
trainer = Trainer(net, opt)
trainer.load('./saved_models/bkp/model-7_flipped.pth')
train_input = pickle.load(open('../data/train-66999_flipped.pkl', 'rb'))
test_input = pickle.load(open('../data/test-9920_flipped.pkl', 'rb'))

print("Training samples %d" % len(train_input))
print("Test samples %d" % len(test_input))

for i in range(8): 
  print("start fold %d" % i)
  if i>0:
    evaluate(train_input, trainer.predict(train_input))
    evaluate(test_input, trainer.predict(test_input))
  trainer.train(train_input, n_epochs=5)
  if not os.path.exists('./saved_models'):
    os.mkdir('./saved_models')
  trainer.save(os.path.join('./saved_models', 'model-%d.pth' % i))
