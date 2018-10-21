#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 10:27:11 2018

@author: zqwu
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
from calculate_frag_mz import get_frag_mz, reverse_one_hot_encode, amino_acid_modified_codes
import os
import pickle
import csv
from pyteomics import mass
from parse_data import amino_acid_name_parse

class Config:
    lr = 0.0001
    batch_size = 8
    max_epoch = 50
    gpu = False
    
opt=Config()

neutral_loss_choices = [0, 17, 18, 35, 36, 44, 46]
n_neutral_losses = len(neutral_loss_choices)
n_charges = 7

net = TestModel(input_dim=24,
                n_tasks=2*n_charges,
                embedding_dim=256,
                hidden_dim_lstm=128,
                hidden_dim_attention=32,
                n_lstm_layers=2,
                n_attention_heads=8,
                gpu=opt.gpu)
trainer = Trainer(net, opt)

trainer.load('./saved_models/model_flipped_bkp.pth')


#####################################
df = pd.read_csv("./syntheticPeptides/synth.peptides.plus.decoys.csv")

seqs = np.array(df['seq'])
Xs = []
X_metas = []
for line in seqs:
  flag, one_hot, charge = amino_acid_name_parse(line)

  Xs.append(np.array(one_hot))
  X_metas.append((flag, charge))
#####################################

sample_input = [(X, X_meta, np.zeros((X.shape[0]-1, 2*n_charges))) for X, X_meta in zip(Xs, X_metas)]
sample_pred = trainer.predict(sample_input)
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

lines = []
lines.append(['acetyl', 'name', 'charge', 'precursor', 'mz', 'intensity',
    'ion', 'position', 'neutral_loss', 'ion_charge', 'delta'])
for i, pred in enumerate(sample_pred):
  total_intensities = np.random.normal(70000, 20000)  
  seq = [list(X) for X in sample_input[i][0]]
  charge = sample_input[i][1][1]
  pep_seq = reverse_one_hot_encode(seq, amino_acid_modified_codes)
  mz = mass.calculate_mass(sequence=pep_seq, charge=charge)
  outputs = [sample_input[i][1][0], str(seq), charge, mz]
  mzs = []
  intensities = []
  ions = []
  positions = []
  neutral_losses = []
  ion_charges = []
  deltas = []
  peaks = np.where(pred > 0.005)
  for position, peak_type in zip(*peaks):
    b_y = (peak_type >= 7) * 1
    ions.append(int(b_y))
    charge = peak_type - b_y * 7 + 1
    if b_y:
      ion_type = 'y'
      pos = (pred.shape[0] + 1) - position - 1
    else:
      ion_type = 'b'
      pos = position + 1
    positions.append(pos)
    neutral_losses.append(0)
    ion_charges.append(charge)
    deltas.append(0)
    mzs.append(get_frag_mz(seq, pos, ion_type, charge))
    intensities.append(pred[position, peak_type] * total_intensities)
  outputs.extend([str(mzs), str(intensities), str(ions), str(positions),
                  str(neutral_losses), str(ion_charges), str(deltas)])
  lines.append(outputs)

with open('./predictions.csv', 'a') as f:
  writer = csv.writer(f)
  for line in lines:
    writer.writerow(line)

  
  
  