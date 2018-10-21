#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 15:40:34 2018

@author: zqwu
"""
from torch import nn
import torch as t
import torch.nn.functional as F
import numpy as np
from scripts.biLSTM import BiLSTM, MultiheadAttention, CombinedConv1D

def CELoss(labels, outs, batch_size=8):
  logits = outs.transpose(0, 1).contiguous().view(batch_size, -1)
  labels = labels.transpose(0, 1).contiguous().view((batch_size, -1))
  log_prob = F.log_softmax(logits, 1)
  ce = -(labels * log_prob).sum()
  return ce
  
class TestModel(nn.Module):
  def __init__(self,
               input_dim=20,
               n_tasks=6,
               embedding_dim=256,
               hidden_dim_lstm=128,
               hidden_dim_attention=32,
               n_lstm_layers=2,
               n_attention_heads=8,
               gpu=True):
    super(TestModel, self).__init__()
    self.input_dim = input_dim
    self.n_tasks = n_tasks
    self.embedding_dim = embedding_dim
    self.hidden_dim_lstm = hidden_dim_lstm
    self.hidden_dim_attention = hidden_dim_attention
    self.n_lstm_layers = n_lstm_layers
    self.n_attention_heads = n_attention_heads
    self.gpu = gpu

    self.embedding_module = t.nn.Embedding(self.input_dim, self.embedding_dim)
    self.lstm_module = BiLSTM(input_dim=self.embedding_dim + 2,
                              hidden_dim=self.hidden_dim_lstm,
                              n_layers=self.n_lstm_layers,
                              gpu=self.gpu)
    self.att_module = MultiheadAttention(Q_dim=self.hidden_dim_lstm,
                                         V_dim=self.hidden_dim_lstm,
                                         head_dim=self.hidden_dim_attention,
                                         n_heads=self.n_attention_heads)
#    self.conv_module = CombinedConv1D(input_dim=self.hidden_dim_lstm,
#                                      conv_dim=self.hidden_dim_attention)
    self.fc = nn.Sequential(
        nn.Linear(self.hidden_dim_lstm * 2, 64),
        nn.ReLU(True),
        nn.Linear(64, self.n_tasks))

  def forward(self, sequence, metas, batch_size=1):
    # sequence of shape: seq_len * batch_size
    embedded_inputs = t.matmul(sequence, self.embedding_module.weight)
    seq_len = embedded_inputs.shape[0]
    metas = metas.view(1, *metas.shape).repeat(seq_len, 1, 1)
    
    lstm_inputs = t.cat([embedded_inputs, metas], 2)
    lstm_outs = self.lstm_module(lstm_inputs, batch_size=batch_size)
    attention_outs = self.att_module(sequence=lstm_outs)
    #conv_outs = self.conv_module(lstm_outs)
    intervals = t.cat([attention_outs[1:], attention_outs[:-1]], 2)
    outs = self.fc(intervals)
    # outs of shape: (seq_len - 1) * batch_size * n_tasks
    return outs

  def predict(self, sequence, metas, batch_size=1, gpu=True):
    assert batch_size == 1
    output = self.forward(sequence, metas, 1)[:, 0, :]
    output_shape = output.shape
    logits = output.view(-1)
    log_prob = F.log_softmax(logits, 0)
    prob = t.exp(log_prob).view(output_shape)
    if gpu:
      prob = prob.cpu()
    prob = prob.data.numpy()
    return prob
  
