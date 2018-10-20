#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 22:06:44 2018

@author: zqwu
"""

from torch import nn
import torch as t
import torch.nn.functional as F
import numpy as np

class BiLSTM(nn.Module):
  def __init__(self, input_dim=256, hidden_dim=128, n_layers=2, gpu=True):
    super(BiLSTM, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.n_layers = n_layers

    self.lstm = nn.LSTM(input_dim, self.hidden_dim//2,
                        num_layers=self.n_layers, bidirectional=True)
    self.gpu = gpu
    self.hidden = self._init_hidden()

  def _init_hidden(self, batch_size=1):
    if self.gpu:
      return (t.randn(2*self.n_layers, batch_size, self.hidden_dim//2).cuda(),
              t.randn(2*self.n_layers, batch_size, self.hidden_dim//2).cuda())
    else:
      return (t.randn(2*self.n_layers, batch_size, self.hidden_dim//2),
              t.randn(2*self.n_layers, batch_size, self.hidden_dim//2))

  def forward(self, sequence, batch_size=1):
    # Get the emission scores from the BiLSTM
    self.hidden = self._init_hidden(batch_size=batch_size)
    inputs = sequence.reshape((-1, batch_size, self.input_dim))
    lstm_out, self.hidden = self.lstm(inputs, self.hidden)
    lstm_out = lstm_out.view(-1, batch_size, self.hidden_dim)
    return lstm_out

class MultiheadAttention(nn.Module):
  def __init__(self, 
               Q_dim=128,
               V_dim=128,
               head_dim=32, 
               n_heads=8):
    # As described in https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf
    super(MultiheadAttention, self).__init__()
    self.Q_dim = Q_dim
    self.K_dim = self.Q_dim
    self.V_dim = V_dim
    self.head_dim = head_dim
    self.n_heads = n_heads

    self.K_linears = nn.ModuleList([nn.Linear(self.K_dim, 
                                              self.head_dim) for i in range(self.n_heads)])
    self.Q_linears = nn.ModuleList([nn.Linear(self.Q_dim, 
                                              self.head_dim) for i in range(self.n_heads)])
    self.V_linears = nn.ModuleList([nn.Linear(self.V_dim, 
                                              self.head_dim) for i in range(self.n_heads)])

    self.post_head_linear = nn.Linear(self.head_dim * self.n_heads, self.Q_dim)
    
    self.fc = nn.Sequential(
        nn.Linear(self.Q_dim, self.Q_dim*4),
        nn.ReLU(True),
        nn.Linear(self.Q_dim*4, self.Q_dim*4),
        nn.ReLU(True),
        nn.Linear(self.Q_dim*4, self.Q_dim))

  def forward(self, sequence=None, K_in=None, Q_in=None, V_in=None):
    # query: seq_len_Q * batch_size * Q_dim
    # key:   seq_len_K * batch_size * Q_dim
    # value: seq_len_K * batch_size * V_dim
    outs = []
    if K_in is None:
      K_in = sequence
    if Q_in is None:
      Q_in = sequence
    if V_in is None:
      V_in = sequence
    for i in range(self.n_heads):
      K = self.K_linears[i](K_in.transpose(0, 1))
      Q = self.Q_linears[i](Q_in.transpose(0, 1))
      V = self.V_linears[i](V_in.transpose(0, 1))
      e = t.matmul(Q, K.transpose(1, 2)) / np.sqrt(self.head_dim)
      a = F.softmax(e, dim=2)
      outs.append(t.matmul(a, V))
    
    att_outs = Q_in.transpose(0, 1) + self.post_head_linear(t.cat(outs, 2))
    outs = att_outs + self.fc(att_outs)
    return outs.transpose(0, 1)

class Attention(nn.Module):
  def __init__(self, 
               Q_dim=128,
               V_dim=2,
               return_attention=False):
    super(Attention, self).__init__()
    self.Q_dim = Q_dim
    self.K_dim = self.Q_dim
    self.V_dim = V_dim
    self.return_attention = return_attention

  def forward(self, sequence=None, K_in=None, Q_in=None, V_in=None):
    # query: seq_len_Q * batch_size * Q_dim
    # key:   seq_len_K * batch_size * Q_dim
    # value: seq_len_K * batch_size * V_dim
    if K_in is None:
      K_in = sequence
    if Q_in is None:
      Q_in = sequence
    if V_in is None:
      V_in = sequence
    
    K = K_in.transpose(0, 1)
    Q = Q_in.transpose(0, 1)
    V = V_in.transpose(0, 1)
    e = t.matmul(Q, K.transpose(1, 2))
    a = F.softmax(e, dim=2)
    out = t.matmul(a, V)
    if self.return_attention:
      return out.transpose(0, 1), a.transpose(0, 1)
    else:
      return out.transpose(0, 1)
  
class CombinedConv1D(nn.Module):
  def __init__(self, input_dim=128, conv_dim=32, n_layers=3):
    # As described in https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf
    super(CombinedConv1D, self).__init__()
    self.input_dim = input_dim
    self.conv_dim = conv_dim
    self.n_layers = n_layers

    self.conv1 = nn.ModuleList([nn.Conv1d(self.input_dim, self.conv_dim, 1)] * n_layers)
    self.conv3 = nn.ModuleList([nn.Conv1d(self.input_dim, self.conv_dim, 3, padding=1)] * n_layers)
    self.conv5 = nn.ModuleList([nn.Conv1d(self.input_dim, self.conv_dim, 5, padding=2)] * n_layers)
    self.pool1_m = nn.ModuleList([nn.MaxPool1d(3, stride=1, padding=1)] * n_layers)
    self.conv1_m = nn.ModuleList([nn.Conv1d(self.input_dim, self.conv_dim, 1)] * n_layers)
    self.conv3_d = nn.ModuleList([nn.Conv1d(self.input_dim, self.conv_dim, 3, padding=2, dilation=2)] * n_layers)
    self.conv3_d2 = nn.ModuleList([nn.Conv1d(self.input_dim, self.conv_dim, 3, padding=4, dilation=4)] * n_layers)

    self.batch_norm = nn.ModuleList([nn.BatchNorm1d(self.conv_dim * 6)] * n_layers)
    self.post_conv_linear = nn.ModuleList([nn.Linear(self.conv_dim * 6, self.input_dim)] * n_layers)

  def forward(self, sequence):
    # sequence in seq_len * batch_size * input_dim
    sequence = sequence.transpose(0, 1).transpose(1, 2) # batch_size * input_dim * seq_len
    for i in range(self.n_layers):
      c1 = self.conv1[i](sequence)
      c3 = self.conv3[i](sequence)
      c5 = self.conv5[i](sequence)
      c1_m = self.conv1_m[i](self.pool1_m[i](sequence))
      c3_d = self.conv3_d[i](sequence)
      c3_d2 = self.conv3_d2[i](sequence)
      bn = self.batch_norm[i](t.cat([c1, c3, c5, c1_m, c3_d, c3_d2], 1))
      bn = F.relu(bn, inplace=True)
      output = self.post_conv_linear[i](bn.transpose(1, 2)).transpose(1, 2)
      sequence = sequence + output

    return sequence.transpose(1, 2).transpose(0, 1)
    return loss