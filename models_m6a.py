# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 15:42:27 2025

@author: DELL
"""

import torch
from torch import nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 4, hidden_dim)  # 128+128=256 -> out: 64
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, 128]
        # encoder_outputs: [batch_size, seq_len, 128]
        timestep = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, timestep, 1)  # [batch, seq_len, 128]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch, seq_len, 64]
        energy = energy @ self.v  # [batch, seq_len]
        attn_weights = F.softmax(energy, dim=1)  # [batch, seq_len]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # [batch, 128]
        return context, attn_weights

class CNN_GRU_Attn_Classifier(nn.Module):
    def __init__(
        self,
        cnn1_in_channels = 100,
        cnn1_out_channels= 16,
        cnn2_out_channels= 128,
        cnn_kernel_size=5,
        cnn_pool_kernel=7,
        gru_input_dim=100,
        gru_hidden_dim=5,
        gru_layers=3,
        dropout_rate=0.3,
        kmer_dim=1024,
        num_classes=2,
        use_fc_layers=True,
        num_fc_layers=2,
        fc_hidden_dims=[512, 128]
    ):
        super(CNN_GRU_Attn_Classifier, self).__init__()
        self.use_fc_layers = use_fc_layers

        # Store CNN config for shape computation
        self.cnn1_in_channels = cnn1_in_channels
        self.cnn1_out_channels = cnn1_out_channels
        self.cnn2_out_channels = cnn2_out_channels
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_pool_kernel = cnn_pool_kernel

        # CNN branch
        self.conv1 = nn.Conv1d(cnn1_in_channels, cnn1_out_channels, cnn_kernel_size, padding=0, stride=1)
        self.conv2 = nn.Conv1d(cnn1_out_channels, cnn2_out_channels, cnn_kernel_size, padding=0, stride=1)
        self.pool = nn.MaxPool1d(kernel_size=cnn_pool_kernel)

        # GRU branch
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate
        )

        # Attention
        self.attn = Attention(hidden_dim=gru_hidden_dim)

        # k-mer branch
        self.kmer_fc_out_dim = fc_hidden_dims[0]
        self.kmer_fc = nn.Sequential(
            nn.Linear(kmer_dim, self.kmer_fc_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Dynamically compute CNN output dimension
        self.cnn_output_dim = self._calculate_cnn_output_dim()

        # Combine features: CNN + GRU-attn + kmer
        self.gru_output_dim = gru_hidden_dim * 2
        self.total_features = self.cnn_output_dim + self.gru_output_dim + self.kmer_fc_out_dim

        # FC layers (optional)
        if use_fc_layers:
            fc_layers = []
            in_dim = self.total_features
            for i in range(num_fc_layers):
                out_dim = fc_hidden_dims[i] if i < len(fc_hidden_dims) else fc_hidden_dims[-1]
                fc_layers.append(nn.Linear(in_dim, out_dim))
                fc_layers.append(nn.ReLU())
                fc_layers.append(nn.Dropout(dropout_rate))
                in_dim = out_dim
            self.fc_layers = nn.Sequential(*fc_layers)
            self.classifier = nn.Linear(in_dim, num_classes)
        else:
            self.fc_layers = nn.Identity()
            self.classifier = nn.Linear(self.total_features, num_classes)

    def _calculate_cnn_output_dim(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.cnn1_in_channels, 201)  # sequence length = 201
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            return x.view(1, -1).shape[1]

    def forward(self, one_hot_input, kmer_input,return_features=False):
        # CNN branch
        x_cnn = one_hot_input.transpose(1, 2)  # [B, 100, 201]
        x_cnn = F.relu(self.conv1(x_cnn))
        x_cnn = F.relu(self.conv2(x_cnn))
        x_cnn = self.pool(x_cnn)
        x_cnn = x_cnn.view(x_cnn.size(0), -1)

        # GRU branch
        gru_out, h_n = self.gru(one_hot_input)
        h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)
        attn_output, attn_weights  = self.attn(h_last, gru_out)

        # K-mer branch
        kmer_out = self.kmer_fc(kmer_input)

        # Combine all features
        combined = torch.cat((x_cnn, attn_output, kmer_out), dim=1)
        x = self.fc_layers(combined)

        if return_features:
            return x  # Return features or attention heatmap (attn_output, attn_weights)

        out = self.classifier(x)
        return out
        
