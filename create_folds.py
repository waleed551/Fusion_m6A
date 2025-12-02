# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 10:28:41 2025

@author: DELL
"""

import os
import torch
import numpy as np
import pandas as pd
from torch import nn, optim
from sklearn.model_selection import KFold  # plain KFold (not stratified)
from kmer_data_process import data_process, MyDataSet

# --- Configuration ---
cell_names = ['HEK293T']
'''cell_names = [
    'liver',
    'brain',
    'kidney',
    'HEK293',
    'HeLa',
    'CD8T',
    'A549',
    'MOLM13',
    'HEK293T',
    'HCT116',
    'HepG2',
]'''
base_path = './data/preprocessed_dataset'
save_dir = './fold_data'  # <-- new folder to save folds
window_size = 201
batch_size = 128
epochs = 200
positive_weight = 0.2
kmer = 5
patience = 20
n_splits = 5  # 5-fold

# --- Make sure folder exists ---
os.makedirs(save_dir, exist_ok=True)

# --- KFold instance ---
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

for cell in cell_names:
    print(f"\n--- Creating 5-Fold splits for cell line: {cell} ---")
    # Load raw data
    train_file = os.path.join(base_path, f"{cell}_train.tsv")
    data, labels = data_process(train_file, window_size)

    # Convert to numpy arrays for indexing
    data = np.array(data)
    # Ensure labels are on CPU before converting
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    else:
        labels = np.array(labels)

    # Folder for this cell line’s folds
    cell_dir = os.path.join(save_dir, f"{cell}")
    os.makedirs(cell_dir, exist_ok=True)

    # Perform K-Fold split and save each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(data), start=1):
        print(f"  Fold {fold}/{n_splits}...")

        # Split data
        X_train, X_val = data[train_idx], data[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        # Save to TSV
        train_df = pd.DataFrame({'text': X_train, 'label': y_train})
        val_df = pd.DataFrame({'text': X_val, 'label': y_val})

        train_path = os.path.join(cell_dir, f"{cell}_fold{fold}_train.tsv")
        val_path = os.path.join(cell_dir, f"{cell}_fold{fold}_val.tsv")

        train_df.to_csv(train_path, sep='\t', index=False)
        val_df.to_csv(val_path, sep='\t', index=False)

    print(f"5-fold data saved in: {cell_dir}")
