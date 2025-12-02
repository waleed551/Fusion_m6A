# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 16:43:40 2025

@author: DELL
"""
import os
import torch
from sklearn import metrics
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm.auto import tqdm
from kmer_data_process import data_process, MyDataSet
from models_m6a import CNN_GRU_Attn_Classifier
from itertools import product
from collections import Counter
import gensim
import pandas as pd

import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef


bases = ['A','C','G','T']
def encoding(seq):
    X = np.zeros((len(seq),len(seq[1]), len(bases)))
    for l,s in enumerate(seq):
        for i, char in enumerate(s):
            if char in bases:
                X[l,i, bases.index(char)] = 1
    return X
# --- Step 2: Extract k-mer features ---
# Generate all possible k-mers from DNA alphabet
def generate_kmer_patterns(k):
    alphabet = ['A', 'C', 'G', 'T']
    return [''.join(p) for p in product(alphabet, repeat=k)]

# Tokenize a single sequence into overlapping k-mers
def generate_kmers(sequence, k=5):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

# Convert tokenized k-mers into normalized count vectors based on patterns
def get_kmer_feature_vectors(sequences, k):
    patterns = generate_kmer_patterns(k)
    features = []
    for seq in sequences:
        kmers = generate_kmers(seq, k)
        counts = Counter(kmers)
        total_kmers = len(kmers)
        if total_kmers == 0:
            feature_vector = [0.0 for _ in patterns]
        else:
            feature_vector = [counts.get(pat, 0) / total_kmers for pat in patterns]
        features.append(feature_vector)
    return np.array(features)

def word2vec(train_data):
    model_wv =gensim.models.word2vec.Word2Vec.load('D:\m6A_data\My_Code\Word2vec\pretrained_models\my_word2vec-model-DNA-5mer')
    train_X = np.zeros((len(train_data),197,100)) # 41 is lenght of sequence 41-2=39, 100 is set in main file length of sentence
    for ix, seq in enumerate(train_data):
        for iy in range(197):
            vec = model_wv.wv[seq[iy:iy+5]]
            train_X[ix,iy,:]=vec
    return train_X
            
hyperparams_dict = {
    'liver': {
        'cnn1_in_channels': 100,
        'cnn1_out_channels': 16,
        'cnn2_out_channels': 128,
        'cnn_kernel_size': 5,
        'gru_input_dim': 100,
        'gru_hidden_dim': 8,
        'gru_layers': 3,
        'dropout_rate': 0.2,
        'kmer_dim': 1024,
        'num_classes': 2,
        'use_fc_layers': True,
        'num_fc_layers': 2,
        'fc_hidden_dims': [256, 128],
    },
    'brain': {
        'cnn1_in_channels': 100,
        'cnn1_out_channels': 16,
        'cnn2_out_channels': 128,
        'cnn_kernel_size': 5,
        'gru_input_dim': 100,
        'gru_hidden_dim': 32,
        'gru_layers': 3,
        'dropout_rate': 0.2,
        'kmer_dim': 1024,
        'num_classes': 2,
        'use_fc_layers': True,
        'num_fc_layers': 2,
        'fc_hidden_dims': [512, 128],
    },
    'kidney': {
        'cnn1_in_channels': 100,
        'cnn1_out_channels': 16,
        'cnn2_out_channels': 128,
        'cnn_kernel_size': 5,
        'gru_input_dim': 100,
        'gru_hidden_dim': 32,
        'gru_layers': 3,
        'dropout_rate': 0.3,
        'kmer_dim': 1024,
        'num_classes': 2,
        'use_fc_layers': True,
        'num_fc_layers': 2,
        'fc_hidden_dims': [512, 128],
    },
    'HEK293': {
        'cnn1_in_channels': 100,
        'cnn1_out_channels': 16,
        'cnn2_out_channels': 32,
        'cnn_kernel_size': 5,
        'gru_input_dim': 100,
        'gru_hidden_dim': 32,
        'gru_layers': 3,
        'dropout_rate': 0.3,
        'kmer_dim': 1024,
        'num_classes': 2,
        'use_fc_layers': True,
        'num_fc_layers': 2,
        'fc_hidden_dims': [512, 128],
    },
    'HeLa': {
        'cnn1_in_channels': 100,
        'cnn1_out_channels': 8,
        'cnn2_out_channels': 16,
        'cnn_kernel_size': 5,
        'gru_input_dim': 100,
        'gru_hidden_dim': 16,
        'gru_layers': 3,
        'dropout_rate': 0.3,
        'kmer_dim': 1024,
        'num_classes': 2,
        'use_fc_layers': True,
        'num_fc_layers': 2,
        'fc_hidden_dims': [512, 128],
    },
    'CD8T': {
        'cnn1_in_channels': 100,
        'cnn1_out_channels': 16,
        'cnn2_out_channels': 128,
        'cnn_kernel_size': 5,
        'gru_input_dim': 100,
        'gru_hidden_dim': 16,
        'gru_layers': 3,
        'dropout_rate': 0.3,
        'kmer_dim': 1024,
        'num_classes': 2,
        'use_fc_layers': True,
        'num_fc_layers': 2,
        'fc_hidden_dims': [512, 128],
    },
    'A549': {
        'cnn1_in_channels': 100,
        'cnn1_out_channels': 16,
        'cnn2_out_channels': 128,
        'cnn_kernel_size': 5,
        'gru_input_dim': 100,
        'gru_hidden_dim': 5,
        'gru_layers': 5,
        'dropout_rate': 0.3,
        'kmer_dim': 1024,
        'num_classes': 2,
        'use_fc_layers': True,
        'num_fc_layers': 2,
        'fc_hidden_dims': [512, 128],
    },
    'MOLM13': {
        'cnn1_in_channels': 100,
        'cnn1_out_channels': 16,
        'cnn2_out_channels': 32,
        'cnn_kernel_size': 5,
        'gru_input_dim': 100,
        'gru_hidden_dim': 32,
        'gru_layers': 3,
        'dropout_rate': 0.3,
        'kmer_dim': 1024,
        'num_classes': 2,
        'use_fc_layers': True,
        'num_fc_layers': 2,
        'fc_hidden_dims': [512, 128],
    },
    'HEK293T': {
        'cnn1_in_channels': 100,
        'cnn1_out_channels': 8,
        'cnn2_out_channels': 32,
        'cnn_kernel_size': 5,
        'gru_input_dim': 100,
        'gru_hidden_dim': 32,
        'gru_layers': 3,
        'dropout_rate': 0.3,
        'kmer_dim': 1024,
        'num_classes': 2,
        'use_fc_layers': True,
        'num_fc_layers': 2,
        'fc_hidden_dims': [512, 128],
    },
    'HCT116': {
        'cnn1_in_channels': 100,
        'cnn1_out_channels': 16,
        'cnn2_out_channels': 32,
        'cnn_kernel_size': 5,
        'gru_input_dim': 100,
        'gru_hidden_dim': 16,
        'gru_layers': 3,
        'dropout_rate': 0.3,
        'kmer_dim': 1024,
        'num_classes': 2,
        'use_fc_layers': True,
        'num_fc_layers': 2,
        'fc_hidden_dims': [512, 128],
    },
    'HepG2': {
        'cnn1_in_channels': 100,
        'cnn1_out_channels': 16,
        'cnn2_out_channels': 32,
        'cnn_kernel_size': 5,
        'gru_input_dim': 100,
        'gru_hidden_dim': 16,
        'gru_layers': 5,
        'dropout_rate': 0.3,
        'kmer_dim': 1024,
        'num_classes': 2,
        'use_fc_layers': True,
        'num_fc_layers': 2,
        'fc_hidden_dims': [256, 128],
    },
}


def evaluate(model, loader, device, criterion):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    total_loss = 0.0

    with torch.no_grad():
        for inputs1,inputs2, labels in loader:
            inputs1,inputs2, labels = inputs1.to(device),inputs2.to(device), labels.to(device)

            outputs = model(inputs1,inputs2)  # shape: [batch_size, 2]
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Convert logits to probabilities
            probs = F.softmax(outputs, dim=1)  # shape: [batch_size, 2]
            preds = torch.argmax(probs, dim=1)  # predicted class index

            # Get probability of class 1 for each sample
            class1_probs = probs[:, 1].cpu().numpy()  # optional

            # Store results
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(class1_probs)

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    pre = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)

    return total_loss / len(loader), acc, pre, rec, f1, mcc

#cell_names = ['A549','brain']

base_path = './data/preprocessed_dataset'  # Not used for folds, kept for compatibility
fold_root = './5_fold_data'                         # where folds were saved
save_dir = './5_fold model'  
save_res = './fold_results'                          # where to save best models (per-fold)
window_size = 201
batch_size = 128
epochs = 200
positive_weight = 0.2
kmer = 5
patience = 20
n_splits = 5

os.makedirs(save_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss()


cell_names = ['liver']
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
          # folder where folds were saved
results_file = os.path.join(save_res, str(cell_names)+"val_results.txt")
n_splits = 5
# overwrite results file header
with open(results_file, 'w') as f:
    f.write("Test Set Results Summary (per-fold)\n\n")

for cell in cell_names:
    print(f"\n--- test for cell line: {cell} (per-fold) ---")

    # prepare model folder and fold folder
    cell_model_dir = os.path.join(save_dir, cell)
    cell_fold_dir = os.path.join(fold_root, cell)

    # if folds folder doesn't exist, try previous behavior (single val file)
    if not os.path.isdir(cell_fold_dir):
        print(f"Fold folder not found: {cell_fold_dir}. Falling back to single test file.")
        val_file = os.path.join(base_path, f"{cell}_test.tsv")
        folds_to_eval = [None]  # single evaluation, will use val_file
    else:
        folds_to_eval = list(range(1, n_splits + 1))

    # collect metrics for averaging
    fold_metrics = []  # will hold tuples (loss, acc, prec, rec, f1, mcc)

    # iterate folds
    for fold in folds_to_eval:
        if fold is None:
            print("Testing if model and fold not avaivle")
            # fallback single test file (old behaviour)
            val_file = os.path.join(base_path, f"{cell}_test.tsv")
            model_path = os.path.join(save_dir, f"{cell}_best_model.pth")
            fold_name = "single_test"
        else:
            val_file = os.path.join(cell_fold_dir, f"{cell}_fold{fold}_val.tsv")
            # prefer per-fold saved model; otherwise fallback to cell-level model
            model_path = os.path.join(cell_model_dir, f"fold{fold}_best_model.pth")
            if not os.path.isfile(model_path):
                model_path = os.path.join(save_dir, f"{cell}_best_model.pth")
            fold_name = f"fold{fold}"

        # check files
        if not os.path.isfile(val_file):
            print(f"  Warning: val file not found for {fold_name}: {val_file}. Skipping.")
            continue
        if not os.path.isfile(model_path):
            print(f"  Warning: model file not found for {fold_name}: {model_path}. Skipping.")
            continue

        print(f"  Evaluating {fold_name} ...")
        # use your existing data_process to get val_data and val_label
        val_data, val_label = data_process(val_file, window_size)

        # compute features (same as before)
        val_wv = word2vec(val_data)
        valX = get_kmer_feature_vectors(val_data, kmer)

        # Build dataset and dataloader (same as your previous code)
        # Ensure val_label is on CPU / numpy if needed inside MyDataSet
        try:
            val_wv_arr = np.array(val_wv)
            valX_arr = np.array(valX)
            val_dataset = MyDataSet(torch.from_numpy(val_wv_arr).float(), torch.from_numpy(valX_arr).float(), val_label, mutation=False)
        except Exception:
            # fallback if features are ragged or already tensors
            val_dataset = MyDataSet(val_wv, valX, val_label, mutation=False)

        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Reinitialize model and load state dict (map to device)
        model_params = hyperparams_dict[cell]
        best_model = CNN_GRU_Attn_Classifier(**model_params).to(device)
        # safe load with map_location
        state = torch.load(model_path, map_location=device)
        # if checkpoint saved as dict with 'state_dict' key, handle that
        if isinstance(state, dict) and 'state_dict' in state:
            best_model.load_state_dict(state['state_dict'])
        else:
            best_model.load_state_dict(state)

        # Evaluate
        val_loss, acc, pre, rec, f1, mcc = evaluate(best_model, val_loader, device=device, criterion=criterion)

        # store metrics for averaging later
        fold_metrics.append((val_loss, acc, pre, rec, f1, mcc))

        # Print results
        print(f"  {fold_name} — Acc: {acc:.4f}, Prec: {pre:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}")

        # Save to file (append)
        with open(results_file, 'a') as f:
            f.write(f"\n--- {cell}_{fold_name}_results ---\n")
            f.write(f"Loss: {val_loss:.4f}\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Precision: {pre:.4f}\n")
            f.write(f"Recall: {rec:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"MCC: {mcc:.4f}\n")

    # After iterating all folds, compute average ± std and write it
    if len(fold_metrics) > 0:
        metrics_arr = np.array(fold_metrics)  # shape: (n_eval_folds, 6)
        mean_metrics = metrics_arr.mean(axis=0)
        std_metrics = metrics_arr.std(axis=0)

        # Print summary
        print(f"\n--- {cell} — Mean ± Std across {metrics_arr.shape[0]} evaluated folds ---")
        print(f"Loss: {mean_metrics[0]:.4f} ± {std_metrics[0]:.4f}")
        print(f"Accuracy: {mean_metrics[1]:.4f} ± {std_metrics[1]:.4f}")
        print(f"Precision: {mean_metrics[2]:.4f} ± {std_metrics[2]:.4f}")
        print(f"Recall: {mean_metrics[3]:.4f} ± {std_metrics[3]:.4f}")
        print(f"F1 Score: {mean_metrics[4]:.4f} ± {std_metrics[4]:.4f}")
        print(f"MCC: {mean_metrics[5]:.4f} ± {std_metrics[5]:.4f}")

        # Append summary to results file
        with open(results_file, 'a') as f:
            f.write(f"\n--- {cell} — Summary across {metrics_arr.shape[0]} folds ---\n")
            f.write(f"Loss: {mean_metrics[0]:.4f} ± {std_metrics[0]:.4f}\n")
            f.write(f"Accuracy: {mean_metrics[1]:.4f} ± {std_metrics[1]:.4f}\n")
            f.write(f"Precision: {mean_metrics[2]:.4f} ± {std_metrics[2]:.4f}\n")
            f.write(f"Recall: {mean_metrics[3]:.4f} ± {std_metrics[3]:.4f}\n")
            f.write(f"F1 Score: {mean_metrics[4]:.4f} ± {std_metrics[4]:.4f}\n")
            f.write(f"MCC: {mean_metrics[5]:.4f} ± {std_metrics[5]:.4f}\n")
    else:
        print(f"No folds were evaluated for {cell}, so no summary will be printed.")

print(f"\nAll per-fold results appended to: {results_file}")


#%%
#Test results 
results_file = os.path.join(save_res, str(cell_names)+"Test_results.txt")
n_splits = 5
batch_size = 128
window_size = 201

# overwrite results file header
with open(results_file, 'w') as f:
    f.write("Independent Test Set Results Summary (evaluated per-fold models)\n\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss()

for cell in cell_names:
    print(f"\n--- Independent test for cell line: {cell} ---")

    # load independent test set from base_path
    test_file = os.path.join(base_path, f"{cell}_test.tsv")
    if not os.path.isfile(test_file):
        print(f"  Error: test file not found: {test_file}. Skipping {cell}.")
        continue

    test_data, test_label = data_process(test_file, window_size)
    test_wv = word2vec(test_data)
    testX = get_kmer_feature_vectors(test_data, kmer)

    # Build test dataset (try numpy -> torch conversion first)
    try:
        test_wv_arr = np.array(test_wv)
        testX_arr = np.array(testX)
        test_dataset = MyDataSet(torch.from_numpy(test_wv_arr).float(), torch.from_numpy(testX_arr).float(), test_label, mutation=False)
    except Exception:
        test_dataset = MyDataSet(test_wv, testX, test_label, mutation=False)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # prepare model folder
    cell_model_dir = os.path.join(save_dir, cell)

    # collect fold metrics
    fold_metrics = []

    for fold in range(1, n_splits + 1):
        # prefer per-fold saved model; otherwise fallback to cell-level model
        fold_model_path = os.path.join(cell_model_dir, f"fold{fold}_best_model.pth")
        fallback_model_path = os.path.join(save_dir, f"{cell}_best_model.pth")

        if os.path.isfile(fold_model_path):
            model_path = fold_model_path
            fold_name = f"fold{fold}"
        elif os.path.isfile(fallback_model_path):
            model_path = fallback_model_path
            fold_name = f"fold{fold}_fallback"
            print(f"  Warning: per-fold model not found for fold{fold}; using fallback model.")
        else:
            print(f"  Warning: no model found for fold{fold} (checked {fold_model_path} and {fallback_model_path}). Skipping fold.")
            continue

        print(f"  Evaluating model: {os.path.basename(model_path)} on independent test set ...")

        # Reinitialize model and load weights (map to device)
        model_params = hyperparams_dict[cell]
        model = CNN_GRU_Attn_Classifier(**model_params).to(device)

        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
            model.load_state_dict(state['state_dict'])
        else:
            model.load_state_dict(state)

        # Evaluate on independent test set
        test_loss, acc, pre, rec, f1, mcc = evaluate(model, test_loader, device=device, criterion=criterion)

        # store metrics
        fold_metrics.append((test_loss, acc, pre, rec, f1, mcc))

        # print & append per-fold result
        print(f"  {fold_name} — Acc: {acc:.4f}, Prec: {pre:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}")
        with open(results_file, 'a') as f:
            f.write(f"\n--- {cell}_{fold_name}_on_independent_test ---\n")
            f.write(f"Loss: {test_loss:.4f}\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Precision: {pre:.4f}\n")
            f.write(f"Recall: {rec:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"MCC: {mcc:.4f}\n")

    # After evaluating all folds, compute mean ± std across folds and save/print
    if len(fold_metrics) > 0:
        arr = np.array(fold_metrics)  # shape: (n_evaluated_folds, 6)
        means = arr.mean(axis=0)
        stds = arr.std(axis=0)

        print(f"\n--- {cell} — Independent test mean ± std across {arr.shape[0]} evaluated folds ---")
        print(f"Loss: {means[0]:.4f} ± {stds[0]:.4f}")
        print(f"Accuracy: {means[1]:.4f} ± {stds[1]:.4f}")
        print(f"Precision: {means[2]:.4f} ± {stds[2]:.4f}")
        print(f"Recall: {means[3]:.4f} ± {stds[3]:.4f}")
        print(f"F1 Score: {means[4]:.4f} ± {stds[4]:.4f}")
        print(f"MCC: {means[5]:.4f} ± {stds[5]:.4f}")

        with open(results_file, 'a') as f:
            f.write(f"\n--- {cell} — Summary on independent test across {arr.shape[0]} folds ---\n")
            f.write(f"Loss: {means[0]:.4f} ± {stds[0]:.4f}\n")
            f.write(f"Accuracy: {means[1]:.4f} ± {stds[1]:.4f}\n")
            f.write(f"Precision: {means[2]:.4f} ± {stds[2]:.4f}\n")
            f.write(f"Recall: {means[3]:.4f} ± {stds[3]:.4f}\n")
            f.write(f"F1 Score: {means[4]:.4f} ± {stds[4]:.4f}\n")
            f.write(f"MCC: {means[5]:.4f} ± {stds[5]:.4f}\n")
    else:
        print(f"  No fold models evaluated for {cell} — nothing to average.")

print(f"\nAll independent-test results appended to: {results_file}")
