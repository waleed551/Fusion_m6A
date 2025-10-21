# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 13:33:23 2025

@author: DELL
"""
import os
import numpy as np
import pandas as pd
import time
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm.auto import tqdm
from kmer_data_process import data_process, MyDataSet
from models_m6a import CNN_GRU_Attn_Classifier
from itertools import product
from collections import Counter
from sklearn.manifold import TSNE
import umap
import gensim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score, average_precision_score

#%%
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
    model_wv =gensim.models.word2vec.Word2Vec.load('word2vec_pretrained_models/my_word2vec-model-DNA-5mer')
    train_X = np.zeros((len(train_data),197,100)) # 41 is lenght of sequence 41-2=39, 100 is set in main file length of sentence
    for ix, seq in enumerate(train_data):
        for iy in range(197):
            vec = model_wv.wv[seq[iy:iy+5]]
            train_X[ix,iy,:]=vec
    return train_X
#%%
def Heatmap_test_plot(metrics_dict,cell_names):
    df = pd.DataFrame(metrics_dict, index=cell_names).T  # Rows: metrics, Columns: cells

    # Plot heatmap
    plt.figure(figsize=(14, 4))
    sns.heatmap(df, annot=True, fmt=".3f", cmap='YlGnBu', linewidths=0.5, linecolor='gray')
    plt.title("Model Performance Metrics on Independent Test")
    plt.xlabel("Tissues & Cells")
    plt.ylabel("Metric")
    plt.tight_layout()
    plt.tight_layout()
    save_path = "Results/Test_Heatmap.png"
    plt.savefig(save_path, dpi=300)
    plt.show()
    
#def cross_test_heatmap():
    
#def  ROC_PR_Curve():
    
#def time_calculate():
    



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

    return total_loss / len(loader), acc, pre, rec, f1, mcc,all_labels,all_preds
def get_true_labels_and_probs(model, loader, device):
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for inputs1, inputs2, labels in loader:
            inputs1, inputs2 = inputs1.to(device), inputs2.to(device)
            outputs = model(inputs1, inputs2)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Prob for class 1
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_probs)


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


#cell_names = ['brain','kidney','HEK293']
cell_names = [
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
]
kmer=5
window_size=201
batch_size=128
base_path = './data/preprocessed_dataset'
save_dir = './trained_weights'
results_file = os.path.join("Test_results.txt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()

with open(results_file, 'w') as f:  # Overwrite file at start
    f.write("Test Set Results Summary\n\n")
metrics_dict = {
'Accuracy': [],
'Precision': [],
'Recall': [],
'F1': [],
'MCC': []
}
for cell in cell_names:
    val_file = os.path.join(base_path, f"{cell}_test.tsv")  # using test as val
    print(f"\n--- test for cell line: {cell} ---")
    val_data, val_label = data_process(val_file, window_size)
    #onh_valX = encoding(val_data)
    val_wv = word2vec(val_data)
    valX =get_kmer_feature_vectors(val_data,kmer)
    val_dataset = MyDataSet(torch.from_numpy(val_wv).float(),torch.from_numpy(valX).float(), val_label, mutation=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    path = os.path.join(save_dir, f"{cell}_best_model.pth")
    
    model_params = hyperparams_dict[cell]
    model = CNN_GRU_Attn_Classifier(**model_params).to(device)
    model.load_state_dict(torch.load(path))
    
    # Evaluate
    val_loss, acc, pre, rec, f1, mcc,true_label,pred_label = evaluate(model, val_loader, device=device, criterion=criterion)
    
    
    metrics_dict['Accuracy'].append(acc)
    metrics_dict['Precision'].append(pre)
    metrics_dict['Recall'].append(rec)
    metrics_dict['F1'].append(f1)
    metrics_dict['MCC'].append(mcc)
    
    
    
    # Print results
    print(f"\n--- Final Evaluation for Cell: {cell} ---")
    #print(f"Loss: {val_loss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {pre:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"MCC: {mcc:.4f}")

    # Save to file
    with open(results_file, 'a') as f:
        f.write(f"\n--- {cell}_results ---\n")
        f.write(f"Loss: {val_loss:.4f}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {pre:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"MCC: {mcc:.4f}\n")
        
#%%
Heatmap_test_plot(metrics_dict,cell_names)
#%%
# Plot headmap for cross testing
# Setup
cell_names = [
    'liver', 'brain', 'kidney', 'HEK293', 'HeLa', 'CD8T',
    'A549', 'MOLM13', 'HEK293T', 'HCT116', 'HepG2'
]
n = len(cell_names)
acc_matrix = np.zeros((n, n))   # Accuracy matrix: source x target
mcc_matrix = np.zeros((n, n))   # MCC matrix

# Cross-test loop
for i, source_cell in enumerate(cell_names):
    print(f"\n=== Loading model trained on {source_cell} ===")
    model_path = os.path.join(save_dir, f"{source_cell}_best_model.pth")
    model_params = hyperparams_dict[source_cell]
    model = CNN_GRU_Attn_Classifier(**model_params).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for j, target_cell in enumerate(cell_names):
        print(f"Testing on {target_cell}")
        test_file = os.path.join(base_path, f"{target_cell}_test.tsv")
        test_data, test_label = data_process(test_file, window_size)
        test_wv = word2vec(test_data)
        testX = get_kmer_feature_vectors(test_data, kmer)
        test_dataset = MyDataSet(
            torch.from_numpy(test_wv).float(),
            torch.from_numpy(testX).float(),
            test_label,
            mutation=False
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Evaluate
        val_loss, acc, pre, rec, f1, mcc,true_label,pred_label = evaluate(model, test_loader, device=device, criterion=criterion)
        acc_matrix[i][j] = acc
        mcc_matrix[i][j] = mcc

# Convert to DataFrames
acc_df = pd.DataFrame(acc_matrix, index=cell_names, columns=cell_names)
mcc_df = pd.DataFrame(mcc_matrix, index=cell_names, columns=cell_names)

# Plot Accuracy Heatmap
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(acc_df, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)
plt.title("Cross-Test Accuracy")
plt.xlabel("Test Tissues & Cells")
plt.ylabel("Trained Models")

# Plot MCC Heatmap
plt.subplot(1, 2, 2)
sns.heatmap(mcc_df, annot=True, fmt=".2f", cmap="PuBuGn", cbar=True)
plt.title("Cross-Test MCC")
plt.xlabel("Test Tissues & Cells")
plt.ylabel("Trained Models")

plt.tight_layout()
save_path = "Results/Cross_Test_Heatmap.png"
plt.savefig(save_path, dpi=300)
plt.show()

#%%
#ROC and PRC 
def get_true_labels_and_probs(model, loader, device):
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for inputs1, inputs2, labels in loader:
            inputs1, inputs2 = inputs1.to(device), inputs2.to(device)
            outputs = model(inputs1, inputs2)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Prob for class 1
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_probs)

# Prepare combined ROC and PRC plots
plt.figure(figsize=(14, 6))

# Subplot 1: ROC
plt.subplot(1, 2, 1)
plt.title("ROC Curve for Independent Test")
plt.plot([0, 1], [0, 1], 'k--', label='Random')

# Subplot 2: PRC
plt.subplot(1, 2, 2)
plt.title("PR Curve for Independent Test")

# Loop through all cell lines
for cell_name in cell_names:
    print(f"Processing: {cell_name}")

    # Load test data
    test_file = os.path.join(base_path, f"{cell_name}_test.tsv")
    test_data, test_label = data_process(test_file, window_size)
    test_wv = word2vec(test_data)
    testX = get_kmer_feature_vectors(test_data, kmer)
    test_dataset = MyDataSet(torch.from_numpy(test_wv).float(),
                             torch.from_numpy(testX).float(),
                             test_label, mutation=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model_path = os.path.join(save_dir, f"{cell_name}_best_model.pth")
    model_params = hyperparams_dict[cell_name]
    model = CNN_GRU_Attn_Classifier(**model_params).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Get true labels and predicted probabilities
    y_true, y_score = get_true_labels_and_probs(model, test_loader, device)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"{cell_name} (AUC={roc_auc:.2f})")

    # PRC
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    prc_auc = average_precision_score(y_true, y_score)

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f"{cell_name} (AUC={prc_auc:.2f})")

# Finalize plots
plt.subplot(1, 2, 1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(False)

plt.subplot(1, 2, 2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(False)

plt.tight_layout()
save_path = "Results/ROC_&PR_Curve.png"
plt.savefig(save_path, dpi=600)
plt.show()
#%%
# To store times
timing_results = {
    "Cell": [],
    "Prediction_Time_sec": [],
    "Num_Samples": [],
    "Avg_Time_per_Sample_ms": [],
    "Max_GPU_Memory_MB": [],
}

for cell_name in cell_names:
    print(f"Processing: {cell_name}")

    # Load test data
    test_file = os.path.join(base_path, f"{cell_name}_test.tsv")
    test_data, test_label = data_process(test_file, window_size)
    test_wv = word2vec(test_data)
    testX = get_kmer_feature_vectors(test_data, kmer)
    test_dataset = MyDataSet(torch.from_numpy(test_wv).float(),
                             torch.from_numpy(testX).float(),
                             test_label, mutation=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model_path = os.path.join(save_dir, f"{cell_name}_best_model.pth")
    model_params = hyperparams_dict[cell_name]
    model = CNN_GRU_Attn_Classifier(**model_params).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Start timing
    start_time = time.time()

    # Run prediction
    y_true, y_score = get_true_labels_and_probs(model, test_loader, device)

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time  # seconds
    # Measure peak GPU memory in MB
    max_gpu_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0.0

    # Save timing
    timing_results["Cell"].append(cell_name)
    timing_results["Prediction_Time_sec"].append(elapsed_time)
    timing_results["Num_Samples"].append(len(test_label))
    timing_results["Avg_Time_per_Sample_ms"].append((elapsed_time / len(test_label)) * 1000)
    timing_results["Max_GPU_Memory_MB"].append(round(max_gpu_memory, 2))
    
# Create DataFrame
timing_df = pd.DataFrame(timing_results)

# Show the table
print("\nPrediction Time & Memory Summary:")
print(timing_df)

# Save to CSV
timing_df.to_csv("Results/GRU_prediction_times.csv", index=False)


#%%
# Read timing result files
timing_before = pd.read_csv("Results/MST_prediction_times.csv")
timing_after = pd.read_csv("Results/GRU_prediction_times.csv")

# Add label columns
timing_before["Version"] = "MST-m6A"
timing_after["Version"] = "Fusion-m6A"

# Combine
timing_df = pd.concat([timing_before, timing_after], ignore_index=True)

# Set style
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# Line plot
ax = sns.lineplot(
    data=timing_df,
    x="Cell",
    y="Prediction_Time_sec",
    hue="Version",
    marker='o',
    linewidth=2,
    markersize=8
)

# Annotate values
for i in range(timing_df.shape[0]):
    row = timing_df.iloc[i]
    ax.text(
        x=i % len(timing_df['Cell'].unique()),  # adjust x to align by cell
        y=row["Prediction_Time_sec"] + 0.01,     # small offset to avoid overlap
        s=f"{row['Prediction_Time_sec']:.2f}s",
        ha='center',
        fontsize=9
    )

# Customize plot
plt.xticks(rotation=45, ha='right')
plt.ylabel("Prediction Time of Test Set  (sec)")
plt.xlabel("Tissue & Cell")
plt.title("Inference Time Comparison per Tissue/Cell")
plt.legend(title="Model")
plt.tight_layout()

# Save and show
plt.savefig("Results/comparison_avg_time_per_sample_annotated.png", dpi=300)
plt.show()
#%%
# Load the two result files
gru_df = pd.read_csv("Results/GRU_prediction_times.csv")
mst_df = pd.read_csv("Results/MST_prediction_times.csv")

# Add model labels
gru_df["Model"] = "Fusion-m6A"
mst_df["Model"] = "MST-m6A"

# Combine both DataFrames
combined_df = pd.concat([gru_df, mst_df], ignore_index=True)


# ==== Bar Plot: GPU Memory Usage ====
plt.figure(figsize=(10, 6))
sns.barplot(
    data=combined_df,
    x="Cell",
    y="Max_GPU_Memory_MB",
    hue="Model",
    palette="Set2"
)

plt.xticks(rotation=45, ha='right')
plt.ylabel("Max GPU Memory Usage (MB)")
plt.xlabel("Cell Line")
plt.title("GPU Memory Comparison (Fusion-m6A vs MST-m6A)")
plt.legend(title="Model")
plt.tight_layout()
plt.savefig("Results/memory_comparison_Fusion-m6A_vs_MST-m6A.png", dpi=300)
plt.show()


# ====  CSV Summary of Improvements ====

# Pivot for each metric
time_pivot = combined_df.pivot(index="Cell", columns="Model", values="Avg_Time_per_Sample_ms")
mem_pivot = combined_df.pivot(index="Cell", columns="Model", values="Max_GPU_Memory_MB")

# Build comparison table
summary_df = pd.DataFrame({
    "GRU_Time_ms": time_pivot["Fusion-m6A"],
    "MST_Time_ms": time_pivot["MST-m6A"],
    "Time_Improvement_% (Fusion-m6A vs MST-m6A)": ((time_pivot["Fusion-m6A"] - time_pivot["MST-m6A"]) / time_pivot["Fusion-m6A"] * 100).round(2),

    "GRU_Memory_MB": mem_pivot["Fusion-m6A"],
    "MST_Memory_MB": mem_pivot["MST-m6A"],
    "Memory_Improvement_% (Fusion-m6A vs MST-m6A)": ((mem_pivot["Fusion-m6A"] - mem_pivot["MST-m6A"]) / mem_pivot["Fusion-m6A"] * 100).round(2)
})

# Save summary
os.makedirs("Results", exist_ok=True)
summary_path = "Results/GRU_vs_MST_performance_comparison.csv"
summary_df.to_csv(summary_path)
print(f"Saved comparison summary to: {summary_path}")
#%%
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")

# Box plot with individual points
sns.boxplot(data=combined_df, x="Model", y="Prediction_Time_sec", palette="pastel", showfliers=False)
sns.swarmplot(data=combined_df, x="Model", y="Prediction_Time_sec", hue="Cell", size=7, dodge=True)

plt.ylabel("Avg Prediction Time per Sample (ms)")
plt.title("Inference Time Distribution Across Models")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Cell Line")
plt.tight_layout()
plt.savefig("Results/inference_time_boxplot.png", dpi=300)
plt.show()

#%%
#Confusion Matrix
def get_true_labels_and_probs(model, loader, device):
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for inputs1, inputs2, labels in loader:
            inputs1, inputs2 = inputs1.to(device), inputs2.to(device)
            outputs = model(inputs1, inputs2)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Prob for class 1
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_probs)

# ----------------------------
# Confusion matrix plotting function
# ----------------------------
def plot_and_save_confusion_matrix(y_true, y_pred, cell_name, save_dir="Results/conf_matrices"):
    cm = confusion_matrix(y_true, y_pred)
    labels = ["N", "T"]

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {cell_name}")
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"conf_matrix_{cell_name}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

# ----------------------------
# Loop over all cell lines
# ----------------------------
cell_names = [
    'liver', 'brain', 'kidney', 'HEK293', 'HeLa', 'CD8T',
    'A549', 'MOLM13', 'HEK293T', 'HCT116', 'HepG2'
]

kmer = 5
window_size = 201
batch_size = 128
base_path = './data/preprocessed_dataset'
save_dir = './trained_weights'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for cell_name in cell_names:
    print(f"Processing: {cell_name}")

    # Load test data
    test_file = os.path.join(base_path, f"{cell_name}_test.tsv")
    test_data, test_label = data_process(test_file, window_size)
    test_wv = word2vec(test_data)
    testX = get_kmer_feature_vectors(test_data, kmer)
    test_dataset = MyDataSet(torch.from_numpy(test_wv).float(),
                             torch.from_numpy(testX).float(),
                             test_label, mutation=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model_path = os.path.join(save_dir, f"{cell_name}_best_model.pth")
    model_params = hyperparams_dict[cell_name]
    model = CNN_GRU_Attn_Classifier(**model_params).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Get true labels and predicted probabilities
    y_true, y_score = get_true_labels_and_probs(model, test_loader, device)

    # Convert probabilities to binary predictions
    y_pred = (y_score >= 0.5).astype(int)

    # Plot and save confusion matrix
    plot_and_save_confusion_matrix(y_true, y_pred, cell_name=cell_name)



#%%%
#Umap and tSNE for one by one 
# ---- Settings ----
cell_names = [
    'liver', 'brain', 'kidney', 'HEK293', 'HeLa', 'CD8T',
    'A549', 'MOLM13', 'HEK293T', 'HCT116', 'HepG2'
]
base_path = './data/preprocessed_dataset'
save_dir = './trained_weights'
window_size = 201
kmer = 5
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Plotting Function ----
def plot_2D(features, labels, method='tsne', file_name='tsne.png', title=''):
    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("method must be 'tsne' or 'umap'")

    reduced = reducer.fit_transform(features)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels,
                    palette={"Negative": "blue", "Positive": "red"}, s=40, alpha=0.8)
    plt.title(f"{title} ({method.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="Class", loc='upper right')
    plt.tight_layout()
    os.makedirs("Results/features", exist_ok=True)
    plt.savefig(f"Results/features/{file_name}", dpi=300)
    plt.close()
    print(f"Saved: Results/features/{file_name}")

# ---- Loop Over Cell Lines ----
for cell_name in cell_names:
    print(f"\n Extracting features for: {cell_name}")

    # Load test data
    test_file = os.path.join(base_path, f"{cell_name}_test.tsv")
    test_data, test_label = data_process(test_file, window_size)
    test_wv = word2vec(test_data)
    testX = get_kmer_feature_vectors(test_data, kmer)

    test_dataset = MyDataSet(torch.from_numpy(test_wv).float(),
                             torch.from_numpy(testX).float(),
                             test_label, mutation=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Load model
    model_path = os.path.join(save_dir, f"{cell_name}_best_model.pth")
    model_params = hyperparams_dict[cell_name]
    model = CNN_GRU_Attn_Classifier(**model_params).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Extract features
    all_features = []
    all_labels = []
    with torch.no_grad():
        for x1, x2, y in test_loader:
            x1, x2 = x1.to(device), x2.to(device)
            features = model(x1, x2, return_features=True)
            all_features.append(features.cpu())
            all_labels.extend(y.cpu().numpy())

    # Prepare tensors
    all_features = torch.cat(all_features, dim=0).numpy()
    all_labels = np.array(["Negative" if l == 0 else "Positive" for l in all_labels])

    print(f"{cell_name} features: {all_features.shape}, labels: {len(all_labels)}")

    # Plot
    plot_2D(all_features, all_labels, method='tsne', file_name=f"{cell_name}_tsne.png", title=cell_name)
    plot_2D(all_features, all_labels, method='umap', file_name=f"{cell_name}_umap.png", title=cell_name)
#%%
# ---- Settings ----
cell_names = [
    'liver', 'brain', 'kidney', 'HEK293', 'HeLa', 'CD8T',
    'A549', 'MOLM13', 'HEK293T', 'HCT116', 'HepG2'
]
base_path = './data/preprocessed_dataset'
save_dir = './trained_weights'
window_size = 201
kmer = 5
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Create Results Directory ----
os.makedirs("Results/attention_plots", exist_ok=True)

# ---- Loop Through Each Cell ----
for cell_name in cell_names:
    print(f" Processing: {cell_name}")

    # ---- Data Preparation ----
    test_file = os.path.join(base_path, f"{cell_name}_test.tsv")
    test_data, test_label = data_process(test_file, window_size)
    test_wv = word2vec(test_data)
    testX = get_kmer_feature_vectors(test_data, kmer)

    test_dataset = MyDataSet(torch.from_numpy(test_wv).float(),
                             torch.from_numpy(testX).float(),
                             test_label, mutation=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # ---- Load Model ----
    model_path = os.path.join(save_dir, f"{cell_name}_best_model.pth")
    model_params = hyperparams_dict[cell_name]
    model = CNN_GRU_Attn_Classifier(**model_params).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # ---- Extract Attention ----
    all_attn_weights = []
    all_labels = []

    with torch.no_grad():
        for x1, x2, y in test_loader:
            x1, x2 = x1.to(device), x2.to(device)
            _,attn = model(x1, x2, return_features=True)  # Expected shape: (B, L)
            all_attn_weights.append(attn.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    # ---- Combine ----
    all_attn_weights = np.concatenate(all_attn_weights, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # ---- Separate by Class ----
    attn_class0 = all_attn_weights[all_labels == 0]
    attn_class1 = all_attn_weights[all_labels == 1]
    mean_attn_0 = np.mean(attn_class0, axis=0)
    mean_attn_1 = np.mean(attn_class1, axis=0)

    # ---- Plot: Average Attention Profiles ----
    plt.figure(figsize=(12, 4))
    plt.plot(mean_attn_0, label='Negative', color='blue')
    plt.plot(mean_attn_1, label='Positive', color='red')
    plt.title(f" Avg. Attention - {cell_name}")
    plt.xlabel("Sequence Position")
    plt.ylabel("Attention Weight")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Results/attention_plots/{cell_name}_attention_profile.png", dpi=300)
    plt.close()

    # ---- Plot: Heatmap ----
    plt.figure(figsize=(14, 6))
    sns.heatmap(all_attn_weights, cmap="YlOrRd", cbar=True)
    plt.title(f" Attention Heatmap - {cell_name}")
    plt.xlabel("Sequence Position")
    plt.ylabel("Sample Index")
    plt.tight_layout()
    plt.savefig(f"Results/attention_plots/{cell_name}_attention_heatmap.png", dpi=300)
    plt.close()

    print(f" Saved for {cell_name}: attention_profile.png + attention_heatmap.png")

