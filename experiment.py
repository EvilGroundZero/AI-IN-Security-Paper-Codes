#!/usr/bin/env python3
"""
=============================================================================
AI-Based Network Intrusion Detection — Full Experiment Script
Course: Applications of Artificial Intelligence in Cyber Security
Dataset: NSL-KDD  (KDDTrain+.txt  /  KDDTest+.txt)
Models:  Random Forest | SVM | MLP (Deep Neural Network)
=============================================================================
HOW TO RUN ON KALI LINUX:
  1. Download dataset (see README at the bottom of this file)
  2. Place KDDTrain+.txt and KDDTest+.txt in the SAME folder as this script
  3. pip3 install pandas numpy scikit-learn matplotlib seaborn --break-system-packages
  4. python3 experiment.py
  All plots are saved as PNG files in the same folder.
=============================================================================
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — works on Kali without display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')

# ── Plot style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150,
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})
COLORS = ['#1a6fa3', '#e05a2b', '#2e8b57', '#8b2be0', '#c0392b']

# ── Column names for NSL-KDD ─────────────────────────────────────────────────
COLUMNS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
    'wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
    'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
    'dst_host_srv_rerror_rate','label','difficulty'
]

# Attack-type to category mapping (5-class)
ATTACK_MAP = {
    'normal': 'Normal',
    # DoS
    'back':'DoS','land':'DoS','neptune':'DoS','pod':'DoS','smurf':'DoS',
    'teardrop':'DoS','apache2':'DoS','udpstorm':'DoS','processtable':'DoS','worm':'DoS',
    # Probe
    'satan':'Probe','ipsweep':'Probe','nmap':'Probe','portsweep':'Probe',
    'mscan':'Probe','saint':'Probe',
    # R2L
    'ftp_write':'R2L','guess_passwd':'R2L','imap':'R2L','multihop':'R2L',
    'phf':'R2L','spy':'R2L','warezclient':'R2L','warezmaster':'R2L',
    'sendmail':'R2L','named':'R2L','snmpgetattack':'R2L','snmpguess':'R2L',
    'xlock':'R2L','xsnoop':'R2L','httptunnel':'R2L',
    # U2R
    'buffer_overflow':'U2R','loadmodule':'U2R','perl':'U2R','rootkit':'U2R',
    'ps':'U2R','sqlattack':'U2R','xterm':'U2R',
}

CLASS_ORDER = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def load_data(train_path='KDDTrain+.txt', test_path='KDDTest+.txt'):
    print("── Loading NSL-KDD dataset ──────────────────────────────────────")
    train = pd.read_csv(train_path, header=None, names=COLUMNS)
    test  = pd.read_csv(test_path,  header=None, names=COLUMNS)
    print(f"   Train samples : {len(train):,}")
    print(f"   Test  samples : {len(test):,}")
    return train, test


def preprocess(df):
    df = df.copy()
    df.drop(columns=['difficulty'], inplace=True, errors='ignore')

    # Map raw labels to 5-class categories
    df['attack_cat'] = df['label'].map(ATTACK_MAP)
    df['attack_cat'].fillna('Other', inplace=True)
    df = df[df['attack_cat'] != 'Other'].copy()

    # Binary label: 0 = Normal, 1 = Attack
    df['binary_label'] = (df['attack_cat'] != 'Normal').astype(int)

    # Encode categorical features
    cat_cols = ['protocol_type', 'service', 'flag']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return df


def get_feature_matrix(df):
    drop_cols = ['label', 'attack_cat', 'binary_label']
    X = df.drop(columns=drop_cols).values
    return X


def scale(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler


# ═══════════════════════════════════════════════════════════════════════════
# 2. MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_models():
    return {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=None, n_jobs=-1, random_state=42
        ),
        'SVM': SVC(
            kernel='rbf', C=10, gamma='scale', probability=True, random_state=42
        ),
        'MLP (Neural Net)': MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_model(name, model, X_train, y_train, X_test, y_test, task='binary'):
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(X_test)
    infer_time = time.time() - t0

    avg = 'binary' if task == 'binary' else 'weighted'
    results = {
        'name':      name,
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average=avg, zero_division=0),
        'recall':    recall_score(y_test, y_pred, average=avg, zero_division=0),
        'f1':        f1_score(y_test, y_pred, average=avg, zero_division=0),
        'train_time': train_time,
        'infer_time': infer_time,
        'y_pred':    y_pred,
        'model':     model,
    }
    if hasattr(model, 'predict_proba'):
        results['y_prob'] = model.predict_proba(X_test)
    print(f"   {name:<22} Acc={results['accuracy']:.4f}  F1={results['f1']:.4f}  "
          f"Train={train_time:.1f}s")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 4. PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_class_distribution(train_df, test_df, save='fig1_class_distribution.png'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, df, title in zip(axes, [train_df, test_df], ['Training Set', 'Test Set']):
        counts = df['attack_cat'].value_counts()[CLASS_ORDER]
        bars = ax.bar(counts.index, counts.values, color=COLORS, edgecolor='white', linewidth=0.8)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Attack Category')
        ax.set_ylabel('Sample Count')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
                    f'{val:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    fig.suptitle('NSL-KDD Dataset — Class Distribution', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save}")


def plot_confusion_matrix(y_test, y_pred, labels, title, save):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, ttl in zip(
        axes,
        [cm, cm_norm],
        ['d', '.2f'],
        ['Counts', 'Normalised (row %)']
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    linewidths=0.5, linecolor='white', ax=ax,
                    cbar_kws={'shrink': 0.8})
        ax.set_xlabel('Predicted Label', labelpad=8)
        ax.set_ylabel('True Label', labelpad=8)
        ax.set_title(f'{title}\n{ttl}', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save}")


def plot_performance_comparison(all_results, save='fig3_performance_comparison.png'):
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    labels  = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    model_names = [r['name'] for r in all_results]
    x = np.arange(len(metrics))
    width = 0.22

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (res, color) in enumerate(zip(all_results, COLORS)):
        vals = [res[m] for m in metrics]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=res['name'],
                      color=color, alpha=0.88, edgecolor='white', linewidth=0.7)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    all_vals = [r[m] for r in all_results for m in metrics]
    ymin = max(0, min(all_vals) - 0.05)
    ax.set_ylim(ymin, 1.01)
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison — Binary Classification (NSL-KDD)', fontweight='bold')
    ax.legend(loc='lower right')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save}")


def plot_roc_curves(all_results, y_test_bin, save='fig4_roc_curves.png'):
    fig, ax = plt.subplots(figsize=(8, 6))
    for res, color in zip(all_results, COLORS):
        if 'y_prob' not in res:
            continue
        prob = res['y_prob']
        if prob.ndim == 2:
            prob = prob[:, 1]
        fpr, tpr, _ = roc_curve(y_test_bin, prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{res['name']} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random Classifier (AUC = 0.50)')
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — Binary Intrusion Detection', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save}")


def plot_feature_importance(rf_model, feature_names, save='fig5_feature_importance.png'):
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]   # top 15 features

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices][::-1],
            color=COLORS[0], alpha=0.85, edgecolor='white', linewidth=0.7)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]])
    ax.set_xlabel('Feature Importance (Mean Decrease in Impurity)')
    ax.set_title('Top 15 Most Important Features — Random Forest Classifier', fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save}")


def plot_training_time(all_results, save='fig6_training_time.png'):
    names = [r['name'] for r in all_results]
    times = [r['train_time'] for r in all_results]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, times, color=COLORS[:len(names)], edgecolor='white', linewidth=0.8)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{t:.1f}s', ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Model Training Time Comparison', fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save}")


def plot_multiclass_f1(all_results_mc, classes, save='fig7_multiclass_f1.png'):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(classes))
    width = 0.26

    for i, (res, color) in enumerate(zip(all_results_mc, COLORS)):
        report = classification_report(
            res['y_test'], res['y_pred'],
            labels=classes, output_dict=True, zero_division=0
        )
        f1_scores = [report.get(c, {}).get('f1-score', 0) for c in classes]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, f1_scores, width, label=res['name'],
                      color=color, alpha=0.88, edgecolor='white', linewidth=0.7)
        for bar, val in zip(bars, f1_scores):
            if val > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('F1-Score')
    ax.set_title('Per-Class F1-Score — Multi-Class Classification', fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*65)
    print("  AI-BASED NETWORK INTRUSION DETECTION — NSL-KDD EXPERIMENT")
    print("="*65 + "\n")

    # ── Load data ──────────────────────────────────────────────────────
    train_df, test_df = load_data()
    train_df = preprocess(train_df)
    test_df  = preprocess(test_df)

    print(f"\n── Class distribution (train) ──────────────────────────────────")
    print(train_df['attack_cat'].value_counts().to_string())

    # ── Feature matrix ─────────────────────────────────────────────────
    feature_names = [c for c in train_df.columns
                     if c not in ['label', 'attack_cat', 'binary_label']]

    X_train_raw = get_feature_matrix(train_df)
    X_test_raw  = get_feature_matrix(test_df)
    X_train, X_test, _ = scale(X_train_raw, X_test_raw)

    y_train_bin = train_df['binary_label'].values
    y_test_bin  = test_df['binary_label'].values
    y_train_mc  = train_df['attack_cat'].values
    y_test_mc   = test_df['attack_cat'].values

    # ── Fig 1: Class distribution ───────────────────────────────────────
    print("\n── Generating plots ────────────────────────────────────────────")
    plot_class_distribution(train_df, test_df)

    # ── Binary classification ───────────────────────────────────────────
    print("\n── Binary Classification (Normal vs Attack) ────────────────────")
    models = get_models()
    binary_results = []
    for name, model in models.items():
        res = evaluate_model(name, model, X_train, y_train_bin,
                             X_test, y_test_bin, task='binary')
        binary_results.append(res)

    # ── Fig 2: Confusion matrix for Random Forest (binary) ─────────────
    plot_confusion_matrix(
        y_test_bin, binary_results[0]['y_pred'],
        labels=[0, 1],
        title='Random Forest — Binary Classification',
        save='fig2_confusion_matrix_rf_binary.png'
    )

    # ── Fig 3: Performance comparison ──────────────────────────────────
    plot_performance_comparison(binary_results)

    # ── Fig 4: ROC curves ──────────────────────────────────────────────
    plot_roc_curves(binary_results, y_test_bin)

    # ── Fig 5: Feature importance ───────────────────────────────────────
    rf_model = binary_results[0]['model']
    plot_feature_importance(rf_model, feature_names)

    # ── Multi-class classification ──────────────────────────────────────
    print("\n── Multi-Class Classification (5 categories) ───────────────────")
    models_mc = get_models()
    mc_results = []
    for name, model in models_mc.items():
        res = evaluate_model(name, model, X_train, y_train_mc,
                             X_test, y_test_mc, task='multiclass')
        res['y_test'] = y_test_mc
        mc_results.append(res)

    # ── Fig 6: Training time ────────────────────────────────────────────
    plot_training_time(binary_results)

    # ── Fig 7: Per-class F1 ─────────────────────────────────────────────
    plot_multiclass_f1(mc_results, CLASS_ORDER)

    # ── Fig 8: Best model (RF) multi-class confusion matrix ────────────
    plot_confusion_matrix(
        y_test_mc, mc_results[0]['y_pred'],
        labels=CLASS_ORDER,
        title='Random Forest — Multi-Class Classification',
        save='fig8_confusion_matrix_rf_multiclass.png'
    )

    # ── Final results table ─────────────────────────────────────────────
    print("\n" + "="*65)
    print("  BINARY CLASSIFICATION RESULTS SUMMARY")
    print("="*65)
    print(f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-"*65)
    for r in binary_results:
        print(f"{r['name']:<22} {r['accuracy']:>9.4f} {r['precision']:>10.4f} "
              f"{r['recall']:>8.4f} {r['f1']:>8.4f}")

    print("\n" + "="*65)
    print("  MULTI-CLASS CLASSIFICATION RESULTS SUMMARY")
    print("="*65)
    print(f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-"*65)
    for r in mc_results:
        print(f"{r['name']:<22} {r['accuracy']:>9.4f} {r['precision']:>10.4f} "
              f"{r['recall']:>8.4f} {r['f1']:>8.4f}")

    print("\n✔  All figures saved. Update your paper with the numbers above.")
    print("="*65 + "\n")

    # ── Save numeric results to CSV for convenience ──────────────────────
    rows = []
    for r in binary_results:
        rows.append({'Model': r['name'], 'Task': 'Binary',
                     'Accuracy': round(r['accuracy'],4),
                     'Precision': round(r['precision'],4),
                     'Recall': round(r['recall'],4),
                     'F1': round(r['f1'],4),
                     'TrainTime_s': round(r['train_time'],2)})
    for r in mc_results:
        rows.append({'Model': r['name'], 'Task': 'MultiClass',
                     'Accuracy': round(r['accuracy'],4),
                     'Precision': round(r['precision'],4),
                     'Recall': round(r['recall'],4),
                     'F1': round(r['f1'],4),
                     'TrainTime_s': round(r['train_time'],2)})
    pd.DataFrame(rows).to_csv('results_summary.csv', index=False)
    print("   Results saved to: results_summary.csv\n")


if __name__ == '__main__':
    main()

# ═══════════════════════════════════════════════════════════════════════════
# README — HOW TO GET THE DATASET
# ═══════════════════════════════════════════════════════════════════════════
#
# Option 1 (Official — recommended):
#   Website : https://www.unb.ca/cic/datasets/nsl.html
#   Files to download:
#     - KDDTrain+.txt
#     - KDDTest+.txt
#   Place both in the same directory as this script.
#
# Option 2 (Kaggle — requires free account):
#   https://www.kaggle.com/datasets/hassan06/nslkdd
#   kaggle datasets download -d hassan06/nslkdd
#
# Option 3 (Direct GitHub mirror):
#   wget https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt
#   wget https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt
#
# ═══════════════════════════════════════════════════════════════════════════