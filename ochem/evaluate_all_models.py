#!/usr/bin/env python3
"""
Script to evaluate all trained models and display performance metrics.
This script reads the result files and compares them with true values from the original data files.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import glob
import warnings
warnings.filterwarnings('ignore')

def load_data_and_predictions(data_file, result_file):
    """
    Load true values from data file and predictions from result file.
    
    Args:
        data_file (str): Path to the original data file with true values
        result_file (str): Path to the result file with predictions
    
    Returns:
        tuple: (y_true, y_pred, y_pred_proba) or (None, None, None) if files don't exist
    """
    try:
        # Load true values from original data file
        if not os.path.exists(data_file):
            return None, None, None
        
        data_df = pd.read_csv(data_file)
        if 'Result0' not in data_df.columns:
            print(f"❌ 'Result0' column not found in {data_file}")
            return None, None, None
        
        y_true = data_df['Result0'].values
        
        # Load predictions from result file
        if not os.path.exists(result_file):
            return None, None, None
        
        result_df = pd.read_csv(result_file)
        if len(result_df.columns) == 0:
            return None, None, None
        
        # Get predictions (first column should be the predictions)
        y_pred_proba = result_df.iloc[:, 0].values
        
        # Convert probabilities to binary predictions (threshold 0.5)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        return y_true, y_pred, y_pred_proba
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None

def calculate_metrics(y_true, y_pred, y_pred_proba, dataset_name):
    """
    Calculate performance metrics for a model.
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        y_pred_proba (array): Predicted probabilities
        dataset_name (str): Name of the dataset (train/test)
    
    Returns:
        dict: Dictionary with all metrics
    """
    try:
        metrics = {
            'dataset': dataset_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5,
            'n_samples': len(y_true),
            'n_positive': np.sum(y_true == 1),
            'n_negative': np.sum(y_true == 0)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error calculating metrics: {e}")
        return None

def print_metrics(metrics, architecture, dataset_name):
    """
    Print formatted metrics for a model.
    
    Args:
        metrics (dict): Metrics dictionary
        architecture (str): Architecture name
        dataset_name (str): Dataset name (train/test)
    """
    if metrics is None:
        print(f"❌ {architecture} - {dataset_name}: FAILED")
        return
    
    print(f"\n{'='*80}")
    print(f"🏗️  {architecture} - {dataset_name.upper()} SET")
    print(f"{'='*80}")
    
    print(f"📊 Basic Metrics:")
    print(f"   Accuracy:           {metrics['accuracy']:.4f}")
    print(f"   Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
    print(f"   Precision:          {metrics['precision']:.4f}")
    print(f"   Recall:             {metrics['recall']:.4f}")
    print(f"   F1-Score:           {metrics['f1_score']:.4f}")
    print(f"   ROC-AUC:            {metrics['roc_auc']:.4f}")
    
    print(f"\n📈 Dataset Info:")
    print(f"   Total samples:      {metrics['n_samples']}")
    print(f"   Positive samples:   {metrics['n_positive']}")
    print(f"   Negative samples:   {metrics['n_negative']}")
    
    print(f"\n🔢 Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"   Predicted")
    print(f"Actual    0    1")
    print(f"    0   {cm[0,0]:4d} {cm[0,1]:4d}")
    print(f"    1   {cm[1,0]:4d} {cm[1,1]:4d}")

def main():
    # Configuration
    data_folder = "../../data"
    train_data_file = f"{data_folder}/odorless_train_desc.csv"
    test_data_file = f"{data_folder}/odorless_test_desc.csv"
    
    # List of all architectures
    architectures = [
        "CoAttentiveFP", "AttFP", "MoE", "ContrastiveGIN", "ContrastiveGAT", 
        "ContrastiveGATv2", "ContrastiveDMPNN", "ContrastiveAttFP", 
        "ContrastiveAddGNN", "ContrastivePNA", "DMPNNAttention", "GAT", 
        "GIN", "GINE", "rGIN", "rGINE", "RGCN", "NMPN", "CMPNN", "DGIN", 
        "DMPNN", "AddGNN", "PNA", "GRPE"
    ]
    
    print("🎯 Evaluating All Models Performance")
    print("=" * 80)
    print(f"📁 Data folder: {data_folder}")
    print(f"📁 Train data: {train_data_file}")
    print(f"📁 Test data: {test_data_file}")
    print("=" * 80)
    
    # Store all results for summary
    all_results = []
    
    for architecture in architectures:
        print(f"\n🏗️  Processing: {architecture}")
        
        # File paths for this architecture
        train_result_file = f"{data_folder}/odorless_results_train_desc_{architecture}.csv"
        test_result_file = f"{data_folder}/odorless_results_test_desc_{architecture}.csv"
        
        # Evaluate train set
        y_true_train, y_pred_train, y_pred_proba_train = load_data_and_predictions(
            train_data_file, train_result_file
        )
        
        if y_true_train is not None:
            train_metrics = calculate_metrics(y_true_train, y_pred_train, y_pred_proba_train, "train")
            print_metrics(train_metrics, architecture, "train")
            if train_metrics:
                all_results.append({
                    'architecture': architecture,
                    'dataset': 'train',
                    **{k: v for k, v in train_metrics.items() if k not in ['dataset', 'confusion_matrix']}
                })
        else:
            print(f"❌ {architecture} - TRAIN: FAILED")
            all_results.append({
                'architecture': architecture,
                'dataset': 'train',
                'accuracy': np.nan,
                'balanced_accuracy': np.nan,
                'precision': np.nan,
                'recall': np.nan,
                'f1_score': np.nan,
                'roc_auc': np.nan,
                'n_samples': 0,
                'n_positive': 0,
                'n_negative': 0
            })
        
        # Evaluate test set
        y_true_test, y_pred_test, y_pred_proba_test = load_data_and_predictions(
            test_data_file, test_result_file
        )
        
        if y_true_test is not None:
            test_metrics = calculate_metrics(y_true_test, y_pred_test, y_pred_proba_test, "test")
            print_metrics(test_metrics, architecture, "test")
            if test_metrics:
                all_results.append({
                    'architecture': architecture,
                    'dataset': 'test',
                    **{k: v for k, v in test_metrics.items() if k not in ['dataset', 'confusion_matrix']}
                })
        else:
            print(f"❌ {architecture} - TEST: FAILED")
            all_results.append({
                'architecture': architecture,
                'dataset': 'test',
                'accuracy': np.nan,
                'balanced_accuracy': np.nan,
                'precision': np.nan,
                'recall': np.nan,
                'f1_score': np.nan,
                'roc_auc': np.nan,
                'n_samples': 0,
                'n_positive': 0,
                'n_negative': 0
            })
    
    # Create summary DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save detailed results
    results_file = f"{data_folder}/model_evaluation_summary.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Display summary table
    print(f"\n{'='*100}")
    print("📊 SUMMARY TABLE - ALL MODELS")
    print(f"{'='*100}")
    
    # Create pivot table for better visualization
    summary_df = results_df.pivot(index='architecture', columns='dataset', 
                                 values=['accuracy', 'balanced_accuracy', 'f1_score', 'roc_auc'])
    
    # Flatten column names
    summary_df.columns = [f"{col[1]}_{col[0]}" for col in summary_df.columns]
    
    # Sort by test accuracy (descending)
    if 'test_accuracy' in summary_df.columns:
        summary_df = summary_df.sort_values('test_accuracy', ascending=False, na_position='last')
    
    # Display the summary
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(summary_df.round(4))
    
    # Save summary table
    summary_file = f"{data_folder}/model_evaluation_summary_table.csv"
    summary_df.to_csv(summary_file)
    print(f"\n💾 Summary table saved to: {summary_file}")
    
    # Show top performers
    print(f"\n{'='*80}")
    print("🏆 TOP PERFORMERS (Test Set)")
    print(f"{'='*80}")
    
    test_results = results_df[results_df['dataset'] == 'test'].copy()
    test_results = test_results.sort_values('accuracy', ascending=False, na_position='last')
    
    print("Rank | Architecture          | Accuracy | Balanced Acc | F1-Score | ROC-AUC")
    print("-" * 75)
    
    for i, (_, row) in enumerate(test_results.head(10).iterrows(), 1):
        if pd.isna(row['accuracy']):
            print(f"{i:4d} | {row['architecture']:<20} | FAILED")
        else:
            print(f"{i:4d} | {row['architecture']:<20} | {row['accuracy']:.4f}    | {row['balanced_accuracy']:.4f}      | {row['f1_score']:.4f}    | {row['roc_auc']:.4f}")
    
    print(f"\n{'='*80}")
    print("✅ Evaluation completed!")
    print(f"📁 Results saved in: {data_folder}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 