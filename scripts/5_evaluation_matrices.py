"""
Evaluation Metric Calculation Script
Computes evaluation metrics (Precision, Recall, F1) based on error localization results.
"""

import os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

def calculate_metrics(df):
    """
    Calculate precision, recall, and F1 from TP, FP, and FN values
    """
    metrics_list = []

    error_types = df['Error_Type'].unique()
    models = df['Model'].unique()
    speakers = df['Speaker'].unique()
    voice_types = df['Voice_Type'].unique() if 'Voice_Type' in df.columns else ['all']

    for error_type in error_types:
        for model in models:
            model_type_filter = (df['Model'] == model) & (df['Error_Type'] == error_type)

            for speaker in speakers:
                for voice_type in voice_types:
                    if 'Voice_Type' in df.columns:
                        filter_condition = model_type_filter & (df['Speaker'] == speaker) & (df['Voice_Type'] == voice_type)
                    else:
                        filter_condition = model_type_filter & (df['Speaker'] == speaker)

                    subset = df[filter_condition]
                    
                    if not subset.empty:
                        tp_sum = subset['TP'].sum()
                        fp_sum = subset['FP'].sum()
                        fn_sum = subset['FN'].sum()

                        precision = tp_sum / max(1, (tp_sum + fp_sum))
                        recall = tp_sum / max(1, (tp_sum + fn_sum))
                        f1 = 2 * precision * recall / max(0.00001, (precision + recall))

                        metrics_dict = {
                            'Error_Type': error_type,
                            'Model': model,
                            'Speaker': speaker,
                            'TP': tp_sum,
                            'FP': fp_sum,
                            'FN': fn_sum,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1
                        }

                        if 'Voice_Type' in df.columns:
                            metrics_dict['Voice_Type'] = voice_type

                        metrics_list.append(metrics_dict)

    metrics_df = pd.DataFrame(metrics_list)
    return metrics_df

def generate_model_comparison_plots(metrics_df, output_dir):
    """
    Generate performance comparison plots by model and error type
    """
    os.makedirs(output_dir, exist_ok=True)

    model_group_avg = metrics_df.groupby(['Model', 'Error_Type']).agg({
        'precision': 'mean',
        'recall': 'mean',
        'f1': 'mean'
    }).reset_index()

    # F1 score by error type and model
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Error_Type', y='f1', hue='Model', data=model_group_avg)
    plt.title('F1 Score by Error Type and Model Size', fontsize=14)
    plt.xlabel('Error Type', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Model Size')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_by_error_type_and_model.png'), dpi=300)

    # Precision and Recall boxplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    sns.boxplot(x='Error_Type', y='precision', data=metrics_df, ax=axes[0])
    axes[0].set_title('Precision by Error Type', fontsize=14)
    axes[0].set_xlabel('Error Type', fontsize=12)
    axes[0].set_ylabel('Precision', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)

    sns.boxplot(x='Error_Type', y='recall', data=metrics_df, ax=axes[1])
    axes[1].set_title('Recall by Error Type', fontsize=14)
    axes[1].set_xlabel('Error Type', fontsize=12)
    axes[1].set_ylabel('Recall', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_boxplots.png'), dpi=300)

    # Overall model comparison
    plt.figure(figsize=(10, 6))
    model_avg = metrics_df.groupby('Model').agg({
        'precision': 'mean',
        'recall': 'mean',
        'f1': 'mean'
    }).reset_index()

    metrics_to_plot = pd.melt(model_avg, id_vars=['Model'], 
                             value_vars=['precision', 'recall', 'f1'],
                             var_name='Metric', value_name='Value')

    sns.barplot(x='Model', y='Value', hue='Metric', data=metrics_to_plot)
    plt.title('Model Performance Comparison', fontsize=14)
    plt.xlabel('Model Size', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_performance_comparison.png'), dpi=300)

    print(f"Performance plots saved to: {output_dir}")
    
    return model_group_avg

def main():
    parser = argparse.ArgumentParser(description='Script for computing and visualizing evaluation metrics')
    parser.add_argument('--input', type=str, default='./evaluation/final_counts/final_counts_with_matching_logs.xlsx',
                      help='Excel file containing TP/FP/FN counts')
    parser.add_argument('--output_dir', type=str, default='./evaluation/figures',
                      help='Directory to save figures')
    parser.add_argument('--summary_file', type=str, default='./evaluation/summary_stats.xlsx',
                      help='Path to save evaluation summary statistics')
    args = parser.parse_args()

    print(f"Loading results from: {args.input}")
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return

    results_df = pd.read_excel(args.input)

    required_columns = ['TP', 'FP', 'FN', 'Error_Type', 'Model', 'Speaker']
    missing_columns = [col for col in required_columns if col not in results_df.columns]

    if missing_columns:
        column_mapping = {
            'Error Type': 'Error_Type',
            'error_type': 'Error_Type',
            'group': 'Error_Type',
            'model_size': 'Model',
            'voice_type': 'Voice_Type'
        }

        for old_col, new_col in column_mapping.items():
            if old_col in results_df.columns and new_col not in results_df.columns:
                results_df.rename(columns={old_col: new_col}, inplace=True)

        missing_columns = [col for col in required_columns if col not in results_df.columns]
        if missing_columns:
            print(f"Error: Required columns {missing_columns} are missing from the input file.")
            return

    print("Calculating metrics...")
    metrics_df = calculate_metrics(results_df)

    os.makedirs(os.path.dirname(args.summary_file), exist_ok=True)
    metrics_df.to_excel(args.summary_file, index=False)
    print(f"Evaluation metrics saved to: {args.summary_file}")

    print("Generating visualizations...")
    generate_model_comparison_plots(metrics_df, args.output_dir)

    print("Evaluation completed successfully!")

if __name__ == "__main__":
    main()
