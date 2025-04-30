"""
Error Matrix Analysis Script
Generates and analyzes a confusion matrix between ASR error types and therapist-labeled categories.
"""

import os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re

def parse_error_report(file_path):
    """
    Extract error information from ASR error report.
    """
    error_events = []
    pattern = re.compile(
        r"ERROR\s*\((?P<error_type>.*?)\):\s*"
        r"(?P<start>[\d\.]+)-(?P<end>[\d\.]+):\s*"
        r"(?:ASR=(?P<asr_word>[^|]+)(?:\|\s*)?)?"
        r"(?:Original=(?P<original_word>.*))?$"
    )
    
    try:
        with open(file_path, 'r', encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("ERROR"):
                    m = pattern.search(line)
                    if m:
                        error_type = m.group("error_type").strip().lower()
                        start = float(m.group("start"))
                        end = float(m.group("end"))
                        asr_word = m.group("asr_word").strip() if m.group("asr_word") else ""
                        original_word = m.group("original_word").strip() if m.group("original_word") else ""
                        
                        error_events.append({
                            'start': start,
                            'end': end,
                            'type': error_type,
                            'asr_word': asr_word,
                            'original_word': original_word
                        })
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
    
    return error_events

def parse_group_labels(file_path):
    """
    Extract grouped label information from therapist label file.
    """
    try:
        df = pd.read_excel(file_path, header=0)
        
        # Normalize column names
        if 'Category' in df.columns:
            df.rename(columns={'Category': 'group'}, inplace=True)
        if 'Start Time' in df.columns and 'End Time' in df.columns:
            df.rename(columns={'Start Time': 'start', 'End Time': 'end'}, inplace=True)
        
        # Normalize group labels (lowercase, remove "errors")
        if 'group' in df.columns:
            df['group'] = df['group'].str.lower().str.replace(r'\berrors\b', '', regex=True).str.strip()
        
        return df
    except Exception as e:
        print(f"Error loading group labels from {file_path}: {e}")
        return pd.DataFrame()

def intervals_overlap(start1, end1, start2, end2):
    """
    Check if two time intervals overlap.
    """
    if start2 == end2:
        return start1 <= start2 <= end1
    else:
        return (start1 <= end2) and (start2 <= end1)

def create_confusion_matrix(model, speaker, voice_type, error_events, grouped_df):
    """
    Generate confusion matrix between ASR-detected error types and therapist labels.
    """
    therapist_groups = grouped_df['group'].unique()
    asr_groups = list(set(e['type'] for e in error_events))

    error_matrix = {
        therapist_group: {
            asr_group: 0 for asr_group in asr_groups + ["Not Detected"]
        } for therapist_group in therapist_groups
    }

    for therapist_group in therapist_groups:
        error_matrix[therapist_group]["Total Count"] = 0

    log_data = []

    for _, row in grouped_df.iterrows():
        group_start = row['start']
        group_end = row['end']
        group_type = row['group']
        group_desc = row['Label'] if 'Label' in row else ""

        matched_asr_errors = [
            e for e in error_events 
            if intervals_overlap(group_start, group_end, e['start'], e['end'])
        ]

        log_entry = {
            "Therapist_Error_Group": group_type,
            "Start_Time": group_start,
            "End_Time": group_end,
            "Matched_ASR_Errors": [],
            "Count_Details": [],
            "Not_Detected": False
        }

        error_matrix[group_type]["Total Count"] += 1

        if matched_asr_errors:
            counted_types = set()

            for e in matched_asr_errors:
                asr_type = e['type']
                if asr_type not in counted_types:
                    counted_types.add(asr_type)
                    error_matrix[group_type][asr_type] += 1
                    log_entry["Matched_ASR_Errors"].append({
                        "Type": asr_type,
                        "Start": e['start'],
                        "End": e['end'],
                        "ASR_Word": e['asr_word'],
                        "Original_Word": e['original_word']
                    })
                    log_entry["Count_Details"].append(f"{asr_type} +1 (Detected)")
        else:
            error_matrix[group_type]["Not Detected"] += 1
            log_entry["Not_Detected"] = True
            log_entry["Count_Details"].append("Not Detected +1")

        log_data.append(log_entry)

    df_result = pd.DataFrame.from_dict(error_matrix, orient='index')
    df_log = pd.DataFrame(log_data)

    return df_result, df_log

def process_file_pair(asr_file, label_file, model, speaker, voice_type, output_dir):
    """
    Process one pair of ASR error report and therapist label file.
    """
    print(f"Processing: {os.path.basename(asr_file)} + {os.path.basename(label_file)}")

    error_events = parse_error_report(asr_file)
    grouped_df = parse_group_labels(label_file)

    if error_events and not grouped_df.empty:
        df_result, df_log = create_confusion_matrix(
            model, speaker, voice_type, error_events, grouped_df
        )

        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, f"{model}_{speaker}_{voice_type}_Result.csv")
        log_file = os.path.join(output_dir, f"{model}_{speaker}_{voice_type}_Log.csv")

        df_result.to_csv(result_file, index=True)
        df_log.to_csv(log_file, index=False)

        print(f"Results saved to: {result_file}, {log_file}")
        return df_result
    else:
        print("Skipping: No data found in files")
        return None

def generate_summary_matrices(output_dir):
    """
    Generate summary matrices: by model, by speaker, and overall.
    """
    csv_files = [f for f in os.listdir(output_dir) if f.endswith("_Result.csv")]

    if not csv_files:
        print("No result files found for summary generation")
        return

    speaker_summary = {}
    model_summary = {}
    all_dfs = []

    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        parts = filename.split("_")
        if len(parts) < 3:
            continue

        model = parts[0]
        speaker = parts[1]

        filepath = os.path.join(output_dir, csv_file)
        df = pd.read_csv(filepath, index_col=0)
        all_dfs.append(df)

        if speaker not in speaker_summary:
            speaker_summary[speaker] = df.copy()
        else:
            speaker_summary[speaker] += df

        if model not in model_summary:
            model_summary[model] = df.copy()
        else:
            model_summary[model] += df

    summary_dir = os.path.join(output_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    for speaker, df_sum in speaker_summary.items():
        output_path = os.path.join(summary_dir, f"{speaker}_Summary.xlsx")
        df_sum.to_excel(output_path, sheet_name=speaker)

    for model, df_sum in model_summary.items():
        output_path = os.path.join(summary_dir, f"{model}_Summary.xlsx")
        df_sum.to_excel(output_path, sheet_name=model)

    if all_dfs:
        total_summary = pd.concat(all_dfs).groupby(level=0).sum()
        total_output_path = os.path.join(summary_dir, "Total_Summary.xlsx")
        total_summary.to_excel(total_output_path, sheet_name="Total")

    print(f"Summary matrices saved to: {summary_dir}")

def visualize_confusion_matrix(matrix_file, output_dir):
    """
    Visualize confusion matrix as a heatmap.
    """
    df = pd.read_excel(matrix_file)

    if "Total Count" in df.columns:
        df_matrix = df.drop(columns=["Total Count"])
    else:
        df_matrix = df

    row_sums = df_matrix.sum(axis=1)
    df_normalized = df_matrix.div(row_sums, axis=0) * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_normalized, annot=True, cmap="YlGnBu", fmt=".1f",
                linewidths=.5, cbar_kws={'label': 'Percentage (%)'})

    plt.title('Confusion Matrix (ASR Error Type vs Therapist Label)', fontsize=14)
    plt.xlabel('ASR Detected Error Type', fontsize=12)
    plt.ylabel('Therapist Label', fontsize=12)
    plt.tight_layout()

    filename = os.path.basename(matrix_file).replace('.xlsx', '_heatmap.png')
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300)

    print(f"Confusion matrix visualization saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Confusion Matrix Analysis Script')
    parser.add_argument('--asr_dir', type=str, default='./evaluation/error_reports',
                      help='Directory containing ASR error reports')
    parser.add_argument('--label_dir', type=str, default='./data/label/grouped',
                      help='Directory containing grouped therapist labels')
    parser.add_argument('--output_dir', type=str, default='./evaluation/matrix/matrix_53group',
                      help='Directory to save output results')
    parser.add_argument('--models', type=str, default='tiny,base,small,medium,large,turbo',
                      help='Comma-separated list of models to process')
    parser.add_argument('--speakers', type=str, default='SA001,SA002,SA003,SA004,SA005,SA006',
                      help='Comma-separated list of speakers to process')
    parser.add_argument('--visualize', action='store_true',
                      help='Generate visualization from summary')
    args = parser.parse_args()

    model_sizes = args.models.split(',')
    speakers = args.speakers.split(',')
    voice_types = ["grandfather", "rainbow"]

    os.makedirs(args.output_dir, exist_ok=True)

    for speaker in speakers:
        for voice_type in voice_types:
            for model in tqdm(model_sizes, desc=f"Processing {speaker}_{voice_type}"):
                asr_file = os.path.join(args.asr_dir, f"{model}_{speaker}_{voice_type}_Error_Report.txt")
                label_file = os.path.join(args.label_dir, f"grouped_{speaker}_{voice_type}_Labels 1.xlsx")

                if not os.path.exists(asr_file):
                    print(f"Skipping: ASR file not found - {asr_file}")
                    continue

                if not os.path.exists(label_file):
                    print(f"Skipping: Label file not found - {label_file}")
                    continue

                process_file_pair(asr_file, label_file, model, speaker, voice_type, args.output_dir)

    generate_summary_matrices(args.output_dir)

    if args.visualize:
        summary_dir = os.path.join(args.output_dir, "summary")
        total_summary = os.path.join(summary_dir, "Total_Summary.xlsx")

        if os.path.exists(total_summary):
            visualize_confusion_matrix(total_summary, args.output_dir)

        for model in model_sizes:
            model_summary = os.path.join(summary_dir, f"{model}_Summary.xlsx")
            if os.path.exists(model_summary):
                visualize_confusion_matrix(model_summary, args.output_dir)

if __name__ == "__main__":
    main()
