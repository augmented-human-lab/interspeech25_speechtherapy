"""
Time-Based Error Localization Script
Performs time-based matching between ASR errors and therapist labels to compute evaluation metrics.
"""

import re
import os
import pandas as pd
import argparse
from tqdm import tqdm

def parse_error_report(file_path):
    """
    Extract error information from ASR error report
    - Error type (substitution, deletion, insertion)
    - Start and end time
    - ASR and original words
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
    Extract error label data from therapist's grouped label file
    - Start and end time
    - Error type
    - Label description
    """
    try:
        df = pd.read_excel(file_path, header=0)

        # Standardize column names
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
    Check if two time intervals overlap
    - Includes boundary values
    - Special handling if interval is a point (start == end)
    """
    if start2 == end2:
        return start1 <= start2 <= end1
    else:
        return (start1 <= end2) and (start2 <= end1)

def calculate_counts_and_log(model, speaker, voice_type, error_events, grouped_df):
    """
    Match ASR errors with therapist labels and calculate TP, FP, FN
    
    Returns:
    - results: Dictionary with TP/FP/FN counts by error type
    - log_df: DataFrame containing matching log
    """
    therapist_error_types = set(grouped_df['group'].unique())
    
    results = {et: {"TP": 0, "FP": 0, "FN": 0, "Model_Size": model} for et in therapist_error_types}
    log_data = []
    matched_asr_errors = set()

    print(f"\n=== Matching: Model={model}, Speaker={speaker}, Voice_Type={voice_type} ===")
    print(f"Therapist labels: {sorted(therapist_error_types)}")
    
    # 1. Match from therapist label side (TP, FN)
    for _, row in grouped_df.iterrows():
        group_start = row['start']
        group_end = row['end']
        group_type = row['group']
        group_desc = row['Label'] if 'Label' in row else ""

        print(f"\n[Checking therapist label] {group_type} ({group_start:.2f} ~ {group_end:.2f}) - Desc: {group_desc[:30]}...")
        
        matching_errors = [e for e in error_events 
                         if e['type'].lower() == group_type.lower() and 
                            intervals_overlap(group_start, group_end, e['start'], e['end'])]
        
        tp_increment = 1 if matching_errors else 0
        fn_increment = 1 if not matching_errors else 0
        
        if group_type in results:
            results[group_type]["TP"] += tp_increment
            results[group_type]["FN"] += fn_increment

        log_entry = {
            "Model_Size": model,
            "Speaker": speaker,
            "Voice_Type": voice_type,
            "Therapist_Error_Group": group_type,
            "Therapist_Error_Description": group_desc,
            "Therapist_Label_Start": group_start,
            "Therapist_Label_End": group_end,
            "ASR_Error_Type": matching_errors[0]['type'] if matching_errors else "N/A",
            "ASR_Error_Start": matching_errors[0]['start'] if matching_errors else "N/A",
            "ASR_Error_End": matching_errors[0]['end'] if matching_errors else "N/A",
            "ASR_Word": matching_errors[0]['asr_word'] if matching_errors else "N/A",
            "Original_Word": matching_errors[0]['original_word'] if matching_errors else "N/A",
            "Match_Status": "Matched" if matching_errors else "No Match",
            "TP_Increment": tp_increment,
            "FN_Increment": fn_increment,
            "FP_Increment": 0
        }
        
        log_data.append(log_entry)

        if matching_errors:
            matched_asr_errors.add((matching_errors[0]['start'], matching_errors[0]['end']))
    
    # 2. Match from ASR side (FP)
    for e in error_events:
        if (e['start'], e['end']) in matched_asr_errors:
            continue
            
        if not any(intervals_overlap(row['start'], row['end'], e['start'], e['end']) 
                  for _, row in grouped_df.iterrows()):
            
            if e['type'] in results:
                results[e['type']]["FP"] += 1
            
            log_data.append({
                "Model_Size": model,
                "Speaker": speaker,
                "Voice_Type": voice_type,
                "Therapist_Error_Group": "N/A",
                "Therapist_Error_Description": "N/A",
                "Therapist_Label_Start": "N/A",
                "Therapist_Label_End": "N/A",
                "ASR_Error_Type": e['type'],
                "ASR_Error_Start": e['start'],
                "ASR_Error_End": e['end'],
                "ASR_Word": e['asr_word'],
                "Original_Word": e['original_word'],
                "Match_Status": "No Match (FP)",
                "TP_Increment": 0,
                "FN_Increment": 0,
                "FP_Increment": 1
            })
    
    df_log = pd.DataFrame(log_data)
    
    df_log["Therapist_Label_Start"] = pd.to_numeric(df_log["Therapist_Label_Start"], errors="coerce")
    df_log["ASR_Error_Start"] = pd.to_numeric(df_log["ASR_Error_Start"], errors="coerce")
    df_log = df_log.sort_values(by=["Therapist_Label_Start", "ASR_Error_Start"], ascending=[True, True])
    
    return results, df_log

def process_file_pair(asr_file, label_file, model, speaker, voice_type, output_dir):
    """
    Process one ASR file and corresponding therapist label file
    """
    print(f"Processing: {os.path.basename(asr_file)} + {os.path.basename(label_file)}")
    
    error_events = parse_error_report(asr_file)
    grouped_df = parse_group_labels(label_file)
    
    if error_events and not grouped_df.empty:
        error_counts, log_df = calculate_counts_and_log(model, speaker, voice_type, error_events, grouped_df)
        
        log_output_file = os.path.join(output_dir, f"{model}_{speaker}_{voice_type}_Matching_Log.csv")
        os.makedirs(output_dir, exist_ok=True)
        log_df.to_csv(log_output_file, index=False)
        
        print(f"Results saved to: {log_output_file}")
        return error_counts, log_df
    else:
        print(f"Skipping: No data found in files")
        return None, None

def main():
    parser = argparse.ArgumentParser(description='Time-based error matching script')
    parser.add_argument('--asr_dir', type=str, default='./evaluation/error_reports',
                      help='Directory for ASR error reports')
    parser.add_argument('--label_dir', type=str, default='./data/label/grouped',
                      help='Directory for grouped therapist labels')
    parser.add_argument('--output_dir', type=str, default='./evaluation/final_counts',
                      help='Directory to save results')
    parser.add_argument('--models', type=str, default='tiny,base,small,medium,large,turbo',
                      help='List of models to evaluate (comma-separated)')
    parser.add_argument('--speakers', type=str, default='SA001,SA002,SA003,SA004,SA005,SA006',
                      help='List of speakers to evaluate (comma-separated)')
    args = parser.parse_args()
    
    model_sizes = args.models.split(',')
    speakers = args.speakers.split(',')
    voice_types = ["grandfather", "rainbow"]
    
    results_list = []
    log_list = []
    
    print(f"Processing {len(model_sizes)} models × {len(speakers)} speakers × {len(voice_types)} voice types...")
    
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
                
                error_counts, log_df = process_file_pair(
                    asr_file, label_file, model, speaker, voice_type, args.output_dir
                )
                
                if error_counts:
                    for error_type, counts in error_counts.items():
                        results_list.append({
                            "Speaker": speaker,
                            "Voice_Type": voice_type,
                            "Model": model,
                            "Error_Type": error_type,
                            "TP": counts["TP"],
                            "FP": counts["FP"],
                            "FN": counts["FN"]
                        })
                
                if log_df is not None:
                    log_list.append(log_df)
    
    if results_list:
        results_df = pd.DataFrame(results_list)
        results_output = os.path.join(args.output_dir, "final_counts_with_matching_logs.xlsx")
        results_df.to_excel(results_output, index=False)
        print(f"Final counts saved to: {results_output}")
    
    if log_list:
        combined_log = pd.concat(log_list, ignore_index=True)
        log_output = os.path.join(args.output_dir, "final_matching_log.csv")
        combined_log.to_csv(log_output, index=False)
        print(f"Combined matching log saved to: {log_output}")

if __name__ == "__main__":
    main()
