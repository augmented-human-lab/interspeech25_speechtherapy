"""
ASR Error Report Generation Script
Generates error reports by comparing Whisper ASR results with original transcripts.
"""

import json
import os
import re
from difflib import SequenceMatcher

def main():
    # Set folder paths
    folder_path = './data'
    json_directory = os.path.join(folder_path, "asr")
    output_directory = os.path.join('./evaluation', "error_reports")
    
    # Create output directory if it does not exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Load original transcript text based on file type
    def load_transcript(file_type):
        """ Load the appropriate transcript file based on the given type (grandfather or rainbow) """
        file_name = "grandfather.txt" if file_type == "grandfather" else "rainbow.txt"
        file_path = os.path.join(folder_path, "transcripts", file_name)
        
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                transcript = file.read()
            print(f"Loaded transcript: {file_name}")
            return transcript
        except FileNotFoundError:
            print(f"Error: Transcript file '{file_path}' not found.")
            return None
    
    # Convert ASR JSON to label format
    def convert_json_to_labels(json_file, output_file):
        """ Convert JSON ASR output into label format (start time, end time, word) """
        try:
            with open(json_file, 'r') as infile, open(output_file, 'w') as outfile:
                data = json.load(infile)
                
                for segment in data.get("segments", []):
                    for word in segment.get("words", []):
                        start = word.get("start", 0)
                        end = word.get("end", 0)
                        text = word.get("text", "")
                        
                        outfile.write(f"{start:.6f}\t{end:.6f}\t{text}\n")
                
            print(f"JSON converted to label format: {output_file}")
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    # Compare ASR JSON output with original transcript and generate error report
    def align_asr_with_transcript(json_file, transcript, output_file):
        """ Align ASR result with original transcript and generate error analysis report """
        
        if transcript is None:
            print(f"Skipping ASR alignment for {json_file} (Transcript missing)")
            return
        
        try:
            def preprocess_text(text):
                return re.sub(r'[^\w\s]', '', text).strip().upper().split()
            
            transcript_cleaned = preprocess_text(transcript)
            
            with open(json_file, 'r') as infile, open(output_file, 'w') as outfile:
                data = json.load(infile)
                
                asr_words = [
                    (
                        re.sub(r'[^\w\s]', '', (word.get("text", "") or "").upper()),
                        word.get("start", 0),
                        word.get("end", 0)
                    )
                    for segment in data.get("segments", [])
                    for word in segment.get("words", [])
                ]
                
                asr_texts = [word[0] for word in asr_words]
                
                matcher = SequenceMatcher(None, transcript_cleaned, asr_texts)
                match_blocks = matcher.get_opcodes()
                
                for tag, i1, i2, j1, j2 in match_blocks:
                    if tag == 'equal':
                        for k in range(i2 - i1):
                            original_word = transcript_cleaned[i1 + k]
                            asr_word, start, end = asr_words[j1 + k]
                            outfile.write(f"MATCH: {start:.6f}-{end:.6f}: {asr_word}\n")
                    elif tag == 'replace':
                        for original_word, (asr_word, start, end) in zip(
                            transcript_cleaned[i1:i2], asr_words[j1:j2]
                        ):
                            if asr_word == original_word:
                                outfile.write(f"MATCH: {start:.6f}-{end:.6f}: {asr_word}\n")
                            else:
                                outfile.write(
                                    f"ERROR (Substitution): {start:.6f}-{end:.6f}: ASR='{asr_word}' | Original='{original_word}'\n"
                                )
                    elif tag == 'delete':
                        for original_word in transcript_cleaned[i1:i2]:
                            outfile.write(f"ERROR (Deletion): [MISSING]: Original='{original_word}'\n")
                    elif tag == 'insert':
                        for asr_word, start, end in asr_words[j1:j2]:
                            outfile.write(f"ERROR (Insertion): {start:.6f}-{end:.6f}: ASR='{asr_word}' | Original='[NOT IN TRANSCRIPT]'\n")
            
            print(f"ASR alignment completed. Output saved: {output_file}")
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    # Process ASR JSON files
    if os.path.exists(json_directory):
        for filename in os.listdir(json_directory):
            if filename.endswith("_ASR.json"):  # Only process ASR output files
                json_file_path = os.path.join(json_directory, filename)
                
                # Extract model name, audio ID, and category (grandfather/rainbow) from filename
                parts = filename.replace("_ASR.json", "").split('_')
                
                if len(parts) != 3:
                    print(f"Warning: Unexpected filename format {filename}, skipping...")
                    continue  # Skip files with unexpected format
                
                model_name = parts[0]      # e.g., 'turbo'
                audio_name = parts[1]      # e.g., 'SA004'
                category = parts[2]        # 'grandfather' or 'rainbow'
                
                if category not in ["grandfather", "rainbow"]:
                    print(f"Warning: Unknown category in {filename}, skipping...")
                    continue
                
                transcript = load_transcript(category)
                
                # Generate label file
                output_label_file = os.path.join(output_directory, f"{model_name}_{audio_name}_{category}_Labels.txt")
                convert_json_to_labels(json_file_path, output_label_file)
                
                # Generate error analysis report
                output_report_file = os.path.join(output_directory, f"{model_name}_{audio_name}_{category}_Error_Report.txt")
                align_asr_with_transcript(json_file_path, transcript, output_report_file)
    else:
        print(f"Warning: JSON directory '{json_directory}' does not exist!")

if __name__ == "__main__":
    main()
