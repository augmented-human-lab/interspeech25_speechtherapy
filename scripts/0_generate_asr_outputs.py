"""
ASR Output Generation Script
Generates ASR results from audio files using the Whisper model.
"""

import whisper_timestamped as whisper
import json
import os
import torch

def main():
    # Check if GPU is available and set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Set folder paths
    folder_path = './data'
    audio_directory = os.path.join(folder_path, "raw_audio")
    json_directory = os.path.join(folder_path, "asr")
    
    # Create result directory if it does not exist
    if not os.path.exists(json_directory):
        os.makedirs(json_directory)
    
    # Function to load original transcript text
    def load_transcript(file_type):
        """ Load the appropriate transcript file based on file type (grandfather or rainbow) """
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
    
    # List of Whisper model sizes
    model_list = ['tiny', 'base', 'small', 'medium', 'large', 'turbo']
    
    if os.path.exists(audio_directory):
        for model_size in model_list:
            print(f"Loading model: {model_size}")
            model = whisper.load_model(model_size, device=device)
            
            for filename in os.listdir(audio_directory):
                if filename.endswith(".wav"):
                    audio_path = os.path.join(audio_directory, filename)
                    audio_name = filename.split('.')[0]
                    
                    # Determine if the file is 'grandfather' or 'rainbow'
                    file_type = "grandfather" if "grandfather" in filename else "rainbow"
                    
                    print(f"Processing {filename} with {model_size} model...")
                    result = whisper.transcribe(model, audio_path, language="en")
                    
                    # Save the result as a JSON file
                    output_file = os.path.join(json_directory, f"{model_size}_{audio_name}_ASR.json")
                    with open(output_file, "w", encoding="utf-8") as file:
                        json.dump(result, file, indent=2, ensure_ascii=False)
                    
                    print(f"Transcription saved: {output_file}")
    else:
        print(f"Warning: Audio directory '{audio_directory}' does not exist!")

if __name__ == "__main__":
    main()
