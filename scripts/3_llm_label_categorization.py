"""
LLM-based Label Classification Script
Automatically categorizes original therapist labels by error type using the OpenAI API.
"""

import os
import pandas as pd
import argparse
import openai
import time
from tqdm import tqdm

def setup_openai_api():
    """Setup OpenAI API"""
    # Load API key (from environment variable or user input)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Enter your OpenAI API key: ")
    
    openai.api_key = api_key
    return openai.Client(api_key=api_key)

def generate_category(client, label):
    """Classify label using the OpenAI API"""
    prompt = f"""
You are a world-class Speech-Language Pathologist specializing in speech disorder classification and ASR (Automatic Speech Recognition) error analysis. Your task is to categorize the given speech error label into one of the predefined categories with high accuracy.

### **Strict Categorization Guidelines**
- **Follow the categorization rules precisely. No assumptions or reinterpretations.**
- **If a label contains multiple error types, select the most specific category based on hierarchy.**
- **Do not include explanations, reasoning, or any additional text. Return only the category name.**

---

### **Categorization Rules (Hierarchy-Based)**
1. **Substitution Errors**
   - A phoneme or word is replaced with another phoneme or word.
   - MUST be used if "phonemic sub", "word sub", or "transposition" appears in the label.
   - **Examples:**
     - "look, took, phonemic sub 'l' to 'k' or word sub" → "Substitution Errors"
     - "horizon, phonemic sub/distortion of vowel" → "Substitution Errors"

2. **Deletion Errors**
   - A phoneme or word is missing.
   - MUST be used if "phonemic del" or "deletion" appears in the label.
   - **Examples:**
     - "raindrops, raindops, phonemic del" → "Deletion Errors"
     - "act, ac, phonemic del (SCE), strained voice" → "Deletion Errors"

3. **Insertion Errors**
   - An extra phoneme or word is added.
   - MUST be used if "addition" appears in the label.
   - **Examples:**
     - "word addition 'big'" → "Insertion Errors"
     - "phonemic addition its -> biz" → "Insertion Errors"

4. **Repetition Errors**
   - A phoneme or word is repeated.
   - MUST be used if "repetition" appears in the label.
   - **Examples:**
     - "his, his his, repetition" → "Repetition Errors"
     - "prism, prism.prisms, repetition+ phonemic addition" → "Repetition Errors"
     - "word repetition 'look'" → "Repetition Errors"

5. **Prosodic Errors**
   - Irregular pauses, unnatural intonation, or breaks between words.
   - MUST be used if the label includes: "pause", "intonation", "break", "timing issue".
   - **Examples:**
     - "gold at one end, gold...at one end, irregular break between words" → "Prosodic Errors"
     - "strained voice" → "Prosodic Errors"

---

### **Input**
"{label}"

### **Output Format**
- **Return only one category name:**
  Substitution Errors, Deletion Errors, Insertion Errors, Repetition Errors, or Prosodic Errors.
- **Do NOT include explanations, reasoning, or additional text.**
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a speech disorder classification expert."},
                      {"role": "user", "content": prompt}],
            temperature=0
        )
        category = response.choices[0].message.content.strip()
        return category
    except Exception as e:
        print(f"API call error: {e}")
        return f"Error: {str(e)}"

def process_label_file(client, input_file, output_file):
    """
    Process label file and save classification results
    """
    try:
        # Read file (Start Time, End Time, Label)
        with open(input_file, "r", encoding="utf-8") as f:
            lines = [line.strip().split("\t") for line in f.readlines() if line.strip()]

        results = []

        print(f"\nProcessing: {input_file}")
        
        # Classify using LLM with progress bar
        for line in tqdm(lines, desc="Classifying Labels"):
            if len(line) < 3:
                continue  # Skip invalid lines

            start_time, end_time, label = line[0], line[1], line[2]
            
            # Prevent rapid repeated API calls
            time.sleep(0.5)
            
            category = generate_category(client, label)
            
            results.append({
                "Start Time": start_time, 
                "End Time": end_time, 
                "Label": label, 
                "Category": category
            })

        # Create DataFrame and save as Excel
        df = pd.DataFrame(results)
        
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        df.to_excel(output_file, index=False, engine="openpyxl")
        print(f"Grouping completed: {output_file}")
        
    except Exception as e:
        print(f"File processing error: {input_file} - {str(e)}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Automatic classification of therapist labels using LLM')
    parser.add_argument('--input_dir', type=str, default='./data/label/origin',
                      help='Directory containing original label files')
    parser.add_argument('--output_dir', type=str, default='./data/label/grouped',
                      help='Directory to save grouped classification results')
    parser.add_argument('--file', type=str, default=None,
                      help='Process a single file (optional)')
    args = parser.parse_args()
    
    # Setup OpenAI API
    client = setup_openai_api()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process either a single file or all files in the input directory
    if args.file:
        input_file = os.path.join(args.input_dir, args.file) if not os.path.isabs(args.file) else args.file
        file_basename = os.path.basename(input_file)
        output_file = os.path.join(args.output_dir, f"grouped_{file_basename}.xlsx")
        process_label_file(client, input_file, output_file)
    else:
        for filename in os.listdir(args.input_dir):
            if filename.endswith(".txt"):
                input_file = os.path.join(args.input_dir, filename)
                output_file = os.path.join(args.output_dir, f"grouped_{filename}.xlsx")
                process_label_file(client, input_file, output_file)

if __name__ == "__main__":
    main()
