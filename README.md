# Towards Temporally Explainable Dysarthric Speech Clarity Assessment

**Accepted at Interspeech 2025**

This repository contains the implementation code for our paper "Towards Temporally Explainable Dysarthric Speech Clarity Assessment," which presents a three-stage framework for automated mispronunciation evaluation in dysarthric speech.

[Paper](https://arxiv.org/abs/2506.00454)
[Supplementary material](https://apps.ahlab.org/interspeech25_speechtherapy/)

## Overview

Our framework provides:
1. Overall clarity scoring of passages or utterances
2. Mispronunciation localization to identify regions with reduced intelligibility  
3. Mispronunciation type classification into specific error categories

We systematically analyze pretrained Automatic Speech Recognition (ASR) models across these three stages, offering clinically relevant insights for automating actionable feedback in pronunciation assessment.

## Framework Architecture

```
Audio Input → ASR Processing → Error Analysis → Clinical Feedback
    ↓              ↓              ↓              ↓
Stage 1:     Stage 2:        Stage 3:      Applications:
Clarity      Temporal        Error Type    Speech Therapy
Scoring      Localization    Classification    Support
```

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- OpenAI API key (for label categorization)

### Setup
```bash
git clone https://github.com/augmented-human-lab/interspeech25_speechtherapy.git
cd interspeech25_speechtherapy
pip install -r requirements.txt
```

## Usage Guide

### Pipeline Overview
Our analysis pipeline consists of 7 sequential scripts that process audio data through the complete framework:

```
0_generate_asr_outputs.py → 1_generate_error_reports.py → 2_forced_alignment.py → 
3_llm_label_categorization.py → 4_error_localization.py → 5_evaluation_matrices.py → 
6_error_matrix_analysis.py
```

### Script Descriptions

#### Stage 1: ASR Processing

**0_generate_asr_outputs.py** - Generate ASR transcriptions using Whisper models
```bash
python scripts/0_generate_asr_outputs.py
```
- Processes audio files with multiple Whisper model sizes (tiny, base, small, medium, large, turbo)
- Outputs timestamped JSON transcriptions
- Input: Audio files (.wav)
- Output: ASR results with word-level timestamps

**1_generate_error_reports.py** - Compare ASR outputs with ground truth transcripts
```bash
python scripts/1_generate_error_reports.py
```
- Aligns ASR transcriptions with original transcripts
- Identifies substitution, deletion, and insertion errors
- Input: ASR JSON files, reference transcripts
- Output: Detailed error reports with timestamps

#### Stage 2: Temporal Analysis

**2_forced_alignment.py** - Perform word-level forced alignment
```bash
python scripts/2_forced_alignment.py --audio_dir ./data/raw_audio --output_dir ./data/alignments
```
- Uses Wav2Vec2 for precise word-level timing
- Input: Audio files
- Output: Word-level alignment files

**3_llm_label_categorization.py** - Categorize therapist labels using LLM
```bash
export OPENAI_API_KEY="your-api-key-here"
python scripts/3_llm_label_categorization.py --input_dir ./data/label/origin --output_dir ./data/label/grouped
```
- Automatically categorizes therapist annotations into 5 error types:
  - Substitution Errors
  - Deletion Errors  
  - Insertion Errors
  - Repetition Errors
  - Prosodic Errors
- Input: Raw therapist label files
- Output: Categorized labels in Excel format

#### Stage 3: Evaluation and Analysis

**4_error_localization.py** - Compute temporal matching metrics
```bash
python scripts/4_error_localization.py --asr_dir ./evaluation/error_reports --label_dir ./data/label/grouped --output_dir ./evaluation/final_counts
```
- Matches ASR-detected errors with therapist labels temporally
- Calculates TP, FP, FN counts for each error type
- Input: Error reports, grouped therapist labels
- Output: Matching logs and count matrices

**5_evaluation_matrices.py** - Generate performance metrics and visualizations
```bash
python scripts/5_evaluation_matrices.py --input ./evaluation/final_counts/final_counts_with_matching_logs.xlsx --output_dir ./evaluation/figures
```
- Computes precision, recall, and F1 scores
- Generates comparison plots across models and error types
- Input: TP/FP/FN count matrices
- Output: Performance metrics and visualization plots

**6_error_matrix_analysis.py** - Create confusion matrices
```bash
python scripts/6_error_matrix_analysis.py --asr_dir ./evaluation/error_reports --label_dir ./data/label/grouped --output_dir ./evaluation/matrix --visualize
```
- Generates confusion matrices between ASR and therapist classifications
- Creates summary statistics by model and speaker
- Input: Error reports, therapist labels
- Output: Confusion matrices and heatmap visualizations

### Running the Complete Pipeline

For a full analysis, run the scripts sequentially:

```bash
# Stage 1: ASR Processing
python scripts/0_generate_asr_outputs.py
python scripts/1_generate_error_reports.py

# Stage 2: Alignment and Categorization  
python scripts/2_forced_alignment.py
python scripts/3_llm_label_categorization.py

# Stage 3: Evaluation
python scripts/4_error_localization.py
python scripts/5_evaluation_matrices.py
python scripts/6_error_matrix_analysis.py --visualize
```

## Key Findings

- ASR-based clarity scores strongly correlate with dysarthria severity levels (r > 0.9)
- Larger Whisper models achieve higher precision in temporal localization
- Substitution and deletion errors are detected more accurately than prosodic errors
- 70.1% exact error match rate between ASR and therapist annotations

## Repository Structure

```
├── scripts/                 # Analysis pipeline scripts
│   ├── 0_generate_asr_outputs.py
│   ├── 1_generate_error_reports.py
│   ├── 2_forced_alignment.py
│   ├── 3_llm_label_categorization.py
│   ├── 4_error_localization.py
│   ├── 5_evaluation_matrices.py
│   └── 6_error_matrix_analysis.py
├── plots/                   # Generated visualization plots
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Expected Data Structure

When using your own data, organize it as follows:
```
data/
├── raw_audio/              # Original audio recordings (.wav)
├── transcripts/            # Reference transcripts (grandfather.txt, rainbow.txt)
├── label/
│   ├── origin/            # Original therapist annotations (.txt)
│   └── grouped/           # LLM-categorized labels (.xlsx)
├── asr/                   # ASR output files (.json)
└── alignments/            # Forced alignment results (.txt)
```

## Clinical Applications

This framework enables:
- Automated speech assessment for patients with dysarthria
- Targeted therapy recommendations based on specific error types
- Progress tracking through objective clarity measurements
- Scalable evaluation reducing therapist workload

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{your2025interspeech,
  title={Towards Temporally Explainable Dysarthric Speech Clarity Assessment},
  author={Seohyun Park*, Chitralekha Gupta*, Michelle Kah Yian Kwan, Xinhui Fung, Alexander Wenjun Yip, Suranga Nanayakkara (*equal contributors)},
  booktitle={Proceedings of Interspeech 2025},
  year={2025}
}
```

## Contact

For questions about the code or methodology, please contact `emily21@korea.ac.kr` .

## Data Availability

The dysarthric speech dataset with expert therapist annotations will be made available for research purposes following institutional review and approval processes. Please check back for updates or contact us for information about data access.

## License

[License information to be added]

---

**Note**: This repository currently contains the analysis scripts and visualization tools. The complete dataset will be released pending ethical approval and privacy considerations.
