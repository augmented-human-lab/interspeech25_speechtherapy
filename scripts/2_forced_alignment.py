"""
Forced Alignment Script
Performs word-level alignment using Wav2Vec2.
"""

import torch
import torchaudio
from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H
from dataclasses import dataclass
import re
import os
import argparse

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float

@dataclass
class Segment:
    word: str
    start: float
    end: float
    score: float

def load_transcript(text_path=None):
    """
    Load transcript from file or use default grandfather/rainbow text
    """
    if text_path and os.path.exists(text_path):
        with open(text_path, 'r', encoding='utf-8') as f:
            transcript = f.read()
    else:
        # Default transcript (modify as needed)
        transcript = """You wished to know all about my grandfather. Well, he is nearly ninety-three years old; 
        he dresses himself in an ancient black frock coat, usually minus several buttons; yet he still thinks as swiftly as ever. 
        A long, flowing beard clings to his chin, giving those who observe him a pronounced feeling of the utmost respect. 
        When he speaks, his voice is just a bit cracked and quivers a trifle. Twice each day he plays skillfully and with zest upon our small organ. 
        Except in the winter when the ooze or snow or ice prevents, he slowly takes a short walk in the open air each day. 
        We have often urged him to walk more and smoke less, but he always answers, \"Banana oil!\" Grandfather likes to be modern in his language. 
        When the sunlight strikes raindrops in the air, they act like a prism and form a rainbow. 
        The rainbow is a division of white light into many beautiful colors. 
        These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. 
        There is, according to legend, a boiling pot of gold at one end. People look but no one ever finds it. 
        When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow."""
    
    return transcript

def get_trellis(emission, tokens, blank_id=0):
    """
    Create trellis matrix using dynamic programming
    """
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    trellis = torch.full((num_frame, num_tokens), -float("inf"))
    trellis[0, 0] = 0  # Starting point

    for t in range(num_frame - 1):
        trellis[t + 1, 0] = trellis[t, 0] + emission[t, blank_id]
        for j in range(1, num_tokens):
            trellis[t + 1, j] = max(
                trellis[t, j] + emission[t, blank_id],  # Stay
                trellis[t, j - 1] + emission[t, tokens[j]]  # Move
            )
    return trellis

def backtrack(trellis, emission, tokens, blank_id=0):
    """
    Perform Viterbi backtracking to get alignment path
    """
    t, j = trellis.size(0) - 1, trellis.size(1) - 1
    path = [Point(j, t, emission[t, blank_id].exp().item())]

    while j > 0:
        assert t > 0
        p_stay = emission[t - 1, blank_id]
        p_change = emission[t - 1, tokens[j]]

        stayed = trellis[t - 1, j] + p_stay
        changed = trellis[t - 1, j - 1] + p_change

        t -= 1
        if changed > stayed:
            j -= 1

        prob = (p_change if changed > stayed else p_stay).exp().item()
        path.append(Point(j, t, prob))

    while t > 0:
        prob = emission[t - 1, blank_id].exp().item()
        path.append(Point(j, t - 1, prob))
        t -= 1

    return path[::-1]

def get_word_segments_from_ground_truth(path, transcript, frame_rate, labels, tokens):
    """
    Extract word-level segments from alignment path
    """
    segments = []
    words = transcript.split()  # Exact word splits
    word_index = 0
    word_start_time = None
    word_scores = []
    current_word = ""

    for i in range(len(path) - 1):
        token_index = path[i].token_index
        if token_index == 0:  # Skip blank token
            continue

        char = labels[tokens[token_index]]

        if char == '|':  # End of word
            if current_word:
                start_time = word_start_time / frame_rate
                end_time = path[i].time_index / frame_rate
                avg_score = sum(word_scores) / len(word_scores)
                segments.append(Segment(words[word_index], start_time, end_time, avg_score))

                word_index += 1
                current_word = ""
                word_start_time = None
                word_scores = []
        else:
            if not current_word:
                word_start_time = path[i].time_index
            current_word += char
            word_scores.append(path[i].score)

    # Handle last word
    if word_index < len(words):
        start_time = word_start_time / frame_rate
        end_time = path[-1].time_index / frame_rate
        avg_score = sum(word_scores) / len(word_scores)
        segments.append(Segment(words[word_index], start_time, end_time, avg_score))

    return segments

def process_audio(audio_path, output_path, transcript_path=None):
    """
    Process audio file and perform forced alignment
    """
    print(f"Processing: {audio_path}")
    
    # Load Wav2Vec2 model
    bundle = WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)
    labels = bundle.get_labels()

    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)

    # Resample if needed
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=bundle.sample_rate)(waveform)

    waveform = waveform.to(device)

    # Generate emissions
    with torch.inference_mode():
        emissions, _ = model(waveform)
        emissions = torch.log_softmax(emissions, dim=-1)

    emission = emissions[0].cpu().detach()

    # Compute frame rate
    frame_rate = bundle.sample_rate / 320  # Wav2Vec2 stride size

    # Load and preprocess transcript
    transcript = load_transcript(transcript_path)
    transcript = re.sub(r'[^\w\s]', '', transcript).upper()
    words = transcript.split()

    # Tokenize transcript
    dictionary = {c: i for i, c in enumerate(labels)}
    tokens = []
    for word in words:
        tokens.extend([dictionary[char] for char in word if char in dictionary])
        tokens.append(dictionary['|'])  # Word separator

    # Create trellis
    trellis = get_trellis(emission, tokens)

    # Backtrack to find alignment path
    path = backtrack(trellis, emission, tokens)

    # Extract word segments
    word_segments = get_word_segments_from_ground_truth(path, transcript, frame_rate, labels, tokens)

    # Save results
    with open(output_path, "w") as f:
        for segment in word_segments:
            f.write(f"{segment.start:.6f}\t{segment.end:.6f}\t{segment.word}\t{segment.score:.6f}\n")
    
    print(f"Alignment completed. Results saved to: {output_path}")
    return word_segments

def main():
    parser = argparse.ArgumentParser(description='Perform forced alignment on audio files')
    parser.add_argument('--audio_dir', type=str, default='./data/raw_audio',
                        help='Directory containing audio files')
    parser.add_argument('--output_dir', type=str, default='./data/alignments',
                        help='Directory to save alignment results')
    parser.add_argument('--transcript', type=str, default=None, 
                        help='Path to transcript file (optional)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each audio file
    for filename in os.listdir(args.audio_dir):
        if filename.endswith('.wav'):
            audio_path = os.path.join(args.audio_dir, filename)
            speaker_id = filename.split('.')[0]  # Filename without extension
            output_path = os.path.join(args.output_dir, f"{speaker_id}_alignment.txt")
            
            # Find corresponding transcript file
            transcript_path = None
            if args.transcript:
                if os.path.isdir(args.transcript):
                    for tfile in os.listdir(args.transcript):
                        if speaker_id in tfile and tfile.endswith('.txt'):
                            transcript_path = os.path.join(args.transcript, tfile)
                            break
                else:
                    transcript_path = args.transcript
            
            # Run forced alignment
            process_audio(audio_path, output_path, transcript_path)

if __name__ == "__main__":
    main()
