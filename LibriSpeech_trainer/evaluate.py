
import torch
import torch.nn.functional as F
from pathlib import Path
import librosa
import numpy as np
import random
from tqdm import tqdm

from model import SpeakerEncoder # Import our model class
from train import audio_to_spectrogram # Import our spectrogram function

# --- Config ---
TEST_DIR = "data/LibriSpeech/test-clean"
MODEL_PATH = "speaker_encoder.pth"
NUM_PAIRS = 1000 # Number of pairs to test (e.g., 1000 positive, 1000 negative)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Helper function to get an embedding ---
def get_embedding(file_path, model):
    """Loads an audio file and returns its embedding from the model."""
    # 1. Load audio and create spectrogram
    # We add .unsqueeze(0) to create a batch of 1
    spec = audio_to_spectrogram(file_path).unsqueeze(0)
    
    # 2. Permute and add channel dim (like in training)
    # (N, Time, Mels) -> (N, 1, Mels, Time)
    spec = spec.permute(0, 2, 1).unsqueeze(1).to(device)

    # 3. Get embedding from model
    # We set return_embedding=True
    # The model now returns (classification, embedding)
    # We take the embedding (the [1] index)
    with torch.no_grad(): # Disable gradient calculation
        _, embedding = model(spec)
    
    return embedding

# --- 2. Load Data ---
print("Scanning test-clean directory...")
test_path = Path(TEST_DIR)
speaker_files = {} # Dictionary to map speaker_id -> list of files

# Group all files by their speaker ID
for file_path in test_path.rglob("*.flac"):
    speaker_id = file_path.parent.parent.name
    if speaker_id not in speaker_files:
        speaker_files[speaker_id] = []
    speaker_files[speaker_id].append(file_path)

speaker_ids = list(speaker_files.keys())
print(f"Found {len(speaker_ids)} speakers in test set.")

# --- 3. Load Model ---
# We need to know the number of speakers the model was trained on.
# We can find this by scanning the *training* directory again.
# (A better way is to save this info, but this works)
train_speakers = set(f.parent.parent.name for f in Path("data/LibriSpeech/train-clean-100").rglob("*.flac"))
num_speakers_trained = len(train_speakers)

print(f"Loading model trained on {num_speakers_trained} speakers.")
model = SpeakerEncoder(num_speakers=num_speakers_trained).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only =True))
model.eval() # Set model to evaluation mode (disables dropout, etc.)

# --- 4. Generate and Test Pairs ---
positive_scores = []
negative_scores = []
print(f"Generating and testing {NUM_PAIRS} positive and negative pairs...")

for _ in tqdm(range(NUM_PAIRS)):
    # --- Positive Pair ---
    # 1. Pick a random speaker
    speaker_id = random.choice(speaker_ids)
    # 2. Pick two different files from that speaker
    file_a, file_b = random.sample(speaker_files[speaker_id], 2)
    
    # 3. Get embeddings and calculate similarity
    embed_a = get_embedding(file_a, model)
    embed_b = get_embedding(file_b, model)
    score = F.cosine_similarity(embed_a, embed_b).item()
    positive_scores.append(score)

    # --- Negative Pair ---
    # 1. Pick two different speakers
    speaker_a, speaker_b = random.sample(speaker_ids, 2)
    # 2. Pick one file from each
    file_a = random.choice(speaker_files[speaker_a])
    file_b = random.choice(speaker_files[speaker_b])
    
    # 3. Get embeddings and calculate similarity
    embed_a = get_embedding(file_a, model)
    embed_b = get_embedding(file_b, model)
    score = F.cosine_similarity(embed_a, embed_b).item()
    negative_scores.append(score)

# --- 5. Show Results ---
print("\n--- Evaluation Complete ---")
print(f"Average Similarity (Same Speaker):   {np.mean(positive_scores):.4f}")
print(f"Average Similarity (Diff. Speaker):  {np.mean(negative_scores):.4f}")
