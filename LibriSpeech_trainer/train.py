import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence # We need this for padding!
from pathlib import Path
import librosa
import numpy as np
from tqdm import tqdm # For a nice progress bar

# Import our model
from model import SpeakerEncoder

# --- 1. Audio Preprocessing ---
# (This is the function from our check_data.py)
def audio_to_spectrogram(file_path, n_mels=128):
    y, sr = librosa.load(file_path, sr=16000)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=2048, hop_length=512)
    S_db = librosa.power_to_db(S, ref=np.max)
    # We transpose the spectrogram to be (Time, Mel)
    # This is more standard for padding.
    spec_tensor = torch.tensor(S_db.T, dtype=torch.float32)
    return spec_tensor

# --- 2. The Custom Dataset Class ---

class LibriSpeechDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        
        # Find all .flac files
        print("Finding audio files...")
        self.audio_files = list(self.data_dir.rglob("*.flac"))
        
        # Create a mapping from speaker_id (str) to an integer index
        # We find all speaker folders (which are the parents' parents of the files)
        self.speaker_ids = sorted(list(set(f.parent.parent.name for f in self.audio_files)))
        self.speaker_to_int = {speaker: i for i, speaker in enumerate(self.speaker_ids)}
        
        self.num_speakers = len(self.speaker_ids)
        print(f"Found {len(self.audio_files)} files from {self.num_speakers} speakers.")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        file_path = self.audio_files[idx]
        
        # Get speaker_id and convert to integer label
        speaker_id = file_path.parent.parent.name
        label = self.speaker_to_int[speaker_id]
        
        # Load and convert audio to spectrogram
        # The spectrogram will have shape (Time, n_mels)
        spectrogram = audio_to_spectrogram(file_path)
        
        return spectrogram, label

# --- 3. The Collate Function (for Padding) ---

def pad_collate_fn(batch):
    """
    This function handles padding for a batch of spectrograms.
    Since audio clips have different lengths, their spectrograms will have
    different time dimensions. This function pads them all to the same
    length.
    """
    # batch is a list of tuples: [(spec1, label1), (spec2, label2), ...]
    
    # 1. Separate spectrograms and labels
    spectrograms = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # 2. Pad spectrograms
    # pad_sequence expects a list of tensors and pads them
    # batch_first=True makes the output shape (Batch_Size, Time, n_mels)
    spectrograms_padded = pad_sequence(spectrograms, batch_first=True, padding_value=0.0)
    
    # 3. We must permute the spectrograms to be (Batch_Size, Channels, Mels, Time)
    # Our model expects (N, 1, Mels, Time).
    # Librosa gives (Time, Mels). pad_sequence makes it (N, Time, Mels).
    # We swap Mels and Time, and add a channel dimension
    spectrograms_padded = spectrograms_padded.permute(0, 2, 1).unsqueeze(1)
    
    # 4. Convert labels to a tensor
    labels = torch.tensor(labels, dtype=torch.long)
    
    return spectrograms_padded, labels


# --- 4. The Main Training Script ---

if __name__ == "__main__":
    # --- Config ---
    DATA_DIR = "data/LibriSpeech/train-clean-100"
    BATCH_SIZE = 32
    NUM_EPOCHS = 40
    LEARNING_RATE = 0.001
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    dataset = LibriSpeechDataset(DATA_DIR)
    
    # The collate_fn is the magic key to handling variable-length audio
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=pad_collate_fn
    )

    # --- Model Setup ---
    model = SpeakerEncoder(num_speakers=dataset.num_speakers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train() # Set model to training mode
        running_loss = 0.0
        
        # Use tqdm for a progress bar
        for specs, labels in tqdm(data_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            # Move data to GPU
            specs = specs.to(device)
            labels = labels.to(device)
            
            # 1. Zero the gradients
            optimizer.zero_grad()
            
            # 2. Forward pass
            outputs, _ = model(specs)
            
            # 3. Calculate loss
            loss = criterion(outputs, labels)
            
            # 4. Backward pass
            loss.backward()
            
            # 5. Optimize
            optimizer.step()
            
            running_loss += loss.item()
        
        # Print epoch statistics
        epoch_loss = running_loss / len(data_loader)
        print(f"Epoch {epoch+1} finished. Average Loss: {epoch_loss:.4f}")

    # --- Save the Model ---
    model_save_path = "speaker_encoder.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Training complete. Model saved to {model_save_path}")