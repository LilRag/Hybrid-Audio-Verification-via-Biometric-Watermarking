import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import librosa
import numpy as np
import random
from tqdm import tqdm

# Import our three models
from gan_models import Generator, Extractor, Discriminator

# --- 1. Hyperparameters ---
LIBRI_DIR = "data/LibriSpeech/train-clean-100"
MUSAN_DIR = "data/musan"
BATCH_SIZE = 32
NUM_EPOCHS = 40
LEARNING_RATE = 0.0001
MESSAGE_BITS = 64
AUDIO_CLIP_LEN_SEC = 1 # We will train on 1-second clips
SAMPLE_RATE = 16000
AUDIO_CLIP_LEN_SAMPLES = SAMPLE_RATE * AUDIO_CLIP_LEN_SEC

# Loss weights (This is the most important part to tune!)
# We want to force message recovery to be the most important.
L_MSG_WEIGHT = 10.0 # Message Recovery (Extractor)
L_AUD_WEIGHT = 1.0  # Audio Quality (Generator)
L_ADV_WEIGHT = 0.1  # Fooling the Adversary (Generator)

# --- 2. The Audio Datasets ---

class AudioClipDataset(Dataset):
    """
    Dataset that loads audio files and provides random 1-second clips.
    """
    def __init__(self, data_dir, clip_len_samples):
        self.data_dir = Path(data_dir)
        self.clip_len = clip_len_samples
        print(f"Scanning {data_dir}...")
        self.audio_files = list(self.data_dir.rglob("*.flac"))
        if not self.audio_files:
             # Add .wav and .ogg for MUSAN
             self.audio_files = list(self.data_dir.rglob("*.wav")) + list(self.data_dir.rglob("*.ogg"))
        print(f"Found {len(self.audio_files)} files.")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        file_path = self.audio_files[idx]
        
        try:
            # Load the full audio file
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            
            # Pad if the audio is too short
            if len(y) < self.clip_len:
                y = np.pad(y, (0, self.clip_len - len(y)), 'constant')
            
            # Take a random 1-second clip
            start_idx = random.randint(0, len(y) - self.clip_len)
            y_clip = y[start_idx : start_idx + self.clip_len]
            
            # Normalize to [-1, 1] for tanh output
            y_clip = y_clip / np.max(np.abs(y_clip) + 1e-6)
            
            # Add channel dimension
            y_tensor = torch.tensor(y_clip, dtype=torch.float32).unsqueeze(0)
            
            return y_tensor

        except Exception as e:
            # print(f"Warning: Skipping broken file {file_path}: {e}")
            # If a file is broken, just get the next one
            return self.__getitem__((idx + 1) % len(self))

# --- 3. The Noise Helper Function ---
def add_robust_noise(watermarked_batch, noise_batch):
    """
    Adds noise from the MUSAN batch to the watermarked batch at a
    random low volume. This forces the Extractor to be robust.
    """
    # --- START OF FIX ---
    # Get the length of the watermarked audio (e.g., 15872)
    target_length = watermarked_batch.size(2)
    
    # Crop the noise batch to match the watermarked audio's length
    noise_batch_cropped = noise_batch[:, :, :target_length]
    # --- END OF FIX ---
    
    # Pick a random noise level (e.g., 0% to 50% noise volume)
    noise_level = torch.rand(watermarked_batch.size(0), 1, 1, device=watermarked_batch.device) * 0.5
    
    # Mix them (now the shapes match)
    noisy_audio = (watermarked_batch * (1.0 - noise_level)) + (noise_batch_cropped * noise_level)
    
    # Clamp to ensure audio is still in [-1, 1] range
    noisy_audio = torch.clamp(noisy_audio, -1.0, 1.0)
    
    return noisy_audio


# --- 4. Main Training Script ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Dataloaders ---
    libri_dataset = AudioClipDataset(LIBRI_DIR, AUDIO_CLIP_LEN_SAMPLES)
    musan_dataset = AudioClipDataset(MUSAN_DIR, AUDIO_CLIP_LEN_SAMPLES)
    
    libri_loader = DataLoader(libri_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    musan_loader = DataLoader(musan_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # --- Models ---
    generator = Generator(MESSAGE_BITS).to(device)
    extractor = Extractor(MESSAGE_BITS).to(device)
    discriminator = Discriminator().to(device)

    # --- Losses ---
    criterion_msg = nn.MSELoss().to(device)     # For message recovery
    criterion_audio = nn.L1Loss().to(device)   # For audio fidelity
    criterion_gan = nn.BCEWithLogitsLoss().to(device) # For adversary

    # --- Optimizers (One for G+E, one for D) ---
    optimizer_ge = optim.Adam(list(generator.parameters()) + list(extractor.parameters()), lr=LEARNING_RATE)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

    print("Starting GAN training...")
    for epoch in range(NUM_EPOCHS):
        # We need to re-create the musan iterator every epoch
        musan_iter = iter(musan_loader)
        
        # We will track the losses
        gen_loss_total = 0.0
        disc_loss_total = 0.0
        
        for clean_batch in tqdm(libri_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            try:
                # Get a matching batch of noise
                noise_batch = next(musan_iter)
            except StopIteration:
                # Reset iterator if MUSAN runs out
                musan_iter = iter(musan_loader)
                noise_batch = next(musan_iter)


            # Get the batch size of our (possibly partial) clean audio batch
            current_batch_size = clean_batch.size(0)
            
            # Crop the noise batch to match and move both to the device
            noise_batch = noise_batch[:current_batch_size, :, :].to(device)
            clean_batch = clean_batch.to(device)
            
            # --- Phase A: Train Generator & Extractor ---
            optimizer_ge.zero_grad()
            
            # 1. Create a random secret message
            # (Using 0s and 1s is fine for MSELoss)
            secret_message = torch.randint(0, 2, (clean_batch.size(0), MESSAGE_BITS), dtype=torch.float32).to(device)

            # 2. Generator: Create watermarked audio
            watermarked_audio = generator(clean_batch, secret_message)
            
            # 3. Add robustness noise from MUSAN
            noisy_watermarked_audio = add_robust_noise(watermarked_audio, noise_batch)
            
            # 4. Extractor: Recover the message
            recovered_message = extractor(noisy_watermarked_audio)
            
            # 5. Discriminator: Guess if the watermarked audio is fake
            discriminator_guess_fake = discriminator(watermarked_audio)
            
            # --- Calculate all Generator/Extractor Losses ---
            
            # Loss 1: Message Loss (Did we recover the message?)
            loss_e = criterion_msg(recovered_message, secret_message)
            
            # Loss 2: Audio Loss (Does it sound the same?)
            clean_batch_cropped = clean_batch[:, :, :watermarked_audio.size(2)]
            loss_g_audio = criterion_audio(watermarked_audio, clean_batch_cropped)
            
            # Loss 3: Adversarial Loss (Did we fool the discriminator?)
            real_labels = torch.ones_like(discriminator_guess_fake).to(device)
            loss_g_adv = criterion_gan(discriminator_guess_fake, real_labels)
            
            # --- Combine G/E losses with our weights ---
            total_loss_ge = (loss_e * L_MSG_WEIGHT) + \
                              (loss_g_audio * L_AUD_WEIGHT) + \
                              (loss_g_adv * L_ADV_WEIGHT)
            
            # Backpropagate G/E
            total_loss_ge.backward()
            optimizer_ge.step()
            
            # --- Phase B: Train Discriminator ---
            optimizer_d.zero_grad()
            
            # 1. Real Loss (on the clean, original audio)
            guess_real = discriminator(clean_batch)
            real_labels = torch.ones_like(guess_real).to(device)
            loss_d_real = criterion_gan(guess_real, real_labels)
            
            # 2. Fake Loss (on the watermarked audio)
            # We .detach() the audio so gradients don't flow back to the Generator
            guess_fake = discriminator(watermarked_audio.detach())
            fake_labels = torch.zeros_like(guess_fake).to(device)
            loss_d_fake = criterion_gan(guess_fake, fake_labels)
            
            # Combine D losses
            total_loss_d = (loss_d_real + loss_d_fake) / 2
            
            # Backpropagate D
            total_loss_d.backward()
            optimizer_d.step()
            
            # --- Record losses for the epoch ---
            gen_loss_total += total_loss_ge.item()
            disc_loss_total += total_loss_d.item()

        # --- End of Epoch ---
        avg_g_loss = gen_loss_total / len(libri_loader)
        avg_d_loss = disc_loss_total / len(libri_loader)
        
        print(f"Epoch {epoch+1} Complete.")
        print(f"  Avg. Generator/Extractor Loss: {avg_g_loss:.4f}")
        print(f"  Avg. Discriminator Loss: {avg_d_loss:.4f}")

    # --- End of Training ---
    print("GAN training complete.")
    # Save the two models we need for the final system
    torch.save(generator.state_dict(), "generator.pth")
    torch.save(extractor.state_dict(), "extractor.pth")
    torch.save(discriminator.state_dict(), "discriminator.pth")
    print("Generator, Extractor, and Discriminator models saved.")