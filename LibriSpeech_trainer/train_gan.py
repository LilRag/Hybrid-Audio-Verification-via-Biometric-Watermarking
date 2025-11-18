import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import librosa
import numpy as np
import random
from tqdm import tqdm

# Import our three models
# Ensure gan_models.py is in the same directory
from gan_models import Generator, Extractor, Discriminator

# --- 1. Hyperparameters ---
LIBRI_DIR = "data/LibriSpeech/train-clean-100"
MUSAN_DIR = "data/musan"
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 0.0001
MESSAGE_BITS = 64
AUDIO_CLIP_LEN_SEC = 1 # We will train on 1-second clips
SAMPLE_RATE = 16000
AUDIO_CLIP_LEN_SAMPLES = SAMPLE_RATE * AUDIO_CLIP_LEN_SEC

# Loss weights
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
        exts = ("*.flac", "*.wav", "*.ogg", "*.mp3")
        files = []
        for e in exts:
            files.extend(list(self.data_dir.rglob(e)))
        self.audio_files = sorted(files)
        print(f"Found {len(self.audio_files)} files.")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        file_path = self.audio_files[idx]
        try:
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            if len(y) < self.clip_len:
                y = np.pad(y, (0, self.clip_len - len(y)), 'constant')
            start_idx = random.randint(0, len(y) - self.clip_len)
            y_clip = y[start_idx : start_idx + self.clip_len]
            # Normalize
            y_clip = y_clip / (np.max(np.abs(y_clip)) + 1e-6)
            y_tensor = torch.tensor(y_clip, dtype=torch.float32).unsqueeze(0)  # (1, L)
            return y_tensor
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

# --- 3. Noise mixing helper ---
def add_robust_noise(watermarked_batch, noise_batch=None):
    """
    Mixes noise_batch into watermarked_batch with random low volume.
    Handles mismatched batch sizes and lengths by sampling/padding/cropping.
    Inputs expected shape: (B,1,L)
    """
    target_length = watermarked_batch.size(2)
    B = watermarked_batch.size(0)
    device = watermarked_batch.device
    dtype = watermarked_batch.dtype

    if noise_batch is None:
        noise = torch.randn_like(watermarked_batch, device=device, dtype=dtype) * 0.005
    else:
        # ensure tensor on correct device/dtype
        if not isinstance(noise_batch, torch.Tensor):
            noise = torch.tensor(noise_batch, device=device, dtype=dtype)
        else:
            noise = noise_batch.to(device=device, dtype=dtype)

        # If noise batch has different batch size, sample with replacement to match B
        nB = noise.size(0)
        if nB == 0:
            # fallback to gaussian if dataset provided no samples
            noise = torch.randn_like(watermarked_batch, device=device, dtype=dtype) * 0.005
        elif nB != B:
            idx = torch.randint(0, nB, (B,), device=device)
            noise = noise[idx]

        # Match length: pad or crop
        nL = noise.size(2)
        if nL < target_length:
            pad = target_length - nL
            noise = torch.nn.functional.pad(noise, (0, pad))
        elif nL > target_length:
            noise = noise[:, :, :target_length]

    # Random per-sample mixing level in a small range
    noise_level = (torch.rand(B, 1, 1, device=device, dtype=dtype) * 0.5 * 0.05)  # e.g. up to ~0.025
    noisy_audio = (watermarked_batch * (1.0 - noise_level)) + (noise * noise_level)
    return torch.clamp(noisy_audio, -1.0, 1.0)

# --- 4. Random attack function (UPDATED) ---
def apply_random_attack(batch_tensor, sample_rate=SAMPLE_RATE, noise_batch=None):
    """
    batch_tensor: (B,1,L) torch tensor on device.
    noise_batch: optional tensor (nB,1,nL) from MUSAN. This will be sampled/padded/cropped
                 to match B and L so shapes always align.
    Returns attacked (B,1,L) on same device/dtype.
    """
    device = batch_tensor.device
    dtype = batch_tensor.dtype
    B, C, L = batch_tensor.size()
    attacked = torch.zeros_like(batch_tensor, device=device, dtype=dtype)

    # Prepare noise to have shape (B,1,L)
    if noise_batch is None:
        base_noise = torch.randn(B, 1, L, device=device, dtype=dtype) * 0.005
    else:
        # ensure tensor on correct device/dtype
        nb = noise_batch
        if not isinstance(nb, torch.Tensor):
            nb = torch.tensor(nb, device=device, dtype=dtype)
        else:
            nb = nb.to(device=device, dtype=dtype)

        nB = nb.size(0)
        nL = nb.size(2)

        # sample indices to match B
        if nB == 0:
            base_noise = torch.randn(B, 1, L, device=device, dtype=dtype) * 0.005
        else:
            if nB != B:
                idx = torch.randint(0, nB, (B,), device=device)
                nb = nb[idx]
            # match length
            if nL < L:
                pad = L - nL
                nb = torch.nn.functional.pad(nb, (0, pad))
            elif nL > L:
                nb = nb[:, :, :L]
            base_noise = nb

    # Per-sample attacks (use numpy/librosa where needed)
    for i in range(B):
        y = batch_tensor[i, 0].detach().cpu().numpy()
        attack = random.choice(["none", "noise", "resample", "clip", "quantize", "compress"])
        if attack == "none":
            y_a = y
        elif attack == "noise":
            # mix with prepared MUSAN or gaussian noise for this sample
            n = base_noise[i, 0].detach().cpu().numpy()
            scale = random.uniform(0.001, 0.03)
            y_a = y + n * scale
        elif attack == "resample":
            try:
                factor = random.uniform(0.6, 0.95)
                target_sr = max(1000, int(sample_rate * factor))
                y_ds = librosa.resample(y, orig_sr=sample_rate, target_sr=target_sr)
                y_a = librosa.resample(y_ds, orig_sr=target_sr, target_sr=sample_rate)
            except Exception:
                y_a = y
        elif attack == "compress":
            try:
                y_ds = librosa.resample(y, orig_sr=sample_rate, target_sr=8000)
                y_a = librosa.resample(y_ds, orig_sr=8000, target_sr=sample_rate)
            except Exception:
                y_a = y
        elif attack == "clip":
            clip_level = random.uniform(0.6, 1.0)
            y_a = np.clip(y, -clip_level, clip_level)
            if np.max(np.abs(y_a)) > 0:
                y_a = y_a / (np.max(np.abs(y_a)) + 1e-9)
        elif attack == "quantize":
            bits = random.choice([8, 12, 16])
            levels = 2 ** bits - 1
            y_a = np.round((y + 1.0) / 2.0 * levels) / levels * 2.0 - 1.0
        else:
            y_a = y

        # ensure length L
        if len(y_a) < L:
            y_a = np.pad(y_a, (0, L - len(y_a)), 'constant')
        elif len(y_a) > L:
            y_a = y_a[:L]
        y_a = np.nan_to_num(y_a).astype(np.float32)

        attacked[i, 0] = torch.from_numpy(y_a).to(device=device, dtype=dtype)

    # Optionally add a small portion of prepared base_noise to every attacked sample
    mix_lvl = torch.rand(B, 1, 1, device=device, dtype=dtype) * 0.01
    attacked = attacked + (base_noise * mix_lvl)
    return torch.clamp(attacked, -1.0, 1.0)

# --- 5. Main Training Script ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    # Datasets & loaders
    libri_dataset = AudioClipDataset(LIBRI_DIR, AUDIO_CLIP_LEN_SAMPLES)
    musan_dataset = AudioClipDataset(MUSAN_DIR, AUDIO_CLIP_LEN_SAMPLES)

    num_workers = 0
    libri_loader = DataLoader(libri_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    
    if len(musan_dataset) == 0:
        print("Warning: MUSAN empty — will use gaussian noise fallback.")
        musan_loader = None
    else:
        musan_loader = DataLoader(musan_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)

    # Models
    generator = Generator(message_bits=MESSAGE_BITS).to(device)
    extractor = Extractor(message_bits=MESSAGE_BITS).to(device)
    discriminator = Discriminator().to(device)

    # Losses & optimizers
    criterion_msg = nn.MSELoss().to(device)
    criterion_audio = nn.L1Loss().to(device)
    criterion_gan = nn.BCEWithLogitsLoss().to(device)

    g_optimizer = torch.optim.Adam(list(generator.parameters()) + list(extractor.parameters()), lr=LEARNING_RATE)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

    print("Starting GAN training...")
    for epoch in range(NUM_EPOCHS):
        gen_loss_total = 0.0
        disc_loss_total = 0.0

        # musan iterator for this epoch
        musan_iter = iter(musan_loader) if musan_loader is not None else None

        for clean_batch in tqdm(libri_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            clean_batch = clean_batch.to(device)  # (B,1,L)

            # get noise batch if available
            noise_batch = None
            if musan_iter is not None:
                try:
                    noise_batch = next(musan_iter).to(device)
                except StopIteration:
                    musan_iter = iter(musan_loader)
                    noise_batch = next(musan_iter).to(device)

            # --- Phase A: Train Generator + Extractor (on attacked audio) ---
            g_optimizer.zero_grad()

            # random message bits
            secret_message = torch.randint(0, 2, (clean_batch.size(0), MESSAGE_BITS), dtype=torch.float32, device=device)

            # generator creates watermarked audio (clean shape)
            watermarked_audio = generator(clean_batch, secret_message)  # expected (B,1,L')

            # ensure same length as clean for losses
            L = min(clean_batch.size(2), watermarked_audio.size(2))
            clean_crop = clean_batch[:, :, :L]
            watermarked_audio = watermarked_audio[:, :, :L]

            # 1. Add basic robustness noise (general degradation)
            robust_audio = add_robust_noise(watermarked_audio, noise_batch)

            # 2. Apply the specific Random Attack (The new function!)
            # We pass noise_batch in case the random choice selects "noise"
            attacked_audio = apply_random_attack(robust_audio, noise_batch=noise_batch)

            # extractor tries to recover from attacked audio
            recovered_message = extractor(attacked_audio)

            # discriminator output on attacked audio for generator adv loss
            d_on_attacked = discriminator(attacked_audio)

            # Message loss (force extractor to recover from attacked audio)
            loss_msg = criterion_msg(recovered_message, secret_message)

            # Audio fidelity loss (between watermarked audio and clean)
            loss_audio = criterion_audio(watermarked_audio, clean_crop)

            # Adversarial loss (generator wants discriminator to believe attacked audio is real)
            real_labels_for_g = torch.ones_like(d_on_attacked).to(device)
            loss_g_adv = criterion_gan(d_on_attacked, real_labels_for_g)

            total_loss_ge = (loss_msg * L_MSG_WEIGHT) + (loss_audio * L_AUD_WEIGHT) + (loss_g_adv * L_ADV_WEIGHT)

            total_loss_ge.backward()
            g_optimizer.step()

            # --- Phase B: Train Discriminator (on clean vs attacked audio) ---
            d_optimizer.zero_grad()

            # real = clean audio
            real_audio = clean_crop
            guess_real = discriminator(real_audio)
            real_labels = torch.ones_like(guess_real).to(device)
            loss_d_real = criterion_gan(guess_real, real_labels)

            # fake = attacked audio (detach to avoid G gradients)
            guess_fake = discriminator(attacked_audio.detach())
            fake_labels = torch.zeros_like(guess_fake).to(device)
            loss_d_fake = criterion_gan(guess_fake, fake_labels)

            total_loss_d = (loss_d_real + loss_d_fake) / 2.0
            total_loss_d.backward()
            d_optimizer.step()

            # record losses
            gen_loss_total += float(total_loss_ge.detach().cpu().item())
            disc_loss_total += float(total_loss_d.detach().cpu().item())

        # End of epoch stats
        avg_g_loss = gen_loss_total / len(libri_loader)
        avg_d_loss = disc_loss_total / len(libri_loader)
        print(f"Epoch {epoch+1} Complete. Avg G/E Loss: {avg_g_loss:.4f}, Avg D Loss: {avg_d_loss:.4f}")

    # Save models
    torch.save(generator.state_dict(), "generator.pth")
    torch.save(extractor.state_dict(), "extractor.pth")
    torch.save(discriminator.state_dict(), "discriminator.pth")
    print("Generator, Extractor, and Discriminator models saved.")