import torch
import torch.nn.functional as F
from pathlib import Path
import librosa
import numpy as np
import random
import hashlib

# Import all our models
from model import SpeakerEncoder
from gan_models import Generator, Extractor
from train import audio_to_spectrogram # For the Encoder
from train_gan import AudioClipDataset # For the GAN models

# --- 1. Config ---
MODEL_ENCODER = "speaker_encoder.pth"
MODEL_GENERATOR = "generator.pth"
MODEL_EXTRACTOR = "extractor.pth"

# Use the 'test-clean' set to find a file
TEST_AUDIO_DIR = "data/LibriSpeech/test-clean"
TRAIN_AUDIO_DIR = "data/LibriSpeech/train-clean-100" # Need this to get num_speakers

MESSAGE_BITS = 64
AUDIO_CLIP_LEN_SEC = 1
SAMPLE_RATE = 16000
AUDIO_CLIP_LEN_SAMPLES = SAMPLE_RATE * AUDIO_CLIP_LEN_SEC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. Load All Three Models ---

print("Loading models...")

# Load Speaker Encoder (Phase 1)
# We need to know the num_speakers it was trained on
train_speakers = set(f.parent.parent.name for f in Path(TRAIN_AUDIO_DIR).rglob("*.flac"))
num_speakers_trained = len(train_speakers)
speaker_encoder = SpeakerEncoder(num_speakers=num_speakers_trained).to(device)
speaker_encoder.load_state_dict(torch.load(MODEL_ENCODER, map_location=device, weights_only = True))
speaker_encoder.eval() # Set to evaluation mode

# Load Generator (Phase 2)
generator = Generator(MESSAGE_BITS).to(device)
generator.load_state_dict(torch.load(MODEL_GENERATOR, map_location=device,weights_only = True))
generator.eval()

# Load Extractor (Phase 2)
extractor = Extractor(MESSAGE_BITS).to(device)
extractor.load_state_dict(torch.load(MODEL_EXTRACTOR, map_location=device, weights_only = True))
extractor.eval()

print("All models loaded.")

# --- 3. Get a Sample Audio File ---

# We'll use the same dataset class from train_gan.py to get a 1-sec clip
# This ensures the audio is in the correct format (1-sec, normalized)
test_dataset = AudioClipDataset(TEST_AUDIO_DIR, AUDIO_CLIP_LEN_SAMPLES)
audio_clip_waveform = test_dataset[0].unsqueeze(0).to(device) # Get first sample, add batch dim

# Check the audio shape
print(f"Loaded audio clip with shape: {audio_clip_waveform.shape}")

# --- 4. Run the Full End-to-End System ---

with torch.no_grad(): # Disable gradients for all operations
    
    # === STEP 1: Get the Voiceprint (Phase 1) ===
    # The encoder needs a spectrogram
    
    # We can't use the audio_to_spectrogram function because it loads from a file.
    # We'll compute the spectrogram from the waveform tensor we already have.
    y = audio_clip_waveform.squeeze(0).cpu().numpy() # [1, 16000] -> [16000]
    S = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=128, n_fft=2048, hop_length=512)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Prep spec for encoder: (Mels, Time) -> (N, 1, Mels, Time)
    spec_tensor = torch.tensor(S_db, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Run the encoder
    _, voiceprint = speaker_encoder(spec_tensor) # voiceprint shape: [1, 256]
    print("Step 1: Voiceprint generated.")
    
    # === STEP 2: Create the Secret Message ===
    # We need to convert the 256-dim voiceprint into our 64-bit message.
    # We'll use a simple but effective "hash": binarize the first 64 values.
    
    voiceprint_subset = voiceprint[0, :MESSAGE_BITS] # Take first 64 values
    secret_message = (voiceprint_subset > 0).float().unsqueeze(0) # [1, 64]
    
    print(f"Step 2: Created 64-bit secret message from voiceprint.")
    # print(f"  Secret Message: {secret_message.cpu().numpy().astype(int)}")

    # === STEP 3: Embed the Watermark (Phase 2) ===
    # The generator needs the raw waveform
    # We need to crop the audio to match the generator's output size (15872)
    # to avoid the error we saw during training.
    
    # Get the expected output size by doing a "dry run"
    dummy_msg = torch.zeros_like(secret_message)
    dummy_audio = torch.zeros_like(audio_clip_waveform)
    target_len = generator(dummy_audio, dummy_msg).size(2) # e.g., 15872
    
    audio_clip_cropped = audio_clip_waveform[:, :, :target_len]
    
    # Now, run the actual generator
    watermarked_audio = generator(audio_clip_cropped, secret_message)
    print(f"Step 3: Watermark embedded. Output audio shape: {watermarked_audio.shape}")

    # === STEP 4: Recover the Message (Phase 2) ===
    # The extractor takes the watermarked audio
    recovered_message_raw = extractor(watermarked_audio)
    
    # Binarize the output to get the message bits
    recovered_message = (recovered_message_raw > 0.5).float() # [1, 64]
    print("Step 4: Message extracted from watermarked audio.")
    # print(f"  Recovered Msg: {recovered_message.cpu().numpy().astype(int)}")

    # === STEP 5: Final Verification ===
    
    # Calculate the Bit Error Rate (BER)
    # This is the percentage of bits that are wrong.
    bit_errors = torch.sum(torch.abs(secret_message - recovered_message))
    ber = (bit_errors / MESSAGE_BITS) * 100
    
    print("\n--- FINAL VERIFICATION ---")
    print(f"Original Message Hash:  {secret_message.cpu().numpy().astype(int)}")
    print(f"Recovered Message Hash: {recovered_message.cpu().numpy().astype(int)}")
    print(f"Bit Errors: {bit_errors.item()} / {MESSAGE_BITS}")
    print(f"Bit Error Rate (BER): {ber.item():.2f}%")
    
    if ber.item() == 0.0:
        print("\nSUCCESS: The system is working! The voiceprint was embedded and recovered perfectly.")
    else:
        print(f"\nNOTE: The system is partially working, but has {ber.item():.2f}% error.")