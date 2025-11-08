import torch
import torch.nn.functional as F
from pathlib import Path
import librosa
import numpy as np
import sys

# Import all our models
from model import SpeakerEncoder
from gan_models import Generator, Extractor, Discriminator
from train_gan import AUDIO_CLIP_LEN_SAMPLES

# --- 1. Config ---
MODEL_ENCODER = "speaker_encoder.pth"
MODEL_EXTRACTOR = "extractor.pth"
MODEL_DISCRIMINATOR = "discriminator.pth"
TRAIN_SPEAKERS_DIR = "data/LibriSpeech/train-clean-100" # Dir for speaker count

MESSAGE_BITS = 64
SAMPLE_RATE = 16000

# This is our "pass/fail" threshold.
# 1.56% BER (from your test) is 1 bit. We'll allow up to 2 bit errors.
BIT_ERROR_THRESHOLD = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. Helper Functions ---

def load_audio_clip(file_path, expected_len):
    """
    Loads an audio file and ensures it's the correct length.
    """
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None

    # Pad or crop to the exact length our GAN models expect
    if len(y) < expected_len:
        y = np.pad(y, (0, expected_len - len(y)), 'constant')
    elif len(y) > expected_len:
        y = y[:expected_len]
    
    # Normalize and add batch/channel dimensions: [1, 1, 16000]
    y = y / np.max(np.abs(y) + 1e-6)
    audio_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    return audio_tensor

def get_voiceprint_hash(audio_waveform, speaker_encoder):
    """
    Generates the 64-bit hash from an audio waveform
    using the Speaker Encoder.
    """
    # 1. Convert waveform to spectrogram
    # Squeeze to [1, 16000] -> [16000] for librosa
    y = audio_waveform.squeeze().cpu().numpy()
    S = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=128, n_fft=2048, hop_length=512)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # 2. Prep spec for encoder: (Mels, Time) -> (N, C, Mels, Time)
    spec_tensor = torch.tensor(S_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # 3. Run the encoder
    _, voiceprint = speaker_encoder(spec_tensor) # voiceprint shape: [1, 256]
    
    # 4. Create the hash (same logic as integrate.py)
    voiceprint_subset = voiceprint[0, :MESSAGE_BITS]
    secret_hash = (voiceprint_subset > 0).float().unsqueeze(0) # [1, 64]
    
    return secret_hash

def get_watermark_hash(audio_waveform, extractor):
    """
    Recovers the 64-bit hash from an audio waveform
    using the Extractor.
    """
    recovered_message_raw = extractor(audio_waveform)
    recovered_hash = (recovered_message_raw > 0.5).float() # [1, 64]
    return recovered_hash

# --- 3. Main Verification Logic ---

def main(file_to_verify):
    print(f"--- Verifying Audio File: {file_to_verify} ---")
    
    # --- Load Models ---
    print("Loading models...")
    try:
        # Load Speaker Encoder
        train_speakers = set(f.parent.parent.name for f in Path(TRAIN_SPEAKERS_DIR).rglob("*.flac"))
        num_speakers_trained = len(train_speakers)
        speaker_encoder = SpeakerEncoder(num_speakers=num_speakers_trained).to(device)
        speaker_encoder.load_state_dict(torch.load(MODEL_ENCODER, map_location=device, weights_only=True))
        speaker_encoder.eval()

        # Load Extractor
        extractor = Extractor(MESSAGE_BITS).to(device)
        extractor.load_state_dict(torch.load(MODEL_EXTRACTOR, map_location=device, weights_only=True))
        extractor.eval()

        # Load Discriminator
        discriminator = Discriminator().to(device)
        discriminator.load_state_dict(torch.load(MODEL_DISCRIMINATOR, map_location=device, weights_only=True))
        discriminator.eval()
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: Could not find model file: {e.filename}")
        print("Please ensure all .pth files are in the same directory.")
        return
    print("All models loaded.")

    # --- Load Audio ---
    # We need to know the *exact* length the Generator was trained on.
    # We'll run a dummy input to get the length (e.g., 15872)
    with torch.no_grad():
        dummy_gen = Generator(MESSAGE_BITS).to(device)
        dummy_audio = torch.zeros(1, 1, AUDIO_CLIP_LEN_SAMPLES).to(device)
        dummy_msg = torch.zeros(1, MESSAGE_BITS).to(device)
        target_len = dummy_gen(dummy_audio, dummy_msg).size(2)
        del dummy_gen, dummy_audio, dummy_msg # Free memory

    print(f"Models expect audio length of {target_len} samples.")
    audio_clip = load_audio_clip(file_to_verify, target_len)
    
    if audio_clip is None:
        return # Error already printed in load_audio_clip

    # --- Run Verification ---
    with torch.no_grad():
        
        # === Step 1: Check for Watermark (Factor 2) ===
        # We use the Discriminator. It was trained to output
        # a LOW score for "real" (unwatermarked)
        # a HIGH score for "fake" (watermarked)
        
        discriminator_score = discriminator(audio_clip).item()
        
        # We'll set our threshold at 0.0
        if discriminator_score > 0.0:
            print("\n--- FINAL VERDICT ---")
            print("🔴 UNVERIFIED (Untrusted Source)")
            print(f"(Reason: Discriminator score {discriminator_score:.2f} is high. File appears to be an unwatermarked original.)")
            return

        # If the score is LOW (like your -6.64), it's watermarked. We can proceed.
        print(f"Discriminator score: {discriminator_score:.2f}. File appears watermarked. Proceeding...")

        # === Step 2: Verify Voiceprint (Factor 1) ===
        
        # Get the "original" hash hidden in the watermark
        hash_original = get_watermark_hash(audio_clip, extractor)
        
        # Get the "current" hash by analyzing the audio
        hash_current = get_voiceprint_hash(audio_clip, speaker_encoder)

        # === Step 3: Compare Hashes ===
        bit_errors = torch.sum(torch.abs(hash_original - hash_current)).item()
        
        print(f"\nOriginal Hash:  {hash_original.cpu().numpy().astype(int)}")
        print(f"Current Hash: {hash_current.cpu().numpy().astype(int)}")
        print(f"Bit Errors: {bit_errors} / {MESSAGE_BITS}")
        
        if bit_errors <= BIT_ERROR_THRESHOLD:
            print("\n--- FINAL VERDICT ---")
            print("✅ VERIFIED")
            print("(Reason: Watermark detected and voiceprint hash matches.)")
        else:
            print("\n--- FINAL VERDICT ---")
            print("❌ TAMPERED (Voice Altered / Deepfake)")
            print(f"(Reason: Watermark detected, but voiceprint hash does NOT match. {bit_errors} bits are different.)")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Enter the path to the audio file you want to verify: ")
    
    main(file_path)