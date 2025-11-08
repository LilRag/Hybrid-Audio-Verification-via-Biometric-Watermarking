import torch
import torch.nn.functional as F
from pathlib import Path
import librosa
import numpy as np
import sys
import soundfile as sf # Need this to save the .wav file

# Import all our models
from model import SpeakerEncoder
from gan_models import Generator
from train_gan import AUDIO_CLIP_LEN_SAMPLES

# --- 1. Config ---
MODEL_ENCODER = "speaker_encoder.pth"
MODEL_GENERATOR = "generator.pth"
TRAIN_SPEAKERS_DIR = "data/LibriSpeech/train-clean-100"

MESSAGE_BITS = 64
SAMPLE_RATE = 16000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. Helper Functions (Copied from verify_audio.py) ---

def load_audio_clip(file_path, expected_len):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None
    
    if len(y) < expected_len:
        y = np.pad(y, (0, expected_len - len(y)), 'constant')
    elif len(y) > expected_len:
        y = y[:expected_len]
    
    y = y / np.max(np.abs(y) + 1e-6)
    audio_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    return audio_tensor

def get_voiceprint_hash(audio_waveform, speaker_encoder):
    y = audio_waveform.squeeze().cpu().numpy()
    S = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=128, n_fft=2048, hop_length=512)
    S_db = librosa.power_to_db(S, ref=np.max)
    spec_tensor = torch.tensor(S_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    _, voiceprint = speaker_encoder(spec_tensor)
    voiceprint_subset = voiceprint[0, :MESSAGE_BITS]
    secret_hash = (voiceprint_subset > 0).float().unsqueeze(0)
    return secret_hash

# --- 3. Main Sealing Logic ---

def main(file_to_seal, output_file):
    print(f"--- Sealing Audio File: {file_to_seal} ---")
    
    # --- Load Models ---
    print("Loading models...")
    try:
        # Load Speaker Encoder
        train_speakers = set(f.parent.parent.name for f in Path(TRAIN_SPEAKERS_DIR).rglob("*.flac"))
        num_speakers_trained = len(train_speakers)
        speaker_encoder = SpeakerEncoder(num_speakers=num_speakers_trained).to(device)
        speaker_encoder.load_state_dict(torch.load(MODEL_ENCODER, map_location=device, weights_only=True))
        speaker_encoder.eval()

        # Load Generator
        generator = Generator(MESSAGE_BITS).to(device)
        generator.load_state_dict(torch.load(MODEL_GENERATOR, map_location=device, weights_only=True))
        generator.eval()
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Could not find model file: {e.filename}")
        return
    print("All models loaded.")

    # --- Load Audio & Get Target Length ---
    with torch.no_grad():
        dummy_gen = Generator(MESSAGE_BITS).to(device)
        dummy_audio = torch.zeros(1, 1, AUDIO_CLIP_LEN_SAMPLES).to(device)
        dummy_msg = torch.zeros(1, MESSAGE_BITS).to(device)
        target_len = dummy_gen(dummy_audio, dummy_msg).size(2)
        del dummy_gen, dummy_audio, dummy_msg

    audio_clip = load_audio_clip(file_to_seal, target_len)
    if audio_clip is None: return

    # --- Run Enrollment ---
    with torch.no_grad():
        # Step 1: Get the voiceprint hash
        secret_message = get_voiceprint_hash(audio_clip, speaker_encoder)
        print("Step 1: Voiceprint hash generated.")

        # Step 2: Embed the watermark
        watermarked_audio = generator(audio_clip, secret_message)
        print("Step 2: Watermark embedded.")
        
    # --- Save the Sealed File ---
    # We must save the audio in a format that verify_audio can read.
    # .wav is perfect.
    
    # Squeeze to 1D array and move to CPU
    watermarked_audio_np = watermarked_audio.squeeze().cpu().numpy()
    sf.write(output_file, watermarked_audio_np, SAMPLE_RATE)
    
    print(f"\n--- SUCCESS ---")
    print(f"Sealed file saved to: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        file_path_in = sys.argv[1]
        file_path_out = sys.argv[2]
    else:
        file_path_in = input("Enter path to the CLEAN audio file to seal: ")
        file_path_out = input("Enter path to save the NEW SEALED file (e.g., sealed_audio.wav): ")
    
    main(file_path_in, file_path_out)