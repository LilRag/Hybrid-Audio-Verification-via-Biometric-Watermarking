import os
import sys
import torch
import librosa
import soundfile as sf
import numpy as np

from model import SpeakerEncoder
from gan_models import Extractor, Discriminator

# --- CONFIG ---
MODEL_ENCODER = "speaker_encoder.pth"
MODEL_DISCRIMINATOR = "discriminator.pth"
# Priority load fixed models
MODEL_EXTRACTOR = "extractor_fixed.pth" if os.path.exists("extractor_fixed.pth") else "extractor.pth"

MESSAGE_BITS = 64
SAMPLE_RATE = 16000
BIT_ERROR_THRESHOLD = 0.20
EMBED_SIM_THRESHOLD = 0.70
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_audio_clip(file_path, target_len=None):
    try:
        y, sr = sf.read(file_path)
    except Exception as e:
        return None
    if y.ndim > 1: y = np.mean(y, axis=1)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
    y = y.astype(np.float32)
    if np.max(np.abs(y)) > 0: y = y / (np.max(np.abs(y)) + 1e-9)
    if target_len is not None:
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), "constant")
        elif len(y) > target_len:
            y = y[:target_len]
    return torch.tensor(y, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

def extract_voiceprint_vec(audio_tensor, speaker_encoder):
    y = audio_tensor.squeeze().cpu().numpy()
    if len(y) < 512: y = np.pad(y, (0, 512 - len(y)))
    S = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=128, n_fft=2048, hop_length=512)
    S_db = librosa.power_to_db(S, ref=np.max)
    spec_tensor = torch.tensor(S_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        if hasattr(speaker_encoder, 'get_embedding'): out = speaker_encoder.get_embedding(spec_tensor)
        else: out = speaker_encoder(spec_tensor)
        
        if isinstance(out, (tuple, list)):
            found = None
            for item in out:
                if item.numel() == 256:
                    found = item
                    break
            out = found if found is not None else out[0]

        return out.squeeze().cpu().numpy()

def get_watermark_bits(audio_tensor, extractor):
    # --- REALISM FIX: Add slight noise to simulate real-world transmission ---
    # This ensures we aren't just reading memorized values 
    noise_level = 0.001 # Small amount of static
    audio_noisy = audio_tensor + (torch.randn_like(audio_tensor) * noise_level)
    
    with torch.no_grad(): 
        out = extractor(audio_noisy)
        
    probs = torch.sigmoid(out) if (out.min() < 0 or out.max() > 1) else out
    return (probs > 0.5).int().cpu().squeeze().numpy()

def compute_ber(gt, pred):
    gt = gt.flatten(); pred = pred.flatten()
    n = min(len(gt), len(pred))
    errors = np.sum(gt[:n] != pred[:n])
    return float(errors / n)

def cos_sim(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    if a.shape != b.shape: return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def verify_file(file_path):
    print(f"\n--- VERIFYING: {file_path} ---")
    try:
        ckpt = torch.load(MODEL_ENCODER, map_location=device)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        num = int(state["fc2.weight"].shape[0]) if "fc2.weight" in state else 251
        encoder = SpeakerEncoder(num_speakers=num).to(device)
        try: encoder.load_state_dict(state)
        except: encoder.load_state_dict(state, strict=False)
        encoder.eval()

        extractor = Extractor(MESSAGE_BITS).to(device)
        extractor.load_state_dict(torch.load(MODEL_EXTRACTOR, map_location=device))
        extractor.eval()

        discriminator = Discriminator().to(device)
        discriminator.load_state_dict(torch.load(MODEL_DISCRIMINATOR, map_location=device))
        discriminator.eval()
    except Exception as e:
        print(f"Model Error: {e}"); return

    # Verify
    audio = load_audio_clip(file_path, 16000)
    
    # Step 1: Discriminator
    disc = discriminator(audio).item()
    print(f"Discriminator: {disc:.4f}")
    
    base = str(file_path)
    if not os.path.exists(base + ".payload.npy"): 
        print("Error: No metadata found.")
        return
    
    gt_bits = np.load(base + ".payload.npy").astype(int)
    
    # Step 2: Extract bits (with noise added internally)
    pred_bits = get_watermark_bits(audio, extractor)
    ber = compute_ber(gt_bits, pred_bits)
    
    # Phase check
    ber_inv = compute_ber(gt_bits, 1-pred_bits)
    if ber_inv < ber: 
        ber = ber_inv
        # pred_bits = 1 - pred_bits # optional visualize

    # Step 3: Voiceprint
    sim = 0.0
    if os.path.exists(base + ".voiceprint_emb.npy"):
        saved = np.load(base + ".voiceprint_emb.npy")
        curr = extract_voiceprint_vec(audio, encoder)
        sim = cos_sim(saved, curr)

    # --- Debug Prints ---
    print(f"Expected Bits (First 10): {gt_bits[:10]}")
    print(f"Recovered Bits (First 10): {pred_bits[:10]}")
    print("-" * 20)
    print(f"BER: {ber:.4f} (Threshold < {BIT_ERROR_THRESHOLD})")
    print(f"Sim: {sim:.4f} (Threshold > {EMBED_SIM_THRESHOLD})")
    print("-" * 20)

    if ber <= BIT_ERROR_THRESHOLD and sim >= EMBED_SIM_THRESHOLD:
        print("✅ VERDICT: [ AUTHENTIC ]")
    elif ber <= BIT_ERROR_THRESHOLD:
        print("⚠️ VERDICT: [ SUSPICIOUS - VOICE MISMATCH ]")
    elif sim >= EMBED_SIM_THRESHOLD:
        print("⚠️ VERDICT: [ CORRUPTED WATERMARK ]")
    else:
        print("❌ VERDICT: [ FAKE ]")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_audio.py <file>")
    else:
        verify_file(sys.argv[1])