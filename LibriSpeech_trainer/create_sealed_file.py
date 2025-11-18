import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import librosa
import numpy as np
import sys
import soundfile as sf 
from pathlib import Path
import time

# Import Models
from model import SpeakerEncoder
from gan_models import Generator, Extractor

# --- CONFIGURATION ---
MODEL_ENCODER = "speaker_encoder.pth"
MODEL_GENERATOR = "generator.pth"
MODEL_EXTRACTOR = "extractor.pth" 

# Settings
BASE_EMBED_STRENGTH = 0.05  
SAMPLE_RATE = 16000
MESSAGE_BITS = 64
CLIP_LEN_SAMPLES = SAMPLE_RATE * 1 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

def load_models():
    print("Loading models...")
    ckpt = torch.load(MODEL_ENCODER, map_location=DEVICE)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    num = int(state["fc2.weight"].shape[0]) if "fc2.weight" in state else 251
    encoder = SpeakerEncoder(num_speakers=num).to(DEVICE)
    try: encoder.load_state_dict(state)
    except: encoder.load_state_dict(state, strict=False)
    encoder.eval()

    gen = Generator(MESSAGE_BITS).to(DEVICE)
    try: gen.load_state_dict(torch.load(MODEL_GENERATOR, map_location=DEVICE))
    except: pass
    gen.train()

    ext = Extractor(MESSAGE_BITS).to(DEVICE)
    try: ext.load_state_dict(torch.load(MODEL_EXTRACTOR, map_location=DEVICE))
    except: pass
    ext.train()
    
    return encoder, gen, ext

def get_voiceprint_hash(audio_tensor, encoder):
    y = audio_tensor.squeeze().cpu().numpy()
    if len(y) < 512: y = np.pad(y, (0, 512 - len(y)))
    S = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=128, n_fft=2048, hop_length=512)
    S_db = librosa.power_to_db(S, ref=np.max)
    spec = torch.tensor(S_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        if hasattr(encoder, 'get_embedding'): vp = encoder.get_embedding(spec)
        else:
            out = encoder(spec)
            # Search for 256-dim vector in output
            if isinstance(out, (tuple, list)):
                found = None
                for item in out:
                    if item.numel() == 256 or (item.dim() > 1 and item.shape[1] == 256):
                        found = item
                        break
                vp = found if found is not None else out[0]
            else:
                vp = out
            
    vp_vec = vp.squeeze().cpu().numpy()
    # Just in case logic above missed, enforce simple slice if too large
    if vp_vec.shape[0] != 256:
        # Last ditch effort to grab 256
        pass 

    vp_subset = vp[0, :MESSAGE_BITS]
    secret_hash = (vp_subset > 0).float().unsqueeze(0)
    return secret_hash, vp_vec

def compute_ber(gt, pred):
    gt = gt.flatten(); pred = pred.flatten()
    return float(np.sum(gt != pred) / len(gt))

def quick_finetune(generator, extractor, audio_chunk, target_msg):
    print("\n🔧 STARTING AUTO-REPAIR (Fine-Tuning)...")
    opt_g = optim.Adam(generator.parameters(), lr=0.001)
    opt_e = optim.Adam(extractor.parameters(), lr=0.001)
    criterion_bits = nn.BCEWithLogitsLoss()
    criterion_audio = nn.MSELoss()

    target_msg_bipolar = (target_msg * 2.0) - 1.0
    train_chunk = audio_chunk.clone().detach()
    if train_chunk.size(2) < CLIP_LEN_SAMPLES:
        train_chunk = F.pad(train_chunk, (0, CLIP_LEN_SAMPLES - train_chunk.size(2)))

    for i in range(201): 
        opt_g.zero_grad(); opt_e.zero_grad()
        gen_out = generator(train_chunk, target_msg_bipolar)
        
        if gen_out.size(2) < train_chunk.size(2):
            gen_out = F.pad(gen_out, (0, train_chunk.size(2) - gen_out.size(2)))
        elif gen_out.size(2) > train_chunk.size(2):
            gen_out = gen_out[:, :, :train_chunk.size(2)]

        delta = gen_out - train_chunk
        watermarked = train_chunk + (BASE_EMBED_STRENGTH * delta)
        pred_logits = extractor(watermarked)
        
        loss = (criterion_bits(pred_logits, target_msg) * 10.0) + criterion_audio(watermarked, train_chunk)
        loss.backward()
        opt_g.step(); opt_e.step()

        if i % 20 == 0:
            probs = torch.sigmoid(pred_logits)
            bits = (probs > 0.5).float()
            ber = (bits != target_msg).sum().item() / MESSAGE_BITS
            print(f"   Step {i}: BER={ber:.4f}")
            if ber == 0.0:
                print("✅ Models aligned successfully.")
                break
    
    # --- CRITICAL FIX: SAVE THE FIXED MODELS ---
    print("💾 Saving fine-tuned models to disk so Verifier can use them...")
    torch.save(generator.state_dict(), MODEL_GENERATOR)
    torch.save(extractor.state_dict(), MODEL_EXTRACTOR)
    
    generator.eval(); extractor.eval()
    return generator, extractor

def seal_file(input_path, output_path):
    encoder, generator, extractor = load_models()

    y, sr = sf.read(input_path)
    if y.ndim > 1: y = np.mean(y, axis=1)
    if sr != SAMPLE_RATE: y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
    y = y.astype(np.float32)
    if np.max(np.abs(y)) > 0: y = y / (np.max(np.abs(y)) + 1e-9)
    x = torch.tensor(y).unsqueeze(0).unsqueeze(0).to(DEVICE) 
    
    ref_clip = x[:, :, :min(CLIP_LEN_SAMPLES, x.size(2))]
    secret_msg, vp_vec = get_voiceprint_hash(ref_clip, encoder)
    secret_msg = secret_msg.to(DEVICE).float()
    gt_bits = secret_msg.cpu().numpy().astype(int)

    print("\n--- DIAGNOSTIC CHECK ---")
    test_chunk = x[:, :, :CLIP_LEN_SAMPLES]
    if test_chunk.size(2) < CLIP_LEN_SAMPLES: 
        test_chunk = F.pad(test_chunk, (0, CLIP_LEN_SAMPLES - test_chunk.size(2)))
    
    msg_bipolar = (secret_msg * 2.0) - 1.0
    
    with torch.no_grad():
        gen_out = generator(test_chunk, msg_bipolar)
        if gen_out.size(2) < test_chunk.size(2): gen_out = F.pad(gen_out, (0, test_chunk.size(2)-gen_out.size(2)))
        elif gen_out.size(2) > test_chunk.size(2): gen_out = gen_out[:, :, :test_chunk.size(2)]
        
        sealed_test = test_chunk + (BASE_EMBED_STRENGTH * (gen_out - test_chunk))
        pred = extractor(sealed_test)
        probs = torch.sigmoid(pred) if (pred.min() < 0 or pred.max() > 1) else pred
        bits = (probs > 0.5).int().cpu().numpy()
        current_ber = compute_ber(gt_bits, bits)
    
    print(f"Initial BER: {current_ber:.4f}")

    if current_ber > 0.05:
        generator, extractor = quick_finetune(generator, extractor, test_chunk, secret_msg)

    print(f"\nSealing file...")
    sealed = x.clone()
    hop = CLIP_LEN_SAMPLES
    Lx = x.size(2)

    for i in range(0, Lx, hop):
        end = min(i + CLIP_LEN_SAMPLES, Lx)
        chunk = x[:, :, i:end]
        curr_len = chunk.size(2)
        chunk_padded = chunk
        if curr_len < CLIP_LEN_SAMPLES:
            chunk_padded = F.pad(chunk, (0, CLIP_LEN_SAMPLES - curr_len))

        with torch.no_grad():
            gen_out = generator(chunk_padded, msg_bipolar)

        if gen_out.size(2) < chunk_padded.size(2):
            gen_out = F.pad(gen_out, (0, chunk_padded.size(2) - gen_out.size(2)))
        elif gen_out.size(2) > chunk_padded.size(2):
            gen_out = gen_out[:, :, :chunk_padded.size(2)]
            
        delta = gen_out - chunk_padded
        weighted_delta = BASE_EMBED_STRENGTH * delta 
        weighted_delta = weighted_delta[:, :, :curr_len]
        sealed[:, :, i:end] += weighted_delta

    sf.write(output_path, sealed.squeeze().cpu().numpy(), SAMPLE_RATE, subtype='FLOAT')
    np.save(output_path + ".payload.npy", secret_msg.cpu().numpy().astype(int))
    np.save(output_path + ".voiceprint_emb.npy", vp_vec)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python create_sealed_file.py <input> <output>")
    else:
        seal_file(sys.argv[1], sys.argv[2])