# 🎙️ Hybrid Biometric Watermarking for Audio Verification

This project is a **deep learning system** designed to combat **audio deepfakes** and **tampering**.  
It implements a unique **"two-factor" audio verification system** that combines **biometric identification** with **audio steganography**.

---

## System Overview

### **Biometric Lock (Speaker Encoder)**
A CNN is trained to generate a unique **"voiceprint"** from a speaker’s voice.

### **Seal of Origin (Steganographic GAN)**
A 1D-CNN GAN is trained to **inaudibly embed a secret message (a watermark)** into an audio file.

>  The system’s uniqueness lies in its integration:  
> The inaudible watermark is a **64-bit hash of the speaker’s own voiceprint**, allowing the system to verify that:
> - The file is from a trusted source, and  
> - The speaker’s voice has not been altered.

---

##  How It Works

The system has **two modes**: **Enrollment** and **Verification**.

---

###  **Enrollment (`create_sealed_file.py`)**
This script “seals” a clean audio file.

1. A clean audio file is fed into the **Speaker Encoder (`speaker_encoder.pth`)** to generate its voiceprint.  
2. This voiceprint is **hashed** into a 64-bit message.  
3. The **Generator (`generator.pth`)** model embeds this hash as an **inaudible watermark** into the original audio, creating a “sealed” file.  

 Example output: `sealed_audio.wav`

---

###  **Verification (`verify_audio.py`)**
This script runs a **3-step check** on a suspect file.

#### Checkpoint 1: The “Bouncer”
- The file is fed to the **Discriminator (`discriminator.pth`)**.  
- If the model scores it as **“clean” (unwatermarked)** → it’s a **forgery**.

**Verdict:** 🔴 **UNVERIFIED (Untrusted Source)**

---

#### Checkpoint 2: The “Two-Factor” Check
If the file **is watermarked**, it’s sent down **two paths simultaneously**:

- **Path A:** The **Extractor (`extractor.pth`)** reads the watermark → **Original Hash**  
- **Path B:** The **Speaker Encoder (`speaker_encoder.pth`)** analyzes the voice → **Current Hash**

---

### Final Verdict

| Condition | Result |
|------------|---------|
| `Original Hash == Current Hash` | ✅ **VERIFIED (Authentic & Untampered)** |
| `Original Hash != Current Hash` | ❌ **TAMPERED (Voice Altered / Deepfake)** |

---

## Getting Started

### Setup

Clone the repository, create a virtual environment, and install dependencies.

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name/LibriSpeech_trainer

# Create and activate a virtual environment
# (On Windows)
python -m venv ..\deepfake
..\deepfake\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 

### Download Datasets 

This project requires the LibriSpeech and MUSAN datasets.
They must be placed in a data/ folder inside the LibriSpeech_trainer directory.
└── LibriSpeech_trainer/
    ├── __pycache__/
    │   └── ... (Python cache files)
    ├── data/
    │   ├── LibriSpeech/
    │   │   ├── dev-clean/
    │   │   ├── test-clean/
    │   │   ├── train-clean-100/
    │   └── musan/
    │       ├── music/
    │       ├── noise/
    │       ├── speech/
    ├── train.py

Dataset Sources

LibriSpeech: OpenSLR SLR12
 → Download train-clean-100.tar.gz and test-clean.tar.gz

MUSAN: OpenSLR SLR17
 → Download musan.tar.gz

### Training the Models

All trained models (`.pth` files) are automatically saved in the root `LibriSpeech_trainer` directory.

---

### **Phase 1: Train the Speaker Encoder (Biometric Lock)**

```bash
python train.py
```

Output:

speaker_encoder.pth

Phase 2: Train the GAN (Seal of Origin)

```bash

python train_gan.py
```

Outputs:
generator.pth
extractor.pth
discriminator.pth

### How to Use the System
Once all models are trained, you can enroll and verify audio.

### To Create a “Sealed” File (Enrollment)
``` bash
python create_sealed_file.py <path/to/clean_audio.flac> <path/to/sealed_output.wav>
```
Example:

```bash
python create_sealed_file.py data/LibriSpeech/test-clean/84/121123/84-121123-0000.flac sealed_file.wav
```

🔍 To Verify a File (Verification)
```bash
python verify_audio.py <path/to/your_file.wav>
```
Example:

```bash
python verify_audio.py sealed_file.wav
```


### Workflow 
####Enrollment Workflow (Creating a "Sealed" File)
A Clean Audio File is provided as input.
The audio is fed into the Speaker Encoder to generate a unique Voiceprint Hash.
The same Clean Audio File is also fed into the Generator.
The Voiceprint Hash is also fed into the Generator.
The Generator combines the audio and hash to produce the final Sealed Audio File.


####Verification Workflow (Checking a "Suspect" File)
A Suspect Audio File is provided as input.
It's first fed into the Discriminator for a "bouncer" check.
If the Discriminator gives a High Score: The file is clean (unwatermarked).
Result: 🔴 UNVERIFIED (Untrusted Source). The process stops.
If the Discriminator gives a Low Score: The file is watermarked and proceeds to Checkpoint 2.
At Checkpoint 2, the file is sent down two paths:
Path A: The file goes to the Extractor to read the watermark.
Output: Original Hash (from watermark)
Path B: The file goes to the Speaker Encoder to analyze the voice.
Output: Current Hash (from voice)
The Original Hash and Current Hash are compared.
If the Hashes Match:
Result: ✅ VERIFIED
If the Hashes Do Not Match:
Result: ❌ TAMPERED (Voice Altered)


