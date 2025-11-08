import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. The Generator (Stamper) ---
# This is an autoencoder that "hides" the message in the audio

class Generator(nn.Module):
    def __init__(self, message_bits=64):
        super(Generator, self).__init__()
        
        # --- Encoder (Downsampling) ---
        # Takes audio [N, 1, 16000] -> [N, 128, 62]
        self.conv_down1 = nn.Conv1d(1, 16, 32, stride=2, padding=15)     # [N, 16, 8000]
        self.bn_down1 = nn.BatchNorm1d(16)
        self.conv_down2 = nn.Conv1d(16, 32, 32, stride=2, padding=15)    # [N, 32, 4000]
        self.bn_down2 = nn.BatchNorm1d(32)
        self.conv_down3 = nn.Conv1d(32, 64, 32, stride=4, padding=14)    # [N, 64, 1000]
        self.bn_down3 = nn.BatchNorm1d(64)
        self.conv_down4 = nn.Conv1d(64, 128, 32, stride=4, padding=14)   # [N, 128, 250]
        self.bn_down4 = nn.BatchNorm1d(128)
        self.conv_down5 = nn.Conv1d(128, 128, 32, stride=4, padding=14)  # [N, 128, 62]
        self.bn_down5 = nn.BatchNorm1d(128)
        
        # --- Message Processor ---
        # Takes message [N, 64] -> [N, 128, 62] (to match audio)
        self.msg_proc = nn.Sequential(
            nn.Linear(message_bits, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * 62) # 62 is the audio feature length
        )
        
        # --- Decoder (Upsampling) ---
        # Takes combined features [N, 256, 62] -> [N, 1, 16000]
        # We use Transposed Convolutions (like a "reverse" convolution)
        self.conv_up1 = nn.ConvTranspose1d(256, 128, 32, stride=4, padding=14) # [N, 128, 250]
        self.bn_up1 = nn.BatchNorm1d(128)
        self.conv_up2 = nn.ConvTranspose1d(128, 64, 32, stride=4, padding=14)  # [N, 64, 1000]
        self.bn_up2 = nn.BatchNorm1d(64)
        self.conv_up3 = nn.ConvTranspose1d(64, 32, 32, stride=4, padding=14)   # [N, 32, 4000]
        self.bn_up3 = nn.BatchNorm1d(32)
        self.conv_up4 = nn.ConvTranspose1d(32, 16, 32, stride=2, padding=15)   # [N, 16, 8000]
        self.bn_up4 = nn.BatchNorm1d(16)
        self.conv_up5 = nn.ConvTranspose1d(16, 1, 32, stride=2, padding=15)    # [N, 1, 16000]
        
    def forward(self, audio, message):
        # 1. Encode the audio
        x = F.relu(self.bn_down1(self.conv_down1(audio)))
        x = F.relu(self.bn_down2(self.conv_down2(x)))
        x = F.relu(self.bn_down3(self.conv_down3(x)))
        x = F.relu(self.bn_down4(self.conv_down4(x)))
        audio_features = F.relu(self.bn_down5(self.conv_down5(x))) # [N, 128, 62]
        
        # 2. Process the message to match the audio feature shape
        msg_features = self.msg_proc(message)
        msg_features = msg_features.view(-1, 128, 62) # [N, 128, 62]

        # 3. Combine them in the "latent space"
        combined_features = torch.cat([audio_features, msg_features], dim=1) # [N, 256, 62]

        # 4. Decode back into an audio signal
        x = F.relu(self.bn_up1(self.conv_up1(combined_features)))
        x = F.relu(self.bn_up2(self.conv_up2(x)))
        x = F.relu(self.bn_up3(self.conv_up3(x)))
        x = F.relu(self.bn_up4(self.conv_up4(x)))
        
        # Final layer: use tanh to keep the output waveform
        # in the [-1, 1] range, just like normalized audio.
        watermarked_audio = torch.tanh(self.conv_up5(x))
        
        return watermarked_audio

# --- 2. The Extractor (Reader) ---
# This is a classifier that reads the audio and finds the message

class Extractor(nn.Module):
    def __init__(self, message_bits=64):
        super(Extractor, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 32, stride=2, padding=15)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, 32, stride=2, padding=15)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, 32, stride=4, padding=14)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, 32, stride=4, padding=14)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128, 256, 32, stride=4, padding=14)
        self.bn5 = nn.BatchNorm1d(256)
        
        # This will pool the features from [N, 256, 62] -> [N, 256, 1]
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier head to get the message back
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, message_bits)

    def forward(self, audio):
        x = F.relu(self.bn1(self.conv1(audio)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1) # Flatten
        
        x = F.relu(self.fc1(x))
        # No activation here, as we'll use MSELoss or L1Loss
        recovered_message = self.fc2(x) 
        
        return recovered_message

# --- 3. The Discriminator (Adversary) ---
# This is a classifier that guesses "Real" vs "Fake"

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # This architecture is almost identical to the Extractor
        self.conv1 = nn.Conv1d(1, 16, 32, stride=2, padding=15)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, 32, stride=2, padding=15)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, 32, stride=4, padding=14)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, 32, stride=4, padding=14)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128, 256, 32, stride=4, padding=14)
        self.bn5 = nn.BatchNorm1d(256)
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier head to get a single "fake" or "real" score
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1) # Single output node

    def forward(self, audio):
        x = F.relu(self.bn1(self.conv1(audio)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1) # Flatten
        
        x = F.relu(self.fc1(x))
        # No activation here, as we'll use BCEWithLogitsLoss
        score = self.fc2(x)
        
        return score