import torch 
import torch.nn as nn
import librosa
import torch.nn.functional as F
import numpy as np

def audio_to_spectogram(file_path , n_mels = 128):
    " audio file -> mel spectogram -> pytorch tensor"
    y,sr = librosa.load(file_path , sr = 16000)
    S = librosa.feature.melspectrogram(y=y, sr=sr , n_mels = n_mels)
    S_db = librosa.power_to_db(S,ref = np.max)
    spec_tensor = torch.tensor(S_db, dtype = torch.float32).unsqueeze(0)

    return spec_tensor

class SpeakerEncoder(nn.Module):
    """
    A simple CNN (ResNet-style) to create speaker embeddings.
    
    Args: 
        num_speakers (int): The number of unique speakers to classify.
        embedding_dim (int): The size of the output voiceprint (e.g., 256).
    """
    def __init__(self, num_speakers, embedding_dim=256):
        super(SpeakerEncoder, self).__init__()
        
        # --- CNN Encoder ---
        # This part extracts features from the spectrogram
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        # --- Adaptive Pooling ---
        # This is the key: it pools the features over time, so we can handle
        # audio clips of any length and get a fixed-size output.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # --- Classifier ---
        # This part takes the fixed-size features and creates the embedding
        # and the final classification.
        
        # The "voiceprint" layer
        self.fc1 = nn.Linear(in_features=64, out_features=embedding_dim)
        
        # The "classifier" layer
        self.fc2 = nn.Linear(in_features=embedding_dim, out_features=num_speakers)
        
    def forward(self, x):
        # x starts as: [batch_size, 1, n_mels, time]
        
        # Pass through CNN blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        # Pool over time
        x = self.adaptive_pool(x)
        
        # Flatten for the linear layers
        x = x.view(x.size(0), -1)  # [batch_size, 64]
        
        # --- This is the key part ---
        # 1. We create the embedding (the "voiceprint")
        embedding = F.relu(self.fc1(x))
        
        # 2. We classify the embedding
        # We return this for training with Cross-Entropy Loss
        classification = self.fc2(embedding)
        
        return classification,embedding 
    