import librosa
from pathlib import Path 
import numpy as np

data_dir = Path("data/LibriSpeech/train-clean-100")
print(f"scanning files in the directory : {data_dir}")


try:
    audio_files = list(data_dir.rglob("*.flac"))
    if not audio_files:
        print("no flac files found in directory ")
    else:
        print(f'found {len(audio_files)} flac files')

        sample_path = audio_files[0]
        print(f'loading sample file : {sample_path}')
        y,sr = librosa.load(sample_path , sr= 16000)

        print(f'Successfully loaded audio.')
        print(f'Sample rate (sr) : {sr} Hz')
        print(f'Audio duration : {len(y)/sr:.2f} Seconds')
        

        speaker_id = sample_path.parent.parent.name
        print(f'Speaker ID : {speaker_id}')
        
except FileNotFoundError:
    print(f"Error: The directory {data_dir} was not found.")
except Exception as e:
    print(f"\nAn error occurred. This might be because ffmpeg is not installed correctly.")
    print(f"Error details: {e}")