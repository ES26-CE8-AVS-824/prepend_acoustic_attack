import torch
import whisper
from datasets import load_dataset
import datasets
import random, os
import librosa
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import os
import warnings
import numpy as np
import json
warnings.filterwarnings("ignore", message=".*non-tuple sequence.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio._backend")
random.seed(100)



# Huggingface ASR dataset to be tested
DATASET = 'google/fleurs'
LANGUAGE = 'en_us'
SPLIT = 'test'
# Whisper model name, can be one of the following: tiny/tiny.en/base/base.en/small/small.en/medium/medium.en
WHISPER_MODEL = 'base'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = whisper.load_model(WHISPER_MODEL).to(DEVICE)
options = dict(language="en", task="transcribe")

dataset = load_dataset(DATASET, LANGUAGE, split=SPLIT, trust_remote_code=True)

muted_audio = np.load(f'audio_attack_segments/{WHISPER_MODEL}.np.npy')
audio_attack_segment = torch.from_numpy(muted_audio).to(DEVICE)

def pad_or_trim(waveform, length=480_000):  # 30s @ 16kHz
    """Whisper-compatible: pad/trim to exact length"""
    if waveform.shape[-1] > length:
        return waveform[..., :length]  # Trim
    else:
        pad = length - waveform.shape[-1]
        return F.pad(waveform, (0, pad))  # Pad right with 0s

class GPUReadyAudioDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
        self.resampler = torchaudio.transforms.Resample(16000)
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        path = os.path.join(os.path.dirname(item['path']), item['audio']['path'])
        waveform, sample_rate = torchaudio.load(path)
        if waveform.shape[0] > 1:  # Stereo → mono
            waveform = waveform.mean(0, keepdim=True)  # [1, T]
        waveform = waveform.squeeze(0)  # [T] 1D strict!
        waveform = pad_or_trim(waveform, 480_000)
        return {'waveform': waveform, 'path': item['path']}

def collate_audio_pinned(batch):
    waveforms = torch.stack([item['waveform'] for item in batch])  # [B, 480k]
    paths = [item['path'] for item in batch]
    return waveforms, paths

dataset = GPUReadyAudioDataset(dataset)
dataloader = DataLoader(dataset, batch_size=16, num_workers=8, pin_memory=True, 
                        prefetch_factor=4, persistent_workers=True, collate_fn=collate_audio_pinned)

model = whisper.load_model(WHISPER_MODEL).to(DEVICE)
log_mel = whisper.log_mel_spectrogram 

text = {}
with torch.no_grad():
    for audio_batch, paths in dataloader:
        audio_batch = audio_batch.to(DEVICE, non_blocking=True)  # [B, 480k] pinned
        
        mels = [whisper.log_mel_spectrogram(audio) for audio in audio_batch]  # [80, 3000] each
        mels = torch.stack(mels)  # [B, 80, 3000]
        # Batch decoder/encoder (90% time, fully parallel)
        options = whisper.DecodingOptions()
        results = model.decode(mels, options)
        
        for path, res in zip(paths, results):
            text[path] = ["original: " + res.text]
            
        concat = [torch.cat((audio_attack_segment, audio), dim=0)[:480_000] for audio in audio_batch]
        mels_muted = [whisper.log_mel_spectrogram(cat) for cat in concat]
        mels_muted = torch.stack(mels_muted)  # [B, 80, 3000]
        results_muted = model.decode(mels_muted, options)
        for path, res in zip(paths, results_muted):
            text[path].append("muted: " + res.text)
torch.cuda.empty_cache()
print(text)
with open(f"transcriptions_{WHISPER_MODEL}.json", "w") as f:
    json.dump(text, f, indent=2)