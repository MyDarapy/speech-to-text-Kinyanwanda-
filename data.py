from datasets import load_dataset, Audio
from huggingface_hub import login
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor
from audiomentations import Compose, AddGaussianNoise, AddBackgroundNoise, TimeStretch, PitchShift, Gain, PolarityInversion, Normalize, OneOf
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import io
import torchaudio
from torchaudio.transforms import Resample
import subprocess
import torchaudio
import numpy as np
import io


login(os.getenv('HF_TOKEN'))

# Load processor
model_id = "openai/whisper-large-v3"
processor = WhisperProcessor.from_pretrained(model_id)

# Load dataset non-streaming
train_data = load_dataset("Oluwadara/kinyarwanda-asr-track-a", split="train")
test_data = load_dataset("Oluwadara/kinyarwanda-asr-track-a", split="validation")

'''# Load dataset non-streaming
train_data = load_dataset("babs/kinyarwada-kaggle", split="train")
test_data = load_dataset("babs/kinyarwada-kaggle", split="test") '''

# Don't decode yet
train_data = train_data.cast_column("audio", Audio(decode=False))
test_data = test_data.cast_column("audio", Audio(decode=False))

noise_dir = "/workspace/background_noises/musan/noise/"
# Augmentations
augmentations = Compose([
    TimeStretch(min_rate=0.8, max_rate=1.2, p=0.2, leave_length_unchanged=False),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
    Gain(min_gain_db=-6, max_gain_db=6, p=0.1),
    PolarityInversion(p=0.05),
    OneOf(
            [
                AddBackgroundNoise(
                    sounds_path=noise_dir, min_snr_db=1.0, max_snr_db=5.0, noise_transform=PolarityInversion(), p=1.0
                ),
                AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=1.0),
            ],
            p=0.2,
        ),
])

def webm_bytes_to_numpy(audio_bytes):
    """Convert webm bytes to a 1D NumPy float32 waveform array."""
    process = subprocess.run(
        ["ffmpeg", "-i", "pipe:0", "-f", "wav", "pipe:1", "-loglevel", "quiet"],
        input=audio_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    wav_bytes = process.stdout
    if not wav_bytes:
        raise ValueError("FFmpeg failed to decode the input bytes.")
     
    waveform, sample_rate = torchaudio.load(io.BytesIO(wav_bytes))
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
        
    waveform = waveform.squeeze(0).numpy().astype(np.float32)  # shape: (T,)
    return waveform, sample_rate

# Preprocessing function
def preprocess(example, apply_augment=False):
    try:
        audio_bytes = example["audio"]["bytes"]
        if not audio_bytes or len(audio_bytes) < 1000:
            raise ValueError("Empty or corrupt bytes")

        waveform, sr = webm_bytes_to_numpy(audio_bytes)

        if sr != 16000:
            waveform = torch.tensor(waveform).unsqueeze(0)
            resampler = Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
            sr = 16000

        if apply_augment:
            waveform = augmentations(waveform.numpy().astype(np.float32), sample_rate=sr)

        input_features = processor(waveform.tolist(), sampling_rate=sr, return_tensors="pt")["input_features"][0]
        labels = processor.tokenizer(example["text"], return_tensors="pt").input_ids[0]

        return {"input_features": input_features, "labels": labels}
        
    except Exception as e:
        print(f"Failed to process example: {e}")
        return {"input_features": None, "labels": None}

''' train_data = train_data.map(lambda x: preprocess(x, apply_augment=True))
test_data = test_data.map(preprocess)

train_data = train_data.filter(lambda x: x["input_features"] is not None)
test_data = test_data.filter(lambda x: x["input_features"] is not None) '''

class ASRDataset(Dataset):
    def __init__(self, hf_dataset, processor, apply_augment=False):
        self.dataset = hf_dataset
        self.processor = processor
        self.apply_augment = apply_augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        try:
            audio_bytes = example["audio"]["bytes"]
            if not audio_bytes or len(audio_bytes) < 1000:
                raise ValueError("Empty or corrupt bytes")

            # Decode audio
            waveform, sr = webm_bytes_to_numpy(audio_bytes)

            # Resample
            if sr != 16000:
                waveform = torch.tensor(waveform).unsqueeze(0)
                resampler = Resample(orig_freq=sr, new_freq=16000)
                waveform = resampler(waveform).squeeze(0).numpy()
                sr = 16000

            # Augment if required
            if self.apply_augment:
                if waveform.ndim > 1:
                    waveform = waveform.flatten()
                waveform = augmentations(waveform.astype(np.float32), sample_rate=sr)

            # Process
            input_features = self.processor(waveform.tolist(), sampling_rate=sr, return_tensors="pt")["input_features"][0]
            labels = self.processor.tokenizer(example["text"], return_tensors="pt").input_ids[0]

            #print(f"input_features: {input_features}, labels: {labels}")

            return {
                "input_features": input_features,
                "labels": labels
            }

        except Exception as e:
            print(f"Failed to process example at index {idx}: {e}")
            return {"input_features": None, "labels": None}

def collate_batch(batch):
    batch = [item for item in batch if item["input_features"] is not None and item["labels"] is not None]
    
    if len(batch) == 0:
        print("Warning: Entire batch was filtered out due to corrupted samples.")
        return None

    input_feats = [item["input_features"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    input_feats = pad_sequence(input_feats, batch_first=True)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return {"input_features": input_feats, "labels": labels}


# Dataloaders
def get_loaders(batch_size=8):
    train_dataset = ASRDataset(train_data, processor, apply_augment=True)
    test_dataset = ASRDataset(test_data, processor, apply_augment=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_batch, num_workers=8)
    return train_loader, test_loader

# Example usage
if __name__ == "__main__":
    train_loader, test_loader = get_loaders(batch_size=8)
    for batch in train_loader:
        #print(batch['input_features'])
        print(batch["input_features"].shape)
        print(batch["labels"].shape)
        break