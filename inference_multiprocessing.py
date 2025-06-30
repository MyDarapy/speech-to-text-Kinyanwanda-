import os
import pandas as pd
import subprocess
import torchaudio
import torch
import io
from tqdm import tqdm
import torch.nn.functional as F
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import concurrent.futures

# --- Configurations ---
AUDIO_BASE_FOLDER = "/kaggle/input/kinyarwanda-asr-b-4"
CSV_PATH = "/kaggle/input/kinyarwanda-asr-b-4/test.json"
BATCH_SIZE = 5
TARGET_SR = 16000
MAX_LEN = 3000
MODEL_ID = "babs/vlfm"
GIT_COMMIT_HASH = None

# --- Device and dtype ---
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"Using device: {device} with dtype: {torch_dtype}")

# --- Load Processor & Model ---
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID,
    revision=GIT_COMMIT_HASH,
    torch_dtype=torch_dtype
).to(device)
model.eval()

if hasattr(model.generation_config, "forced_decoder_ids"):
    model.generation_config.forced_decoder_ids = None

# --- Decoder prompt tokens ---
decoder_start_token_id = model.config.decoder_start_token_id
transcribe_token_id = processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")
notimestamps_token_id = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")

prompt_ids = torch.tensor(
    [[decoder_start_token_id, transcribe_token_id, notimestamps_token_id]],
    device=device
)
decoder_prompt_len = prompt_ids.shape[-1]
max_new_tokens = model.config.max_target_positions - decoder_prompt_len
print(f"Adjusted max_new_tokens: {max_new_tokens}")

# --- Load test data ---
df = pd.read_json(CSV_PATH) if CSV_PATH.endswith(".json") else pd.read_csv(CSV_PATH)
audio_keys = list(df.keys())
audio_paths = [df[k]["audio_path"] for k in audio_keys]

# --- Parallelized audio decoding ---
def load_audio_bytes(path):
    full_path = os.path.join(AUDIO_BASE_FOLDER, path)
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", full_path, "-f", "wav", "pipe:1", "-loglevel", "quiet"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        waveform, sr = torchaudio.load(io.BytesIO(result.stdout))
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != TARGET_SR:
            waveform = torchaudio.transforms.Resample(sr, TARGET_SR)(waveform)
        return path, waveform.squeeze(0).numpy()
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return path, None

print("Decoding audio in parallel...")
wav_dict = {}
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(load_audio_bytes, path): path for path in audio_paths}
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Decoding Audio"):
        path, waveform = future.result()
        wav_dict[path] = waveform

# --- Inference ---
transcriptions = []
batch_features = []
batch_paths = []

for path in tqdm(audio_paths, desc="Transcribing"):
    waveform = wav_dict.get(path)
    if waveform is None:
        transcriptions.append("")
        continue

    features = processor(waveform, sampling_rate=TARGET_SR, return_tensors="pt").input_features[0].to(torch_dtype)
    features = F.pad(features, (0, max(0, MAX_LEN - features.shape[1])), value=0)
    features = features[:, :MAX_LEN]  # Truncate if too long

    batch_features.append(features)
    batch_paths.append(path)

    if len(batch_features) == BATCH_SIZE:
        input_batch = torch.stack(batch_features).to(device)
        decoder_inputs = prompt_ids.expand(input_batch.size(0), -1)

        with torch.no_grad():
            pred_ids = model.generate(
                input_batch,
                decoder_input_ids=decoder_inputs,
                max_new_tokens=max_new_tokens
            )
        decoded = processor.batch_decode(pred_ids, skip_special_tokens=True)
        transcriptions.extend(decoded)
        batch_features.clear()
        batch_paths.clear()

# Handle last batch
if batch_features:
    input_batch = torch.stack(batch_features).to(device)
    decoder_inputs = prompt_ids.expand(input_batch.size(0), -1)

    with torch.no_grad():
        pred_ids = model.generate(
            input_batch,
            decoder_input_ids=decoder_inputs,
            max_new_tokens=max_new_tokens
        )
    decoded = processor.batch_decode(pred_ids, skip_special_tokens=True)
    transcriptions.extend(decoded)

# --- Save Results ---
ids = [os.path.splitext(os.path.basename(p))[0] for p in audio_paths]
pd.DataFrame({"id": ids, "transcription": transcriptions}).to_csv("transcribed_results.csv", index=False)
print("✅ Done! Saved to transcribed_results.csv")
