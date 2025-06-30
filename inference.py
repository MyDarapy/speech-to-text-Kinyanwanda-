import torch
import torch.nn.functional as F
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio
from tqdm import tqdm
import pandas as pd
import os
from huggingface_hub import login

login(os.getenv("HF_TOKEN"))

MODEL_ID = "Oluwadara/finetuned-whisper-asr-track-a"
GIT_COMMIT_HASH = "f39bcdc73aa3f04ae57016c3d43c9b53dcec01a5"
BATCH_SIZE = 4
TARGET_SR = 16000
MAX_LEN = 3000

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Using device: {device} with dtype: {torch_dtype}")

processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID,
    revision=GIT_COMMIT_HASH,
    torch_dtype=torch_dtype
).to(device)
model.eval()

# Override forced decoder ids and timestamp behavior
if hasattr(model.generation_config, "forced_decoder_ids"):
    model.generation_config.forced_decoder_ids = None

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

data = load_dataset("Oluwadara/audio-test", split="train").cast_column("audio", Audio(decode=True))

transcriptions = []
batch_features = []
audio_paths = []

for sample in tqdm(data, desc="Transcribing"):
    audio = sample["audio"]
    waveform = audio["array"]
    sr = audio["sampling_rate"]
    audio_paths.append(audio["path"])

    if sr != TARGET_SR:
        waveform = torchaudio.transforms.Resample(sr, TARGET_SR)(torch.tensor(waveform)).numpy()

    features = processor(waveform, sampling_rate=TARGET_SR, return_tensors="pt").input_features[0].to(torch_dtype)

    if features.shape[1] < MAX_LEN:
        padded = F.pad(features, (0, MAX_LEN - features.shape[1]), value=0)
    else:
        padded = features[:, :MAX_LEN]

    batch_features.append(padded)

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
        batch_features = []

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

ids = [os.path.splitext(os.path.basename(p))[0] for p in audio_paths]
df = pd.DataFrame({"id": ids, "transcription": transcriptions})
df.to_csv("transcribed_results.csv", index=False)
print("Done!")
