from typing import Any

import os
import gradio as gr
import torch
import transformers
import numpy as np
import librosa
import requests

# ----------------------------
# Device selection
# ----------------------------
num_gpus = torch.cuda.device_count()
device_whisper = "cuda:0" if num_gpus >= 1 else "cpu"
print(f"Using {device_whisper} for Whisper ASR")
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

# ----------------------------
# Whisper ASR
# ----------------------------
dtype = torch.float32 
model_id = "openai/whisper-small"  # heavy on CPU; consider "openai/whisper-medium" if slow

transcribe_model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    
    use_safetensors=True
).to(device_whisper)

processor = transformers.AutoProcessor.from_pretrained(model_id)

pipe0 = transformers.pipeline(
    "automatic-speech-recognition",
    model=transcribe_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size= 1,
    return_timestamps=False,  # you don't use timestamps; faster
    torch_dtype=dtype,
    device=device_whisper,
)

def preprocess_audio(y: np.ndarray, sr: int) -> np.ndarray:
    if y.dtype != np.float32:
        y = y.astype(np.float32)

    # Convert stereo to mono if needed
    if y.ndim > 1:
        y = librosa.to_mono(y.T) if y.shape[0] < y.shape[1] else librosa.to_mono(y)

    # Resample to 16k for Whisper pipeline input
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)

    # Normalize safely
    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak > 0:
        y = y / peak

    return y

def transcribe_audio(audio, language_code: str) -> str:
    sr, y = audio  # audio comes as (sample_rate, np.array) because type="numpy"
    y = preprocess_audio(y, sr)

    result: Any = pipe0(
        y,
        generate_kwargs={"language": language_code, "temperature": 0.5, "top_p": 0.9}
    )
    text = result["text"]
    print(f"Transcribed ({language_code}): {text}")
    return text

# ----------------------------
# GMI Translation
# ----------------------------
GMI_API_KEY = os.getenv("GMI_API_KEY")  # set this in your environment
GMI_URL = "https://api.gmi-serving.com/v1/chat/completions"

def translate_text_gmi(text: str) -> str:
    if not GMI_API_KEY:
        return "Missing GMI_API_KEY. Set it as an environment variable first."

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GMI_API_KEY}",
    }

    payload = {
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "messages": [
            {"role": "system", "content": "Translate the following Zulu or Xhosa into English. Output ONLY the translation."},
            {"role": "user", "content": text},
        ],
        "temperature": 0,
        "max_tokens": 500,
    }

    try:
        resp = requests.post(GMI_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        translated = data["choices"][0]["message"]["content"].strip()
        print("Translated:", translated)
        return translated
    except requests.HTTPError:
        return f"GMI HTTP error: {resp.status_code} - {resp.text}"
    except Exception as e:
        return f"GMI request failed: {e}"

# ----------------------------
# Gradio function
# ----------------------------
def voice_to_text(audio, language):
    if audio is None:
        return ""

    sr, y = audio
    print("sr:", sr, "len(y):", len(y))
    print("max abs:", float(np.max(np.abs(y))) if len(y) else 0)
    print("language:", language)

    # Check for short audio (less than 1 second)
    y16 = preprocess_audio(y, sr)
    if len(y16) < 16000:
        return f"Listening… ({len(y16)/16000:.2f}s)"


    # Whisper language codes
    lang_code = "zulu" if language == "Zulu" else "xhosa"

    transcribed = transcribe_audio(audio, language_code=lang_code)
    #translated = translate_text_gmi(transcribed)
    #return translated
    return transcribed


def quick_translation_test():
    sample = "Sawubona, ngubani igama lakho?"
    print("Translation test input:", sample)
    print("Translation test output:", translate_text_gmi(sample))

quick_translation_test()

# ----------------------------
# UI
# ----------------------------
demo = gr.Interface(
    fn=voice_to_text,
    inputs=[
        gr.Audio(sources=["microphone"], type="numpy"),
        gr.Dropdown(choices=["Zulu", "Xhosa"], value="Zulu", label="Source Language")
    ],
    outputs=gr.Textbox(label="English Translation"),
    title="Voice-to-Text Translator (Zulu/Xhosa → English)",
    description="Select your language, speak in Zulu or Xhosa, and see the English translation.",
    live=True
)

if __name__ == "__main__":
    demo.launch(share=True)
