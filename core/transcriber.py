from urllib import response

import requests
import whisper
import os
from utils.audio_processor import AudioSegment
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
_model= None

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v4")
SARVAM_API_URL = "https://api.sarvam.ai/speech-to-text"

def load_model():
    global _modelff
    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL}...")
        _model = whisper.load_model(WHISPER_MODEL)
    return _model

def transcribe_chunk_whisper(chunk_path :str) -> str:
    model = load_model()
    result = model.transcribe(chunk_path, task="transcribe")

    return result["text"]

def transcribe_piece_sarvam(piece_path :str) -> str:
    
    headers = {"api-subscription-key": SARVAM_API_KEY}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": SARVAM_STT_MODEL, "with_diarization":"false"}
        response = requests.post(SARVAM_API_URL, headers=headers, files=files, data=data, timeout=120,)

    if not response.ok:
        print(f"\n Sarvam returned an error: {response.status_code}")
        print(f"Response: {response.text}")
        response.raise_for_status()
    return response.json().get("transcription", "")

def transcribe_chunk(chunk_path :str , language : str = "english") -> str:
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_whisper(chunk_path)
    
def transcribe_chunk_sarvam(chunk_path :str) -> str:
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY is not set in the environment variables/ .env")
    audio= AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_API_MAX_DURATION * 1000
    full_text = ""
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms
    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start : start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")
        try:
            print(f" -> Sarvam piece {i+1}/{total_pieces} of chunk {chunk_path}...")
            full_text += transcribe_piece_sarvam(piece_path) + " "
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)
    return full_text.strip()

def transcribe_all(chunks :list, language : str = "english") -> str:
    full_transcription=""

    engine = "Sarvam AI" if language.lower() == "hinglish" else "Whisper"
    print(f"Transcribing {len(chunks)} chunk(s) using {engine}...")
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i+1}/{len(chunks)}...")
        transcription = transcribe_chunk(chunk, language=language)
        full_transcription += transcription + " "

    print("Transcription complete.")

    return full_transcription.strip()


