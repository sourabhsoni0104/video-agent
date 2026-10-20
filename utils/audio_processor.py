import yt_dlp
from pydub import AudioSegment
import os
from pathlib import Path

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_youtube_audio(url :str) ->str:      #downloads audio from a YouTube video in WAV format
   output_path = os.path.join(DOWNLOAD_DIR, '%(title)s.%(ext)s')
   ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
        'noprogress': True,
        'no_warnings': True,
    }
   with yt_dlp.YoutubeDL(ydl_opts) as ydl:
       info = ydl.extract_info(url, download=True)
       filename = ydl.prepare_filename(info).replace('.webm', '.wav').replace('.m4a', '.wav')
   return filename

data=download_youtube_audio("https://www.youtube.com/watch?v=dQw4w9WgXcQ")



def convert_audio_to_wav(input_path: str) -> str:        #converts the audio into mono channel and 16kHz
    """CConverts an audio file to WAV format using pydub."""
    output_path=os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio=audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path

def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms=chunk_minutes*60*1000
    chunks=[]

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav" )
        chunks.append(chunk_path)
    return chunks
data_final=convert_audio_to_wav(data)
print(chunk_audio(data_final))
os.remove(data)
os.remove(data_final)

