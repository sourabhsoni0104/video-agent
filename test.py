from utils.audio_processor import process_input
from core.transcriber import transcribe_all

source = "https://www.youtube.com/watch?v=inwPi1cX0n4"  # Replace with your YouTube URL or local file path
chunks = process_input(source)
transcription = transcribe_all(chunks, translate=False)
print("Final Transcription:")
print(transcription)    