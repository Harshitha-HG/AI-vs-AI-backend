from fastapi import APIRouter, UploadFile, File
from transformers import pipeline
import soundfile as sf
import numpy as np
import io

from text_verify import verify_text_logic

router = APIRouter()

# Load Whisper ASR
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base"
)

@router.post("/verify-audio-text")
async def verify_audio_text(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()

        audio, sr = sf.read(io.BytesIO(audio_bytes))

        # Convert stereo → mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # 🔥 TRIM TO FIRST 30 SECONDS
        max_samples = sr * 30
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # Speech-to-text
        transcription = asr(
            {"array": audio, "sampling_rate": sr}
        )["text"]

        if not transcription.strip():
            return {
                "status": "error",
                "message": "No speech detected in audio"
            }

        analysis = verify_text_logic(transcription)

        return {
            "status": "success",
            "transcribed_text": transcription,
            "analysis": analysis
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
