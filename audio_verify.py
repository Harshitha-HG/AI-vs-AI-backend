from fastapi import APIRouter, UploadFile, File
from transformers import pipeline
import soundfile as sf
import numpy as np
import io

router = APIRouter()

# ----------------------------------
# Load Audio AI Detection Model
# ----------------------------------
audio_detector = pipeline(
    "audio-classification",
    model="superb/wav2vec2-base-superb-ks"
)

# ----------------------------------
# AUDIO AI VERIFICATION API
# ----------------------------------
@router.post("/verify-audio")
async def verify_audio(file: UploadFile = File(...)):
    try:
        # Read uploaded audio file as bytes
        audio_bytes = await file.read()

        # Decode audio safely (no aifc issue)
        audio, sr = sf.read(io.BytesIO(audio_bytes))

        # Convert stereo to mono if needed
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Run audio AI detection
        result = audio_detector(audio)[0]
        ai_prob = int(result["score"] * 100)

        # Decide verdict
        if ai_prob >= 80:
            verdict = "Likely AI-Generated Audio"
        elif ai_prob >= 50:
            verdict = "Possibly AI-Generated Audio"
        else:
            verdict = "Likely Human Voice"

        return {
            "status": "success",
            "audio_ai_probability": ai_prob,
            "verdict": verdict
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
