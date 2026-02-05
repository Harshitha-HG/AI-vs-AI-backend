from fastapi import APIRouter, UploadFile, File
import cv2
import tempfile
import os
import subprocess
import pytesseract
from PIL import Image

from text_verify import verify_text_logic

router = APIRouter()

@router.post("/verify-video-text")
async def verify_video_text(file: UploadFile = File(...)):
    video_path = None
    audio_path = None

    try:
        # ---------------- SAVE VIDEO ----------------
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await file.read())
            video_path = tmp.name

        # ---------------- OCR FROM FRAMES ----------------
        cap = cv2.VideoCapture(video_path)
        texts = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 60 == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                text = pytesseract.image_to_string(pil_img).strip()
                if text:
                    texts.append(text)

            frame_count += 1
            if len(texts) >= 5:
                break

        cap.release()

        # ---------------- EXTRACT AUDIO ----------------
        audio_path = video_path.replace(".mp4", ".wav")

        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                audio_path
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # ---------------- TRANSCRIBE AUDIO ----------------
        from transformers import pipeline
        import soundfile as sf
        import numpy as np

        asr = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base"
        )

        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        audio = audio[: sr * 30]  # limit 30 sec
        audio_text = asr({"array": audio, "sampling_rate": sr})["text"].strip()

        # ---------------- MERGE TEXT ----------------
        merged_text = " ".join(texts)
        if audio_text:
            merged_text += " " + audio_text

        if not merged_text.strip():
            return {"status": "error", "message": "No text found in video"}

        analysis = verify_text_logic(merged_text)

        return {
            "status": "success",
            "extracted_text": merged_text,
            "analysis": analysis
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        # ---------------- CLEANUP (WINDOWS SAFE) ----------------
        try:
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
        except:
            pass
