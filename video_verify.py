from fastapi import APIRouter, UploadFile, File
import cv2
import tempfile
import os
import numpy as np
from PIL import Image

from image_verify import image_detector

router = APIRouter()

@router.post("/verify-video")
async def verify_video(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(await file.read())
            video_path = temp_video.name

        cap = cv2.VideoCapture(video_path)
        scores = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 30 == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                result = image_detector(pil_img)[0]
                scores.append(result["score"] * 100)

            frame_count += 1
            if len(scores) >= 10:
                break

        cap.release()
        os.remove(video_path)

        if not scores:
            return {"status": "error", "message": "No frames extracted"}

        avg = int(np.mean(scores))

        verdict = (
            "Likely AI-Generated Video" if avg >= 80 else
            "Possibly AI-Generated Video" if avg >= 50 else
            "Likely Real Video"
        )

        return {
            "status": "success",
            "video_ai_probability": avg,
            "verdict": verdict
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
