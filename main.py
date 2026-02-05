import os
from dotenv import load_dotenv
from pymongo import MongoClient

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Existing routers (UNCHANGED)
from text_verify import router as text_router
from image_verify import router as image_router
from image_ocr import router as image_ocr_router 
from audio_verify import router as audio_router
from audio_ocr import router as audio_ocr_router
from video_verify import router as video_router
from video_ocr import router as video_ocr_router

# ✅ FIXED AUTH IMPORT (UNCHANGED)
from routes.auth_routes import router as auth_router

# 🔹 ADD HERE
load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("DB_NAME")]

app = FastAPI()

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(text_router)
app.include_router(image_router)
app.include_router(image_ocr_router)
app.include_router(audio_router)
app.include_router(audio_ocr_router)
app.include_router(video_router)
app.include_router(video_ocr_router)

# Auth routes
app.include_router(auth_router, prefix="/api/auth")

@app.get("/")
def root():
    return {"message": "TruthGuard Backend Running"}
