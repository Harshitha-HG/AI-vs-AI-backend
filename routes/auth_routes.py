from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from database import users_collection
import bcrypt

router = APIRouter()

class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str

@router.post("/register")
async def register_user(user: RegisterRequest):

    existing_user = await users_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")

    hashed_password = bcrypt.hashpw(
        user.password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")

    await users_collection.insert_one({
        "name": user.name,
        "email": user.email,
        "password": hashed_password
    })

    return {"message": "User registered successfully"}
