from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

def setup_cors(app: FastAPI):
    """Configure CORS middleware"""
    allowed_origins = os.getenv(
        "ALLOWED_ORIGINS", 
        "http://localhost:8080,http://localhost:5173,http://localhost:3000"
    ).split(",")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    ) 