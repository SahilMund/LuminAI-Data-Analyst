# backend/app/api/routes/__init__.py

from fastapi import APIRouter
from .hello_router import hello_router  # Correct import of the hello router

# Create the main API router
api_router = APIRouter()

# Include the hello router in the main API router
api_router.include_router(hello_router)
