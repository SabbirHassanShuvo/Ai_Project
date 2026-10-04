from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.gemma import router as gemma_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
)

app.include_router(gemma_router)

@app.get("/")
async def root():
    return {"message": "Hello World"}

    
    
