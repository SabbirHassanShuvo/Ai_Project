# pyrefly: ignore [missing-import]
from fastapi import FastAPI
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
from app.gamma import router as gemma_router
from app.llama import router as llama_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
)

app.include_router(gemma_router)
app.include_router(llama_router)

@app.get("/")
async def root():
    return {"message": "Hello World"}


    
    
