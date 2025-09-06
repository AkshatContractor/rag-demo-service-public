import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.routers import ingest, ask_que

app = FastAPI()

cors_origins_str = os.getenv("ALLOWED_CORS_ORIGINS", "http://localhost:3000")
origins = [origin.strip() for origin in cors_origins_str.split(',')]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Hello, world!"}


app.include_router(ingest.router)
app.include_router(ask_que.router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
