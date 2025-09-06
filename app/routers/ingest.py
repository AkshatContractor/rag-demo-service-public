from fastapi import APIRouter
from app.services.store_service import ingest_documents

router = APIRouter()

@router.post("/ingest")
async def ingest():
    chunks_count = ingest_documents()
    return {"chunks_count": chunks_count}
