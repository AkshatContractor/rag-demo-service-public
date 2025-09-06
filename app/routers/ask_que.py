from fastapi import APIRouter
from pydantic import BaseModel
from app.services.chat_service import ask_question

router = APIRouter(prefix="/qa", tags=["QA"])

class Question(BaseModel):
    query: str

@router.post("/ask")
async def ask(payload: Question):
    return ask_question(payload.query)
