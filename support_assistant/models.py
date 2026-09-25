from typing import List, TypedDict
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)


class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float = Field(..., ge=0.0, le=1.0)


class GraphState(TypedDict, total=False):
    query: str
    intent: str
    retrieved_chunks: List[dict]
    answer: str
    sources: List[str]
    confidence: float
