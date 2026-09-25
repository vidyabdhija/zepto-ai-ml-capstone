from dotenv import load_dotenv

load_dotenv()
from fastapi import FastAPI

from .models import AskRequest, AskResponse
from .rag import ask_question


app = FastAPI(
    title="Zepto Policy Support Assistant",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "service": "Zepto Policy Support Assistant",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "service": "Zepto Policy Support Assistant",
        "status": "healthy",
        "mock_llm": True,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    return ask_question(request.query)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "support_assistant.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
