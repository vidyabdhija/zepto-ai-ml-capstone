FROM python:3.11-slim

WORKDIR /app

COPY support_assistant/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY support_assistant /app/support_assistant

RUN rm -rf /app/support_assistant/chroma_db

RUN python -m support_assistant.ingest

ENV MOCK_LLM=1

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "support_assistant.main:app", "--host", "0.0.0.0", "--port", "7860"]
