FROM python:3.11-slim

WORKDIR /app

# copy requirements from repo root
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy only project files
COPY . .

# default container port for local dev (Cloud Run will set PORT=8080)
EXPOSE 8080

CMD exec uvicorn app.main:app --host 0.0.0.0 --port $PORT


