FROM python:3.9-slim

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
# onnxruntime-gpu если есть GPU, иначе onnxruntime
RUN pip install --no-cache-dir -r requirements.txt

# Копируем веса и код
COPY model.onnx .
COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
