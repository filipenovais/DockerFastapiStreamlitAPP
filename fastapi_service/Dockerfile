FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install fastapi uvicorn torch torchvision pillow python-multipart
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
