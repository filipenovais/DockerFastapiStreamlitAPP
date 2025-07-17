FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install streamlit requests pillow
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.runOnSave=true"]
