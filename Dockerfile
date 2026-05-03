FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
# Pin scikit-learn in the RUN command to be safe and use --no-cache
RUN pip3 install --no-cache-dir -r requirements.txt
COPY app.py ./
COPY model_and_preprocessing_artifacts.pkl ./
# Required for HF Spaces Docker to work properly
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
# Add healthcheck to help HF monitor the container
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health
EXPOSE 7860
ENTRYPOINT ["streamlit", "run", "app.py"]
