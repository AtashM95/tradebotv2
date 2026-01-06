FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt -r requirements-ai.txt
CMD ["python","-m","scripts.run_bot","--mode","paper","--dry-run"]
