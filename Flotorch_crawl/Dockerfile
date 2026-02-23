FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src/ src/
RUN pip install -e .

ENV PORT=8081
EXPOSE 8081

CMD ["python", "-m", "flotorch_crawl"]
