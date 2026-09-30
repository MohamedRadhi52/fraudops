FROM python:3.14-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY src/fraudops/__init__.py src/fraudops/api.py fraudops/
COPY models/lightgbm.txt models/

RUN useradd --no-create-home api
USER api
EXPOSE 8000
CMD ["uvicorn", "fraudops.api:app", "--host", "0.0.0.0", "--port", "8000"]
