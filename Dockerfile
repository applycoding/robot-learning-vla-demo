FROM python:3.11-slim

WORKDIR /app

# CPU wheel first so a later dependency install does not pull a CUDA build.
COPY pyproject.toml README.md requirements.txt ./
COPY simreach ./simreach
COPY tests ./tests
COPY docs ./docs

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir "gymnasium>=0.29" "stable-baselines3>=2.3" "h5py>=3.10" \
        "matplotlib>=3.8" "pillow>=10" "transformers>=4.44" "pytest>=8.0" \
    && pip install --no-cache-dir --no-deps .

COPY . .

CMD ["python", "-m", "simreach", "benchmark", "--quick", "--out", "/tmp/simreach-results"]
