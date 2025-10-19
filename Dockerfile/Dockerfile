FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libcurl4-openssl-dev \
    libatlas-base-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt

RUN pip install --upgrade pip && pip install -r /workspace/requirements.txt

CMD ["bash"]
