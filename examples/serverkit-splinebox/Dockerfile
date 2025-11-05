# Doc: https://github.com/conda-forge/miniforge-images
FROM condaforge/miniforge3:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y \
    && apt upgrade -y \
    && apt-get install -yq --no-install-recommends \
    python3-dev \
    python3-venv \
    gcc \
    g++ \
    && apt-get autoremove --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && find /var/log -type f -exec cp /dev/null \{\} \;

ARG PYTHON_VERSION=3.10
RUN conda install python=$PYTHON_VERSION -y

WORKDIR /app

COPY . .

RUN python -m pip install -r requirements.txt

EXPOSE 8000

CMD ["python3", "main.py"]