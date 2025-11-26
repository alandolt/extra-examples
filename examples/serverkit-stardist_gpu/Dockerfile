# for GPU support, use the tensorflow image with GPU support
FROM tensorflow/tensorflow:2.20.0-gpu 

# for CPU support, use the tensorflow image without GPU support
# FROM tensorflow/tensorflow:2.20.0

RUN python -m pip install imaging-server-kit stardist numpy==1.26.4

WORKDIR /app

COPY . .

EXPOSE 8000

CMD ["python3", "main.py"]