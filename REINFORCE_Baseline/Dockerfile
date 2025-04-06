FROM python:3.8-slim

WORKDIR /app

# OpenCV와 pygame 의존성 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsdl2-2.0-0 \
    libsdl2-image-2.0-0 \
    libsdl2-mixer-2.0-0 \
    libsdl2-ttf-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# 로그와 비디오 디렉토리 생성
RUN mkdir -p /app/logs /app/videos

COPY . .

CMD ["python", "main.py"] 