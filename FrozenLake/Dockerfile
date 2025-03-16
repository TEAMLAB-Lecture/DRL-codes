FROM python:3.11-slim

# 시스템 패키지 업데이트 및 필요한 라이브러리 설치
RUN apt-get update && apt-get install -y \
    python3-opengl \
    ffmpeg \
    xvfb \
    freeglut3-dev \
    libglu1-mesa \
    libglu1-mesa-dev \
    xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# pip 업그레이드 및 Python 패키지 설치
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --verbose -r requirements.txt

# 코드 복사
COPY . .

# 가상 디스플레이 설정 및 스크립트 실행 권한 부여
RUN chmod +x /app/run.sh

# 컨테이너 실행 시 스크립트 실행
ENTRYPOINT ["/app/run.sh"] 