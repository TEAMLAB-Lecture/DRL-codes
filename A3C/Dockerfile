FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip python3-dev libgl1-mesa-glx libglib2.0-0 xvfb cmake zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt 복사 및 Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Gymnasium 설치 확인 (디버깅용) ---
RUN echo "Listing Gymnasium wrappers directory:" && \
    ls -l /usr/local/lib/python3.9/site-packages/gymnasium/wrappers/ || echo "Wrappers directory not found or ls failed" && \
    echo "Checking for frame_stack.py:" && \
    ls -l /usr/local/lib/python3.9/site-packages/gymnasium/wrappers/frame_stack.py || echo "frame_stack.py not found"

# 소스 코드 복사
COPY . .

# 결과물을 저장할 볼륨 설정
VOLUME ["/app/output_a3c"]

# 실행 명령
CMD ["python", "a3c_pong.py"]