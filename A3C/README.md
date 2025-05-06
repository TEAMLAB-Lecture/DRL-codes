# A3C (Asynchronous Advantage Actor-Critic) 구현

이 프로젝트는 A3C 알고리즘을 사용하여 Atari 게임 중 하나인 Pong을 학습하는 코드입니다.

## 주요 기능

- 멀티프로세싱을 통한 비동기 학습
- Actor-Critic 아키텍처 사용
- CNN 기반 특징 추출
- 학습 과정 영상 저장
- 주기적인 모델 저장

## Docker를 사용한 실행 방법

### Windows 환경
```bash
run.bat
```

### Linux/Mac 환경
```bash
chmod +x run.sh
./run.sh
```

## 수동 설치 및 실행 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. Atari 환경 설치:
```bash
pip install gymnasium[atari]
pip install gymnasium[accept-rom-license]
```

3. 코드 실행:
```bash
python a3c_pong.py
```

## 주요 하이퍼파라미터

- `MAX_GLOBAL_STEPS`: 총 학습 스텝 수 (기본값: 200,000)
- `RECORD_INTERVAL_GLOBAL_STEPS`: 영상 녹화 간격 (기본값: 50,000)
- `N_WORKERS`: 워커 수 (기본값: 4)
- `LEARNING_RATE`: 학습률 (기본값: 1e-4)
- `GAMMA`: 할인율 (기본값: 0.99)
- `ENTROPY_BETA`: 엔트로피 계수 (기본값: 0.01)

## 출력 디렉토리 구조

```
output_a3c/
├── models/         # 학습된 모델 저장
├── logs/          # 학습 로그 저장
└── videos/        # 학습 과정 영상 저장
```

## Docker 환경 구성

Docker 환경은 다음과 같은 구성요소를 포함합니다:

1. Python 3.9 기반 이미지
2. 필요한 시스템 패키지 (OpenGL, Xvfb 등)
3. Python 패키지 (requirements.txt에 명시된 모든 패키지)
4. 볼륨 마운트를 통한 로컬 디렉토리 접근

## 주의사항

1. Docker를 사용할 경우 `pyvirtualdisplay`가 자동으로 설정됩니다.
2. GPU를 사용하려면 Docker 실행 시 `--gpus all` 옵션을 추가하고, NVIDIA Container Toolkit이 설치되어 있어야 합니다.
3. 학습 시간은 하드웨어 성능에 따라 크게 달라질 수 있습니다.
4. Docker 컨테이너 내부에서 생성된 파일들은 호스트 시스템의 `output_a3c` 디렉토리에 저장됩니다. 