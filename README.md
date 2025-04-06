# 강화학습 알고리즘 구현 모음

이 저장소는 다양한 강화학습 알고리즘과 환경에 대한 구현을 포함하고 있습니다.

## 프로젝트 구조

```
.
├── FrozenLake/             # FrozenLake 환경에서의 Q-Learning 구현
│   ├── frozen_lake_qlearning.py  # Q-Learning 알고리즘 구현
│   ├── requirements.txt     # 필요한 패키지 목록
│   ├── Dockerfile          # Docker 설정 파일
│   ├── README.md           # 프로젝트 설명
│   ├── run_docker.bat      # Docker 실행 스크립트 (Windows)
│   ├── run_simple.bat      # 간단한 실행 스크립트 (Windows)
│   ├── run_quick_test.bat  # 빠른 테스트 스크립트 (Windows)
│   ├── docker_utils.bat    # Docker 환경 관리 유틸리티 (Windows)
│   └── run.sh             # Docker 실행 스크립트 (Linux/Mac)
│
├── REINFORCE_CartPole/        # CartPole 환경에서의 REINFORCE 구현
│   ├── main.py              # 메인 학습 코드
│   ├── requirements.txt     # 필요한 패키지 목록
│   ├── run.bat             # 실행 스크립트
│   └── Dockerfile          # Docker 설정 파일
│
├── REINFORCE/               # 기본 REINFORCE 알고리즘 구현
│   ├── main.py              # 메인 학습 코드
│   ├── requirements.txt     # 필요한 패키지 목록
│   ├── run.bat             # 실행 스크립트
│   ├── Dockerfile          # Docker 설정 파일
│   ├── policy_net.pth      # 학습된 모델
│   ├── game_progress.gif   # 학습 과정 시각화
│   ├── game_progress.mp4   # 학습 과정 비디오
│   ├── frame.png           # 게임 프레임
│   └── REINFOCE_example_code*.ipynb  # Jupyter 노트북 예제
│
└── REINFORCE_Baseline/      # Baseline이 추가된 REINFORCE 구현
    ├── main.py              # 메인 학습 코드
    ├── requirements.txt     # 필요한 패키지 목록
    ├── run.bat             # 실행 스크립트
    ├── Dockerfile          # Docker 설정 파일
    ├── README.md           # 프로젝트 설명
    ├── logs/               # 학습 로그 저장
    └── videos/             # 학습 과정 비디오 저장
```

## 구현된 알고리즘

### 1. Q-Learning (FrozenLake)
- 위치: [FrozenLake/](FrozenLake/)
- 설명: 테이블 기반 Q-Learning을 사용하여 FrozenLake 환경을 해결
- 특징:
  - 4x4 및 8x8 맵 지원
  - 미끄러운/미끄럽지 않은 버전 지원
  - 학습 과정 시각화
  - 다양한 실행 옵션 제공 (Docker, 간단 실행, 빠른 테스트)
  - Windows/Linux/Mac 호환성

### 2. REINFORCE (CartPole)
- 위치: [REINFORCE_CartPole/](REINFORCE_CartPole/)
- 설명: REINFORCE 알고리즘을 사용하여 OpenAI Gym의 CartPole-v1 환경을 학습
- 특징:
  - PyTorch를 사용한 신경망 기반 정책 네트워크
  - GPU 가속 지원
  - Docker 컨테이너화 지원

### 3. 기본 REINFORCE
- 위치: [REINFORCE/](REINFORCE/)
- 설명: 기본적인 REINFORCE 알고리즘 구현
- 특징:
  - Jupyter 노트북을 통한 예제 코드 제공
  - 학습 과정 시각화 (GIF, MP4)
  - Docker 컨테이너화 지원

### 4. REINFORCE with Baseline
- 위치: [REINFORCE_Baseline/](REINFORCE_Baseline/)
- 설명: Baseline이 추가된 REINFORCE 알고리즘 구현
- 특징:
  - 학습 로그 및 비디오 저장
  - 상세한 실행 스크립트
  - Docker 컨테이너화 지원

## 공통 특징
- 모든 프로젝트는 Docker를 통해 실행 가능
- 필요한 패키지는 requirements.txt에 명시
- 실행 스크립트(run.bat) 제공

## 실행 방법
각 프로젝트 디렉토리에서 다음 명령어를 실행:
```bash
# Docker를 사용하는 경우
docker build -t rl-project .
docker run -it rl-project

# 또는 직접 실행
python main.py  # 또는 frozen_lake_qlearning.py
```

## 환경 요구사항
- Python 3.6+
- PyTorch (REINFORCE 프로젝트)
- OpenAI Gym
- NumPy
- Matplotlib
- Docker (선택사항) 