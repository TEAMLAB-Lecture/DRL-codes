# 심층 강화학습(Deep Reinforcement Learning) 실습 코드 모음

이 저장소는 심층 강화학습(DRL) 수업에서 다루는 다양한 알고리즘과 환경에 대한 실습 코드를 제공합니다. 각 폴더는 서로 다른 강화학습 알고리즘 또는 환경에 대한 구현을 포함하고 있으며, Docker를 통해 쉽게 실행할 수 있도록 구성되어 있습니다.

## 목차

- [소개](#소개)
- [프로젝트 구조](#프로젝트-구조)
- [시작하기](#시작하기)
- [구현된 알고리즘](#구현된-알고리즘)
- [환경](#환경)
- [기여 방법](#기여-방법)
- [라이센스](#라이센스)

## 소개

이 저장소는 강화학습의 기초부터 심층 강화학습까지 다양한 알고리즘을 실습할 수 있는 코드를 제공합니다. 각 구현은 Docker 환경에서 실행할 수 있도록 설계되어 있어, 환경 설정의 번거로움 없이 바로 실습을 시작할 수 있습니다.

주요 특징:
- Docker를 통한 간편한 환경 설정
- 다양한 강화학습 알고리즘 구현
- 학습 과정 시각화 및 결과 분석 도구
- 하이퍼파라미터 조정을 위한 유틸리티

## 프로젝트 구조

```
.
├── FrozenLake/                # FrozenLake 환경에서의 Q-Learning 구현
│   ├── Dockerfile             # Docker 이미지 빌드 파일
│   ├── frozen_lake_qlearning.py  # Q-Learning 알고리즘 구현
│   ├── run_docker.bat         # Docker 실행 스크립트 (Windows)
│   ├── run_simple.bat         # 간단한 실행 스크립트 (Windows)
│   ├── run_quick_test.bat     # 빠른 테스트 스크립트 (Windows)
│   ├── docker_utils.bat       # Docker 환경 관리 유틸리티 (Windows)
│   ├── run.sh                 # Docker 실행 스크립트 (Linux/Mac)
│   └── README.md              # FrozenLake 프로젝트 설명
│
├── CartPole/                  # CartPole 환경에서의 DQN 구현 (예정)
│
├── Atari/                     # Atari 게임에서의 DQN 구현 (예정)
│
└── README.md                  # 메인 README 파일
```

## 시작하기

### 요구 사항

- [Docker](https://www.docker.com/get-started) 설치
- Windows 사용자의 경우 WSL2 또는 Docker Desktop 설치 권장
- Git

### 저장소 클론

```bash
git clone https://github.com/your-username/DRL-class-codes.git
cd DRL-class-codes
```

### 특정 알고리즘 실행하기

각 폴더에는 해당 알고리즘을 실행하기 위한 지침이 포함된 README.md 파일이 있습니다. 예를 들어, FrozenLake 환경에서 Q-Learning을 실행하려면:

```bash
cd FrozenLake

# Windows
run_docker.bat

# Linux/Mac
chmod +x run.sh
./run.sh
```

## 구현된 알고리즘

현재 구현된 알고리즘 목록:

1. **Q-Learning**
   - 위치: [FrozenLake/](FrozenLake/)
   - 설명: 테이블 기반 Q-Learning을 사용하여 FrozenLake 환경 해결
   - 특징: 학습 과정 시각화, 하이퍼파라미터 조정, 결과 분석

향후 추가될 알고리즘:
- Deep Q-Network (DQN)
- Double DQN
- Dueling DQN
- Policy Gradient
- Actor-Critic
- Proximal Policy Optimization (PPO)
- Soft Actor-Critic (SAC)

## 환경

현재 지원되는 환경:

1. **FrozenLake**
   - OpenAI Gym의 FrozenLake-v1 환경
   - 4x4 및 8x8 맵 지원
   - 미끄러운/미끄럽지 않은 버전 지원

향후 추가될 환경:
- CartPole
- MountainCar
- Atari 게임 (Breakout, Pong 등)
- MuJoCo 환경

## 기여 방법

이 프로젝트에 기여하고 싶으시다면:

1. 이 저장소를 포크합니다.
2. 새로운 브랜치를 생성합니다 (`git checkout -b feature/amazing-feature`).
3. 변경 사항을 커밋합니다 (`git commit -m 'Add some amazing feature'`).
4. 브랜치에 푸시합니다 (`git push origin feature/amazing-feature`).
5. Pull Request를 생성합니다.

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요. 