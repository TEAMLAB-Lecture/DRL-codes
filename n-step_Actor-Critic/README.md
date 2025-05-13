# n-step Actor-Critic (LunarLander-v2)

이 프로젝트는 n-step Actor-Critic 알고리즘을 LunarLander-v2 환경에 적용한 것입니다. Actor-Critic 알고리즘은 정책 기반(Actor)과 가치 기반(Critic) 방법을 결합한 강화학습 알고리즘입니다.

## 환경 설명

LunarLander-v2는 달 착륙선을 조종하여 안전하게 착륙하는 것이 목표인 환경입니다:
- 상태 공간: 8차원 (위치, 속도, 각도, 각속도 등)
- 행동 공간: 4가지 (아무것도 하지 않음, 왼쪽 엔진 점화, 메인 엔진 점화, 오른쪽 엔진 점화)
- 보상: 안전 착륙 +200, 충돌 -100, 연료 소모 -0.3, 엔진 사용 -0.03

## 주요 특징

- n-step TD 학습을 통한 보상 전파 가속화
- Actor와 Critic 네트워크의 동시 학습
- 경험 버퍼를 통한 효율적인 학습
- 학습 과정 시각화 및 로깅

## 설치 방법

1. Docker를 사용하여 실행:
```bash
docker build -t n-step-actor-critic .
docker run -it n-step-actor-critic
```

2. 로컬에서 직접 실행:
```bash
pip install -r requirements.txt
python main.py
```

## 주요 파라미터

- `n_steps`: n-step TD 학습에서 사용할 스텝 수 (기본값: 5)
- `gamma`: 할인율 (기본값: 0.99)
- `lr_actor`: Actor 네트워크의 학습률 (기본값: 0.0003)
- `lr_critic`: Critic 네트워크의 학습률 (기본값: 0.0003)
- `hidden_dim`: 신경망 은닉층 크기 (기본값: 256)

## 출력

- 학습 과정 로그: `logs_[timestamp]/training.log`
- 에피소드 비디오: `videos_[timestamp]/episode_*.mp4`
- 보상 그래프: `rewards_[timestamp].png`

## 참고 사항

- LunarLander-v2는 CartPole보다 더 복잡한 환경이므로 더 큰 신경망과 더 작은 학습률을 사용합니다.
- n-step TD 학습은 1-step TD와 Monte Carlo 방법의 중간 지점에 해당합니다.
- 더 큰 n 값은 더 긴 시간 스케일의 보상 전파를 가능하게 하지만, 더 많은 메모리를 필요로 합니다.
- 성공적인 착륙을 위해서는 보통 100-200 에피소드의 학습이 필요합니다. 