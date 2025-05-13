# TRPO (Trust Region Policy Optimization)

이 프로젝트는 TRPO 알고리즘을 사용하여 Pendulum-v1 환경을 제어하는 구현체입니다.

## 설치 방법

필요한 패키지들을 설치하기 위해 다음 명령어를 실행하세요:

```bash
pip install -r requirements.txt
```

## 실행 방법

학습을 시작하려면 다음 명령어를 실행하세요:

```bash
python main.py
```

## 주요 하이퍼파라미터

- `batch_size`: 2000
- `gamma`: 0.95 (할인율)
- `lmda`: 0.95 (GAE 람다)
- `delta`: 0.01 (KL 제약)
- `value_lr`: 0.003 (가치 함수 학습률)
- `hidden_dims`: (64, 64) (신경망 은닉층 크기)

## 구현 세부사항

- GAE(Generalized Advantage Estimation)를 사용하여 어드밴티지 계산
- Conjugate Gradient 방법을 사용하여 자연 그래디언트 계산
- Backtracking line search를 통한 스텝 크기 결정
- 가우시안 정책을 사용한 연속 행동 공간 제어

## 학습 곡선

학습이 완료되면 `trpo_pendulum_learning_curves.png` 파일에 다음 세 가지 그래프가 저장됩니다:
1. 평균 리턴
2. 정책 손실
3. 가치 함수 손실 