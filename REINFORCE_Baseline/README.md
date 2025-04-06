# REINFORCE with Baseline

이 프로젝트는 REINFORCE 알고리즘에 baseline을 추가하여 구현한 강화학습 예제입니다. CartPole 환경에서 실행되며, 학습 과정을 비디오로 기록합니다.

## 특징

- Policy Network와 Value Network를 사용하여 baseline을 구현
- 각 에피소드의 학습 과정을 비디오로 저장
- 학습 진행 상황을 그래프로 시각화
- Docker를 통한 환경 구성

## 설치 및 실행

1. Docker 이미지 빌드:
```bash
docker build -t reinforce_baseline .
```

2. Docker 컨테이너 실행:
```bash
docker run -it --rm -v $(pwd)/videos:/app/videos reinforce_baseline
```

## 구현 세부사항

- Policy Network: 상태를 입력으로 받아 각 행동의 확률을 출력
- Value Network: 상태를 입력으로 받아 가치를 예측
- Baseline: Value Network의 예측값을 baseline으로 사용
- Advantage: 실제 리턴과 baseline의 차이를 계산하여 정책 업데이트에 사용

## 결과

- 100 에피소드마다 학습 과정을 비디오로 저장
- 학습이 완료되면 에피소드별 보상 그래프를 생성
- 비디오와 그래프는 `videos_[timestamp]` 디렉토리에 저장 