# FrozenLake 강화학습 Docker 환경 상세 설명

## 코드 구조 및 기능

이 프로젝트는 OpenAI Gym의 FrozenLake 환경에서 Q-Learning 알고리즘을 사용한 강화학습을 Docker 환경에서 실행할 수 있도록 구성되어 있습니다.

### 주요 파일 구성
- `frozen_lake_qlearning.py`: 메인 Q-Learning 알고리즘 구현
- `Dockerfile`: Docker 이미지 빌드 파일
- `requirements.txt`: 필요한 Python 패키지 목록
- `run_docker.bat`, `run_simple.bat`, `run_quick_test.bat`: Windows 환경에서 Docker 실행을 위한 배치 파일
- `docker_utils.bat`: Docker 환경 관리 유틸리티

### frozen_lake_qlearning.py 주요 기능

1. **환경 설정**
   - 환경 변수를 통해 하이퍼파라미터 설정 (EPISODES, LEARNING_RATE, DISCOUNT_FACTOR 등)
   - 출력 디렉토리 구조 생성 (models, logs, videos)
   - 가상 디스플레이 설정 (Docker 환경에서 GUI 없이 렌더링)

2. **Q-Learning 알고리즘 구현**
   - `initialize_q_table()`: Q-테이블 초기화
   - `epsilon_greedy_policy()`: 입실론-그리디 정책 구현
   - `train_agent()`: Q-Learning 알고리즘을 사용한 에이전트 훈련

3. **평가 및 시각화**
   - `evaluate_during_training()`: 훈련 중 주기적으로 에이전트 평가
   - `evaluate_agent()`: 최종 훈련된 에이전트 평가
   - `plot_training_history()`: 학습 과정 그래프 생성
   - `save_video()`: 에이전트 행동 영상 저장

4. **영상 저장 기능**
   - 학습 중간 과정 영상 저장
   - 개별 평가 영상 저장
   - 모든 평가 영상을 하나로 합치는 기능

## Docker 환경 설명

### Dockerfile 구성
Dockerfile은 Python 환경과 필요한 라이브러리를 설치하고, 가상 디스플레이를 설정하여 GUI 없이도 렌더링이 가능하도록 구성되어 있습니다.

### 실행 스크립트 설명

1. **run_docker.bat**
   - 다양한 옵션으로 Docker 환경을 실행할 수 있는 메뉴 제공
   - 명령줄 인수를 통한 옵션 설정 (`rebuild`, `--no-videos` 등)
   - 다음과 같은 실행 옵션 제공:
     - 기본 실행
     - 출력 결과 저장
     - 하이퍼파라미터 조정
     - 8x8 맵에서 학습
     - 미끄럽지 않은 환경에서 학습
     - 빠른 학습 테스트
     - 사용자 정의 실행

2. **run_simple.bat**
   - 간단한 옵션으로 Docker 환경을 빠르게 실행
   - 기본 설정으로 학습 진행
   - 출력 결과를 호스트 시스템에 저장

3. **run_quick_test.bat**
   - 빠른 테스트를 위한 설정으로 실행
   - 에피소드 수와 평가 에피소드 수를 줄여 빠르게 결과 확인 가능

4. **docker_utils.bat**
   - Docker 환경 관리를 위한 유틸리티 제공
   - 이미지 다시 빌드, 삭제, 캐시 정리 등의 기능
   - 학습 과정 영상 파일 관리 기능 (개별 평가 영상 삭제, 모든 학습 과정 영상 삭제)

### 환경 변수 설정

Docker 실행 시 다음 환경 변수를 통해 학습 파라미터를 조정할 수 있습니다:

- `EPISODES`: 학습 에피소드 수 (기본값: 20000)
- `LEARNING_RATE`: 학습률 (기본값: 0.8)
- `DISCOUNT_FACTOR`: 할인 계수 (기본값: 0.95)
- `EPSILON`: 초기 입실론 값 (기본값: 1.0)
- `EPSILON_DECAY`: 입실론 감소율 (기본값: 0.999)
- `MIN_EPSILON`: 최소 입실론 값 (기본값: 0.01)
- `EVAL_EPISODES`: 평가 에피소드 수 (기본값: 100)
- `EVAL_INTERVAL`: 평가 간격 (기본값: 1000)
- `SLIPPERY`: 미끄러운 환경 여부 (기본값: True)
- `MAP_SIZE`: 맵 크기 (기본값: '4x4', 옵션: '4x4' 또는 '8x8')
- `SAVE_ALL_VIDEOS`: 모든 평가 영상 저장 여부 (기본값: True)

## 실행 방법 및 예시

### 기본 실행
```bash
run_docker.bat
```
메뉴에서 원하는 옵션을 선택하여 실행할 수 있습니다.

### 영상 저장 없이 실행
```bash
run_docker.bat --no-videos
# 또는
run_docker.bat -nv
```

### 이미지 다시 빌드 후 실행
```bash
run_docker.bat rebuild
# 또는
run_docker.bat --rebuild
# 또는
run_docker.bat -r
```

### 간단한 실행
```bash
run_simple.bat
```

### 빠른 테스트 실행
```bash
run_quick_test.bat
```

### Docker 환경 관리
```bash
docker_utils.bat
```
메뉴에서 원하는 관리 옵션을 선택할 수 있습니다.

## 출력 결과

학습 완료 후 다음 결과물이 `output` 디렉토리에 저장됩니다:

1. **모델 파일**
   - `output/models/best_q_table_ep{episode}_{timestamp}.pkl`: 최적 성능의 Q-테이블
   - `output/final_q_table_{timestamp}.pkl`: 최종 학습된 Q-테이블

2. **로그 및 그래프**
   - `output/logs/training_history_{timestamp}.json`: 학습 과정 데이터
   - `output/logs/training_history_{timestamp}.png`: 학습 과정 그래프 (성공률, 입실론, 평균 보상)

3. **영상 파일**
   - `output/videos/episode_{episode}_{timestamp}.mp4`: 개별 평가 시점의 영상
   - `output/learning_progress_{timestamp}.mp4`: 모든 평가 영상을 하나로 합친 파일
   - `output/frozen_lake_solution_{timestamp}.mp4`: 최종 학습된 에이전트의 성능 영상

## 학습 과정 시각화

학습 과정은 다음과 같은 방식으로 시각화됩니다:

1. **그래프**
   - 성공률 변화: 학습이 진행됨에 따라 에이전트의 성공률 변화
   - 입실론 감소: 탐색(exploration)에서 활용(exploitation)으로 전환되는 과정
   - 평균 보상: 최근 100개 에피소드의 평균 보상 변화

2. **영상**
   - 개별 평가 영상: 각 평가 시점에서 에이전트의 행동을 보여주는 영상
   - 학습 과정 영상: 모든 평가 영상을 시간 순서대로 합쳐 학습 진행 과정을 한눈에 볼 수 있는 영상
   - 최종 솔루션 영상: 최종 학습된 에이전트의 성능을 보여주는 영상

## 주요 알고리즘 설명

### Q-Learning 알고리즘
Q-Learning은 모델 없는(model-free) 강화학습 알고리즘으로, 환경과의 상호작용을 통해 최적 정책을 학습합니다.

1. **Q-테이블 업데이트 규칙**:
   ```
   Q(s, a) = (1 - α) * Q(s, a) + α * (r + γ * max(Q(s', a')))
   ```
   여기서:
   - `Q(s, a)`: 상태 s에서 행동 a를 취했을 때의 예상 보상
   - `α`: 학습률 (LEARNING_RATE)
   - `r`: 즉각적인 보상
   - `γ`: 할인 계수 (DISCOUNT_FACTOR)
   - `max(Q(s', a'))`: 다음 상태에서 가능한 최대 Q-값

2. **입실론-그리디 정책**:
   - 확률 ε로 무작위 행동 선택 (탐색)
   - 확률 (1-ε)로 현재 Q-테이블에서 최대 Q-값을 가진 행동 선택 (활용)
   - ε은 학습이 진행됨에 따라 감소 (EPSILON_DECAY)

이 프로젝트는 Docker 환경에서 FrozenLake 문제를 Q-Learning으로 해결하는 과정을 시각적으로 보여주며, 다양한 하이퍼파라미터 설정을 통해 학습 성능을 실험할 수 있는 환경을 제공합니다. 