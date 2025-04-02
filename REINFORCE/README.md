# REINFORCE 알고리즘 예제

이 프로젝트는 REINFORCE 알고리즘을 사용하여 그리드 월드 환경에서 에이전트를 학습시키는 예제입니다.

## 환경 설정

- Python 3.8
- PyTorch
- NumPy
- Matplotlib
- ImageIO

## 실행 방법

1. Docker가 설치되어 있어야 합니다.
2. `run.bat` 파일을 실행합니다.
3. 학습이 완료되면 `game_progress.gif`와 `game_progress.mp4` 파일이 생성됩니다.

## 프로젝트 구조

- `main.py`: 메인 코드
- `Dockerfile`: Docker 환경 설정
- `requirements.txt`: 필요한 Python 패키지 목록
- `run.bat`: 실행 스크립트

## 알고리즘 설명

REINFORCE 알고리즘은 정책 경사법(Policy Gradient)의 대표적인 예시입니다. 이 알고리즘은 다음과 같은 단계로 작동합니다:

1. 정책 네트워크를 사용하여 에피소드 생성
2. 각 에피소드에서 얻은 보상으로 할인된 누적 보상 계산
3. 정책 네트워크의 손실 함수 계산 및 업데이트

## 결과

학습이 완료되면 에이전트가 시작점(0,0)에서 목표점(6,5)까지 최적의 경로를 찾아 이동하는 것을 시각화한 GIF와 MP4 파일이 생성됩니다. 