@echo off
echo A3C Pong 학습을 시작합니다...

REM 현재 디렉토리의 output_a3c 폴더를 Docker 볼륨으로 마운트
docker build -t a3c-pong .
docker run --rm -v "%cd%/output_a3c:/app/output_a3c" a3c-pong

echo 학습이 완료되었습니다.
echo 결과물은 output_a3c 폴더에서 확인할 수 있습니다. 