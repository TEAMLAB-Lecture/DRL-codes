@echo off
chcp 65001 > nul
echo REINFORCE with Baseline 실행을 시작합니다...

REM Docker 이미지 빌드
echo Docker 이미지를 빌드합니다...
docker build -t reinforce_baseline .

REM 필요한 디렉토리 생성
if not exist videos mkdir videos
if not exist logs mkdir logs

REM Docker 컨테이너 실행
echo Docker 컨테이너를 실행합니다...
docker run -it --rm ^
    -v %cd%\videos:/app/videos ^
    -v %cd%\logs:/app/logs ^
    reinforce_baseline

echo 실행이 완료되었습니다.
pause 