@echo off
chcp 65001 > nul
echo FrozenLake 강화학습 Docker 환경 실행 스크립트 (간단 버전)
echo =======================================

REM 명령줄 인수 확인
set REBUILD=0
set SAVE_VIDEOS=1
if "%1"=="rebuild" set REBUILD=1
if "%1"=="--rebuild" set REBUILD=1
if "%1"=="-r" set REBUILD=1
if "%1"=="--no-videos" set SAVE_VIDEOS=0
if "%1"=="-nv" set SAVE_VIDEOS=0
if "%2"=="--no-videos" set SAVE_VIDEOS=0
if "%2"=="-nv" set SAVE_VIDEOS=0

REM 이미지가 존재하는지 확인
docker image inspect frozen-lake-rl >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Docker 이미지를 빌드합니다...
    docker build --progress=plain -t frozen-lake-rl .
    if %ERRORLEVEL% NEQ 0 (
        echo Docker 이미지 빌드에 실패했습니다.
        echo 자세한 로그를 확인하려면 아무 키나 누르세요...
        pause > nul
        echo 빌드 로그를 다시 확인합니다...
        docker build --no-cache --progress=plain -t frozen-lake-rl .
        if %ERRORLEVEL% NEQ 0 (
            echo 빌드 실패. 종료합니다.
            pause
            exit /b 1
        ) else (
            echo 두 번째 시도에서 빌드 성공!
        )
    ) else (
        echo 빌드 성공!
    )
) else (
    if %REBUILD%==1 (
        echo Docker 이미지를 강제로 다시 빌드합니다...
        docker build --no-cache --progress=plain -t frozen-lake-rl .
        if %ERRORLEVEL% NEQ 0 (
            echo Docker 이미지 빌드에 실패했습니다.
            pause
            exit /b 1
        ) else (
            echo 빌드 성공!
        )
    ) else (
        echo Docker 이미지가 이미 존재합니다. 다시 빌드하려면 'run_simple.bat rebuild' 명령을 사용하세요.
    )
)

REM 출력 디렉토리 생성
if not exist output mkdir output

echo 기본 설정으로 실행 중...
docker run -it --rm -v "%cd%\output:/app/output" -e EVAL_INTERVAL=1000 -e SAVE_ALL_VIDEOS=%SAVE_VIDEOS% frozen-lake-rl

echo.
echo 실행이 완료되었습니다.
pause 