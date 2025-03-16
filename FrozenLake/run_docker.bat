@echo off
chcp 65001 > nul
echo FrozenLake 강화학습 Docker 환경 실행 스크립트
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
        echo Docker 이미지가 이미 존재합니다. 다시 빌드하려면 'run_docker.bat rebuild' 명령을 사용하세요.
    )
)

REM 출력 디렉토리 생성
if not exist output mkdir output

echo.
echo 실행 옵션을 선택하세요:
echo 1. 기본 실행
echo 2. 출력 결과 저장
echo 3. 하이퍼파라미터 조정 (기본값)
echo 4. 8x8 맵에서 학습
echo 5. 미끄럽지 않은 환경에서 학습
echo 6. 빠른 학습 테스트
echo 7. 사용자 정의 실행
echo 8. 종료
echo.

:MENU
set /p OPTION="옵션 번호를 입력하세요: "

set SAVE_VIDEOS_ENV=-e SAVE_ALL_VIDEOS=%SAVE_VIDEOS%

if "%OPTION%"=="1" (
    echo 기본 실행 중...
    docker run -it --rm %SAVE_VIDEOS_ENV% frozen-lake-rl
) else if "%OPTION%"=="2" (
    echo 출력 결과를 저장하며 실행 중...
    docker run -it --rm -v "%cd%\output:/app/output" %SAVE_VIDEOS_ENV% frozen-lake-rl
) else if "%OPTION%"=="3" (
    echo 기본 하이퍼파라미터로 실행 중...
    docker run -it --rm -v "%cd%\output:/app/output" -e EPISODES=10000 -e LEARNING_RATE=0.9 -e EPSILON_DECAY=0.995 %SAVE_VIDEOS_ENV% frozen-lake-rl
) else if "%OPTION%"=="4" (
    echo 8x8 맵에서 학습 중...
    docker run -it --rm -v "%cd%\output:/app/output" -e MAP_SIZE=8x8 %SAVE_VIDEOS_ENV% frozen-lake-rl
) else if "%OPTION%"=="5" (
    echo 미끄럽지 않은 환경에서 학습 중...
    docker run -it --rm -v "%cd%\output:/app/output" -e SLIPPERY=False %SAVE_VIDEOS_ENV% frozen-lake-rl
) else if "%OPTION%"=="6" (
    echo 빠른 학습 테스트 중...
    docker run -it --rm -v "%cd%\output:/app/output" -e EPISODES=1000 -e EVAL_EPISODES=10 %SAVE_VIDEOS_ENV% frozen-lake-rl
) else if "%OPTION%"=="7" (
    echo 사용자 정의 실행을 위한 파라미터를 입력하세요.
    echo 기본값: EPISODES=20000, LEARNING_RATE=0.8, DISCOUNT_FACTOR=0.95, EPSILON=1.0, EPSILON_DECAY=0.999, MIN_EPSILON=0.01
    echo.
    
    set /p EPISODES="EPISODES (기본값: 20000): "
    if "%EPISODES%"=="" set EPISODES=20000
    
    set /p LEARNING_RATE="LEARNING_RATE (기본값: 0.8): "
    if "%LEARNING_RATE%"=="" set LEARNING_RATE=0.8
    
    set /p DISCOUNT_FACTOR="DISCOUNT_FACTOR (기본값: 0.95): "
    if "%DISCOUNT_FACTOR%"=="" set DISCOUNT_FACTOR=0.95
    
    set /p EPSILON="EPSILON (기본값: 1.0): "
    if "%EPSILON%"=="" set EPSILON=1.0
    
    set /p EPSILON_DECAY="EPSILON_DECAY (기본값: 0.999): "
    if "%EPSILON_DECAY%"=="" set EPSILON_DECAY=0.999
    
    set /p MIN_EPSILON="MIN_EPSILON (기본값: 0.01): "
    if "%MIN_EPSILON%"=="" set MIN_EPSILON=0.01
    
    set /p EVAL_EPISODES="EVAL_EPISODES (기본값: 100): "
    if "%EVAL_EPISODES%"=="" set EVAL_EPISODES=100
    
    set /p EVAL_INTERVAL="EVAL_INTERVAL (기본값: 1000): "
    if "%EVAL_INTERVAL%"=="" set EVAL_INTERVAL=1000
    
    set /p SLIPPERY="SLIPPERY (True/False, 기본값: True): "
    if "%SLIPPERY%"=="" set SLIPPERY=True
    
    set /p MAP_SIZE="MAP_SIZE (4x4/8x8, 기본값: 4x4): "
    if "%MAP_SIZE%"=="" set MAP_SIZE=4x4
    
    set /p SAVE_ALL_VIDEOS_INPUT="모든 평가 영상 저장 (True/False, 기본값: True): "
    if "%SAVE_ALL_VIDEOS_INPUT%"=="False" set SAVE_VIDEOS=0
    if "%SAVE_ALL_VIDEOS_INPUT%"=="false" set SAVE_VIDEOS=0
    
    echo 사용자 정의 파라미터로 실행 중...
    docker run -it --rm -v "%cd%\output:/app/output" ^
        -e EPISODES=%EPISODES% ^
        -e LEARNING_RATE=%LEARNING_RATE% ^
        -e DISCOUNT_FACTOR=%DISCOUNT_FACTOR% ^
        -e EPSILON=%EPSILON% ^
        -e EPSILON_DECAY=%EPSILON_DECAY% ^
        -e MIN_EPSILON=%MIN_EPSILON% ^
        -e EVAL_EPISODES=%EVAL_EPISODES% ^
        -e EVAL_INTERVAL=%EVAL_INTERVAL% ^
        -e SLIPPERY=%SLIPPERY% ^
        -e MAP_SIZE=%MAP_SIZE% ^
        -e SAVE_ALL_VIDEOS=%SAVE_VIDEOS% ^
        frozen-lake-rl
) else if "%OPTION%"=="8" (
    echo 종료합니다.
    exit /b 0
) else (
    echo 잘못된 옵션입니다. 다시 선택해주세요.
    goto MENU
)

echo.
echo 실행이 완료되었습니다.
pause
exit /b 0 