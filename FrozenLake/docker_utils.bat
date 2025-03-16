@echo off
chcp 65001 > nul
echo FrozenLake 강화학습 Docker 환경 관리 유틸리티
echo =======================================

if "%1"=="" (
    goto :MENU
) else if "%1"=="clean" (
    goto :CLEAN
) else if "%1"=="rebuild" (
    goto :REBUILD
) else if "%1"=="help" (
    goto :HELP
) else if "%1"=="clean-videos" (
    goto :CLEAN_VIDEOS
) else (
    echo 알 수 없는 명령: %1
    echo 사용 가능한 명령: clean, rebuild, help, clean-videos
    exit /b 1
)

:MENU
echo.
echo 작업을 선택하세요:
echo 1. Docker 이미지 다시 빌드
echo 2. Docker 이미지 삭제
echo 3. Docker 캐시 정리
echo 4. 학습 과정 영상 파일 정리
echo 5. 도움말
echo 6. 종료
echo.

set /p OPTION="옵션 번호를 입력하세요: "

if "%OPTION%"=="1" (
    goto :REBUILD
) else if "%OPTION%"=="2" (
    goto :CLEAN
) else if "%OPTION%"=="3" (
    goto :CLEAN_CACHE
) else if "%OPTION%"=="4" (
    goto :CLEAN_VIDEOS
) else if "%OPTION%"=="5" (
    goto :HELP
) else if "%OPTION%"=="6" (
    exit /b 0
) else (
    echo 잘못된 옵션입니다. 다시 선택해주세요.
    goto :MENU
)

:REBUILD
echo Docker 이미지를 다시 빌드합니다...
docker build --no-cache --progress=plain -t frozen-lake-rl .
if %ERRORLEVEL% NEQ 0 (
    echo Docker 이미지 빌드에 실패했습니다.
    pause
    exit /b 1
) else (
    echo 빌드 성공!
    pause
    exit /b 0
)

:CLEAN
echo Docker 이미지를 삭제합니다...
docker image inspect frozen-lake-rl >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    docker rmi frozen-lake-rl
    echo 이미지가 삭제되었습니다.
) else (
    echo 이미지가 존재하지 않습니다.
)
pause
exit /b 0

:CLEAN_CACHE
echo Docker 캐시를 정리합니다...
docker system prune -f
echo 캐시가 정리되었습니다.
pause
exit /b 0

:CLEAN_VIDEOS
echo.
echo 학습 과정 영상 파일 정리 옵션:
echo 1. 개별 평가 영상만 삭제 (합쳐진 영상 유지)
echo 2. 모든 학습 과정 영상 삭제
echo 3. 취소
echo.

set /p VIDEO_OPTION="옵션 번호를 입력하세요: "

if "%VIDEO_OPTION%"=="1" (
    if exist output\videos (
        echo 개별 평가 영상을 삭제합니다...
        del /q output\videos\*.mp4 2>nul
        echo 개별 평가 영상이 삭제되었습니다.
    ) else (
        echo 영상 디렉토리가 존재하지 않습니다.
    )
) else if "%VIDEO_OPTION%"=="2" (
    if exist output\videos (
        echo 개별 평가 영상을 삭제합니다...
        del /q output\videos\*.mp4 2>nul
    )
    echo 합쳐진 학습 과정 영상을 삭제합니다...
    del /q output\learning_progress_*.mp4 2>nul
    echo 모든 학습 과정 영상이 삭제되었습니다.
) else if "%VIDEO_OPTION%"=="3" (
    echo 취소되었습니다.
) else (
    echo 잘못된 옵션입니다.
)
pause
exit /b 0

:HELP
echo.
echo === 도움말 ===
echo docker_utils.bat             - 메뉴 표시
echo docker_utils.bat rebuild     - Docker 이미지 다시 빌드
echo docker_utils.bat clean       - Docker 이미지 삭제
echo docker_utils.bat clean-videos - 학습 과정 영상 파일 정리
echo.
echo 배치 파일 사용법:
echo run_docker.bat               - 기본 실행
echo run_docker.bat rebuild       - 이미지 다시 빌드 후 실행
echo run_docker.bat --no-videos   - 학습 과정 영상 저장 없이 실행
echo run_simple.bat               - 간단 버전 실행
echo run_simple.bat rebuild       - 이미지 다시 빌드 후 간단 버전 실행
echo run_simple.bat --no-videos   - 학습 과정 영상 저장 없이 간단 버전 실행
echo run_quick_test.bat           - 빠른 테스트 실행
echo run_quick_test.bat rebuild   - 이미지 다시 빌드 후 빠른 테스트 실행
echo run_quick_test.bat --no-videos - 학습 과정 영상 저장 없이 빠른 테스트 실행
echo.
pause
exit /b 0 