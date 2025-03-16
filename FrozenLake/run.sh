#!/bin/bash

# 가상 디스플레이 설정
echo "Starting virtual display..."
Xvfb :1 -screen 0 1400x900x24 &
export DISPLAY=:1

# Python 스크립트 실행
echo "Running FrozenLake Q-Learning script..."
python frozen_lake_qlearning.py 