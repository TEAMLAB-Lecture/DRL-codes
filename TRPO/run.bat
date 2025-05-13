@echo off
docker build -t trpo_pendulum .
docker run --gpus all -v %cd%:/app -v %cd%/videos:/app/videos trpo_pendulum 