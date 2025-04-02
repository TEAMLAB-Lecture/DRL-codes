@echo off
docker build -t reinforce .
docker run --gpus all -v %cd%:/app reinforce 