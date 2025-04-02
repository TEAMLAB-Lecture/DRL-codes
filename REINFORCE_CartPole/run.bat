@echo off
docker build -t reinforce_cartpole .
docker run --gpus all -v %cd%:/app reinforce_cartpole 