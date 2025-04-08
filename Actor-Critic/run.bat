@echo off
docker build -t actor-critic .
docker run -it --rm -v %cd%:/app actor-critic 