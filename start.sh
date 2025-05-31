#!/bin/bash
# start.sh - Render.com용 시작 스크립트

# Xvfb 가상 디스플레이 시작 (GUI 없는 환경에서 MuseScore 실행용)
Xvfb :99 -screen 0 1024x768x24 &

# 잠시 대기 (Xvfb 초기화)
sleep 2

# Music21 설정 확인 및 초기화
python -c "
from music21 import environment
env = environment.Environment()
env['musescoreDirectPNGPath'] = '/usr/bin/musescore3'
env['graphicsPath'] = '/usr/bin/musescore3'
print('Music21 MuseScore 경로 설정 완료')
"

# Gunicorn으로 Flask 앱 시작
exec gunicorn --bind 0.0.0.0:$PORT app:app
