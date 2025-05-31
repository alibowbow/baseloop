#!/bin/bash
# start.sh - Render.com용 시작 스크립트

# Xvfb 가상 디스플레이 시작 (GUI 없는 환경에서 MuseScore 실행용)
# 더 상세한 설정과 로깅 추가
echo "Starting Xvfb virtual display..."
Xvfb :99 -screen 0 1280x1024x24 -ac +extension GLX +render -noreset &
XVFB_PID=$!

# Xvfb가 완전히 시작될 때까지 대기
sleep 5

# Xvfb 상태 확인
if ps -p $XVFB_PID > /dev/null; then
    echo "Xvfb started successfully (PID: $XVFB_PID)"
else
    echo "ERROR: Xvfb failed to start!"
fi

# 환경 변수 설정
export DISPLAY=:99
export QT_QPA_PLATFORM=offscreen
export QT_DEBUG_PLUGINS=1  # Qt 디버깅 활성화

# MuseScore 테스트 실행
echo "Testing MuseScore installation..."
if command -v musescore3 &> /dev/null; then
    echo "MuseScore3 found at: $(which musescore3)"
    musescore3 --version 2>&1 || echo "MuseScore version check failed"
else
    echo "WARNING: MuseScore3 not found in PATH"
fi

# Music21 설정 확인 및 초기화
echo "Configuring Music21..."
python -c "
import os
os.environ['DISPLAY'] = ':99'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

try:
    from music21 import environment
    env = environment.Environment()
    env['musescoreDirectPNGPath'] = '/usr/bin/musescore3'
    env['musicxmlPath'] = '/usr/bin/musescore3'
    env['graphicsPath'] = '/usr/bin/musescore3'
    env['pdfPath'] = '/usr/bin/musescore3'
    
    print('Music21 environment settings:')
    print(f'  musescoreDirectPNGPath: {env[\"musescoreDirectPNGPath\"]}')
    print(f'  musicxmlPath: {env[\"musicxmlPath\"]}')
    print(f'  DISPLAY: {os.environ.get(\"DISPLAY\")}')
    print(f'  QT_QPA_PLATFORM: {os.environ.get(\"QT_QPA_PLATFORM\")}')
    print('Music21 MuseScore configuration completed successfully')
except Exception as e:
    print(f'ERROR configuring Music21: {e}')
"

# 포트 설정 확인
echo "Starting Gunicorn on port ${PORT:-10000}..."

# Gunicorn으로 Flask 앱 시작 (Xvfb는 이미 백그라운드에서 실행 중)
exec gunicorn --bind 0.0.0.0:${PORT:-10000} --timeout 120 --workers 1 app:app
