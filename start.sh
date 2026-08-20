#!/bin/bash
# start.sh - Render.com용 시작 스크립트 (디버깅 강화)

echo "=== Starting application setup ==="

# Xvfb 가상 디스플레이 시작 (GUI 없는 환경에서 MuseScore 실행용)
echo "Starting Xvfb virtual display..."
Xvfb :99 -screen 0 1280x1024x24 -ac +extension GLX +render -noreset &
XVFB_PID=$!

# Xvfb가 완전히 시작될 때까지 대기
sleep 3

# Xvfb 상태 확인
if ps -p $XVFB_PID > /dev/null; then
    echo "✓ Xvfb started successfully (PID: $XVFB_PID)"
else
    echo "✗ ERROR: Xvfb failed to start!"
fi

# 환경 변수 설정
export DISPLAY=:99
export QT_QPA_PLATFORM=offscreen
export PYTHONUNBUFFERED=1  # Python 출력 버퍼링 비활성화

echo "Environment variables set:"
echo "  DISPLAY=$DISPLAY"
echo "  QT_QPA_PLATFORM=$QT_QPA_PLATFORM"
echo "  PORT=${PORT:-10000}"

# Python 및 패키지 확인
echo ""
echo "=== Python environment check ==="
python --version
echo ""
echo "Installed packages:"
pip list
echo ""

# 앱 파일 확인
echo "=== Application files check ==="
for required_file in app.py enhanced_app.py; do
    if [ -f "$required_file" ]; then
        echo "✓ $required_file found"
    else
        echo "✗ $required_file NOT FOUND!"
        ls -la
        exit 1
    fi
done

# Python 모듈 임포트 테스트
echo ""
echo "=== Testing Python imports ==="
python -c "
import sys
print(f'Python path: {sys.path}')
print('Testing imports...')

try:
    import flask
    print('✓ Flask imported successfully')
except Exception as e:
    print(f'✗ Flask import failed: {e}')

try:
    import numpy
    print('✓ NumPy imported successfully')
except Exception as e:
    print(f'✗ NumPy import failed: {e}')

try:
    import music21
    print('✓ Music21 imported successfully')
except Exception as e:
    print(f'✗ Music21 import failed: {e}')

try:
    import mido
    print('✓ Mido imported successfully')
except Exception as e:
    print(f'✗ Mido import failed: {e}')

print('Import tests completed.')
"

# 향상 레이어를 포함한 앱 직접 테스트
echo ""
echo "=== Testing enhanced_app.py directly ==="
python -c "
try:
    import enhanced_app
    print('✓ enhanced_app.py imported successfully')
    print(f'Flask app: {enhanced_app.app}')
except Exception as e:
    print(f'✗ Failed to import enhanced_app.py: {e}')
    import traceback
    traceback.print_exc()
    raise
"

# MuseScore 확인
echo ""
echo "=== MuseScore check ==="
if command -v musescore3 &> /dev/null; then
    echo "✓ MuseScore3 found at: $(which musescore3)"
else
    echo "✗ MuseScore3 not found in PATH"
fi

# 포트 설정 확인
echo ""
echo "=== Starting Gunicorn ==="
echo "Port: ${PORT:-10000}"
echo "Command: gunicorn --bind 0.0.0.0:${PORT:-10000} --timeout 120 --workers 1 --log-level debug enhanced_app:app"
echo ""

# Gunicorn으로 향상 레이어가 포함된 Flask 앱 시작
exec gunicorn --bind 0.0.0.0:${PORT:-10000} --timeout 120 --workers 1 --log-level debug --capture-output --enable-stdio-inheritance enhanced_app:app
