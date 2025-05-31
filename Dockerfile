#!/bin/bash
# start.sh - Render.com용 시작 스크립트 (최종)

# xvfb-run을 사용하여 Flask 앱을 가상 디스플레이 환경에서 실행합니다.
# `exec` 명령어는 현재 쉘 프로세스를 Gunicorn 프로세스로 대체하여,
# Render.com이 Gunicorn 프로세스에 직접 종료 신호를 보낼 수 있도록 합니다.
exec xvfb-run --auto-display gunicorn --bind 0.0.0.0:$PORT app:app

# 참고: 만약 Gunicorn 없이 `python app.py`로 앱을 실행한다면,
# 다음처럼 변경하세요:
# exec xvfb-run --auto-display python app.py
