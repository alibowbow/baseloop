# Dockerfile for Render.com with MuseScore support
FROM python:3.10-slim

# 시스템 의존성 설치 (MuseScore 포함)
RUN apt-get update && apt-get install -y \
    musescore3 \
    xvfb \
    fonts-liberation \
    fonts-dejavu \
    fonts-noto \
    libqt5gui5 \
    libqt5widgets5 \
    libqt5printsupport5 \
    libqt5svg5 \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# Python 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# MuseScore 환경변수 설정
ENV MUSESCORE_PATH=/usr/bin/musescore3
ENV QT_QPA_PLATFORM=offscreen
ENV DISPLAY=:99

# Music21 환경 설정
RUN python -c "from music21 import configure; configure.run()"

# Xvfb 가상 디스플레이 시작 스크립트
COPY start.sh /start.sh
RUN chmod +x /start.sh

# 포트 설정
EXPOSE 5000

# 애플리케이션 시작
CMD ["/start.sh"]
