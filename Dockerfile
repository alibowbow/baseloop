FROM python:3.10-slim

# MuseScore 및 xvfb, 그리고 필요한 추가 시스템 의존성 설치
# --no-install-recommends: 추천 패키지는 설치하지 않아 이미지 크기 감소
RUN apt-get update && apt-get install -y --no-install-recommends \
    musescore3 \
    xvfb \
    fontconfig \
    # 기존에 추가했던Qt, X11, 그래픽 관련 라이브러리들
    libgconf-2-4 \
    libnss3 \
    libasound2 \
    libxrender1 \
    libxtst6 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxrandr2 \
    libgl1 \
    libcups2 \
    libpulse0 \
    # 최종적으로 추가된 라이브러리들 (더 포괄적인 호환성용)
    fonts-liberation \
    fonts-dejavu \
    fonts-noto \
    xfonts-base \
    xfonts-75dpi \
    # 다른 Qt 런타임 의존성 (혹시 빠진 것이 있다면)
    libqt5core5a \
    libqt5dbus5 \
    libqt5network5 \
    libqt5xml5 \
    libqt5svg5 \
    libqt5opengl5 \
    # OpenGL/SDL 관련 (일부 시스템에서 필요)
    libsdl1.2debian \
    libglu1-mesa \
    # GStreamer (미디어 관련, Qt 앱에서 필요할 수 있음)
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    # 이미지 포맷 지원 (Qt에서 로딩할 수 있는 이미지 포맷용)
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    # 추가 X11/Qt 관련 라이브러리
    libxkbcommon-x11-0 \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    # DBus 관련 (Qt 앱에서 필요할 수 있음)
    dbus-x11 \
    # -----------------------------------------------------
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 폰트 캐시 재생성 (헤드리스 환경에서 폰트 문제 방지)
RUN fc-cache -f -v

# 작업 디렉토리 설정
WORKDIR /app

# Python 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# MuseScore 환경변수 설정 (Music21 연동용)
ENV MUSESCORE_PATH=/usr/bin/musescore3
ENV QT_QPA_PLATFORM=offscreen
ENV DISPLAY=:99

# Xvfb 가상 디스플레이 시작 스크립트 복사 및 실행 권한 부여
COPY start.sh /start.sh
RUN chmod +x /start.sh

# 포트 설정
EXPOSE 5000

# 애플리케이션 시작 (컨테이너가 시작될 때 start.sh 스크립트를 실행합니다.)
# start.sh 스크립트 내부에서 xvfb-run을 사용하여 최종 앱을 실행합니다.
CMD ["/start.sh"]
