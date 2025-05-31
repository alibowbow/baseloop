FROM python:3.10-slim

# MuseScore 및 xvfb, 그리고 필요한 추가 시스템 의존성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    musescore3 \
    xvfb \
    fontconfig \
    libgconf-2-4 \
    libnss3 \
    libasound2 \
    # ------ 추가되는 라이브러리들 ------
    libxrender1 \
    libxtst6 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxrandr2 \
    libgl1 \
    libcups2 \
    libpulse0 \
    # ---------------------------------
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MUSESCORE_PATH=/usr/bin/musescore3
ENV QT_QPA_PLATFORM=offscreen
ENV DISPLAY=:99

COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 5000

CMD ["/start.sh"]
