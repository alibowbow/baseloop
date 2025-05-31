FROM python:3.10-slim

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
