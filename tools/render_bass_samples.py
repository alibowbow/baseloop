#!/usr/bin/env python3
"""로컬 베이스 샘플 렌더러 — 확장 Karplus-Strong 플럭 스트링 모델.

외부 CDN(GM 사운드폰트) 의존을 없애기 위해, 물리 모델링 기반의
일렉트릭 베이스 음색을 anchor pitch별로 렌더해 static/bass_samples/에 WAV로 저장한다.
Tone.Sampler가 이 anchor들 사이를 피치시프트해 전 음역을 커버한다.

의존성: numpy (stdlib wave 로 저장 — 인코더 불필요)
사용:   python tools/render_bass_samples.py
"""
import math
import os
import wave

import numpy as np

SR = 22050          # 베이스는 고역이 거의 없어 22.05kHz로 충분(용량 절반)
DUR = 2.2           # 자연 감쇠 포함 길이(초)
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "static", "bass_samples")

# anchor pitch: 이름 → (MIDI, 파일명).  '#' 은 URL 프래그먼트 문자라 파일명은 's' 표기(CDN 관례).
# E1~G4 를 ~5반음 간격으로 커버 → 앱이 내는 전 음역(옥타브 0~4, 최고 C5≈72)에서
# Tone.Sampler가 가장 가까운 앵커로부터 ≤5반음만 피치시프트(치핑 없음).
ANCHORS = [
    ("E1", 28, "E1.wav"),
    ("A1", 33, "A1.wav"),
    ("D2", 38, "D2.wav"),
    ("G2", 43, "G2.wav"),
    ("C3", 48, "C3.wav"),
    ("F3", 53, "F3.wav"),
    ("A3", 57, "A3.wav"),
    ("D4", 62, "D4.wav"),
    ("G4", 67, "G4.wav"),
]


def midi_to_freq(m):
    return 440.0 * (2.0 ** ((m - 69) / 12.0))


def biquad(x, b0, b1, b2, a1, a2):
    """Direct Form I 바이쿼드 필터 (a0=1 정규화)."""
    y = np.zeros_like(x)
    x1 = x2 = y1 = y2 = 0.0
    for n in range(len(x)):
        xn = x[n]
        yn = b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        y[n] = yn
        x2, x1 = x1, xn
        y2, y1 = y1, yn
    return y


def rbj_lowpass(fc, q=0.707):
    w0 = 2 * math.pi * fc / SR
    alpha = math.sin(w0) / (2 * q)
    cw = math.cos(w0)
    b0 = (1 - cw) / 2; b1 = 1 - cw; b2 = (1 - cw) / 2
    a0 = 1 + alpha; a1 = -2 * cw; a2 = 1 - alpha
    return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)


def rbj_highpass(fc, q=0.707):
    w0 = 2 * math.pi * fc / SR
    alpha = math.sin(w0) / (2 * q)
    cw = math.cos(w0)
    b0 = (1 + cw) / 2; b1 = -(1 + cw); b2 = (1 + cw) / 2
    a0 = 1 + alpha; a1 = -2 * cw; a2 = 1 - alpha
    return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)


def rbj_peaking(fc, gain_db, q=0.7):
    a = 10 ** (gain_db / 40)
    w0 = 2 * math.pi * fc / SR
    alpha = math.sin(w0) / (2 * q)
    cw = math.cos(w0)
    b0 = 1 + alpha * a; b1 = -2 * cw; b2 = 1 - alpha * a
    a0 = 1 + alpha / a; a1 = -2 * cw; a2 = 1 - alpha / a
    return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)


def one_pole_phase_delay(d, w0):
    """루프 로우패스 H(z)=(1-d)/(1-d·z⁻¹)가 각주파수 w0에서 더하는 위상 지연(샘플)."""
    return math.atan2(d * math.sin(w0), 1.0 - d * math.cos(w0)) / w0


def karplus_strong(freq, decay=0.9975, damp=0.5, seed=0):
    """확장 Karplus-Strong: 분수 지연 + 루프 로우패스로 자연스러운 플럭 감쇠.

    freq   : 목표 기음(Hz) — 루프 필터 위상 지연을 보정해 정확 튜닝(분수 지연 선형보간)
    decay  : 매 루프 진폭 유지율 — 클수록 서스테인 김
    damp   : 루프 로우패스 계수(클수록 고역 빨리 죽어 어두움)
    """
    n_samples = int(SR * DUR)
    w0 = 2.0 * math.pi * freq / SR
    pd = one_pole_phase_delay(damp, w0)          # 루프 필터가 더하는 지연
    delay = SR / freq - pd                        # 필터 지연만큼 보정 → 정확한 주기
    if delay < 2.0:
        delay = 2.0
    Ni = int(math.floor(delay))
    frac = delay - Ni                             # 분수 지연 선형보간 가중치

    rng = np.random.default_rng(seed)
    # 여기(pluck) = 필터링된 노이즈 버스트로 히스토리 초기화
    exc = rng.uniform(-1.0, 1.0, Ni + 2)
    exc = np.convolve(exc, [0.25, 0.5, 0.25], mode="same")  # 살짝 둥근 피크(핑거 질감)

    y = np.zeros(n_samples + Ni + 2)
    y[: Ni + 2] = exc
    lp = 0.0
    for n in range(Ni + 2, len(y)):
        s = (1.0 - frac) * y[n - Ni] + frac * y[n - Ni - 1]  # 분수 지연
        lp = (1.0 - damp) * s + damp * lp                    # 루프 로우패스(원폴)
        y[n] = decay * lp
    return y[Ni + 2:]


def render_note(midi, seed):
    freq = midi_to_freq(midi)
    # 저음일수록 서스테인 길고 밝기 낮게(실제 베이스 저역 특성)
    decay = 0.9985 if midi < 36 else (0.9978 if midi < 46 else 0.9970)
    damp = 0.52 if midi < 36 else 0.46
    sig = karplus_strong(freq, decay=decay, damp=damp, seed=seed)

    # 바디/픽업 공명 + 대역 정리
    sig = biquad(sig, *rbj_highpass(32))            # DC/럼블 제거
    sig = biquad(sig, *rbj_peaking(115, 3.0, 0.8))  # 로우미드 바디
    sig = biquad(sig, *rbj_peaking(700, -2.0, 1.0)) # 박스 울림 살짝 컷
    sig = biquad(sig, *rbj_lowpass(4200))           # 초고역 정리

    # 어택 트랜지언트(픽업 클릭) 살짝 + 진폭 엔벨로프
    t = np.arange(len(sig)) / SR
    attack = 1.0 - np.exp(-t / 0.004)               # 4ms 어택
    body = 0.55 + 0.45 * np.exp(-t / 1.6)           # 완만한 세틀
    env = attack * body
    # 끝 30ms 페이드아웃(클릭 방지)
    fade = int(0.03 * SR)
    if fade > 0:
        env[-fade:] *= np.linspace(1.0, 0.0, fade)
    sig = sig * env

    # 정규화
    peak = np.max(np.abs(sig)) or 1.0
    sig = sig / peak * 0.92
    return sig


def write_wav(path, sig):
    pcm = np.clip(sig, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    total = 0
    for i, (name, midi, fname) in enumerate(ANCHORS):
        sig = render_note(midi, seed=1000 + i)
        path = os.path.join(OUT_DIR, fname)
        write_wav(path, sig)
        sz = os.path.getsize(path)
        total += sz
        print(f"  {name:>3} (midi {midi}, {midi_to_freq(midi):6.1f}Hz) → {fname}  {sz//1024}KB")
    print(f"총 {len(ANCHORS)}개 · {total//1024}KB → {OUT_DIR}")


if __name__ == "__main__":
    main()
