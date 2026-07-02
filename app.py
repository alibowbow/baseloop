import numpy as np
from scipy.io.wavfile import write as write_wav
from flask import Flask, render_template, request, send_file, Response, jsonify
import io
import re
import os
import glob
import traceback
import logging
import tempfile
import shutil
import base64
import subprocess

# Music21 라이브러리 (악보 생성)
try:
    from music21 import stream, note, meter, tempo, clef, bar, duration, pitch, key, layout
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False
    print("Warning: music21 라이브러리가 설치되지 않았습니다. 전문 악보 생성을 사용할 수 없습니다.")

# ─── Music21 MuseScore 환경 설정 함수 ───────────────────────
_M21_CONFIGURED = False

# 자동 탐색 시 시도할 MuseScore 실행 파일 후보 경로
MUSESCORE_CANDIDATE_PATHS = [
    "/usr/bin/musescore3",
    "/usr/bin/musescore",
    "/usr/local/bin/musescore3",
    "/usr/local/bin/musescore",
    "/snap/bin/musescore",
]


def find_musescore_path() -> str:
    """
    MuseScore 실행 파일 경로를 결정합니다.
    MUSESCORE_PATH 환경 변수를 우선하고, 없으면 알려진 후보 경로를 탐색합니다.
    찾지 못하면 기본값(/usr/bin/musescore3)을 반환합니다.
    """
    env_path = os.getenv("MUSESCORE_PATH")
    if env_path:
        logger.info(f"환경 변수 MUSESCORE_PATH 사용: {env_path}")
        return env_path

    for path in MUSESCORE_CANDIDATE_PATHS:
        if os.path.exists(path) and os.access(path, os.X_OK):
            logger.info(f"MuseScore 발견: {path}")
            return path

    default_path = "/usr/bin/musescore3"
    logger.warning(f"MuseScore를 찾을 수 없습니다. 기본 경로 사용: {default_path}")
    return default_path


def setup_music21_environment() -> bool:
    """
    MuseScore 경로와 Qt 헤드리스 환경 변수를 Music21에 등록합니다.
    """
    global _M21_CONFIGURED
    logger.info(f"Music21 환경 설정 시도. Music21_AVAILABLE: {MUSIC21_AVAILABLE}, 이미 설정됨: {_M21_CONFIGURED}")

    if _M21_CONFIGURED or not MUSIC21_AVAILABLE:
        if not MUSIC21_AVAILABLE:
            logger.warning("Music21 라이브러리를 사용할 수 없어 환경 설정을 건너뜁니다.")
        return _M21_CONFIGURED

    try:
        from music21 import environment
        us = environment.UserSettings()

        musescore_path = find_musescore_path()

        # Music21에 MuseScore 경로 설정
        us["musescoreDirectPNGPath"] = musescore_path
        us["musicxmlPath"] = musescore_path
        us["graphicsPath"] = musescore_path
        us["pdfPath"] = musescore_path

        logger.info(f"Music21 UserSettings에 MuseScore 경로 설정:")
        logger.info(f"  - musescoreDirectPNGPath: {us['musescoreDirectPNGPath']}")
        logger.info(f"  - musicxmlPath: {us['musicxmlPath']}")
        logger.info(f"  - graphicsPath: {us['graphicsPath']}")
        logger.info(f"  - pdfPath: {us['pdfPath']}")

        # Qt/X11 환경 변수 설정
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        os.environ.setdefault("DISPLAY", ":99")
        os.environ.setdefault("QT_XCB_GL_INTEGRATION", "none")  # OpenGL 통합 비활성화
        os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.xcb.xcberror=false")  # XCB 에러 로깅 비활성화

        # Music21 디버그 로깅 활성화
        us['debug'] = True
        us['warnings'] = True  # 경고 메시지 활성화

        logger.info(f"[Music21 설정 완료]")
        logger.info(f"  - MuseScore 경로: {musescore_path}")
        logger.info(f"  - DISPLAY: {os.environ.get('DISPLAY')}")
        logger.info(f"  - QT_QPA_PLATFORM: {os.environ.get('QT_QPA_PLATFORM')}")
        
        # MuseScore 실행 가능 여부 테스트
        try:
            import subprocess
            cmd_version_check = [musescore_path, "--version"]
            logger.info(f"MuseScore 버전 확인 실행: {' '.join(cmd_version_check)}")
            result = subprocess.run(cmd_version_check,
                                  capture_output=True, text=True, timeout=10) # Increased timeout slightly

            # Log stdout and stderr regardless of return code
            if result.stdout:
                logger.info(f"MuseScore 버전 STDOUT: {result.stdout.strip()}")
            else:
                logger.info("MuseScore 버전 STDOUT: (내용 없음)")

            if result.stderr:
                logger.warning(f"MuseScore 버전 STDERR: {result.stderr.strip()}")
            else:
                logger.info("MuseScore 버전 STDERR: (내용 없음)")

            if result.returncode == 0:
                logger.info(f"MuseScore 버전 확인 성공 (Return Code: {result.returncode})")
            else:
                logger.warning(f"MuseScore 버전 확인 실패 (Return Code: {result.returncode})")
        except Exception as e:
            logger.warning(f"MuseScore 테스트 실행 중 예외 발생: {e}")
        
        _M21_CONFIGURED = True
        return True
    except Exception as e:
        logger.error(f"[Music21 설정 오류] MuseScore 경로 설정 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        return False

# MIDI 라이브러리
try:
    import mido
    from mido import MidiFile, MidiTrack, Message
    MIDI_AVAILABLE = True
except ImportError:
    MIDI_AVAILABLE = False
    print("Warning: mido 라이브러리가 설치되지 않았습니다. MIDI 파일 생성을 사용할 수 없습니다.")

# 로컬 개발용 .env 파일 로딩
try:
    from dotenv import load_dotenv
    load_dotenv() 
except ImportError:
    print("Warning: python-dotenv 라이브러리가 설치되지 않았습니다.")

# 로깅 설정
is_production = os.environ.get('RENDER') == 'true' or os.environ.get('ENVIRONMENT') == 'production'

if is_production:
    # 프로덕션 환경: WARNING 이상만 로깅
    logging.basicConfig(level=logging.WARNING)
    # Gunicorn DEBUG 로그 완전히 비활성화
    logging.getLogger('gunicorn.error').setLevel(logging.ERROR)
    logging.getLogger('gunicorn.access').setLevel(logging.ERROR)
else:
    # 개발 환경: INFO 이상 로깅
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

# Flask 앱 초기화
app = Flask(__name__, template_folder='templates', static_folder='static')

# --- 오디오 생성 관련 상수 및 기본 함수 ---
SAMPLE_RATE = 44100
MAX_AMPLITUDE = 0.5 * (2**15 - 1)

def get_note_frequency(note_name_chromatic, octave):
    """
    음표 이름 (C, C#, D, Eb 등 완전한 이름)과 옥타브를 받아 주파수를 계산합니다.
    note_name_chromatic은 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B' 중 하나여야 합니다.
    """
    chromatic_notes_midi_offset = {
        'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
        'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
    }
    
    note_name_std = note_name_chromatic.replace('Db', 'C#').replace('Eb', 'D#') \
                                       .replace('Gb', 'F#').replace('Ab', 'G#') \
                                       .replace('Bb', 'A#').replace('Cb', 'B') \
                                       .replace('B#', 'C') 
    
    if note_name_std not in chromatic_notes_midi_offset:
        raise ValueError(f"유효하지 않은 크로매틱 음표 이름: {note_name_chromatic} (변환 후: {note_name_std})")

    midi_note_number = (octave * 12) + chromatic_notes_midi_offset[note_name_std]
    
    A4_MIDI = 69
    A4_FREQ = 440.0
    
    return A4_FREQ * (2 ** ((midi_note_number - A4_MIDI) / 12.0))


def generate_note_waveform(frequency, duration_seconds, sample_rate, amplitude):
    """부드럽고 자연스러운 베이스 기타 사운드를 생성합니다."""
    num_samples = int(duration_seconds * sample_rate)
    if num_samples <= 0:
        return np.array([])
        
    t = np.linspace(0, duration_seconds, num_samples, endpoint=False)
    
    # 더 부드러운 베이스 기타 하모닉 구조
    fundamental = np.sin(2 * np.pi * frequency * t)
    harmonic_2 = 0.4 * np.sin(2 * np.pi * frequency * 2 * t)  # 2배음 감소
    harmonic_3 = 0.2 * np.sin(2 * np.pi * frequency * 3 * t)  # 3배음 감소
    harmonic_4 = 0.1 * np.sin(2 * np.pi * frequency * 4 * t)  # 4배음 감소
    
    # 서브 하모닉은 더 부드럽게
    sub_harmonic = 0.15 * np.sin(2 * np.pi * frequency * 0.5 * t) * np.exp(-t * 0.5)
    
    # 모든 하모닉 합성 (포화 효과 제거)
    waveform = fundamental + harmonic_2 + harmonic_3 + harmonic_4 + sub_harmonic
    
    # 부드러운 ADSR 엔벨로프
    attack_percent = 0.01   # 조금 더 부드러운 어택
    decay_percent = 0.08    # 더 부드러운 디케이
    sustain_level = 0.8     # 더 높은 서스테인
    release_percent = 0.3   # 더 부드러운 릴리스

    attack_samples = int(num_samples * attack_percent)
    decay_samples = int(num_samples * decay_percent)
    sustain_samples = int(num_samples * (1 - attack_percent - decay_percent - release_percent))
    release_samples = int(num_samples * release_percent)
    
    envelope = np.ones(num_samples)

    # 부드러운 Attack (S-커브 사용)
    if attack_samples > 0:
        attack_curve = np.linspace(0, 1, attack_samples)
        # S-커브로 더 부드러운 어택
        attack_curve = attack_curve * attack_curve * (3 - 2 * attack_curve)
        envelope[:attack_samples] = attack_curve
    
    # 부드러운 Decay
    if decay_samples > 0:
        start_decay = attack_samples
        end_decay = start_decay + decay_samples
        decay_curve = np.linspace(1, sustain_level, decay_samples)
        # 지수적 감소로 더 자연스럽게
        decay_curve = sustain_level + (1 - sustain_level) * np.exp(-3 * np.linspace(0, 1, decay_samples))
        envelope[start_decay:end_decay] = decay_curve
    
    # Sustain
    if sustain_samples > 0:
        start_sustain = attack_samples + decay_samples
        end_sustain = start_sustain + sustain_samples
        envelope[start_sustain:end_sustain] = sustain_level
    
    # 부드러운 Release (지수적 감소)
    if release_samples > 0:
        start_release = num_samples - release_samples
        release_curve = np.linspace(0, 1, release_samples)
        # 지수적 감소로 자연스러운 릴리스
        release_curve = sustain_level * np.exp(-3 * release_curve)
        envelope[start_release:] = release_curve
    
    # 파형에 엔벨로프 적용
    waveform *= envelope
    
    # 클리핑 방지를 위한 소프트 제한
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        # 소프트 클리핑 (tanh 함수 사용)
        waveform = np.tanh(waveform * 0.8) * amplitude * 0.9
    else:
        waveform *= amplitude * 0.9
    
    # 베이스 주파수 강화 (더 부드럽게)
    if frequency < 150:  # 베이스 주파수 영역
        waveform *= 1.1  # 저음역 부스트 감소
    
    return waveform

def parse_note_sequence_string(notes_sequence_str):
    """
    "C2 1.0, G#2 0.5, R 1.0, Eb3 0.25" 형태의 문자열을 파싱하여 
    [('C', 2, 1.0, False, ''), ('G', 2, 0.5, False, '#'), ('R', 4, 1.0, True, ''), ('E', 3, 0.25, False, 'b')] 형태로 변환.
    is_rest 플래그와 accidental (예: '#', 'b') 추가.
    """
    if not notes_sequence_str or not notes_sequence_str.strip():
        raise ValueError("악보 시퀀스가 비어 있습니다.")
        
    parsed_notes = []
    for note_entry in notes_sequence_str.split(','):
        note_entry = note_entry.strip()
        if not note_entry:
            continue
        
        match_note = re.match(r"([A-Ga-g])(#|b)?(\d+)\s+(\d+(?:\.\d+)?)", note_entry, re.IGNORECASE)
        if match_note:
            note_name = match_note.group(1).upper()
            accidental = match_note.group(2) if match_note.group(2) else ''
            octave = int(match_note.group(3))
            duration = float(match_note.group(4))
            
            if not (0 <= octave <= 8):
                raise ValueError(f"옥타브는 0-8 범위여야 합니다: {octave} (음표: {note_entry})")
            if not (0 < duration <= 8):
                raise ValueError(f"음표 길이는 0보다 크고 8보다 작거나 같아야 합니다: {duration} (음표: {note_entry})")
                
            parsed_notes.append((note_name, octave, duration, False, accidental))
            continue

        match_rest = re.match(r"R\s+(\d+(?:\.\d+)?)", note_entry, re.IGNORECASE)
        if match_rest:
            duration = float(match_rest.group(1))
            if not (0 < duration <= 8):
                raise ValueError(f"쉼표 길이는 0보다 크고 8보다 작거나 같아야 합니다: {duration} (쉼표: {note_entry})")
            parsed_notes.append(('R', 4, duration, True, '')) 
            continue
        
        raise ValueError(f"유효하지 않은 음표/쉼표 형식: '{note_entry}'. 'C2 1.0' 또는 'R 1.0' 형식이어야 합니다.")
    
    if not parsed_notes:
        raise ValueError("파싱된 음표가 없습니다. 올바른 형식으로 입력해주세요.")
        
    return parsed_notes


def create_bass_loop_from_parsed_sequence(notes_sequence, bpm, num_loops):
    """
    파싱된 음표 시퀀스 ([('C', 2, 1.0, False, ''), ...])를 기반으로 오디오 버퍼 생성
    """
    if not notes_sequence:
        raise ValueError("음표 시퀀스가 비어 있습니다.")
    
    if bpm <= 0 or bpm > 300:
        raise ValueError(f"BPM은 1-300 범위여야 합니다: {bpm}")
        
    if num_loops <= 0 or num_loops > 100:
        raise ValueError(f"루프 횟수는 1-100 범위여야 합니다: {num_loops}")
    
    quarter_note_duration = 60 / bpm 
    full_loop_waveform = np.array([])

    logger.info(f"생성 중: {len(notes_sequence)}개 음표, BPM {bpm}, {num_loops}회 반복")

    for note_info in notes_sequence:
        note_name, octave_val, duration_units, is_rest, accidental = note_info 
        try:
            if is_rest:
                freq = 0 
            else:
                note_name_for_freq = note_name
                if accidental:
                    note_name_for_freq += accidental 

                freq = get_note_frequency(note_name_for_freq, octave_val)
            
            actual_duration = duration_units * quarter_note_duration
            note_waveform = generate_note_waveform(freq, actual_duration, SAMPLE_RATE, MAX_AMPLITUDE)
            full_loop_waveform = np.concatenate((full_loop_waveform, note_waveform))
        except Exception as e:
            logger.error(f"오디오 음표 생성 실패 {note_name}{accidental}{octave_val} ({duration_units}박): {e}")
            silence_samples = int(duration_units * quarter_note_duration * SAMPLE_RATE)
            silence = np.zeros(silence_samples)
            full_loop_waveform = np.concatenate((full_loop_waveform, silence))
    
    if len(full_loop_waveform) == 0:
        raise ValueError("생성된 오디오 데이터가 없습니다.")
    
    final_waveform = np.tile(full_loop_waveform, num_loops)
    
    # 부드러운 정규화 (RMS 기반)
    rms = np.sqrt(np.mean(final_waveform**2))
    if rms > 0:
        # RMS 정규화로 더 일관된 볼륨
        target_rms = MAX_AMPLITUDE * 0.3  # 더 보수적인 볼륨
        final_waveform = final_waveform * (target_rms / rms)
    
    # 하드 클리핑 방지
    final_waveform = np.clip(final_waveform, -MAX_AMPLITUDE * 0.95, MAX_AMPLITUDE * 0.95)
    
    audio_data_int16 = final_waveform.astype(np.int16)

    buffer = io.BytesIO()
    write_wav(buffer, SAMPLE_RATE, audio_data_int16)
    buffer.seek(0)
    
    return buffer

# --- Music21 기반 전문 악보 생성 함수 ---
def generate_music21_score_with_fallback(
    notes_sequence, 
    bpm,
    key_signature="C",
):
    """
    Music21 + MuseScore → SVG 생성. 실패 시 텍스트 악보 반환.
    (svg_data_base64, text_score) 형태 튜플을 돌려줍니다.
    """
    if not MUSIC21_AVAILABLE:
        logger.warning("Music21 라이브러리가 설치되지 않아 전문 악보 생성을 할 수 없습니다.")
        return None, generate_text_score(notes_sequence, bpm)

    if not setup_music21_environment():
        logger.error("Music21 환경 설정 실패. MuseScore SVG 생성을 건너뛰고 텍스트 악보를 생성합니다.") # Changed PNG to SVG
        return None, generate_text_score(notes_sequence, bpm)

    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()
        logger.info(f"악보 생성을 위한 임시 디렉토리 생성: {temp_dir}")

        logger.info("Music21 악보 객체 생성 시작.")
        score = stream.Score()
        score.append(tempo.MetronomeMark(number=bpm))
        score.append(meter.TimeSignature("4/4"))
        score.append(key.Key(key_signature))

        part = stream.Part()
        part.append(clef.BassClef())
        
        # 베이스 기타 악기 설정
        from music21 import instrument
        bass_instrument = instrument.ElectricBass()
        bass_instrument.instrumentName = 'Bass Guitar'
        bass_instrument.midiProgram = 32  # Acoustic Bass
        part.insert(0, bass_instrument)

        for n_name, octv, dur, is_rest, accidental in notes_sequence:
            if is_rest:
                part.append(note.Rest(quarterLength=dur))
            else:
                try:
                    p = pitch.Pitch(f"{n_name}{accidental}") 
                    n = note.Note(p)
                    n.octave = octv
                    n.duration = duration.Duration(quarterLength=dur)
                    part.append(n)
                except Exception as e:
                    logger.warning(f"Music21 음표 변환 실패 → Rest 처리: {n_name}{accidental}{octv} ({dur}박) / {e}")
                    part.append(note.Rest(quarterLength=dur))

        score.append(part)
        logger.info("Music21 악보 객체 생성 완료. MuseScore SVG 생성을 시도합니다.")

        svg_path = os.path.join(temp_dir, "score.svg") # Changed to SVG

        try:
            # MuseScore로 SVG 직접 생성 (개선된 방법)
            xml_path_for_svg = os.path.join(temp_dir, "score_temp.xml")
            logger.info(f"SVG 생성을 위한 MusicXML 저장: {xml_path_for_svg}")
            score.write("musicxml", fp=xml_path_for_svg)
            
            if not os.path.exists(xml_path_for_svg) or os.path.getsize(xml_path_for_svg) == 0:
                logger.error(f"MusicXML 파일 생성 실패: {xml_path_for_svg}")
                raise RuntimeError(f"Failed to create MusicXML file at {xml_path_for_svg}")
            
            # MuseScore로 SVG 직접 생성
            musescore_exec_path = find_musescore_path()
            cmd_svg = [musescore_exec_path, "-o", svg_path, xml_path_for_svg]
            
            logger.info(f"MuseScore SVG 생성 명령: {' '.join(cmd_svg)}")
            result_svg = subprocess.run(cmd_svg, capture_output=True, text=True, timeout=30)
            
            logger.info(f"MuseScore SVG 생성 Return Code: {result_svg.returncode}")
            if result_svg.stdout:
                logger.info(f"MuseScore SVG STDOUT: {result_svg.stdout.strip()}")
            if result_svg.stderr:
                logger.warning(f"MuseScore SVG STDERR: {result_svg.stderr.strip()}")
            
            if result_svg.returncode != 0:
                logger.warning(f"MuseScore SVG 생성 경고 (Return Code: {result_svg.returncode}), 하지만 파일 확인 중...")
            
            # 파일 생성 확인 (약간의 대기 시간 추가)
            import time
            time.sleep(0.5)

            # MuseScore는 페이지 번호 접미사(예: score-1.svg)로 저장할 수 있으므로 보정한다.
            if not os.path.exists(svg_path):
                svg_candidates = sorted(glob.glob(os.path.join(temp_dir, "*.svg")))
                if svg_candidates:
                    logger.info(f"기대한 SVG 경로({svg_path})가 없어 대체 파일 사용: {svg_candidates[0]}")
                    svg_path = svg_candidates[0]

            if not os.path.exists(svg_path):
                raise RuntimeError(f"SVG 파일이 생성되지 않음: {svg_path}")
            
            if os.path.getsize(svg_path) == 0:
                raise RuntimeError(f"SVG 파일이 비어있음: {svg_path}")
            
            # SVG 파일 내용 검증
            with open(svg_path, 'r', encoding='utf-8') as f:
                svg_content = f.read(200)  # 처음 200자만 확인
                if not svg_content.strip().startswith('<?xml') and not svg_content.strip().startswith('<svg'):
                    raise RuntimeError(f"유효하지 않은 SVG 파일: {svg_path}")
            
            logger.info(f"SVG 생성 성공: {svg_path}, 크기: {os.path.getsize(svg_path)} 바이트")

        except Exception as e_svg:
            logger.error(f"SVG 생성 실패: {e_svg}")
            # 오류 시 텍스트 악보로 폴백
            raise RuntimeError(f"SVG 생성 실패: {e_svg}")


        # Check if SVG file was successfully created by any method
        if os.path.exists(svg_path) and os.path.getsize(svg_path) > 0:
            logger.info(f"MuseScore SVG 파일 생성 성공: {svg_path}, 크기: {os.path.getsize(svg_path)} 바이트")
            with open(svg_path, "rb") as f_svg: # Read as binary for base64
                svg_data_content = f_svg.read()
            return base64.b64encode(svg_data_content).decode('utf-8'), None
        else:
            # This path should ideally be caught by specific exceptions above, but as a final catch
            # This will be caught by the outer try-except block, leading to text score generation
            raise RuntimeError(f"SVG 파일 생성 최종 실패: 파일이 없거나 비어 있습니다. 경로: {svg_path}")

    except Exception as e:
        logger.error(f"SVG 악보 이미지 생성 중 오류 발생: {e}") # Changed from "악보 생성 중 예상치 못한 오류 발생"
        logger.error(traceback.format_exc())
        text_score = generate_text_score(notes_sequence, bpm)
        return None, text_score
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir) 

def generate_text_score(notes_sequence, bpm):
    """텍스트 기반 악보 표현 (Music21 실패시 대안)"""
    text_lines = []
    text_lines.append(f"베이스 라인 악보 (BPM: {bpm})")
    text_lines.append("=" * 40)
    text_lines.append("")
    
    current_measure = 1
    current_beats = 0
    measure_notes_str_list = [] 

    for note_name, octave, duration_val, is_rest, accidental in notes_sequence:
        if current_beats + duration_val > 4.001 and current_beats > 0: 
            if measure_notes_str_list:
                text_lines.append(f"마디 {current_measure}: {' | '.join(measure_notes_str_list)}")
            current_measure += 1
            measure_notes_str_list = []
            current_beats = 0
        
        note_str = ""
        if is_rest:
            note_str = f"R({duration_val})"
        else:
            note_str = f"{note_name}{accidental}{octave}({duration_val})"
            
        measure_notes_str_list.append(note_str)
        current_beats += duration_val
    
    if measure_notes_str_list:
        text_lines.append(f"마디 {current_measure}: {' | '.join(measure_notes_str_list)}")
    
    return "\n".join(text_lines)

# --- MIDI 생성 함수 ---
def note_name_to_midi(note_name, octave, accidental=''):
    """음표 이름을 MIDI 노트 번호로 변환 (accidental 포함)"""
    notes = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    
    midi_base = (octave * 12) + notes[note_name.upper()]
    
    if accidental == '#':
        midi_base += 1
    elif accidental == 'b':
        midi_base -= 1
        
    return midi_base

def generate_midi_file(notes_sequence, bpm):
    """MIDI 파일 생성"""
    if not MIDI_AVAILABLE:
        raise ValueError("Mido 라이브러리가 설치되지 않습니다.")
    
    try:
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        
        track.append(Message('program_change', channel=0, program=33, time=0))  # Electric Bass (finger) - MIDI General MIDI #34
        
        tempo_microseconds = mido.bpm2tempo(bpm)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo_microseconds))
        
        ticks_per_beat = mid.ticks_per_beat
        
        for note_name, octave, duration, is_rest, accidental in notes_sequence: 
            if not is_rest:
                try:
                    midi_note = note_name_to_midi(note_name, octave, accidental)
                    velocity = 80 
                    duration_ticks = int(duration * ticks_per_beat)
                    
                    track.append(Message('note_on', channel=0, note=midi_note, 
                                       velocity=velocity, time=0))
                    track.append(Message('note_off', channel=0, note=midi_note, 
                                       velocity=0, time=duration_ticks))
                except Exception as e:
                    logger.warning(f"MIDI 음표 생성 실패 {note_name}{accidental}{octave}: {e}")
                    duration_ticks = int(duration * ticks_per_beat)
                    track.append(Message('note_on', channel=0, note=60, 
                                       velocity=0, time=duration_ticks))
            else:
                duration_ticks = int(duration * ticks_per_beat)
                track.append(Message('note_on', channel=0, note=60, 
                                   velocity=0, time=duration_ticks))
        
        return mid
        
    except Exception as e:
        logger.error(f"MIDI 생성 실패: {e}")
        raise ValueError(f"MIDI 생성 실패: {str(e)}")

# --- LilyPond 형식 생성 ---
def convert_to_lilypond_notes(notes_sequence):
    """음표 시퀀스를 LilyPond 형식으로 변환"""
    lilypond_notes = []
    
    for note_name, octave, duration, is_rest, accidental in notes_sequence:
        ly_note_str = ""
        
        if is_rest:
            ly_note_str += 'r'
        else:
            ly_note_str += note_name.lower() 
            if accidental == '#':
                ly_note_str += 'is'
            elif accidental == 'b':
                ly_note_str += 'es'
            
            if octave == 3: 
                ly_note_str += ','
            elif octave == 2: 
                ly_note_str += ',,'
            elif octave == 1: 
                ly_note_str += ',,,'
            elif octave == 0: 
                ly_note_str += ',,,,'
            elif octave == 4: 
                pass 
            elif octave == 5: 
                ly_note_str += "'"
            elif octave == 6: 
                ly_note_str += "''"
            else:
                logger.warning(f"LilyPond: 지원하지 않는 옥타브 {octave}. 기본 LilyPond 옥타브 표기법을 따릅니다.")

        if duration == 4.0:
            ly_note_str += '1' 
        elif duration == 3.0: 
            ly_note_str += '2.' 
        elif duration == 2.0:
            ly_note_str += '2' 
        elif duration == 1.5: 
            ly_note_str += '4.' 
        elif duration == 1.0:
            ly_note_str += '4' 
        elif duration == 0.75:
            ly_note_str += '8.' 
        elif duration == 0.5:
            ly_note_str += '8' 
        elif duration == 0.25:
            ly_note_str += '16' 
        else:
            logger.warning(f"LilyPond: 인식할 수 없는 박자 단위 {duration}. 4분음표로 대체합니다.")
            ly_note_str += '4' 
        
        lilypond_notes.append(ly_note_str)
    
    return ' '.join(lilypond_notes)

def generate_lilypond_score(notes_sequence, bpm):
    """LilyPond 형식 악보 텍스트 생성"""
    lilypond_notes_str = convert_to_lilypond_notes(notes_sequence)
    
    lilypond_code = f"""\\version "2.24.0"
\\header {{
  title = "베이스 루프"
  composer = "AI 생성"
  tagline = \\markup {{ \\small "baseloop.onrender.com에서 생성됨" }}
}}

\\score {{
  \\new Staff \\with {{
    instrumentName = "Bass"
  }} {{
    \\clef "bass"
    \\tempo 4 = {bpm}
    \\time 4/4
    {lilypond_notes_str}
  }}
  \\layout {{ }}
  \\midi {{ }}
}}
"""
    return lilypond_code

# --- 스타일 기반 랜덤 생성 함수 (리프 기반 루프) ---
# 실제 베이스 반주의 핵심은 "짧은 리프의 반복"이다:
#   · 1~2마디 리프를 만들어 전체에 반복 (루프 맛)
#   · 마디 첫 박은 항상 루트 (앵커)
#   · 강박은 코드톤, 약박은 이전 음 주변 순차진행 (도약 없는 워킹)
#   · 음역은 루트 아래 5반음 ~ 위 12반음으로 제한 (높은 음 방지)
#   · 4마디마다/마지막 마디에 루트로 향하는 반음 접근 턴어라운드 필
BASS_STYLES = {
    "rock":   {"scale": ["C", "D", "E", "F", "G", "A", "B"],       "intervals": [0, 7, 5, 3],          "rhythms": [1.0, 0.5]},
    "funk":   {"scale": ["C", "D", "Eb", "F", "G", "A", "Bb"],     "intervals": [0, 7, 10, 5],         "rhythms": [0.25, 0.5, 1.0]},
    "pop":    {"scale": ["C", "D", "E", "F", "G", "A", "B"],       "intervals": [0, 7, 3, 5],          "rhythms": [0.5, 1.0]},
    "jazz":   {"scale": ["C", "D", "Eb", "F", "G", "A", "Bb"],     "intervals": [0, 3, 7, 9, 10],      "rhythms": [0.25, 0.5, 0.75, 1.0]},
    "blues":  {"scale": ["C", "Eb", "F", "F#", "G", "Bb"],         "intervals": [0, 3, 5, 6, 7, 10],   "rhythms": [0.5, 1.0]},
    "reggae": {"scale": ["C", "D", "E", "F", "G", "A", "B"],       "intervals": [0, 7, 5, 3],          "rhythms": [0.5, 1.0, 1.5]},
    "hiphop": {"scale": ["C", "D", "Eb", "G", "Ab"],               "intervals": [0, 3, 5, 7, 8, 10],   "rhythms": [0.25, 0.5, 1.0, 2.0]},
    "random": {"scale": ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'],
               "intervals": list(range(12)),                        "rhythms": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]},
}


def _measure_rhythm(rhythms):
    """한 마디(4박)를 채우는 리듬 패턴. 첫 박은 0.5박 이상으로 안정감 있게."""
    pattern = []
    beats = 0.0
    while beats < 3.999:
        remaining = 4.0 - beats
        choices = [r for r in rhythms if r <= remaining + 1e-9]
        if not choices:
            pattern.append(remaining)
            break
        weights = np.array([1.0 / c for c in choices], dtype=float)
        if beats == 0.0:
            weights = weights * np.array([3.0 if c >= 0.5 else 0.4 for c in choices])
        weights = weights / weights.sum()
        dur = float(np.random.choice(choices, p=weights))
        pattern.append(dur)
        beats += dur
    return pattern


def _build_style_riff(scale_pcs, intervals, rhythms, root_midi):
    """1마디 리프 생성: 강박=코드톤(루트 중심), 약박=이전 음 주변 순차진행."""
    lo, hi = root_midi - 5, root_midi + 12   # 베이스 음역 창
    candidates = sorted({m for pc in scale_pcs for m in range(lo, hi + 1) if m % 12 == pc})
    if not candidates:
        candidates = [root_midi]

    riff = []
    beat = 0.0
    cur = root_midi
    for i, dur in enumerate(_measure_rhythm(rhythms)):
        if i == 0:
            midi = root_midi                              # 마디 첫 박 = 루트 (루프 앵커)
        elif beat % 1.0 == 0 and dur >= 0.5:              # 강박: 코드톤
            r = np.random.rand()
            if r < 0.55:
                midi = root_midi
            elif r < 0.8:
                midi = root_midi + 7 if root_midi + 7 <= hi else root_midi - 5
            else:
                midi = root_midi + int(np.random.choice(intervals))
        else:                                             # 약박: 순차진행 워킹
            idx = min(range(len(candidates)), key=lambda k: abs(candidates[k] - cur))
            step = int(np.random.choice([-2, -1, -1, 1, 1, 2]))
            midi = candidates[max(0, min(len(candidates) - 1, idx + step))]
        midi = max(lo, min(hi, midi))
        cur = midi
        riff.append([midi, dur])
        beat += dur
    return riff


def create_random_bass_loop_by_style(style, key_root_note, octave, length_measures, bpm, variation=True):
    """리프 반복 구조의 베이스 루프 생성. variation=False면 순수 반복(테스트용)."""
    logger.info(f"생성 중: {style} 베이스 루프 (키: {key_root_note}{octave}, BPM: {bpm}, 마디: {length_measures})")

    selected = BASS_STYLES.get(style, BASS_STYLES["random"])
    root_pc = NOTE_TO_PC.get(key_root_note.upper())
    if root_pc is None:
        raise ValueError(f"유효하지 않은 루트 음: {key_root_note}")

    scale_pcs = sorted({(NOTE_TO_PC[n.upper()] + root_pc) % 12
                        for n in selected["scale"] if n.upper() in NOTE_TO_PC})
    root_midi = octave * 12 + root_pc
    lo, hi = root_midi - 5, root_midi + 12

    # 리프 1~2마디 생성 후 전체 길이에 반복
    riff_measures = 2 if length_measures >= 4 else 1
    riffs = [_build_style_riff(scale_pcs, selected["intervals"], selected["rhythms"], root_midi)
             for _ in range(riff_measures)]

    output = []   # [midi, dur] 목록
    for m in range(length_measures):
        riff = [list(n) for n in riffs[m % riff_measures]]
        is_phrase_end = (m % 4 == 3) or (m == length_measures - 1)

        if variation and is_phrase_end:
            # 턴어라운드 필: 마지막 1박을 루트로 향하는 반음 상행 접근(root-2 → root-1)으로
            last_midi, last_dur = riff[-1]
            if last_dur >= 1.0:
                riff[-1][1] = last_dur - 1.0
                if riff[-1][1] <= 1e-9:
                    riff.pop()
                riff.append([root_midi - 2, 0.5])
                riff.append([root_midi - 1, 0.5])
            else:
                riff[-1][0] = root_midi - 1
        elif variation and m >= riff_measures and len(riff) >= 3 and np.random.rand() < 0.2:
            # 반복 마디에 약박 음 하나만 살짝 변주 (루프 느낌은 유지)
            j = int(np.random.randint(1, len(riff)))
            riff[j][0] = max(lo, min(hi, riff[j][0] + int(np.random.choice([-2, 2]))))

        output.extend(riff)

    parts = []
    for midi, dur in output:
        pc = midi % 12
        oct_ = max(0, midi // 12)
        name, acc = PC_TO_NAME[pc]
        parts.append(f"{name}{acc}{oct_} {float(dur)}")
    formatted_sequence = ", ".join(parts)
    logger.info(f"생성된 시퀀스: {formatted_sequence}")
    return formatted_sequence


# ============================================================
# 음악성 강화: 코드 진행 기반 그루브 생성
# ============================================================
# 음이름 → 피치 클래스(0=C). 이명동음 표기를 모두 흡수한다.
NOTE_TO_PC = {
    'C': 0, 'B#': 0, 'C#': 1, 'DB': 1, 'D': 2, 'D#': 3, 'EB': 3,
    'E': 4, 'FB': 4, 'F': 5, 'E#': 5, 'F#': 6, 'GB': 6, 'G': 7,
    'G#': 8, 'AB': 8, 'A': 9, 'A#': 10, 'BB': 10, 'B': 11, 'CB': 11,
}
# 피치 클래스 → (음이름, 임시표). 베이스 표기는 플랫을 선호한다.
PC_TO_NAME = {
    0: ('C', ''), 1: ('C', '#'), 2: ('D', ''), 3: ('E', 'b'), 4: ('E', ''),
    5: ('F', ''), 6: ('F', '#'), 7: ('G', ''), 8: ('A', 'b'), 9: ('A', ''),
    10: ('B', 'b'), 11: ('B', ''),
}

# 코드 품질 → 루트 기준 반음 간격 (third, fifth, seventh 위치 파악용)
CHORD_QUALITIES = {
    '': [0, 4, 7], 'maj': [0, 4, 7], 'M': [0, 4, 7],
    'm': [0, 3, 7], 'min': [0, 3, 7], '-': [0, 3, 7],
    '7': [0, 4, 7, 10], 'dom7': [0, 4, 7, 10],
    'maj7': [0, 4, 7, 11], 'M7': [0, 4, 7, 11], 'Δ': [0, 4, 7, 11],
    'm7': [0, 3, 7, 10], 'min7': [0, 3, 7, 10], '-7': [0, 3, 7, 10],
    'dim': [0, 3, 6], '°': [0, 3, 6], 'o': [0, 3, 6],
    'dim7': [0, 3, 6, 9], '°7': [0, 3, 6, 9],
    'm7b5': [0, 3, 6, 10], 'ø': [0, 3, 6, 10],
    'aug': [0, 4, 8], '+': [0, 4, 8],
    'sus2': [0, 2, 7], 'sus4': [0, 5, 7], 'sus': [0, 5, 7],
    '6': [0, 4, 7, 9], 'm6': [0, 3, 7, 9], 'min6': [0, 3, 7, 9],
    '9': [0, 4, 7, 10], 'm9': [0, 3, 7, 10], 'maj9': [0, 4, 7, 11],
}


def parse_chord_symbol(symbol):
    """
    "C", "Am", "G7", "Fmaj7", "D#m7", "Bbdim" 같은 코드 기호를 파싱.
    (root_pc, intervals, normalized_name) 튜플을 반환한다.
    intervals 는 루트 기준 반음 간격 리스트(예: [0,4,7]).
    """
    symbol = symbol.strip()
    if not symbol:
        raise ValueError("빈 코드 기호입니다.")

    # 슬래시(온음) 코드: "C/E" → 코드 C, 베이스 음 E. 슬래시 뒤 음이 실제 베이스 음.
    slash_bass_pc = None
    if '/' in symbol:
        main_part, bass_part = symbol.split('/', 1)
        bm = re.match(r'^([A-Ga-g])([#b]?)$', bass_part.strip())
        if bm:
            slash_bass_pc = NOTE_TO_PC.get((bm.group(1).upper() + bm.group(2).replace('B', 'b')).upper())
        symbol = main_part.strip()

    m = re.match(r'^([A-Ga-g])([#b]?)(.*)$', symbol)
    if not m:
        raise ValueError(f"유효하지 않은 코드 기호: '{symbol}'")

    root_name = m.group(1).upper() + m.group(2).replace('B', 'b')
    quality_raw = m.group(3).strip()

    root_pc = NOTE_TO_PC.get(root_name.upper())
    if root_pc is None:
        raise ValueError(f"유효하지 않은 코드 루트: '{symbol}'")

    intervals = CHORD_QUALITIES.get(quality_raw)
    if intervals is None:
        # 알 수 없는 확장(텐션 등)은 메이저/마이너 트라이어드로 안전하게 축약
        intervals = [0, 3, 7] if quality_raw[:1] in ('m', '-') and quality_raw[:3] != 'maj' else [0, 4, 7]

    bass_pc = slash_bass_pc if slash_bass_pc is not None else root_pc
    name = root_name + quality_raw + (('/' + PC_TO_NAME[bass_pc][0] + PC_TO_NAME[bass_pc][1]) if bass_pc != root_pc else '')
    return root_pc, intervals, bass_pc, name


def parse_chord_progression(progression_str):
    """ "C G Am F" 또는 "C, G, Am, F" 형태의 진행을 코드 리스트로 파싱."""
    if not progression_str or not progression_str.strip():
        raise ValueError("코드 진행이 비어 있습니다.")
    tokens = [t for t in re.split(r'[\s,|]+', progression_str.strip()) if t]
    if not tokens:
        raise ValueError("코드 진행에서 유효한 코드를 찾지 못했습니다.")
    return [parse_chord_symbol(t) for t in tokens]


# 장르별 그루브 템플릿: 한 마디(4박)를 채우는 (박자, 역할) 패턴 목록.
# 역할(role) 의미:
#   root/third/fifth/seventh = 현재 코드 톤, octave = 루트 한 옥타브 위,
#   approach = 다음 코드의 베이스 음으로 향하는 반음 접근음, rest = 쉼표.
#   (마디 첫 음의 root/octave는 슬래시 코드의 베이스 음을 사용)
GROOVE_TEMPLATES = {
    'rock': [
        [(1.0, 'root'), (1.0, 'root'), (1.0, 'fifth'), (1.0, 'fifth')],
        [(1.0, 'root'), (0.5, 'root'), (0.5, 'octave'), (1.0, 'fifth'), (1.0, 'approach')],
        [(0.5, 'root'), (0.5, 'root'), (1.0, 'fifth'), (1.0, 'root'), (1.0, 'approach')],
    ],
    'funk': [
        [(0.5, 'root'), (0.5, 'rest'), (0.5, 'root'), (0.25, 'octave'), (0.25, 'root'),
         (0.5, 'rest'), (0.5, 'fifth'), (0.5, 'approach'), (0.5, 'root')],
        [(0.25, 'root'), (0.25, 'root'), (0.5, 'octave'), (0.5, 'rest'), (0.5, 'fifth'),
         (1.0, 'root'), (0.5, 'third'), (0.5, 'approach')],
    ],
    'pop': [
        [(1.0, 'root'), (1.0, 'fifth'), (1.0, 'root'), (1.0, 'third')],
        [(2.0, 'root'), (1.0, 'fifth'), (1.0, 'approach')],
        [(1.0, 'root'), (0.5, 'root'), (0.5, 'fifth'), (1.0, 'octave'), (1.0, 'approach')],
    ],
    'jazz': [  # 워킹 베이스: 4분음표로 코드톤→접근음
        [(1.0, 'root'), (1.0, 'third'), (1.0, 'fifth'), (1.0, 'approach')],
        [(1.0, 'root'), (1.0, 'fifth'), (1.0, 'seventh'), (1.0, 'approach')],
        [(1.0, 'root'), (1.0, 'third'), (1.0, 'fifth'), (1.0, 'seventh')],
    ],
    'blues': [
        [(1.5, 'root'), (0.5, 'third'), (1.0, 'fifth'), (1.0, 'seventh')],
        [(1.0, 'root'), (1.0, 'third'), (1.0, 'fifth'), (1.0, 'octave')],
        [(1.0, 'root'), (1.0, 'fifth'), (1.0, 'seventh'), (1.0, 'approach')],
    ],
    'reggae': [  # 강박을 비우고 약박을 강조
        [(1.0, 'rest'), (1.0, 'root'), (1.0, 'rest'), (0.5, 'root'), (0.5, 'fifth')],
        [(1.0, 'rest'), (0.5, 'root'), (0.5, 'octave'), (1.0, 'rest'), (1.0, 'fifth')],
    ],
    'hiphop': [
        [(2.0, 'root'), (1.0, 'rest'), (1.0, 'octave')],
        [(1.5, 'root'), (0.5, 'rest'), (1.0, 'fifth'), (1.0, 'root')],
        [(1.0, 'root'), (1.0, 'rest'), (1.0, 'root'), (0.5, 'fifth'), (0.5, 'approach')],
    ],
}


def _role_to_note(role, intervals, root_pc, bass_pc, next_bass_pc, is_first):
    """역할 → (기준 피치클래스, 반음 오프셋, 옥타브 오프셋).

    마디 첫 음의 root/octave 역할은 슬래시 베이스 음(bass_pc)을 쓴다
    (예: C/E → 첫 박을 E로). 3·5·7음은 코드 루트(root_pc) 기준 유지.
    접근음은 다음 코드의 실제 베이스 음으로 향한다.
    """
    if role == 'root':
        return (bass_pc if is_first else root_pc), 0, 0
    if role == 'octave':
        return (bass_pc if is_first else root_pc), 0, 1
    if role == 'third':
        return root_pc, (intervals[1] if len(intervals) > 1 else 4), 0
    if role == 'fifth':
        return root_pc, (intervals[2] if len(intervals) > 2 else 7), 0
    if role == 'seventh':
        return root_pc, (intervals[3] if len(intervals) > 3 else 10), 0
    if role == 'approach':
        # 다음 코드의 베이스 음으로 반음 아래에서 접근 (없으면 5도로 대체)
        if next_bass_pc is None:
            return root_pc, 7, 0
        return root_pc, ((next_bass_pc - 1) - root_pc) % 12, 0
    return root_pc, 0, 0


def generate_bassline_from_chords(progression_str, genre, octave, seed=None,
                                  loops=1):
    """
    코드 진행 + 장르 그루브 템플릿으로 음악적인 베이스 라인을 생성한다.
    각 코드는 한 마디(4박)를 차지한다. 슬래시 코드(C/E)의 베이스 음을 반영한다.
    시드를 주면 동일 입력에 동일 결과. 프론트엔드 시퀀스 문자열("C2 1.0, ...")을 반환한다.
    """
    if seed is not None:
        np.random.seed(int(seed) % (2 ** 32))

    chords = parse_chord_progression(progression_str)
    templates = GROOVE_TEMPLATES.get(genre, GROOVE_TEMPLATES['rock'])
    # 그루브 템플릿은 한 번만 뽑아 전 마디에 반복 → 마디마다 리듬이 바뀌지 않고
    # "같은 그루브가 코드를 따라가는" 진짜 베이스 반주 느낌을 만든다.
    template = templates[int(np.random.randint(len(templates)))]

    notes_out = []
    for _ in range(max(1, int(loops))):
        for idx, (root_pc, intervals, bass_pc, _name) in enumerate(chords):
            next_bass_pc = chords[(idx + 1) % len(chords)][2]
            for j, (dur, role) in enumerate(template):
                if role == 'rest':
                    notes_out.append(('R', 4, float(dur), True, ''))
                    continue
                ref_pc, semitone, oct_off = _role_to_note(
                    role, intervals, root_pc, bass_pc, next_bass_pc, j == 0)
                total = ref_pc + semitone
                pc = total % 12
                note_octave = octave + oct_off + (total // 12)
                note_octave = max(0, min(6, note_octave))
                name, acc = PC_TO_NAME[pc]
                notes_out.append((name, note_octave, float(dur), False, acc))

    formatted = ", ".join(
        [f"{n[0]}{n[4]}{n[1]} {float(n[2])}" if not n[3] else f"R {float(n[2])}"
         for n in notes_out]
    )
    logger.info(f"코드 진행 생성 시퀀스: {formatted}")
    return formatted

# --- Flask 웹 라우트 ---

@app.route('/')
def index_page():
    """메인 페이지 렌더링"""
    return render_template(
        'index.html',
        music21_available=MUSIC21_AVAILABLE,
        midi_available=MIDI_AVAILABLE,
    )

@app.route('/health')
def health_check():
    """헬스체크 엔드포인트 (로깅 최소화)"""
    return {'status': 'ok'}, 200

@app.route('/generate_notes', methods=['POST'])
def generate_notes_route():
    """사용자 요청에 따라 랜덤(스타일) 또는 코드 진행 방식으로 악보 시퀀스를 생성하여 반환"""
    try:
        generation_mode = request.form.get('generation_mode', 'random')

        try:
            bpm = int(request.form.get('bpm_input', 120))
            length_measures = int(request.form.get('length_input', 4))
            octave = int(request.form.get('octave_input', 2))
        except ValueError:
            return jsonify({'status': 'error', 'message': 'BPM, 마디 길이, 옥타브는 숫자여야 합니다.'}), 400

        genre = request.form.get('genre_input', 'rock')
        key_note = request.form.get('key_note_input', 'C')

        if not (30 <= bpm <= 240):
            return jsonify({'status': 'error', 'message': 'BPM은 30-240 범위여야 합니다.'}), 400
        if not (1 <= length_measures <= 16):
            return jsonify({'status': 'error', 'message': '마디 길이는 1-16 범위여야 합니다.'}), 400
        if not (0 <= octave <= 4):
            return jsonify({'status': 'error', 'message': '옥타브는 0-4 범위여야 합니다.'}), 400

        # 시드: 비우면 무작위 시드를 만들어 사용하고 응답으로 돌려준다(재현/공유 가능).
        seed_raw = request.form.get('seed_input', '').strip()
        try:
            seed = int(seed_raw) if seed_raw else int(np.random.randint(0, 2 ** 31 - 1))
        except ValueError:
            return jsonify({'status': 'error', 'message': '시드는 정수여야 합니다.'}), 400
        np.random.seed(seed % (2 ** 32))

        generated_notes_str = ""

        if generation_mode == "random":
            generated_notes_str = create_random_bass_loop_by_style(
                genre, key_note, octave, length_measures, bpm
            )
        elif generation_mode == "chords":
            progression = request.form.get('chord_progression_input', '').strip()
            if not progression:
                return jsonify({'status': 'error', 'message': '코드 진행을 입력해주세요. 예: C G Am F'}), 400
            generated_notes_str = generate_bassline_from_chords(
                progression, genre, octave, seed=seed
            )
        else:
            return jsonify({'status': 'error', 'message': '알 수 없는 생성 모드입니다.'}), 400

        if generated_notes_str:
            return jsonify({'status': 'success', 'notes': generated_notes_str, 'seed': seed})
        else:
            return jsonify({'status': 'error', 'message': '악보 생성에 실패했습니다.'}), 500

    except ValueError as ve:
        logger.error(f"악보 생성 중 ValueError: {ve}")
        return jsonify({'status': 'error', 'message': str(ve)}), 400
    except Exception as e:
        logger.error(f"악보 생성 중 예외 발생: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'status': 'error', 'message': f"서버 오류 발생: {str(e)}"}), 500

@app.route('/generate_audio', methods=['POST'])
def generate_audio_route(): 
    """사용자 입력 악보 시퀀스를 기반으로 베이스 루프 오디오 생성하여 반환"""
    try:
        notes_sequence_str = request.form.get('notes_sequence_input', '').strip()
        
        try:
            bpm = int(request.form.get('bpm_input', 120))
            num_loops = int(request.form.get('num_loops_input', 2))
        except ValueError:
            return Response("BPM과 루프 횟수는 숫자여야 합니다.", status=400, mimetype='text/plain')

        if not notes_sequence_str:
            return Response("악보 시퀀스가 비어 있습니다. 음표를 생성하거나 입력해주세요.", status=400, mimetype='text/plain')

        notes_sequence = parse_note_sequence_string(notes_sequence_str)

        audio_buffer = create_bass_loop_from_parsed_sequence(notes_sequence, bpm, num_loops)

        return Response(audio_buffer.getvalue(), mimetype='audio/wav')

    except ValueError as ve:
        logger.error(f"오디오 생성 중 ValueError: {ve}")
        return Response(f"악보 파싱 오류: {str(ve)}", status=400, mimetype='text/plain')
    except Exception as e:
        logger.error(f"오디오 생성 중 예외 발생: {e}")
        logger.error(traceback.format_exc())
        return Response(f"오류 발생: {str(e)}. 서버 로그를 확인하세요.", status=500, mimetype='text/plain')

@app.route('/generate_score_image', methods=['GET'])
def generate_score_image_route(): 
    """Music21로 생성한 전문 악보 이미지 (Base64) 또는 텍스트 악보 반환"""
    try:
        notes_sequence_str = request.args.get('notes', '')
        bpm = int(request.args.get('bpm', 120))
        key_signature = request.args.get('key', 'C')
        
        if not notes_sequence_str:
            return jsonify({"status": "error", "message": "악보 시퀀스가 필요합니다."}), 400
        
        notes_sequence = parse_note_sequence_string(notes_sequence_str)
        svg_data_base64, text_score = generate_music21_score_with_fallback(notes_sequence, bpm, key_signature) # Changed png_data_base64
        
        if svg_data_base64:
            return jsonify({"status": "success", "image_data": svg_data_base64, "format": "svg"}) # Changed format to svg
        elif text_score:
            return jsonify({"status": "success", "text_data": text_score, "format": "text"})
        else:
            return jsonify({"status": "error", "message": "악보 이미지/텍스트 생성에 실패했습니다."}), 500 # Generic message is fine
            
    except Exception as e:
        logger.error(f"악보 이미지 생성 오류: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": f"악보 이미지 생성 오류: {str(e)}"}), 500 # Changed message from "악보 생성 오류"

@app.route('/generate_midi', methods=['POST'])
def generate_midi_route(): 
    """MIDI 파일 생성 및 다운로드"""
    try:
        notes_sequence_str = request.form.get('notes_sequence_input', '').strip()
        bpm = int(request.form.get('bpm_input', 120))
        
        if not notes_sequence_str:
            return Response("악보 시퀀스가 비어 있습니다.", status=400, mimetype='text/plain')
        
        notes_sequence = parse_note_sequence_string(notes_sequence_str)
        midi_file = generate_midi_file(notes_sequence, bpm)
        
        buffer = io.BytesIO()
        midi_file.save(file=buffer)
        buffer.seek(0)
        
        return Response(
            buffer.getvalue(),
            mimetype='audio/midi',
            headers={'Content-Disposition': 'attachment; filename=bassloop.mid'}
        )
        
    except ValueError as ve:
        logger.error(f"MIDI 생성 중 ValueError: {ve}")
        return Response(f"MIDI 생성 오류: {str(ve)}", status=400, mimetype='text/plain')
    except Exception as e:
        logger.error(f"MIDI 생성 중 예외 발생: {e}")
        logger.error(traceback.format_exc())
        return Response(f"MIDI 생성 오류: {str(e)}", status=500, mimetype='text/plain')

@app.route('/generate_lilypond', methods=['POST'])
def generate_lilypond_route(): 
    """LilyPond 형식 악보 텍스트 생성 및 다운로드"""
    try:
        notes_sequence_str = request.form.get('notes_sequence_input', '').strip()
        bpm = int(request.form.get('bpm_input', 120))
        
        if not notes_sequence_str:
            return Response("악보 시퀀스가 비어 있습니다.", status=400, mimetype='text/plain')
        
        notes_sequence = parse_note_sequence_string(notes_sequence_str)
        lilypond_code = generate_lilypond_score(notes_sequence, bpm)
        
        return Response(
            lilypond_code,
            mimetype='text/plain',
            headers={'Content-Disposition': 'attachment; filename=bassloop.ly'}
        )
        
    except ValueError as ve:
        logger.error(f"LilyPond 생성 중 ValueError: {ve}")
        return Response(f"LilyPond 생성 오류: {str(ve)}", status=400, mimetype='text/plain')
    except Exception as e:
        logger.error(f"LilyPond 생성 중 예외 발생: {e}")
        logger.error(traceback.format_exc())
        return Response(f"LilyPond 생성 오류: {str(e)}", status=500, mimetype='text/plain')

@app.route('/get_features_status')
def get_features_status():
    """라이브러리 설치 상태 확인 API"""
    music21_configured = False
    if MUSIC21_AVAILABLE:
        music21_configured = setup_music21_environment() 

    return jsonify({
        'music21_available': MUSIC21_AVAILABLE,
        'music21_configured_for_png': music21_configured,
        'midi_available': MIDI_AVAILABLE,
    })

@app.route('/debug_musescore')
def debug_musescore():
    """MuseScore 및 Music21 설정 디버깅"""
    import subprocess
    debug_info = {
        "music21_available": MUSIC21_AVAILABLE,
        "musescore_tests": {},
        "environment": {},
        "music21_settings": {}
    }
    
    # 환경 변수 확인
    debug_info["environment"] = {
        "DISPLAY": os.environ.get("DISPLAY"),
        "QT_QPA_PLATFORM": os.environ.get("QT_QPA_PLATFORM"),
        "MUSESCORE_PATH": os.environ.get("MUSESCORE_PATH"),
        "PATH": os.environ.get("PATH")
    }
    
    # MuseScore 찾기
    musescore_paths = [
        "/usr/bin/musescore3",
        "/usr/bin/musescore",
        "/usr/local/bin/musescore3",
        "/usr/local/bin/musescore",
        "/snap/bin/musescore"
    ]
    
    for path in musescore_paths:
        debug_info["musescore_tests"][path] = {
            "exists": os.path.exists(path),
            "executable": os.access(path, os.X_OK) if os.path.exists(path) else False
        }
    
    # MuseScore 실행 테스트
    try:
        result = subprocess.run(["which", "musescore3"], capture_output=True, text=True)
        debug_info["which_musescore3"] = result.stdout.strip()
    except:
        debug_info["which_musescore3"] = "Failed"
    
    # Music21 설정 확인
    if MUSIC21_AVAILABLE:
        try:
            from music21 import environment
            us = environment.UserSettings()
            debug_info["music21_settings"] = {
                "musescoreDirectPNGPath": us.get("musescoreDirectPNGPath", "Not set"),
                "musicxmlPath": us.get("musicxmlPath", "Not set"),
                "graphicsPath": us.get("graphicsPath", "Not set"),
                "debug": us.get("debug", False)
            }
        except Exception as e:
            debug_info["music21_settings"]["error"] = str(e)
    
    # Xvfb 프로세스 확인
    try:
        result = subprocess.run(["pgrep", "-f", "Xvfb"], capture_output=True, text=True)
        debug_info["xvfb_running"] = bool(result.stdout.strip())
        debug_info["xvfb_pids"] = result.stdout.strip()
    except:
        debug_info["xvfb_running"] = False
    
    return jsonify(debug_info)

@app.errorhandler(404)
def not_found_error(error):
    return render_template(
        'index.html',
        music21_available=MUSIC21_AVAILABLE,
        midi_available=MIDI_AVAILABLE,
    ), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 오류: {error}")
    return "Internal server error. Check server logs for more details.", 500

if __name__ == '__main__':
    try:
        load_dotenv()
    except:
        pass 

    port = int(os.environ.get('PORT', 10000))

    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"서버 시작 - 포트: {port}, 디버그: {debug_mode}")
    logger.info(f"Music21 사용 가능: {MUSIC21_AVAILABLE}")
    logger.info(f"MIDI 사용 가능: {MIDI_AVAILABLE}")
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
