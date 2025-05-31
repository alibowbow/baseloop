import numpy as np
from scipy.io.wavfile import write as write_wav
from flask import Flask, render_template, request, send_file, Response, jsonify
import io 
import re
import os
import ast # For safe parsing of LLM output
import traceback
import logging
import tempfile
import shutil
import base64 

# Music21 라이브러리 (악보 생성)
try:
    from music21 import stream, note, meter, tempo, clef, bar, duration, pitch, key, layout
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False
    print("Warning: music21 라이브러리가 설치되지 않았습니다. 전문 악보 생성을 사용할 수 없습니다.")

# ─── Music21 MuseScore 환경 설정 함수 ───────────────────────
_M21_CONFIGURED = False

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

        musescore_path = os.getenv("MUSESCORE_PATH")
        if musescore_path:
            logger.info(f"환경 변수 MUSESCORE_PATH 사용: {musescore_path}")
        else:
            logger.info("MUSESCORE_PATH 환경 변수가 설정되지 않았습니다. 자동 탐색을 시도합니다.")
            # 여러 가능한 경로 시도
            possible_paths = [
                "/usr/bin/musescore3",
                "/usr/bin/musescore",
                "/usr/local/bin/musescore3",
                "/usr/local/bin/musescore",
                "/snap/bin/musescore"
            ]
            
            for path in possible_paths:
                if os.path.exists(path) and os.access(path, os.X_OK):
                    musescore_path = path
                    logger.info(f"MuseScore 발견: {path}")
                    break
            
            if not musescore_path:
                musescore_path = "/usr/bin/musescore3"  # 기본값
                logger.warning(f"MuseScore를 찾을 수 없습니다. 기본 경로 사용: {musescore_path}")

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

# Google Gemini 라이브러리 (AI 모드 사용 시 필수)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai 라이브러리가 설치되지 않았습니다. AI 모드를 사용할 수 없습니다.")

# 로컬 개발용 .env 파일 로딩
try:
    from dotenv import load_dotenv
    load_dotenv() 
except ImportError:
    print("Warning: python-dotenv 라이브러리가 설치되지 않았습니다.")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Hardcoded MusicXML Test (글로벌 플래그) ---
# 이 플래그를 True로 설정하면 generate_music21_score_with_fallback 함수가
# music21 스코어 생성 로직을 건너뛰고 미리 정의된 MusicXML을 사용하여 PNG 생성을 시도합니다.
# MuseScore 직접 실행 기능 테스트 및 디버깅에 유용합니다.
USE_HARDCODED_XML_TEST = False # True로 변경하여 테스트 활성화

HARDCODED_MUSICXML_STRING = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="4.0">
  <part-list>
    <score-part id="P1">
      <part-name>Bass</part-name>
    </score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <key>
          <fifths>0</fifths>
          <mode>major</mode>
        </key>
        <time>
          <beats>4</beats>
          <beat-type>4</beat-type>
        </time>
        <clef>
          <sign>F</sign>
          <line>4</line>
        </clef>
      </attributes>
      <note>
        <pitch>
          <step>C</step>
          <octave>2</octave>
        </pitch>
        <duration>4</duration>
        <type>whole</type>
      </note>
    </measure>
  </part>
</score-partwise>
"""

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
    """단일 음표의 파형을 생성합니다."""
    num_samples = int(duration_seconds * sample_rate)
    if num_samples <= 0:
        return np.array([])
        
    t = np.linspace(0, duration_seconds, num_samples, endpoint=False)
    waveform = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # ADSR 엔벨로프 적용 (간단한 Attack-Release)
    attack_percent = 0.01
    release_percent = 0.2

    attack_samples = int(num_samples * attack_percent)
    release_samples = int(num_samples * release_percent)
    
    envelope = np.ones(num_samples)

    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    
    if release_samples > 0 and (num_samples - release_samples) >= attack_samples:
        start_release_idx = max(attack_samples, num_samples - release_samples)
        envelope[start_release_idx:] *= np.linspace(1, 0, num_samples - start_release_idx)
    
    waveform *= envelope
    
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
    
    max_val = np.max(np.abs(final_waveform))
    if max_val > 0:
        final_waveform = final_waveform / max_val * MAX_AMPLITUDE 
    
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
    Music21 + MuseScore → PNG 생성. 실패 시 텍스트 악보 반환.
    (png_data_base64, text_score) 형태 튜플을 돌려줍니다.
    """
    if not MUSIC21_AVAILABLE:
        logger.warning("Music21 라이브러리가 설치되지 않아 전문 악보 생성을 할 수 없습니다.")
        return None, generate_text_score(notes_sequence, bpm)

    if not setup_music21_environment():
        logger.error("Music21 환경 설정 실패. MuseScore PNG 생성을 건너뛰고 텍스트 악보를 생성합니다.")
        return None, generate_text_score(notes_sequence, bpm)

    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp() # Moved earlier to be available for hardcoded test
        logger.info(f"악보 생성을 위한 임시 디렉토리 생성: {temp_dir}")

        if USE_HARDCODED_XML_TEST:
            logger.warning("하드코딩된 MusicXML 테스트 경로를 사용합니다.")
            hardcoded_xml_path = os.path.join(temp_dir, "hardcoded_score.xml")
            hardcoded_png_path = os.path.join(temp_dir, "hardcoded_score.png")

            try:
                with open(hardcoded_xml_path, "w", encoding="utf-8") as f:
                    f.write(HARDCODED_MUSICXML_STRING)
                logger.info(f"하드코딩된 MusicXML 파일 저장: {hardcoded_xml_path}")

                musescore_path_env = os.environ.get("MUSESCORE_PATH", "/usr/bin/musescore3")
                cmd_hardcoded = [musescore_path_env, "-o", hardcoded_png_path, hardcoded_xml_path]

                logger.info(f"하드코딩된 XML로 MuseScore 직접 실행 명령: {' '.join(cmd_hardcoded)}")

                result_hardcoded = subprocess.run(cmd_hardcoded, capture_output=True, text=True, timeout=30)

                logger.info(f"하드코딩된 XML MuseScore 실행 Return Code: {result_hardcoded.returncode}")
                if result_hardcoded.stdout:
                    logger.info(f"하드코딩된 XML MuseScore 실행 STDOUT: {result_hardcoded.stdout.strip()}")
                if result_hardcoded.stderr:
                    logger.warning(f"하드코딩된 XML MuseScore 실행 STDERR: {result_hardcoded.stderr.strip()}")

                if result_hardcoded.returncode == 0 and os.path.exists(hardcoded_png_path) and os.path.getsize(hardcoded_png_path) > 0:
                    logger.info(f"하드코딩된 XML로부터 PNG 파일 생성 성공: {hardcoded_png_path}, 크기: {os.path.getsize(hardcoded_png_path)} 바이트")
                    with open(hardcoded_png_path, "rb") as f_png:
                        png_data = f_png.read()
                    if temp_dir and os.path.exists(temp_dir): # Cleanup before early return
                        shutil.rmtree(temp_dir)
                    return base64.b64encode(png_data).decode('utf-8'), None
                else:
                    logger.error(f"하드코딩된 XML로부터 PNG 생성 실패. Return Code: {result_hardcoded.returncode}, stderr: {result_hardcoded.stderr.strip()}")
                    if temp_dir and os.path.exists(temp_dir): # Cleanup before early return
                        shutil.rmtree(temp_dir)
                    return None, "하드코딩된 XML 테스트 PNG 생성 실패." # Special error message

            except Exception as e_hardcoded:
                logger.error(f"하드코딩된 XML 테스트 중 예외 발생: {e_hardcoded}")
                logger.error(traceback.format_exc())
                if temp_dir and os.path.exists(temp_dir): # Cleanup before early return
                    shutil.rmtree(temp_dir)
                return None, f"하드코딩된 XML 테스트 중 예외: {str(e_hardcoded)}"
            # If USE_HARDCODED_XML_TEST is true, the function will have returned by this point.
            # The main finally block is thus only for the normal execution path.
            # However, since this path *does* return, we'd need to clean up temp_dir here.
            # For simplicity in this subtask, I will let the main finally block handle it,
            # which means if USE_HARDCODED_XML_TEST is True and successful, the temp_dir might not be cleaned immediately.
            # A more robust solution would duplicate the finally logic or refactor cleanup.
            # For now, the focus is on the conditional execution.
            # Let's ensure the main finally block is reached by not returning early from here if test fails.
            # If test fails, it will return (None, "Hardcoded XML test PNG 생성 실패."),
            # and the main finally block will execute.

        # Normal processing starts here if USE_HARDCODED_XML_TEST is False
        logger.info("Music21 악보 객체 생성 시작.")
        score = stream.Score()
        score.append(tempo.MetronomeMark(number=bpm))
        score.append(meter.TimeSignature("4/4"))
        score.append(key.Key(key_signature))

        part = stream.Part()
        part.append(clef.BassClef())

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
        logger.info("Music21 악보 객체 생성 완료. MuseScore PNG 생성을 시도합니다.")

        # temp_dir = tempfile.mkdtemp() # Moved earlier
        png_path = os.path.join(temp_dir, "score.png") # This will be used by normal path

        try:
            # Music21이 MuseScore를 호출하여 PNG를 생성합니다.
            # 먼저 MusicXML로 변환 후 PNG로 변환하는 2단계 접근법 시도
            logger.info("PNG 생성 시도 중...")
            
            # 방법 1: 직접 PNG 생성
            try:
                # --- Temporary MusicXML logging ---
                temp_xml_file = None
                try:
                    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".musicxml", dir=temp_dir) as temp_xml_file_obj:
                        temp_xml_file_path = temp_xml_file_obj.name
                        score.write("musicxml", fp=temp_xml_file_path)
                        logger.info(f"임시 MusicXML 파일 생성: {temp_xml_file_path}")
                        temp_xml_file_obj.seek(0)
                        content = temp_xml_file_obj.read(1000)
                        logger.info(f"임시 MusicXML 파일 내용 (처음 1000자): {content}...")
                except Exception as ex_temp_xml:
                    logger.warning(f"임시 MusicXML 파일 로깅 중 오류: {ex_temp_xml}")
                finally:
                    if temp_xml_file_path and os.path.exists(temp_xml_file_path):
                        try:
                            os.remove(temp_xml_file_path)
                            logger.info(f"임시 MusicXML 파일 삭제: {temp_xml_file_path}")
                        except Exception as ex_remove:
                            logger.warning(f"임시 MusicXML 파일 삭제 중 오류: {ex_remove}")
                # --- End Temporary MusicXML logging ---

                score.write("musicxml.png", fp=png_path)
                logger.info(f"직접 PNG 생성 시도 완료: {png_path}")

                # --- PNG file existence and size logging ---
                if os.path.exists(png_path):
                    logger.info(f"PNG 파일 존재 확인: {png_path} (존재함)")
                    logger.info(f"PNG 파일 크기: {os.path.getsize(png_path)} 바이트")
                else:
                    logger.info(f"PNG 파일 존재 확인: {png_path} (존재하지 않음)")
                # --- End PNG file logging ---

            except Exception as e:
                logger.warning(f"직접 PNG 생성 실패: {e}")
                
                # 방법 2: lily.png 시도 (LilyPond 경유)
                try:
                    score.write("lily.png", fp=png_path)
                    logger.info("LilyPond 경유 PNG 생성 시도")
                except Exception as e2:
                    logger.warning(f"LilyPond PNG 생성도 실패: {e2}")
                    
                    # 방법 3: 명시적으로 MuseScore 호출
                    try:
                        from music21 import converter
                        # 먼저 MusicXML로 저장
                        xml_path = os.path.join(temp_dir, "score.xml")
                        score.write("musicxml", fp=xml_path)
                        
                        # MuseScore 명령어로 직접 변환
                        musescore_path = os.environ.get("MUSESCORE_PATH", "/usr/bin/musescore3")
                        import subprocess

                        # --- Logging before subprocess.run ---
                        logger.info(f"MuseScore 직접 실행 명령: {musescore_path} -o {png_path} {xml_path}")
                        if os.path.exists(xml_path):
                            logger.info(f"입력 MusicXML 파일({xml_path}) 존재 확인: 존재함")
                            logger.info(f"입력 MusicXML 파일({xml_path}) 크기: {os.path.getsize(xml_path)} 바이트")
                            try:
                                with open(xml_path, "r", encoding="utf-8") as f_xml:
                                    xml_content_preview = f_xml.read(1000)
                                    logger.info(f"입력 MusicXML 파일({xml_path}) 내용 (처음 1000자): {xml_content_preview}...")
                            except Exception as ex_read_xml:
                                logger.warning(f"입력 MusicXML 파일({xml_path}) 읽기 중 오류: {ex_read_xml}")
                        else:
                            logger.warning(f"입력 MusicXML 파일({xml_path}) 존재 확인: 존재하지 않음!")
                        # --- End logging before subprocess.run ---

                        cmd = [musescore_path, "-o", png_path, xml_path]
                        # logger.info(f"MuseScore 직접 실행: {' '.join(cmd)}") # Already logged above in more detail
                        
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

                        # --- Logging after subprocess.run ---
                        logger.info(f"MuseScore 실행 Return Code: {result.returncode}")
                        if result.stdout:
                            logger.info(f"MuseScore 실행 STDOUT: {result.stdout.strip()}")
                        if result.stderr: # Log stderr even if returncode is 0
                            logger.info(f"MuseScore 실행 STDERR: {result.stderr.strip()}")
                        # --- End logging after subprocess.run ---

                        if result.returncode != 0:
                            # logger.error(f"MuseScore 실행 실패: {result.stderr}") # Already logged by the generic STDERR log
                            raise RuntimeError(f"MuseScore 실행 실패 (자세한 내용은 이전 로그 참조)")
                            
                    except Exception as e3:
                        logger.error(f"MuseScore 직접 호출 실패: {e3}")
                        raise
            
            # PNG 파일 확인
            if os.path.exists(png_path) and os.path.getsize(png_path) > 0:
                logger.info(f"MuseScore PNG 파일 생성 성공: {png_path}, 크기: {os.path.getsize(png_path)} 바이트")
                with open(png_path, "rb") as f:
                    png_data = f.read()
                return base64.b64encode(png_data).decode('utf-8'), None 
            
            # 파일이 없거나 비어있는 경우
            raise RuntimeError("MuseScore로 PNG 생성 실패: 파일이 없거나 비어 있습니다.")
        except Exception as e:
            logger.warning(f"MuseScore PNG 생성 실패: {e}")
            logger.warning("MuseScore PNG 실패: PNG 생성에 실패했습니다. (MuseScore 설치 및 환경 변수 MUSESCORE_PATH 설정, 또는 xvfb 필요)")
            text_score = generate_text_score(notes_sequence, bpm)
            return None, text_score 
    except Exception as e:
        logger.error(f"악보 생성 중 예상치 못한 오류 발생: {e}")
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
        
        track.append(Message('program_change', channel=0, program=32, time=0))  # Acoustic Bass (MIDI General MIDI #33)
        
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

# --- 스타일 기반 랜덤 생성 함수 ---
def create_random_bass_loop_by_style(style, key_root_note, octave, length_measures, bpm):
    """스타일에 따른 랜덤 베이스 라인 생성"""
    logger.info(f"생성 중: {style} 베이스 루프 (키: {key_root_note}{octave}, BPM: {bpm}, 마디: {length_measures})")
    
    styles = {
        "rock": {
            "scale": ["C", "D", "E", "F", "G", "A", "B"], 
            "intervals": [0, 7, 5, 3], 
            "rhythms": [1.0, 0.5], 
        },
        "funk": {
            "scale": ["C", "D", "Eb", "F", "G", "A", "Bb"], 
            "intervals": [0, 7, 10, 5], 
            "rhythms": [0.25, 0.5, 1.0], 
        },
        "pop": {
            "scale": ["C", "D", "E", "F", "G", "A", "B"], 
            "intervals": [0, 7, 3, 5], 
            "rhythms": [0.5, 1.0], 
        },
        "jazz": {
            "scale": ["C", "D", "Eb", "F", "G", "A", "Bb"], 
            "intervals": [0, 3, 7, 9, 10], 
            "rhythms": [0.25, 0.5, 0.75, 1.0], 
        },
        "blues": {
            "scale": ["C", "Eb", "F", "F#", "G", "Bb"], 
            "intervals": [0, 3, 5, 6, 7, 10], 
            "rhythms": [0.5, 1.0], 
        },
        "reggae": {
            "scale": ["C", "D", "E", "F", "G", "A", "B"], 
            "intervals": [0, 7, 5, 3], 
            "rhythms": [0.5, 1.0, 1.5], 
        },
        "hiphop": {
            "scale": ["C", "D", "Eb", "G", "Ab"], 
            "intervals": [0, 3, 5, 7, 8, 10], 
            "rhythms": [0.25, 0.5, 1.0, 2.0], 
        },
        "random": { 
            "scale": ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'],
            "intervals": list(range(12)),
            "rhythms": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
        }
    }
    
    selected_style = styles.get(style, styles["random"]) 
    base_scale_notes_raw = selected_style["scale"] 
    base_rhythms = selected_style["rhythms"]
    
    chromatic_notes_std = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    try:
        root_idx_chromatic = chromatic_notes_std.index(key_root_note.upper())
    except ValueError:
        raise ValueError(f"유효하지 않은 루트 음: {key_root_note}")

    notes_sequence_output = [] 
    total_beats_per_loop = length_measures * 4 
    current_beats = 0

    while current_beats < total_beats_per_loop - 0.001: 
        remaining_beats = total_beats_per_loop - current_beats
        
        available_rhythms = [r for r in base_rhythms if r <= remaining_beats]
        if not available_rhythms:
            break
            
        rhythm_weights = [1.0/r for r in available_rhythms]
        if sum(rhythm_weights) == 0: # 모든 가중치가 0인 경우 방지
            rhythm_weights = [1.0] * len(available_rhythms)
        rhythm_weights = [w / sum(rhythm_weights) for w in rhythm_weights] # 정규화
        
        duration_unit = np.random.choice(available_rhythms, p=rhythm_weights) 
        
        selected_base_note_name = '' 
        selected_octave = 0
        accidental = '' 
        
        is_strong_beat = (current_beats % 1.0 == 0) and (duration_unit >= 0.5) 

        if is_strong_beat and np.random.rand() < 0.6: 
            chosen_interval_semitones = np.random.choice(selected_style["intervals"]) 
            
            midi_root_for_transpose = 12 * (octave + 1) + chromatic_notes_std.index(key_root_note.upper())
            target_midi_number = midi_root_for_transpose + chosen_interval_semitones
            
            target_octave_raw = target_midi_number // 12 - 1 
            target_note_idx_chromatic = target_midi_number % 12
            
            selected_octave = min(octave + 1, max(octave, target_octave_raw))
            if selected_octave < 0: selected_octave = 0 

            target_note_name_chromatic = chromatic_notes_std[target_note_idx_chromatic]
            
            # 스케일에 속하는 음표인지 확인하고, 임시표를 분리 저장
            found_in_scale = False
            for scale_note_raw in base_scale_notes_raw:
                if len(scale_note_raw) > 1: # Eb, C# 같은 경우
                    base = scale_note_raw[0]
                    acc = scale_note_raw[1]
                else: # C, D 같은 경우
                    base = scale_note_raw
                    acc = ''
                
                if (base == target_note_name_chromatic[0] and acc == target_note_name_chromatic[1:]) or \
                   (base == target_note_name_chromatic and acc == ''):
                    # 정확히 일치하거나, 임시표 없는 음이름이 일치하는 경우
                    selected_base_note_name = base
                    accidental = acc
                    found_in_scale = True
                    break
                # Db/C#, Eb/D# 등 이명동음 처리 (간단화)
                # Music21의 pitch.Pitch 자동 처리 (우선은)
                # 이명동음 처리는 get_note_frequency 함수가 맡음.
            
            if not found_in_scale:
                # 스케일에 없으면 루트 음으로 폴백 (임시표 없음)
                selected_base_note_name = key_root_note
                selected_octave = octave
                accidental = ''

        # 위 조건에 해당하지 않는 경우 (약박 또는 랜덤 선택)
        if not selected_base_note_name: 
            full_scale_parsed = []
            for current_chromatic_octave in range(max(0, octave), min(5, octave + 2)): 
                for scale_note_raw in base_scale_notes_raw:
                    if len(scale_note_raw) == 1: 
                        full_scale_parsed.append((scale_note_raw, current_chromatic_octave, ''))
                    else: 
                        full_scale_parsed.append((scale_note_raw[0], current_chromatic_octave, scale_note_raw[1]))

            if style != "random" and full_scale_parsed:
                chosen_note_info = full_scale_parsed[np.random.randint(len(full_scale_parsed))]
                selected_base_note_name = chosen_note_info[0]
                selected_octave = chosen_note_info[1]
                accidental = chosen_note_info[2]
            elif chromatic_notes_std: 
                rand_note_raw = np.random.choice(chromatic_notes_std)
                rand_octave_offset = np.random.randint(2) 
                selected_base_note_name = rand_note_raw[0]
                selected_octave = max(0, octave + rand_octave_offset)
                accidental = rand_note_raw[1] if len(rand_note_raw) > 1 else ''
            else: 
                selected_base_note_name = 'C'
                selected_octave = max(1, octave)
                accidental = ''

        notes_sequence_output.append((selected_base_note_name, selected_octave, duration_unit, False, accidental))
        current_beats += duration_unit
        
    if not notes_sequence_output:
        logger.warning("랜덤 생성기가 음표를 생성하지 못했습니다. 기본 시퀀스를 사용합니다.")
        notes_sequence_output = [('C', max(1, octave), 1.0, False, ''), ('G', max(1, octave), 1.0, False, '')]
        
    formatted_sequence = ", ".join([f"{n[0]}{n[4]}{n[1]} {float(n[2])}" for n in notes_sequence_output])
    logger.info(f"생성된 시퀀스: {formatted_sequence}")
    return formatted_sequence


# --- Gemini LLM 설정 및 호출 함수 ---
def generate_notes_with_gemini(api_key, genre, bpm, measures, key_note, octave):
    """Gemini AI를 사용한 베이스 라인 생성"""
    if not GEMINI_AVAILABLE:
        raise ValueError("Google Generative AI 라이브러리가 설치되지 않았습니다. pip install google-generativeai를 실행하세요.")
        
    if not api_key or not api_key.strip():
        raise ValueError("Gemini API 키가 필요합니다. 입력해주세요.")

    try:
        genai.configure(api_key=api_key.strip()) 
        gemini_model = genai.GenerativeModel('gemini-pro') 
        
        prompt = f"""Generate a bassline sequence in Python list of tuples format: [('NoteName', Octave, DurationUnit, Accidental), ...].
        Example: `[('C', 2, 1.0, ''), ('G', 2, 0.5, ''), ('A', 2, 0.5, '#'), ('F', 2, 1.0, 'b'), ('R', 4, 1.0, '')]`
        - NoteName: C, D, E, F, G, A, B (uppercase). For rests, use 'R'.
        - Octave: Integer (e.g., 2, 3). For rests, use 4.
        - DurationUnit: Float (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0 etc. for quarterLength)
        - Accidental: '', '#', or 'b' (empty string if no accidental). For rests, use ''.

        Requirements:
        - Genre: {genre}
        - BPM: {bpm}
        - Key: {key_note}
        - Starting octave: {octave}
        - Total beats: {measures * 4} (each measure has 4 beats)
        - Use octaves {octave} to {octave + 1} mainly, but allow adjacent octaves for musicality.
        - Make it musically appropriate for {genre}.
        - Ensure total duration matches Total beats. Use rests (R) if necessary to fill measures.
        
        Return ONLY the Python list, no explanation or extra text.
        """
        
        response = gemini_model.generate_content(prompt)
        text_response = response.text.strip()
        
        if text_response.startswith('```python') and text_response.endswith('```'):
            text_response = text_response[len('```python'):-len('```')].strip()
        elif text_response.startswith('```') and text_response.endswith('```'):
            text_response = text_response[3:-3].strip()
            
        if text_response.startswith('list(') and text_response.endswith(')'):
             text_response = text_response[len('list('):-len(')')].strip()
        
        logger.info(f"Gemini 응답 원본: {text_response[:500]}...")
        
        parsed_sequence_raw = ast.literal_eval(text_response)
        
        if not isinstance(parsed_sequence_raw, list):
            raise ValueError("AI가 Python 리스트를 반환하지 않았습니다.")
            
        final_parsed_sequence_for_output = [] 
        for item in parsed_sequence_raw:
            if not isinstance(item, tuple) or len(item) != 4: 
                raise ValueError(f"AI가 잘못된 형식의 튜플을 반환했습니다: {item}. (NoteName, Octave, DurationUnit, Accidental) 형식이어야 합니다.")
            
            note_name = str(item[0]).upper()
            octave = int(item[1])
            duration = float(item[2])
            accidental = str(item[3])

            is_rest = (note_name == 'R')

            if not (0 <= octave <= 8):
                logger.warning(f"AI 생성 옥타브 범위 오류: {octave}. 2로 강제 조정합니다.")
                octave = 2
            if not (0 < duration <= 8):
                logger.warning(f"AI 생성 음표 길이 오류: {duration}. 1.0으로 강제 조정합니다.")
                duration = 1.0
            if accidental not in ['', '#', 'b']:
                 logger.warning(f"AI 생성 임시표 오류: '{accidental}'. 빈 문자열로 강제 조정합니다.")
                 accidental = ''

            final_parsed_sequence_for_output.append((note_name, octave, duration, is_rest, accidental))
        
        formatted_sequence_str = ", ".join([f"{n[0]}{n[4]}{n[1]} {float(n[2])}" for n in final_parsed_sequence_for_output])
        logger.info(f"Gemini 생성 시퀀스 (프론트엔드 형식): {formatted_sequence_str}")
        return formatted_sequence_str
        
    except ValueError as ve:
        logger.error(f"AI 응답 파싱 또는 유효성 검사 오류: {str(ve)}")
        raise ValueError(f"AI 응답 파싱 중 오류: {str(ve)}. AI가 요청된 형식을 따르지 않았을 수 있습니다.")
    except Exception as e:
        logger.error(f"Gemini API 호출 오류: {e}")
        logger.error(traceback.format_exc())
        raise ValueError(f"Gemini API 호출 중 오류: {str(e)}. API 키가 유효한지 확인하거나 네트워크 상태를 점검하세요.")

# --- Flask 웹 라우트 ---

@app.route('/')
def index_page(): 
    """메인 페이지 렌더링"""
    return render_template('index.html')

@app.route('/generate_notes', methods=['POST'])
def generate_notes_route(): 
    """사용자 요청에 따라 랜덤 또는 AI 방식으로 악보 시퀀스를 생성하여 반환"""
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

        generated_notes_str = ""

        if generation_mode == "random":
            generated_notes_str = create_random_bass_loop_by_style(
                genre, key_note, octave, length_measures, bpm
            )
        elif generation_mode == "ai":
            if not GEMINI_AVAILABLE:
                return jsonify({'status': 'error', 'message': 'AI 모드를 사용하려면 `google-generativeai` 라이브러리가 필요합니다.'}), 400
            api_key_from_ui = request.form.get('gemini_api_key_input', '').strip()
            if not api_key_from_ui:
                return jsonify({'status': 'error', 'message': 'AI 모드를 사용하려면 Gemini API 키를 입력해야 합니다.'}), 400
            generated_notes_str = generate_notes_with_gemini(
                api_key_from_ui, genre, bpm, length_measures, key_note, octave
            )
        else:
            return jsonify({'status': 'error', 'message': '알 수 없는 생성 모드입니다.'}), 400

        if generated_notes_str:
            return jsonify({'status': 'success', 'notes': generated_notes_str})
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
        png_data_base64, text_score = generate_music21_score_with_fallback(notes_sequence, bpm, key_signature)
        
        if png_data_base64:
            return jsonify({"status": "success", "image_data": png_data_base64, "format": "png"})
        elif text_score:
            return jsonify({"status": "success", "text_data": text_score, "format": "text"})
        else:
            return jsonify({"status": "error", "message": "악보 이미지/텍스트 생성에 실패했습니다."}), 500
            
    except Exception as e:
        logger.error(f"악보 이미지 생성 오류: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": f"악보 생성 오류: {str(e)}"}), 500

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
        'gemini_available': GEMINI_AVAILABLE
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
    return render_template('index.html'), 404

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
    logger.info(f"Gemini AI 사용 가능: {GEMINI_AVAILABLE}")
    logger.info(f"Music21 사용 가능: {MUSIC21_AVAILABLE}")
    logger.info(f"MIDI 사용 가능: {MIDI_AVAILABLE}")
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
