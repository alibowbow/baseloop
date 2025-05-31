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
import base64 # Base64 인코딩/디코딩용 추가

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
    이 함수는 앱 런타임에 Music21이 MuseScore를 찾도록 설정합니다.
    """
    global _M21_CONFIGURED

    if _M21_CONFIGURED or not MUSIC21_AVAILABLE:
        return _M21_CONFIGURED

    try:
        from music21 import environment
        us = environment.UserSettings()

        musescore_path = os.getenv("MUSESCORE_PATH")
        if not musescore_path:
            # Render 환경에서는 /usr/bin/musescore3 이 일반적
            musescore_path = "/usr/bin/musescore3" 
            logger.warning(f"MUSESCORE_PATH 환경 변수가 설정되지 않았습니다. 기본 경로 '{musescore_path}'를 사용합니다. MuseScore가 설치된 실제 경로로 설정해주세요.")

        us["musescoreDirectPNGPath"] = musescore_path
        us["musicxmlPath"] = musescore_path

        # Headless 환경 변수 설정 (Dockerfile 또는 start.sh에서 이미 설정했을 수도 있음)
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        os.environ.setdefault("DISPLAY", ":99")

        logger.info(f"[Music21 설정] MuseScore 경로 등록 완료: {musescore_path}")
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

# Flask 앱 초기화
# template_folder와 static_folder를 명시하여 Render.com 배포 환경에서 파일 경로 문제를 방지합니다.
app = Flask(__name__, template_folder='templates', static_folder='static')

# --- 오디오 생성 관련 상수 및 기본 함수 ---
SAMPLE_RATE = 44100
MAX_AMPLITUDE = 0.5 * (2**15 - 1)

def get_note_frequency(note_name_chromatic, octave):
    """
    음표 이름 (C, C#, D, Eb 등 완전한 이름)과 옥타브를 받아 주파수를 계산합니다.
    note_name_chromatic은 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B' 중 하나여야 합니다.
    """
    # 크로매틱 스케일과 해당 노트의 A4로부터의 반음 간격 (A4=440Hz, MIDI 69)
    # C4 = MIDI 60
    chromatic_notes_midi_offset = {
        'C': -9, 'C#': -8, 'D': -7, 'D#': -6, 'E': -5, 'F': -4,
        'F#': -3, 'G': -2, 'G#': -1, 'A': 0, 'A#': 1, 'B': 2
    }
    
    try:
        # 입력된 음이름을 표준화 (Eb -> D#, Fb -> E, B# -> C 등)
        # Music21의 pitch.Pitch.name 로직과 유사하게, 내부적으로 반음을 #으로 통일하는 경우가 많음
        # 여기서는 get_note_frequency가 C, C#, D, D# 등만 받으므로, 입력값을 그대로 사용
        if note_name_chromatic not in chromatic_notes_midi_offset:
            # Db, Eb, Gb, Ab, Bb 등을 #으로 변환
            if note_name_chromatic == 'Db': note_name_chromatic = 'C#'
            elif note_name_chromatic == 'Eb': note_name_chromatic = 'D#'
            elif note_name_chromatic == 'Fb': note_name_chromatic = 'E' # 실제로는 E
            elif note_name_chromatic == 'Gb': note_name_chromatic = 'F#'
            elif note_name_chromatic == 'Ab': note_name_chromatic = 'G#'
            elif note_name_chromatic == 'Bb': note_name_chromatic = 'A#'
            elif note_name_chromatic == 'B#': note_name_chromatic = 'C' # 실제로는 C
            elif note_name_chromatic == 'Cb': note_name_chromatic = 'B' # 실제로는 B
            
            if note_name_chromatic not in chromatic_notes_midi_offset:
                raise ValueError(f"유효하지 않은 크로매틱 음표 이름: {note_name_chromatic}")

        # A4 (440Hz)는 MIDI 노트 69
        # MIDI 노트 번호 = 69 + 반음 간격 (A4로부터)
        # 옥타브 4의 A는 69 (offset 0)
        # 옥타브 4의 C는 60 (offset -9)
        midi_note_number = (octave * 12) + chromatic_notes_midi_offset[note_name_chromatic] + 57 # C0의 MIDI 노트는 12 (0옥타브 C)
        # A4=440Hz = MIDI 69
        # C4=261.63Hz = MIDI 60
        # Formula: freq = 440 * (2^((midi_note_number - 69)/12))
        
        A4_MIDI = 69
        A4_FREQ = 440.0
        
        return A4_FREQ * (2 ** ((midi_note_number - A4_MIDI) / 12.0))

    except ValueError as ve:
        raise ValueError(f"주파수 계산 오류: {note_name_chromatic}{octave} - {ve}")
    except KeyError:
        raise ValueError(f"지원하지 않는 음표 이름: {note_name_chromatic}")


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
        
        # 음표 파싱 (예: C#2 1.0, Eb3 0.5)
        # ([A-Ga-g])       -> 음이름 (C, D, E, F, G, A, B)
        # (#|b)?           -> # 또는 b (선택적)
        # (\d+)            -> 옥타브 숫자
        # \s+              -> 공백
        # (\d+(?:\.\d+)?)  -> 박자 (1, 0.5, 1.25 등)
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

        # 쉼표 파싱 (예: R 1.0)
        match_rest = re.match(r"R\s+(\d+(?:\.\d+)?)", note_entry, re.IGNORECASE)
        if match_rest:
            duration = float(match_rest.group(1))
            if not (0 < duration <= 8):
                raise ValueError(f"쉼표 길이는 0보다 크고 8보다 작거나 같아야 합니다: {duration} (쉼표: {note_entry})")
            parsed_notes.append(('R', 4, duration, True, '')) # 쉼표는 옥타브와 임시표 중요하지 않음, is_rest = True
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
                freq = 0 # 쉼표는 무음
            else:
                # get_note_frequency 함수가 'C#', 'D#', 'F#', 'G#', 'A#' 또는 'Db', 'Eb' 등으로 변환된 음이름을 받도록 처리
                note_name_for_freq = note_name
                if accidental == '#':
                    note_name_for_freq += '#'
                elif accidental == 'b':
                    # 플랫 음표를 #으로 변환하여 get_note_frequency에 전달
                    chromatic_scale = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                    base_index = chromatic_scale.index(note_name)
                    flat_index = (base_index - 1) % 12
                    note_name_for_freq = chromatic_scale[flat_index] # 예를 들어 Eb는 D#으로
                
                freq = get_note_frequency(note_name_for_freq, octave_val)
            
            actual_duration = duration_units * quarter_note_duration
            note_waveform = generate_note_waveform(freq, actual_duration, SAMPLE_RATE, MAX_AMPLITUDE)
            full_loop_waveform = np.concatenate((full_loop_waveform, note_waveform))
        except Exception as e:
            logger.error(f"오디오 음표 생성 실패 {note_name}{accidental}{octave_val} ({duration_units}박): {e}")
            # 음표 생성 실패시 무음으로 대체
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
                    # Music21 pitch 객체에 임시표 포함 (예: "C#", "Eb")
                    # Music21은 'b'를 플랫 심볼로, '#'를 샵 심볼로 해석합니다.
                    p = pitch.Pitch(f"{n_name}{accidental}") 
                    n = note.Note(p)
                    n.octave = octv
                    n.duration = duration.Duration(quarterLength=dur)
                    part.append(n)
                except Exception as e:
                    logger.warning(f"Music21 음표 변환 실패 → Rest 처리: {n_name}{accidental}{octv} ({dur}박) / {e}")
                    part.append(note.Rest(quarterLength=dur))

        score.append(part)

        temp_dir = tempfile.mkdtemp()
        png_path = os.path.join(temp_dir, "score.png")

        try:
            # score.write("musicxml.png")는 MuseScore를 호출하여 PNG를 만듭니다.
            score.write("musicxml.png", fp=png_path) 
            
            if os.path.exists(png_path) and os.path.getsize(png_path) > 0:
                with open(png_path, "rb") as f:
                    png_data = f.read()
                return base64.b64encode(png_data).decode('utf-8'), None # Base64 인코딩하여 반환
            
            raise RuntimeError("MuseScore로 PNG 생성 실패: 파일이 없거나 비어 있습니다.")
        except Exception as e:
            logger.warning(f"MuseScore PNG 생성 실패: {e}")
            logger.warning("MuseScore PNG 실패: PNG 생성 실패 (MuseScore 설치 및 환경 변수 MUSESCORE_PATH 설정 권장, 또는 xvfb 필요)")
            text_score = generate_text_score(notes_sequence, bpm)
            return None, text_score # PNG 실패 시 텍스트 악보 반환
    except Exception as e:
        logger.error(f"악보 생성 중 예상치 못한 오류 발생: {e}")
        logger.error(traceback.format_exc())
        text_score = generate_text_score(notes_sequence, bpm)
        return None, text_score
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir) # 임시 디렉토리 삭제

def generate_text_score(notes_sequence, bpm):
    """텍스트 기반 악보 표현 (Music21 실패시 대안)"""
    text_lines = []
    text_lines.append(f"베이스 라인 악보 (BPM: {bpm})")
    text_lines.append("=" * 40)
    text_lines.append("")
    
    current_measure = 1
    current_beats = 0
    measure_notes_str_list = [] # 마디 내 음표 문자열 리스트

    for note_name, octave, duration_val, is_rest, accidental in notes_sequence:
        # 현재 음표를 추가할 때 마디 박자를 초과하는지 확인
        if current_beats + duration_val > 4.001 and current_beats > 0: # 4.001로 작은 오차 허용
            # 현재 마디 완료
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
    
    # 마지막 마디 처리
    if measure_notes_str_list:
        text_lines.append(f"마디 {current_measure}: {' | '.join(measure_notes_str_list)}")
    
    return "\n".join(text_lines)

# --- MIDI 생성 함수 ---
def note_name_to_midi(note_name, octave, accidental=''):
    """음표 이름을 MIDI 노트 번호로 변환 (accidental 포함)"""
    # Music21과 유사하게, C, D, E, F, G, A, B에 대한 기본 반음 인덱스
    notes = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    
    # C0 (MIDI 12번)을 기준으로 계산
    midi_base = (octave * 12) + notes[note_name.upper()]
    
    # 임시표 적용
    if accidental == '#':
        midi_base += 1
    elif accidental == 'b':
        midi_base -= 1
        
    return midi_base

def generate_midi_file(notes_sequence, bpm):
    """MIDI 파일 생성"""
    if not MIDI_AVAILABLE:
        raise ValueError("Mido 라이브러리가 설치되지 않았습니다.")
    
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
                    velocity = 80 # 음량 (0-127)
                    duration_ticks = int(duration * ticks_per_beat)
                    
                    track.append(Message('note_on', channel=0, note=midi_note, 
                                       velocity=velocity, time=0))
                    track.append(Message('note_off', channel=0, note=midi_note, 
                                       velocity=0, time=duration_ticks))
                except Exception as e:
                    logger.warning(f"MIDI 음표 생성 실패 {note_name}{accidental}{octave}: {e}")
                    duration_ticks = int(duration * ticks_per_beat)
                    track.append(Message('note_on', channel=0, note=60, # 가상의 노트, velocity=0
                                       velocity=0, time=duration_ticks))
            else:
                duration_ticks = int(duration * ticks_per_beat)
                track.append(Message('note_on', channel=0, note=60, # 가상의 노트, velocity=0
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
            ly_note_str += note_name.lower() # LilyPond는 소문자 음이름 사용
            if accidental == '#':
                ly_note_str += 'is'
            elif accidental == 'b':
                ly_note_str += 'es'
            
            # LilyPond 옥타브 조정 (중앙 C는 c')
            # LilyPond의 기본 옥타브는 c' (C4)입니다. 
            # 베이스 클레프의 낮은 음역을 고려하여 옥타브를 LilyPond 표기법에 맞게 조정
            if octave == 3: # C3, D3, E3... (MIDI 48-59) -> c, d, e...
                ly_note_str += ','
            elif octave == 2: # C2, D2... (MIDI 36-47) -> c,, d,, e,,...
                ly_note_str += ',,'
            elif octave == 1: # C1, D1... (MIDI 24-35) -> c,,, d,,, e,,,...
                ly_note_str += ',,,'
            elif octave == 0: # C0, D0... (MIDI 12-23) -> c,,,, d,,,, e,,,,...
                ly_note_str += ',,,,'
            elif octave == 4: # C4, D4... (MIDI 60-71) -> c', d', e'...
                pass # LilyPond 기본 c' 표기 (c, d, e...)
            elif octave == 5: # C5, D5... (MIDI 72-83) -> c'', d'', e''...
                ly_note_str += "'"
            elif octave == 6: # C6, D6... (MIDI 84-95) -> c''' d''' e'''...
                ly_note_str += "''"
            else:
                logger.warning(f"LilyPond: 지원하지 않는 옥타브 {octave}. 기본 LilyPond 옥타브 표기법을 따릅니다.")

        # 음표 길이 변환 (LilyPond Duration)
        if duration == 4.0:
            ly_note_str += '1' # 온음표
        elif duration == 3.0: 
            ly_note_str += '2.' # 점2분음표
        elif duration == 2.0:
            ly_note_str += '2' # 2분음표
        elif duration == 1.5: 
            ly_note_str += '4.' # 점4분음표
        elif duration == 1.0:
            ly_note_str += '4' # 4분음표
        elif duration == 0.75:
            ly_note_str += '8.' # 점8분음표
        elif duration == 0.5:
            ly_note_str += '8' # 8분음표
        elif duration == 0.25:
            ly_note_str += '16' # 16분음표
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
            "scale": ["C", "D", "E", "F", "G", "A", "B"], # 기본 Major Scale
            "intervals": [0, 7, 5, 3], # Root, 5th, 4th, 3rd (relative to root)
            "rhythms": [1.0, 0.5], # Quarter, Eighth
        },
        "funk": {
            "scale": ["C", "D", "Eb", "F", "G", "A", "Bb"], # Mixolydian or Dorian feel
            "intervals": [0, 7, 10, 5], # Root, 5th, b7th, 4th
            "rhythms": [0.25, 0.5, 1.0], # 16th, Eighth, Quarter
        },
        "pop": {
            "scale": ["C", "D", "E", "F", "G", "A", "B"], # Major Scale
            "intervals": [0, 7, 3, 5], # Root, 5th, 3rd, 4th
            "rhythms": [0.5, 1.0], # Eighth, Quarter
        },
        "jazz": {
            "scale": ["C", "D", "Eb", "F", "G", "A", "Bb"], # Dorian or Minor scales
            "intervals": [0, 3, 7, 9, 10], # Root, b3rd, 5th, 6th, b7th
            "rhythms": [0.25, 0.5, 0.75, 1.0], # 16th, Eighth, Dotted Eighth, Quarter
        },
        "blues": {
            "scale": ["C", "Eb", "F", "F#", "G", "Bb"], # Blues Scale
            "intervals": [0, 3, 5, 6, 7, 10], # Root, b3rd, 4th, b5th, 5th, b7th
            "rhythms": [0.5, 1.0], # Eighth, Quarter
        },
        "reggae": {
            "scale": ["C", "D", "E", "F", "G", "A", "B"], 
            "intervals": [0, 7, 5, 3], 
            "rhythms": [0.5, 1.0, 1.5], # Eighth, Quarter, Dotted Quarter (often off-beat)
        },
        "hiphop": {
            "scale": ["C", "D", "Eb", "G", "Ab"], # Minor Pentatonic or similar
            "intervals": [0, 3, 5, 7, 8, 10], 
            "rhythms": [0.25, 0.5, 1.0, 2.0], # 16th, Eighth, Quarter, Half
        },
        "random": { 
            "scale": ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'], # Full chromatic
            "intervals": list(range(12)), # Any interval
            "rhythms": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0], # Various rhythms
        }
    }
    
    selected_style = styles.get(style, styles["random"]) 
    base_scale_notes_raw = selected_style["scale"] # 예를 들어 'Eb'가 포함될 수 있음
    base_rhythms = selected_style["rhythms"]
    
    # Music21 또는 오디오 계산에 필요한 표준 크로매틱 음표 리스트
    chromatic_notes_std = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # 루트 노트가 유효한지 확인
    try:
        root_idx_chromatic = chromatic_notes_std.index(key_root_note.upper())
    except ValueError:
        raise ValueError(f"유효하지 않은 루트 음: {key_root_note}")

    notes_sequence_output = [] # List of (note_name, octave, duration, is_rest, accidental)
    total_beats_per_loop = length_measures * 4 
    current_beats = 0

    while current_beats < total_beats_per_loop - 0.001: # 약간의 오차 허용
        remaining_beats = total_beats_per_loop - current_beats
        
        available_rhythms = [r for r in base_rhythms if r <= remaining_beats]
        if not available_rhythms:
            break
            
        rhythm_weights = [1.0/r for r in available_rhythms]
        rhythm_weights = [w / sum(rhythm_weights) for w in rhythm_weights]
        
        duration_unit = np.random.choice(available_rhythms, p=rhythm_weights) 
        
        selected_note_name = ''
        selected_octave = 0
        accidental = '' # 임시표
        
        is_strong_beat = (current_beats % 1.0 == 0) and (duration_unit >= 0.5) 

        if is_strong_beat and np.random.rand() < 0.6: # 강박에 주요음 선택 비중 높임
            chosen_interval_semitones = np.random.choice(selected_style["intervals"]) 
            
            root_midi_note_base_val = 12 * (octave + 1) + chromatic_notes_std.index(key_root_note.upper())
            target_midi_number = root_midi_note_base_val + chosen_interval_semitones
            
            # MIDI 번호에서 옥타브와 음이름 추출 (임시표 포함 여부는 여기서 결정)
            target_octave_raw = target_midi_number // 12 - 1 
            target_note_idx_chromatic = target_midi_number % 12
            
            # 옥타브 범위 제한 (선택 옥타브 및 그 위 옥타브까지)
            selected_octave = min(octave + 1, max(octave, target_octave_raw))
            if selected_octave < 0: selected_octave = 0 # 최소 옥타브 보장

            # Music21에 전달할 수 있는 형태로 음이름과 임시표 분리
            target_note_name_chromatic = chromatic_notes_std[target_note_idx_chromatic]
            
            # 베이스 스케일 노트에 해당하는 음이름이 있으면 그대로 사용
            # (예: "Eb" 스케일이 있다면, D# 대신 Eb를 선택하게)
            if target_note_name_chromatic in base_scale_notes_raw:
                selected_note_name = target_note_name_chromatic
            elif target_note_name_chromatic.replace('b', '#') in base_scale_notes_raw: # Eb -> D#
                selected_note_name = target_note_name_chromatic.replace('b', '#')
            else: # 스케일에 없는 경우, 가장 가까운 스케일 음으로 대체하거나 루트 음으로
                # 간단화를 위해, 스케일에 없으면 루트 음으로 폴백
                selected_note_name = key_root_note
                selected_octave = octave
                accidental = ''

            # 선택된 음이름에서 임시표 추출
            if len(selected_note_name) > 1:
                accidental = selected_note_name[1] # # 또는 b
                selected_note_name = selected_note_name[0] # 기본 음이름 (예: "C#") -> "C"

        # 위 조건에 맞지 않거나, 옥타브가 유효하지 않은 경우
        if not selected_note_name: 
            # 풀 스케일 리스트를 생성 (Music21에 맞는 형태: (음이름, 옥타브, 임시표))
            full_scale_parsed = []
            for current_chromatic_octave in range(max(0, octave), min(5, octave + 2)): 
                for scale_note_raw in base_scale_notes_raw:
                    if len(scale_note_raw) == 1: # C, D, E
                        full_scale_parsed.append((scale_note_raw, current_chromatic_octave, ''))
                    else: # C#, Eb
                        full_scale_parsed.append((scale_note_raw[0], current_chromatic_octave, scale_note_raw[1]))

            if style != "random" and full_scale_parsed:
                chosen_note_info = full_scale_parsed[np.random.randint(len(full_scale_parsed))]
                selected_note_name = chosen_note_info[0]
                selected_octave = chosen_note_info[1]
                accidental = chosen_note_info[2]
            elif chromatic_notes_std: # 마지막 비상 fallback (아무 음이나)
                rand_note_idx = np.random.randint(len(chromatic_notes_std))
                rand_octave_offset = np.random.randint(2) 
                selected_note_name = chromatic_notes_std[rand_note_idx]
                selected_octave = max(0, octave + rand_octave_offset)
                accidental = '' # 랜덤이므로 임시표는 일단 없다고 가정
                if len(selected_note_name) > 1: # C# 같은 경우
                    accidental = selected_note_name[1]
                    selected_note_name = selected_note_name[0]
            else: # 정말 아무것도 생성 못할 때의 비상
                selected_note_name = 'C'
                selected_octave = max(1, octave)
                accidental = ''

        notes_sequence_output.append((selected_note_name, selected_octave, duration_unit, False, accidental))
        current_beats += duration_unit
        
    if not notes_sequence_output:
        logger.warning("랜덤 생성기가 음표를 생성하지 못했습니다. 기본 시퀀스를 사용합니다.")
        notes_sequence_output = [('C', max(1, octave), 1.0, False, ''), ('G', max(1, octave), 1.0, False, '')]
        
    # 프론트엔드 형식으로 변환 (예: C#2 1.0)
    formatted_sequence = ", ".join([f"{n[0]}{n[4]}{n[1]} {float(n[2])}" for n in notes_sequence_output]) # n[4]는 accidental
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
        
        # 프롬프트 변경: accidental을 따로 반환하도록 요청
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
        
        # Markdown 코드 블록 제거
        if text_response.startswith('```python') and text_response.endswith('```'):
            text_response = text_response[len('```python'):-len('```')].strip()
        elif text_response.startswith('```') and text_response.endswith('```'):
            text_response = text_response[3:-3].strip()
        
        # list() 생성자 형식 제거 (예: list([('C', 2, 1.0, '')]))
        if text_response.startswith('list(') and text_response.endswith(')'):
             text_response = text_response[len('list('):-len(')')].strip()
        
        logger.info(f"Gemini 응답 원본: {text_response[:500]}...") # 로그 길이 늘림
        
        # 안전한 파싱
        parsed_sequence_raw = ast.literal_eval(text_response) # 일단 AI가 준 그대로 파싱
        
        if not isinstance(parsed_sequence_raw, list):
            raise ValueError("AI가 Python 리스트를 반환하지 않았습니다.")
            
        final_parsed_sequence_for_output = [] # 실제 앱에서 사용될 파싱된 시퀀스
        for item in parsed_sequence_raw:
            if not isinstance(item, tuple) or len(item) != 4: # 정확히 4개의 원소를 가진 튜플
                raise ValueError(f"AI가 잘못된 형식의 튜플을 반환했습니다: {item}. (NoteName, Octave, DurationUnit, Accidental) 형식이어야 합니다.")
            
            note_name = str(item[0]).upper()
            octave = int(item[1])
            duration = float(item[2])
            accidental = str(item[3]) # Accidental 필드 그대로 사용

            is_rest = (note_name == 'R')

            # 유효성 검사 (AI가 이상한 값 줄 경우 대비)
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
        
        # 최종적으로 프론트엔드 형식으로 변환 (accidental 포함)
        formatted_sequence_str = ", ".join([f"{n[0]}{n[4]}{n[1]} {float(n[2])}" for n in final_parsed_sequence_for_output]) # n[4]는 accidental
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
    # 프론트엔드 HTML 파일은 `templates` 폴더 안에 있어야 합니다.
    # 예: `templates/index.html`
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
        # Music21이 설치되어 있다면, MuseScore 환경 설정도 시도합니다.
        # 이 호출은 _M21_CONFIGURED 플래그를 업데이트할 것입니다.
        music21_configured = setup_music21_environment() 

    return jsonify({
        'music21_available': MUSIC21_AVAILABLE, # Music21 라이브러리 import 성공 여부
        'music21_configured_for_png': music21_configured, # MuseScore 경로 설정 성공 여부
        'midi_available': MIDI_AVAILABLE, # Mido 라이브러리 import 성공 여부
        'gemini_available': GEMINI_AVAILABLE # Gemini 라이브러리 import 성공 여부
    })

@app.errorhandler(404)
def not_found_error(error):
    # Flask가 template_folder를 모르기 때문에 명시해줌
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 오류: {error}")
    # Render.com에서 상세한 오류 메시지를 제공하지 않으므로 일반적인 메시지 반환
    return "Internal server error. Check server logs for more details.", 500

if __name__ == '__main__':
    try:
        load_dotenv()
    except:
        pass # .env 파일이 없어도 앱 실행에는 지장 없음

    # Render.com은 $PORT 환경 변수를 통해 포트를 할당합니다.
    port = int(os.environ.get('PORT', 10000)) # 기본값 10000

    # 디버그 모드는 개발 환경에서만 True로 설정해야 합니다.
    # Render.com 배포 환경에서는 False로 설정하는 것이 좋습니다.
    # FLASK_DEBUG 환경 변수로 제어 가능
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"서버 시작 - 포트: {port}, 디버그: {debug_mode}")
    logger.info(f"Gemini AI 사용 가능: {GEMINI_AVAILABLE}")
    logger.info(f"Music21 사용 가능: {MUSIC21_AVAILABLE}")
    logger.info(f"MIDI 사용 가능: {MIDI_AVAILABLE}")
    
    # 0.0.0.0으로 바인딩하여 외부 접속 허용 (Render.com에 필요)
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
