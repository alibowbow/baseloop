import numpy as np
from scipy.io.wavfile import write as write_wav
from flask import Flask, render_template, request, send_file, Response
import io 
import re
import os
import ast # For safe parsing of LLM output
import traceback
import logging
import tempfile
import base64

# Music21 라이브러리 (악보 생성)
try:
    from music21 import stream, note, meter, tempo, clef, bar, duration, pitch, key, layout
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False
    print("Warning: music21 라이브러리가 설치되지 않았습니다. 전문 악보 생성을 사용할 수 없습니다.")

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

app = Flask(__name__)

# --- 오디오 생성 관련 상수 및 기본 함수 ---
SAMPLE_RATE = 44100
MAX_AMPLITUDE = 0.5 * (2**15 - 1)

def get_note_frequency(note_name, octave):
    """음표 이름과 옥타브를 받아 주파수를 계산합니다."""
    notes_in_octave = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    try:
        note_index = notes_in_octave.index(note_name.upper())
    except ValueError:
        raise ValueError(f"유효하지 않은 음표 이름: {note_name}. 다음 중 하나여야 합니다: {notes_in_octave}")

    A4_FREQ = 440.0
    midi_note_offset = (octave - 4) * 12 + note_index - notes_in_octave.index('A')
    
    return A4_FREQ * (2 ** (midi_note_offset / 12.0))

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
        envelope[num_samples - release_samples:] *= np.linspace(1, 0, release_samples)
    
    waveform *= envelope
    
    return waveform

def parse_note_sequence_string(notes_sequence_str):
    """
    "C2 1.0, G2 0.5" 형태의 문자열을 파싱하여 [('C', 2, 1.0), ...] 형태로 변환
    """
    if not notes_sequence_str or not notes_sequence_str.strip():
        raise ValueError("악보 시퀀스가 비어 있습니다.")
        
    notes_sequence = []
    for note_entry in notes_sequence_str.split(','):
        note_entry = note_entry.strip()
        if not note_entry:
            continue
        
        # "C#2 1.0" 형태의 문자열 파싱 (음표 이름, 옥타브, 길이)
        match = re.match(r"([A-Ga-g]#?)(\d+)\s+(\d+(\.\d+)?)", note_entry)
        if not match:
            raise ValueError(f"유효하지 않은 음표 형식: '{note_entry}'. 'C2 1.0' 형식이어야 합니다.")

        note_name = match.group(1)
        octave = int(match.group(2))
        duration = float(match.group(3))
        
        # 유효성 검사
        if octave < 0 or octave > 8:
            raise ValueError(f"옥타브는 0-8 범위여야 합니다: {octave}")
        if duration <= 0 or duration > 8:
            raise ValueError(f"음표 길이는 0보다 크고 8보다 작아야 합니다: {duration}")
            
        notes_sequence.append((note_name, octave, duration))
    
    if not notes_sequence:
        raise ValueError("파싱된 음표가 없습니다. 올바른 형식으로 입력해주세요.")
        
    return notes_sequence

def create_bass_loop_from_parsed_sequence(notes_sequence, bpm, num_loops):
    """
    파싱된 음표 시퀀스 ([('C', 2, 1.0), ...])를 기반으로 오디오 버퍼 생성
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
        note_name, octave_val, duration_units = note_info
        try:
            freq = get_note_frequency(note_name, octave_val)
            actual_duration = duration_units * quarter_note_duration
            note_waveform = generate_note_waveform(freq, actual_duration, SAMPLE_RATE, MAX_AMPLITUDE)
            full_loop_waveform = np.concatenate((full_loop_waveform, note_waveform))
        except Exception as e:
            logger.error(f"음표 생성 실패 {note_name}{octave_val}: {e}")
            # 음표 생성 실패시 무음으로 대체
            silence_samples = int(duration_units * quarter_note_duration * SAMPLE_RATE)
            silence = np.zeros(silence_samples)
            full_loop_waveform = np.concatenate((full_loop_waveform, silence))
    
    if len(full_loop_waveform) == 0:
        raise ValueError("생성된 오디오 데이터가 없습니다.")
    
    # 루프 반복
    final_waveform = np.tile(full_loop_waveform, num_loops)
    
    # 정규화
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
    (png_path, text_score) 형태 튜플을 돌려줍니다.
    """
    if not MUSIC21_AVAILABLE:
        raise ValueError("Music21 라이브러리가 설치되지 않았습니다.")

    # 첫 호출 시 Music21 환경 세팅
    if not setup_music21_environment():
        raise RuntimeError("Music21 환경 설정 실패")

    try:
        score = stream.Score()
        score.append(tempo.MetronomeMark(number=bpm))
        score.append(meter.TimeSignature("4/4"))
        score.append(music21_key.Key(key_signature))  # ✅ KeySignature→Key

        part = stream.Part()
        part.append(clef.BassClef())
        part.append(tempo.MetronomeMark(number=bpm))
        part.append(meter.TimeSignature("4/4"))
        part.append(music21_key.Key(key_signature))

        for n_name, octv, dur in notes_sequence:
            if n_name.upper() == "R":
                part.append(note.Rest(quarterLength=dur))
            else:
                try:
                    n = note.Note(f"{n_name}{octv}")
                    n.duration = duration.Duration(quarterLength=dur)
                    part.append(n)
                except Exception as e:
                    logger.warning(f"음표 변환 실패 → Rest 처리: {n_name}{octv} / {e}")
                    part.append(note.Rest(quarterLength=dur))

        score.append(part)

        tmp_dir = tempfile.mkdtemp()
        png_path = os.path.join(tmp_dir, "score.png")

        try:
            score.write("musicxml.png", fp=png_path)
            if os.path.exists(png_path) and os.path.getsize(png_path) > 0:
                return png_path, None
            raise RuntimeError("PNG 생성 실패")
        except Exception as e:
            logger.warning(f"MuseScore PNG 실패: {e}")
            text_score = generate_text_score(notes_sequence, bpm)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return None, text_score
    except Exception as e:
        logger.error(f"악보 생성 실패: {e}")
        raise

def generate_text_score(notes_sequence, bpm):
    """텍스트 기반 악보 표현 (Music21 실패시 대안)"""
    text_lines = []
    text_lines.append(f"베이스 라인 악보 (BPM: {bpm})")
    text_lines.append("=" * 40)
    text_lines.append("")
    
    current_measure = 1
    current_beats = 0
    measure_notes = []
    
    for note_name, octave, duration_val in notes_sequence:
        if current_beats + duration_val > 4:
            # 현재 마디 완료
            if measure_notes:
                text_lines.append(f"마디 {current_measure}: {' | '.join(measure_notes)}")
                current_measure += 1
                measure_notes = []
                current_beats = 0
        
        note_str = f"{note_name}{octave}({duration_val})"
        measure_notes.append(note_str)
        current_beats += duration_val
    
    # 마지막 마디
    if measure_notes:
        text_lines.append(f"마디 {current_measure}: {' | '.join(measure_notes)}")
    
    return "\n".join(text_lines)

# --- MIDI 생성 함수 ---
def note_name_to_midi(note_name, octave):
    """음표 이름을 MIDI 노트 번호로 변환"""
    notes = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 
             'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
    return (octave + 1) * 12 + notes[note_name.upper()]

def generate_midi_file(notes_sequence, bpm):
    """MIDI 파일 생성"""
    if not MIDI_AVAILABLE:
        raise ValueError("Mido 라이브러리가 설치되지 않았습니다.")
    
    try:
        # MIDI 파일 생성
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        
        # 프로그램 체인지 (베이스 악기 설정)
        track.append(Message('program_change', channel=0, program=32, time=0))  # Acoustic Bass
        
        # 템포 설정
        tempo_microseconds = mido.bpm2tempo(bpm)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo_microseconds))
        
        # 음표들을 MIDI 이벤트로 변환
        ticks_per_beat = mid.ticks_per_beat
        
        for note_name, octave, duration in notes_sequence:
            if note_name.upper() != 'R':  # 쉼표가 아닌 경우
                try:
                    midi_note = note_name_to_midi(note_name, octave)
                    velocity = 80
                    duration_ticks = int(duration * ticks_per_beat)
                    
                    # Note On
                    track.append(Message('note_on', channel=0, note=midi_note, 
                                       velocity=velocity, time=0))
                    # Note Off
                    track.append(Message('note_off', channel=0, note=midi_note, 
                                       velocity=0, time=duration_ticks))
                except Exception as e:
                    logger.warning(f"MIDI 음표 생성 실패 {note_name}{octave}: {e}")
                    # 쉼표로 처리
                    duration_ticks = int(duration * ticks_per_beat)
                    track.append(Message('note_on', channel=0, note=60, 
                                       velocity=0, time=duration_ticks))
            else:
                # 쉼표 처리
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
    
    for note_name, octave, duration in notes_sequence:
        if note_name.upper() == 'R':
            ly_note = 'r'
        else:
            ly_note = note_name.lower().replace('#', 'is').replace('b', 'es')
            # 옥타브 조정 (LilyPond 표기법)
            if octave <= 2:
                ly_note += ',' * (3 - octave)
            elif octave > 3:
                ly_note += "'" * (octave - 3)
        
        # 음표 길이 변환
        if duration == 4.0:
            ly_note += '1'
        elif duration == 2.0:
            ly_note += '2'
        elif duration == 1.0:
            ly_note += '4'
        elif duration == 0.5:
            ly_note += '8'
        elif duration == 0.25:
            ly_note += '16'
        else:
            # 기타 길이는 4분음표로 기본 설정
            ly_note += '4'
        
        lilypond_notes.append(ly_note)
    
    return ' '.join(lilypond_notes)

def generate_lilypond_score(notes_sequence, bpm):
    """LilyPond 형식 악보 텍스트 생성"""
    lilypond_notes = convert_to_lilypond_notes(notes_sequence)
    
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
    {lilypond_notes}
  }}
  \\layout {{ }}
  \\midi {{ }}
}}
"""
    return lilypond_code

# --- 스타일 기반 랜덤 생성 함수 (기존 코드 유지) ---
def create_random_bass_loop_by_style(style, key_root_note, octave, length_measures, bpm):
    """스타일에 따른 랜덤 베이스 라인 생성 (기존 코드 유지)"""
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
    base_scale_notes = selected_style["scale"]
    base_rhythms = selected_style["rhythms"]
    
    full_scale = []
    
    notes_in_chromatic_octave = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # 루트 노트가 유효한지 확인
    try:
        root_idx_chromatic = notes_in_chromatic_octave.index(key_root_note.upper())
    except ValueError:
        raise ValueError(f"유효하지 않은 루트 음: {key_root_note}")

    for current_chromatic_octave in range(max(0, octave), min(5, octave + 2)): 
        for note_name_in_chromatic in notes_in_chromatic_octave:
            if note_name_in_chromatic in base_scale_notes:
                full_scale.append((note_name_in_chromatic, current_chromatic_octave))

    notes_sequence = []
    total_beats_per_loop = length_measures * 4 
    current_beats = 0

    while current_beats < total_beats_per_loop:
        # 남은 박자 수 계산
        remaining_beats = total_beats_per_loop - current_beats
        
        # 가능한 리듬 중에서 남은 박자를 초과하지 않는 것만 선택
        available_rhythms = [r for r in base_rhythms if r <= remaining_beats]
        if not available_rhythms:
            # 남은 박자에 맞는 리듬이 없으면 나머지를 쉼표로 채움
            break
            
        rhythm_weights = [1.0/r for r in available_rhythms]
        rhythm_weights = [w / sum(rhythm_weights) for w in rhythm_weights]
        
        duration_unit = np.random.choice(available_rhythms, p=rhythm_weights) 
        
        selected_note_info = None
        
        is_strong_beat = (current_beats % 1.0 == 0) and (duration_unit >= 0.5) 

        if is_strong_beat and np.random.rand() < 0.6: 
            chosen_interval_semitones = np.random.choice(selected_style["intervals"]) 
            
            root_midi_note_base_val = 12 * (octave + 1) + notes_in_chromatic_octave.index(key_root_note.upper())
            target_midi_number = root_midi_note_base_val + chosen_interval_semitones
            
            target_octave = target_midi_number // 12 - 1 
            target_note_name = notes_in_chromatic_octave[target_midi_number % 12]

            # 옥타브 범위 제한
            if target_octave > octave + 1:
                target_octave = octave + 1
            elif target_octave < max(0, octave):
                target_octave = max(0, octave)
                
            selected_note_info = (target_note_name, target_octave)
        
        if selected_note_info is None or selected_note_info[1] < 0: 
            if style != "random" and len(full_scale) > 0:
                selected_note_info = full_scale[np.random.randint(len(full_scale))]
            elif len(notes_in_chromatic_octave) > 0:
                rand_note_idx = np.random.randint(len(notes_in_chromatic_octave))
                rand_octave_offset = np.random.randint(2)
                selected_note_info = (notes_in_chromatic_octave[rand_note_idx], max(0, octave + rand_octave_offset))
            else:
                selected_note_info = ('C', max(1, octave))

        note_name, final_octave = selected_note_info
        
        notes_sequence.append((note_name, final_octave, duration_unit))
        current_beats += duration_unit
        
    if not notes_sequence:
        # 최소한의 기본 시퀀스 생성
        notes_sequence = [('C', max(1, octave), 1.0), ('G', max(1, octave), 1.0)]
        
    formatted_sequence = ", ".join([f"{n[0]}{n[1]} {float(n[2])}" for n in notes_sequence])
    logger.info(f"생성된 시퀀스: {formatted_sequence}")
    return formatted_sequence

# --- Gemini LLM 설정 및 호출 함수 (기존 코드 유지) ---
def generate_notes_with_gemini(api_key, genre, bpm, measures, key_note, octave):
    """Gemini AI를 사용한 베이스 라인 생성"""
    if not GEMINI_AVAILABLE:
        raise ValueError("Google Generative AI 라이브러리가 설치되지 않았습니다. pip install google-generativeai를 실행하세요.")
        
    if not api_key or not api_key.strip():
        raise ValueError("Gemini API 키가 필요합니다. 입력해주세요.")

    try:
        genai.configure(api_key=api_key.strip()) 
        gemini_model = genai.GenerativeModel('gemini-pro') 
        
        prompt = f"""Generate a bassline sequence in Python tuple list format: [('NoteName', Octave, DurationUnit), ...].
        Example: `[('C', 2, 1.0), ('G', 2, 0.5), ('A', 2, 0.5), ('F', 2, 1.0)]`
        
        Requirements:
        - Genre: {genre}
        - BPM: {bpm}
        - Key: {key_note}
        - Starting octave: {octave}
        - Total beats: {measures * 4} (each measure has 4 beats)
        - Use octaves {octave} to {octave + 1} mainly
        - Duration units should be: 0.25, 0.5, 1.0, 1.5, 2.0, etc.
        - Make it musically appropriate for {genre}
        
        Return ONLY the Python list, no explanation or extra text.
        """
        
        response = gemini_model.generate_content(prompt)
        text_response = response.text.strip()
        
        # 응답 정리
        if text_response.startswith('```python') and text_response.endswith('```'):
            text_response = text_response[len('```python'):-len('```')].strip()
        elif text_response.startswith('```') and text_response.endswith('```'):
            text_response = text_response[3:-3].strip()
            
        if text_response.startswith('list(') and text_response.endswith(')'):
             text_response = text_response[len('list('):-len(')')].strip()
        
        logger.info(f"Gemini 응답: {text_response[:100]}...")
        
        # 안전한 파싱
        parsed_sequence = ast.literal_eval(text_response)
        
        if not isinstance(parsed_sequence, list):
            raise ValueError("AI가 Python 리스트를 반환하지 않았습니다.")
            
        for item in parsed_sequence:
            if not (isinstance(item, tuple) and len(item) == 3 and
                    isinstance(item[0], str) and isinstance(item[1], int) and isinstance(item[2], (float, int))):
                raise ValueError(f"AI가 잘못된 형식을 반환했습니다: {item}. ('NoteName', Octave, DurationUnit) 형식이어야 합니다.")
        
        # 추가 유효성 검사
        for note_name, oct, dur in parsed_sequence:
            if oct < 0 or oct > 8:
                raise ValueError(f"옥타브 범위 오류: {oct}")
            if dur <= 0 or dur > 8:
                raise ValueError(f"음표 길이 오류: {dur}")
                
        formatted_sequence = ", ".join([f"{n[0]}{n[1]} {float(n[2])}" for n in parsed_sequence])
        logger.info(f"Gemini 생성 시퀀스: {formatted_sequence}")
        return formatted_sequence
        
    except ValueError as ve:
        raise ValueError(f"AI 응답 파싱 또는 유효성 검사 오류: {str(ve)}")
    except Exception as e:
        logger.error(f"Gemini API 호출 오류: {e}")
        raise ValueError(f"Gemini API 호출 중 오류: {str(e)}. API 키가 유효한지 확인하세요.")

# --- Flask 웹 라우트 ---

@app.route('/')
def index_page(): 
    """메인 페이지 렌더링"""
    default_values = {
        'default_bpm': 120,
        'default_loops': 2,
        'default_length': 4,
        'default_genre': "rock",
        'default_key_note': "C",
        'default_octave': 2,
        'default_generation_mode': "random",
        'recommended_notes_str': "C2 1.0, G2 1.0, A2 1.0, F2 1.0",
        'music21_available': MUSIC21_AVAILABLE,
        'midi_available': MIDI_AVAILABLE,
        'gemini_available': GEMINI_AVAILABLE
    }
    
    return render_template('index.html', **default_values)

@app.route('/generate_notes', methods=['POST'])
def generate_notes():
    """사용자 요청에 따라 랜덤 또는 AI 방식으로 악보 시퀀스를 생성하여 반환"""
    try:
        generation_mode = request.form.get('generation_mode', 'random') 
        
        # 입력값 검증
        try:
            bpm = int(request.form.get('bpm_input', 120))
            length_measures = int(request.form.get('length_input', 4))
            octave = int(request.form.get('octave_input', 2))
        except ValueError:
            return {'status': 'error', 'message': 'BPM, 마디 길이, 옥타브는 숫자여야 합니다.'}, 400
        
        genre = request.form.get('genre_input', 'rock')
        key_note = request.form.get('key_note_input', 'C')

        # 범위 검증
        if not (30 <= bpm <= 240):
            return {'status': 'error', 'message': 'BPM은 30-240 범위여야 합니다.'}, 400
        if not (1 <= length_measures <= 16):
            return {'status': 'error', 'message': '마디 길이는 1-16 범위여야 합니다.'}, 400
        if not (0 <= octave <= 4):
            return {'status': 'error', 'message': '옥타브는 0-4 범위여야 합니다.'}, 400

        generated_notes_str = ""

        if generation_mode == "random":
            generated_notes_str = create_random_bass_loop_by_style(
                genre, key_note, octave, length_measures, bpm
            )
        elif generation_mode == "ai":
            api_key_from_ui = request.form.get('gemini_api_key_input', '').strip()
            generated_notes_str = generate_notes_with_gemini(
                api_key_from_ui, genre, bpm, length_measures, key_note, octave
            )
        else:
            return {'status': 'error', 'message': '알 수 없는 생성 모드입니다.'}, 400

        if generated_notes_str:
            return {'status': 'success', 'notes': generated_notes_str}
        else:
            return {'status': 'error', 'message': '악보 생성에 실패했습니다.'}, 500

    except ValueError as ve:
        logger.error(f"악보 생성 중 ValueError: {ve}")
        return {'status': 'error', 'message': str(ve)}, 400
    except Exception as e:
        logger.error(f"악보 생성 중 예외 발생: {e}")
        logger.error(traceback.format_exc())
        return {'status': 'error', 'message': f"서버 오류 발생: {str(e)}"}, 500

@app.route('/generate_audio', methods=['POST'])
def generate_audio():
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

        # 악보 파싱
        notes_sequence = parse_note_sequence_string(notes_sequence_str)

        # 오디오 생성
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
def generate_score_image():
    """Music21으로 생성한 전문 악보 이미지 반환"""
    try:
        notes_sequence_str = request.args.get('notes', '')
        bpm = int(request.args.get('bpm', 120))
        key_signature = request.args.get('key', 'C')
        
        if not notes_sequence_str:
            return Response("악보 시퀀스가 필요합니다.", status=400, mimetype='text/plain')
        
        notes_sequence = parse_note_sequence_string(notes_sequence_str)
        png_path, text_score = generate_music21_score_with_fallback(notes_sequence, bpm, key_signature)
        
        if png_path and os.path.exists(png_path):
            return send_file(png_path, mimetype='image/png')
        elif text_score:
            # 텍스트 악보를 이미지로 변환하여 반환
            return Response(text_score, mimetype='text/plain')
        else:
            return Response("악보 이미지 생성에 실패했습니다.", status=500, mimetype='text/plain')
            
    except Exception as e:
        logger.error(f"악보 이미지 생성 오류: {e}")
        return Response(f"악보 생성 오류: {str(e)}", status=500, mimetype='text/plain')

@app.route('/generate_midi', methods=['POST'])
def generate_midi():
    """MIDI 파일 생성 및 다운로드"""
    try:
        notes_sequence_str = request.form.get('notes_sequence_input', '').strip()
        bpm = int(request.form.get('bpm_input', 120))
        
        if not notes_sequence_str:
            return Response("악보 시퀀스가 비어 있습니다.", status=400, mimetype='text/plain')
        
        notes_sequence = parse_note_sequence_string(notes_sequence_str)
        midi_file = generate_midi_file(notes_sequence, bpm)
        
        # MIDI 파일을 바이트 스트림으로 변환
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
def generate_lilypond():
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
    return {
        'music21': MUSIC21_AVAILABLE,
        'midi': MIDI_AVAILABLE,
        'gemini': GEMINI_AVAILABLE
    }

@app.errorhandler(404)
def not_found_error(error):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 오류: {error}")
    return "Internal server error", 500

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
