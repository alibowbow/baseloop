import numpy as np
from scipy.io.wavfile import write as write_wav
from flask import Flask, render_template, request, send_file
import io 
import re
import os
import ast # For safe parsing of LLM output

# Google Gemini 라이브러리 (AI 모드 사용 시 필수)
import google.generativeai as genai

# 로컬 개발용 .env 파일 로딩 (배포 시에는 Render 환경 변수 사용)
from dotenv import load_dotenv
load_dotenv() 

app = Flask(__name__)

# --- 오디오 생성 관련 상수 및 기본 함수 ---
SAMPLE_RATE = 44100
MAX_AMPLITUDE = 0.5 * (2**15 - 1)

def get_note_frequency(note_name, octave):
    notes_in_octave = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    try:
        note_index = notes_in_octave.index(note_name.upper())
    except ValueError:
        raise ValueError(f"유효하지 않은 음표 이름: {note_name}. 다음 중 하나여야 합니다: {notes_in_octave}")

    A4_FREQ = 440.0
    midi_note_offset = (octave - 4) * 12 + note_index - notes_in_octave.index('A')
    
    return A4_FREQ * (2 ** (midi_note_offset / 12.0))

def generate_note_waveform(frequency, duration_seconds, sample_rate, amplitude):
    num_samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, num_samples, endpoint=False)
    waveform = amplitude * np.sin(2 * np.pi * frequency * t)
    
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
        notes_sequence.append((note_name, octave, duration))
    return notes_sequence


def create_bass_loop_from_parsed_sequence(notes_sequence, bpm, num_loops):
    """
    파싱된 음표 시퀀스 ([('C', 2, 1.0), ...])를 기반으로 오디오 버퍼 생성
    """
    quarter_note_duration = 60 / bpm 
    full_loop_waveform = np.array([])

    for note_info in notes_sequence:
        note_name, octave_val, duration_units = note_info
        freq = get_note_frequency(note_name, octave_val)
        actual_duration = duration_units * quarter_note_duration
        note_waveform = generate_note_waveform(freq, actual_duration, SAMPLE_RATE, MAX_AMPLITUDE)
        full_loop_waveform = np.concatenate((full_loop_waveform, note_waveform))
    
    final_waveform = np.tile(full_loop_waveform, num_loops)
    if len(final_waveform) > 0: # 정규화 전에 비어있지 않은지 확인
        final_waveform = final_waveform / np.max(np.abs(final_waveform)) * MAX_AMPLITUDE 
    audio_data_int16 = final_waveform.astype(np.int16)

    buffer = io.BytesIO()
    write_wav(buffer, SAMPLE_RATE, audio_data_int16)
    buffer.seek(0)
    
    return buffer


# --- 스타일 기반 랜덤 생성 함수 (랜덤 모드) ---
def create_random_bass_loop_by_style(
    style, key_root_note, octave, length_measures, bpm
):
    print(f"Generating {style} bass loop (Key: {key_root_note}{octave}, BPM: {bpm}, Measures: {length_measures})")
    
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
    root_idx_chromatic = notes_in_chromatic_octave.index(key_root_note.upper())

    for current_chromatic_octave in range(octave, octave + 2): # 보통 베이스는 2옥타브 정도의 음역대를 사용
        for note_name_in_chromatic in notes_in_chromatic_octave:
            if note_name_in_chromatic in base_scale_notes:
                full_scale.append((note_name_in_chromatic, current_chromatic_octave))

    notes_sequence = []
    total_beats_per_loop = length_measures * 4 
    current_beats = 0

    while current_beats < total_beats_per_loop:
        rhythm_weights = [1.0/r for r in base_rhythms]
        rhythm_weights = [w / sum(rhythm_weights) for w in rhythm_weights]
        
        duration_unit = np.random.choice(base_rhythms, p=rhythm_weights) 
        
        selected_note_info = None
        
        is_strong_beat = (current_beats % 1.0 == 0) and (duration_unit >= 0.5) 

        if is_strong_beat and np.random.rand() < 0.6: 
            chosen_interval_semitones = np.random.choice(selected_style["intervals"])
            
            root_midi_note_base_val = 12 * (octave + 1) + notes_in_chromatic_octave.index(key_root_note.upper())
            target_midi_number = root_midi_note_base_val + chosen_interval_semitones
            
            target_octave = target_midi_number // 12 - 1 
            target_note_name = notes_in_chromatic_octave[target_midi_number % 12]

            # 최종 옥타브 보정 (너무 높으면 내림)
            if target_octave > octave + 1:
                target_octave = octave + 1 # max 1 octave higher than base
            elif target_octave < octave:
                target_octave = octave # min at base octave
                
            selected_note_info = (target_note_name, target_octave)
        
        if selected_note_info is None or selected_note_info[1] < octave: 
            if style != "random" and len(full_scale) > 0:
                selected_note_info = full_scale[np.random.randint(len(full_scale))]
            elif len(notes_in_chromatic_octave) > 0:
                rand_note_idx = np.random.randint(len(notes_in_chromatic_octave))
                rand_octave_offset = np.random.randint(2)
                selected_note_info = (notes_in_chromatic_octave[rand_note_idx], octave + rand_octave_offset)
            else:
                selected_note_info = ('C', octave)

        note_name, final_octave = selected_note_info
        
        notes_sequence.append((note_name, final_octave, duration_unit))
        current_beats += duration_unit
        
    formatted_sequence = ", ".join([f"{n[0]}{n[1]} {float(n[2])}" for n in notes_sequence])
    return formatted_sequence


# --- Gemini LLM 설정 및 호출 함수 (AI 모드) ---
def generate_notes_with_gemini(api_key, genre, bpm, measures, key_note, octave):
    # API 키를 프런트엔드에서 받음 (보안 경고)
    if not api_key:
        raise ValueError("Gemini API 키가 필요합니다. 입력해주세요.")

    genai.configure(api_key=api_key) 
    gemini_model = genai.GenerativeModel('gemini-pro') 
    
    prompt = f"""Generate a bassline sequence in Python tuple list format: [('NoteName', Octave, DurationUnit), ...].
    Example: `[('C', 2, 1.0), ('G', 2, 0.5), ('A', 2, 0.5), ('F', 2, 1.0)]`
    The genre is {genre}. BPM is {bpm}. The bassline should start around {key_note}{octave} and be approximately {measures * 4} beats long (each measure has 4 beats).
    Focus on common bass notes in lower octaves (mostly octave {octave} to {octave + 1}) and rhythms appropriate for {genre}.
    Ensure the sequence can be parsed as a Python list of tuples. Do not include any explanation or extra text, just the Python list.
    Provide about {measures * 4} beats of notes in the sequence (e.g. four 1.0 duration notes for a 1-measure loop).
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        text_response = response.text.strip()
        
        if text_response.startswith('```python') and text_response.endswith('```'):
            text_response = text_response[len('```python'):-len('```')].strip()
        if text_response.startswith('list(') and text_response.endswith(')'):
             text_response = text_response[len('list('):-len(')')].strip()
        
        parsed_sequence = ast.literal_eval(text_response)
        
        if not isinstance(parsed_sequence, list):
            raise ValueError("AI did not return a Python list.")
        for item in parsed_sequence:
            if not (isinstance(item, tuple) and len(item) == 3 and
                    isinstance(item[0], str) and isinstance(item[1], int) and isinstance(item[2], (float, int))):
                raise ValueError(f"AI returned invalid item format: {item}. Expected ('NoteName', Octave, DurationUnit)")
                
        return ", ".join([f"{n[0]}{n[1]} {float(n[2])}" for n in parsed_sequence])
        
    except ValueError as ve:
        raise ValueError(f"AI 응답 파싱 또는 유효성 검사 오류: {str(ve)}. API 키가 유효하거나 AI 응답 형식을 확인하세요.")
    except Exception as e:
        raise ValueError(f"Gemini API 호출 중 오류: {str(e)}. API 키가 유효하거나 요청 할당량을 확인하세요.")


# --- Flask 웹 라우트 ---

@app.route('/')
def index_page(): # 함수 이름을 더 일반적인 이름으로 변경
    """메인 페이지 렌더링 - 이제 'index.html'을 렌더링합니다."""
    default_bpm = 120
    default_loops = 2
    default_length = 4 # 마디 단위
    default_genre = "rock" 
    default_key_note = "C"
    default_octave = 2 
    default_generation_mode = "random" 
    
    return render_template('index.html', # <--- 이 부분을 'index.html'로 수정!
                           default_bpm=default_bpm, 
                           default_loops=default_loops,
                           default_length=default_length,
                           default_genre=default_genre,
                           default_key_note=default_key_note,
                           default_octave=default_octave,
                           default_generation_mode=default_generation_mode,
                           recommended_notes_str="C2 1.0, G2 1.0, A2 1.0, F2 1.0" # 기본 베이스라인 예시
                          )


@app.route('/generate_notes', methods=['POST'])
def generate_notes():
    """사용자 요청에 따라 랜덤 또는 AI 방식으로 악보 시퀀스를 생성하여 반환"""
    try:
        generation_mode = request.form.get('generation_mode', 'random') 
        
        bpm = int(request.form.get('bpm_input', 120))
        length_measures = int(request.form.get('length_input', 4))
        genre = request.form.get('genre_input', 'rock')
        key_note = request.form.get('key_note_input', 'C')
        octave = int(request.form.get('octave_input', 2))

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
        print(f"Error during note generation: {ve}")
        return {'status': 'error', 'message': str(ve)}, 400
    except Exception as e:
        import traceback
        print(traceback.format_exc()) 
        return {'status': 'error', 'message': f"서버 오류 발생: {str(e)}"}, 500


@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    """사용자 입력 악보 시퀀스를 기반으로 베이스 루프 오디오 생성"""
    try:
        notes_sequence_str = request.form['notes_sequence_input']
        bpm = int(request.form['bpm_input'])
        num_loops = int(request.form['num_loops_input'])

        # 파싱 함수를 재활용
        notes_sequence = parse_note_sequence_string(notes_sequence_str)

        if not notes_sequence:
            return "악보 시퀀스가 비어 있습니다. 음표를 생성하거나 입력해주세요.", 400

        audio_buffer = create_bass_loop_from_parsed_sequence(notes_sequence, bpm, num_loops)

        return send_file(audio_buffer, 
                         mimetype='audio/wav', 
                         as_attachment=True, 
                         download_name='bass_loop.wav')

    except ValueError as ve:
        return f"악보 파싱 오류: {str(ve)}", 400
    except Exception as e:
        import traceback
        print(traceback.format_exc()) 
        return f"오류 발생: {str(e)}. 서버 로그를 확인하세요.", 500

if __name__ == '__main__':
    load_dotenv() 

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
