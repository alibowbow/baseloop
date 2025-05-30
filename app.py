import numpy as np
from scipy.io.wavfile import write as write_wav
from flask import Flask, render_template, request, send_file
import io 
import re
import os

app = Flask(__name__)

# --- 오디오 생성 관련 상수 및 함수 (이전 코드에서 복사) ---
SAMPLE_RATE = 44100
MAX_AMPLITUDE = 0.5 * (2**15 - 1)

def get_note_frequency(note_name, octave):
    notes_in_octave = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    try:
        note_index = notes_in_octave.index(note_name.upper())
    except ValueError:
        raise ValueError(f"Invalid note name: {note_name}. Must be one of: {notes_in_octave}")

    A4_FREQ = 440.0
    A4_MIDI_NUM = 69
    
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

# --- 새로운 랜덤 스타일 기반 베이스 루프 생성 함수 ---
def create_random_bass_loop_by_style(
    style,        # "rock", "funk", "pop", "random"
    key_root_note, # 'C', 'G', 'A' 등 (시작음)
    octave,       # 옥타브
    length_measures, # 루프 길이 (마디)
    bpm,
    num_loops,
    target_filename="generated_bass_loop.wav"
):
    print(f"Generating {style} bass loop (Key: {key_root_note}{octave}, BPM: {bpm}, Measures: {length_measures})")
    
    # --- 스타일별 규칙 정의 ---
    # 각 스타일에 따라 사용될 수 있는 음정 (스케일)과 리듬 패턴의 경향성을 정의
    styles = {
        "rock": {
            "scale": ["C", "D", "E", "F", "G", "A", "B"], # Major scale as base, for simple demonstration
            "intervals": [0, 7, 5, 3], # Root, 5th, 4th, 3rd (Major scale intervals for power/rock feel)
            "rhythms": [1.0, 0.5], # 4분음표, 8분음표 위주
            "complexity": 0.3 # 낮은 복잡도
        },
        "funk": {
            "scale": ["C", "D", "Eb", "F", "G", "A", "Bb"], # Mixolydian/Dorian for funk, or pentatonic
            "intervals": [0, 7, 10, 5], # Root, 5th, flat 7th, 4th (typical funky)
            "rhythms": [0.25, 0.5, 1.0], # 16분음표, 8분음표 위주 (syncopation)
            "complexity": 0.7 # 높은 복잡도
        },
        "pop": {
            "scale": ["C", "D", "E", "F", "G", "A", "B"], # Major scale
            "intervals": [0, 7, 3, 5], # Root, 5th, 3rd, 4th
            "rhythms": [0.5, 1.0], # 8분음표, 4분음표 (규칙적)
            "complexity": 0.4 # 중간 복잡도
        },
        "jazz": {
            "scale": ["C", "D", "Eb", "F", "G", "A", "Bb"], # Dorian or other jazz scales
            "intervals": [0, 3, 7, 9, 10], # Root, minor 3rd, 5th, 6th, flat 7th
            "rhythms": [0.25, 0.5, 0.75, 1.0], # 더 다양한 리듬 (스윙 가능성)
            "complexity": 0.8
        }
    }
    
    selected_style = styles.get(style, styles["rock"]) # 기본은 rock
    base_scale_notes = selected_style["scale"] # 선택된 스타일의 기준 스케일 (음표 이름)
    base_rhythms = selected_style["rhythms"] # 선택된 스타일의 기준 리듬
    
    # 현재 시작음을 기준으로 음계 전체 만들기
    # 예: 'C' Major -> C, D, E, F, G, A, B, C...
    full_scale = []
    root_idx = base_scale_notes.index(key_root_note.upper())
    
    # 2개 옥타브에 걸친 스케일 생성 (낮은 베이스 기타 음정을 위함)
    for current_octave_offset in range(2): 
        current_octave = octave + current_octave_offset
        for i in range(len(base_scale_notes)):
            # Rotate scale to start from key_root_note
            scale_note_name = base_scale_notes[(root_idx + i) % len(base_scale_notes)]
            full_scale.append((scale_note_name, current_octave if (root_idx + i) < len(base_scale_notes) else current_octave + 1))
            
    # 너무 길게 만들지 않기 위해 유효한 범위 설정
    # 예를 들어 C2 시작이라면, C2-G3 정도까지만 허용. (베이스는 너무 고음은 안 씀)
    # 단순화하여 2개 옥타브 (e.g. C2-C4)로 설정

    # 베이스라인 음표 시퀀스 생성
    notes_sequence = []
    total_beats_per_loop = length_measures * 4 # 4/4 박자 기준 한 마디 4박자
    current_beats = 0

    while current_beats < total_beats_per_loop:
        # 리듬 선택: 스타일별 경향성 반영하여 랜덤 선택
        duration_unit = np.random.choice(base_rhythms, p=np.array(base_rhythms)/np.sum(base_rhythms)) 
        
        # 음정 선택: 시작음을 중심으로 스타일별 음정을 고려하여 선택
        # 여기서는 단순히 풀 스케일에서 랜덤으로 뽑지만, 실제로는 코드 진행을 따르거나, 
        # 중요한 박자에 루트음을 배치하는 등의 고급 로직이 필요
        
        # 더 프로페셔널하게: 주요 박자에 루트/5도를 우선적으로 배치하는 경향 추가
        selected_note_info = None
        
        # 첫 박이나 중요한 박자 (quarter_note에 맞춰 1.0, 2.0 등)에는 키루트나 5도 선택 확률 높이기
        is_strong_beat = (current_beats % 1.0 == 0) and (duration_unit >= 0.5) # 4분음표 또는 8분음표 시작점

        if is_strong_beat and np.random.rand() < 0.7: # 70% 확률로 루트나 5도 우선
            chosen_interval = np.random.choice([0, 7]) # Root (0), 5th (7 semitones)
            
            # Key_root_note에 따라 midi_note_number로 변환 후 offset
            midi_note_offset = (octave - 4) * 12 + get_note_frequency(key_root_note, 4) - get_note_frequency('A', 4) # Adjust for correct root midi index
            # This logic needs a better way to map interval to a specific note based on the scale and root note
            
            # For simplicity with notes_in_octave mapping
            notes_in_octave = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            
            root_midi = notes_in_octave.index(key_root_note.upper()) + octave * 12 # Roughly correct base MIDI note
            
            if chosen_interval == 0:
                target_midi_note = root_midi
            elif chosen_interval == 7: # A 5th up from root
                target_midi_note = root_midi + 7
            
            # Convert midi note back to (name, octave) for consistency
            target_octave = target_midi_note // 12 - 1 # Roughly. (midi_note 12=C0, so oct=1, then C1.. A0=21)
            target_note_name = notes_in_octave[target_midi_note % 12]
            
            selected_note_info = (target_note_name, target_octave)

            # Re-adjust selected_note_info to be within a reasonable bass octave range like C2-G3
            # If target_octave too high, lower it by 12 semitones or use lower octave mapping
            if target_octave > octave + 1 and target_note_name != key_root_note: # Simple heuristic
                 selected_note_info = (target_note_name, octave) # Stick to the lower selected octave

        if selected_note_info is None: # 아니면 그냥 랜덤하게 선택
            selected_note_info = full_scale[np.random.randint(len(full_scale))]

        note_name, final_octave = selected_note_info

        # Optional: ensure final_octave stays in bass range
        if not (octave <= final_octave <= octave + 1): # Example range C2-C4 (octave 2, 3)
            final_octave = octave # default to base octave if outside preferred range

        notes_sequence.append((note_name, final_octave, duration_unit))
        current_beats += duration_unit
        
    print(f"Generated Sequence: {notes_sequence}")

    # 실제 오디오 데이터 생성 부분 (기존과 동일)
    quarter_note_duration = 60 / bpm
    full_loop_waveform = np.array([])
    
    for note_info in notes_sequence:
        note_name, octave_val, duration_units = note_info
        freq = get_note_frequency(note_name, octave_val)
        actual_duration = duration_units * quarter_note_duration
        note_waveform = generate_note_waveform(freq, actual_duration, SAMPLE_RATE, MAX_AMPLITUDE)
        full_loop_waveform = np.concatenate((full_loop_waveform, note_waveform))
    
    final_waveform = np.tile(full_loop_waveform, num_loops)
    final_waveform = final_waveform / np.max(np.abs(final_waveform)) * MAX_AMPLITUDE
    audio_data_int16 = final_waveform.astype(np.int16)
    
    buffer = io.BytesIO()
    write_wav(buffer, SAMPLE_RATE, audio_data_int16)
    buffer.seek(0)
    
    return buffer

# --- Flask 웹 라우트 ---

@app.route('/')
def index_style_generator():
    """메인 페이지 렌더링 (스타일 기반 생성)"""
    default_bpm = 100
    default_loops = 2
    default_length = 4 # 마디 단위
    default_genre = "rock" 
    default_key_note = "C"
    default_octave = 2 # 베이스 기타 기본 옥타브
    return render_template('index_style_generator.html', 
                           default_bpm=default_bpm, 
                           default_loops=default_loops,
                           default_length=default_length,
                           default_genre=default_genre,
                           default_key_note=default_key_note,
                           default_octave=default_octave)

@app.route('/generate_styled_bass', methods=['POST'])
def generate_styled_bass():
    """스타일 기반 랜덤 베이스 루프를 생성하고 WAV 파일로 반환"""
    try:
        bpm = int(request.form['bpm_input'])
        num_loops = int(request.form['num_loops_input'])
        length_measures = int(request.form['length_input'])
        genre = request.form['genre_input']
        key_note = request.form['key_note_input']
        octave = int(request.form['octave_input'])

        audio_buffer = create_random_bass_loop_by_style(
            genre, key_note, octave, length_measures, bpm, num_loops
        )

        return send_file(audio_buffer, 
                         mimetype='audio/wav', 
                         as_attachment=True, 
                         download_name=f"{genre}_bass_loop.wav")

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
