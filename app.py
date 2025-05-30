import numpy as np
from scipy.io.wavfile import write as write_wav
from flask import Flask, render_template, request, send_file
import io 
import re
import os # 이 줄을 추가합니다.

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

def create_bass_loop_in_memory(notes_sequence, bpm, num_loops):
    quarter_note_duration = 60 / bpm 
    full_loop_waveform = np.array([])

    for note_info in notes_sequence:
        note_name, octave, duration_units = note_info
        freq = get_note_frequency(note_name, octave)
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
def index():
    default_notes = "C2 1.0, G2 1.0, A2 1.0, F2 1.0"
    default_bpm = 100
    default_loops = 2
    return render_template('index.html', 
                           default_notes=default_notes, 
                           default_bpm=default_bpm, 
                           default_loops=default_loops)

@app.route('/generate_bass', methods=['POST'])
def generate_bass():
    try:
        notes_sequence_str = request.form['notes_sequence_input']
        bpm = int(request.form['bpm_input'])
        num_loops = int(request.form['num_loops_input'])

        notes_sequence = []
        for note_entry in notes_sequence_str.split(','):
            note_entry = note_entry.strip()
            if not note_entry:
                continue
            match = re.match(r"([A-Ga-g]#?)(\d+)\s+(\d+(\.\d+)?)", note_entry)
            if not match:
                return f"Invalid note format: '{note_entry}'. It should be like 'C2 1.0'.", 400

            note_name = match.group(1)
            octave = int(match.group(2))
            duration = float(match.group(3))

            notes_sequence.append((note_name, octave, duration))

        if not notes_sequence:
            return "Please enter a note sequence.", 400

        audio_buffer = create_bass_loop_in_memory(notes_sequence, bpm, num_loops)

        return send_file(audio_buffer, 
                        mimetype='audio/wav', 
                        as_attachment=True,
                        download_name='bass_loop.wav')

    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    # Heroku/Render 환경에서는 PORT 환경 변수를 사용하고, 아니면 로컬 개발용 기본 포트 5000 사용
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
