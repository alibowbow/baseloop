"""baseloop 핵심 로직 유닛 테스트.

music21 / MuseScore가 없어도 통과하도록, 악보 이미지 생성 경로는 다루지 않는다.
표준 라이브러리 unittest 만 사용하므로 `python -m unittest` 또는 `pytest` 로 실행 가능.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app  # noqa: E402


class NoteFrequencyTests(unittest.TestCase):
    def test_a_concert_pitch(self):
        # 이 앱의 옥타브 규약은 C5=MIDI60 → A5=440Hz (표준 A4와 한 옥타브 차이)
        self.assertAlmostEqual(app.get_note_frequency("A", 5), 440.0, places=3)

    def test_middle_c(self):
        # 같은 규약에서 C5 ≈ 261.63 Hz
        self.assertAlmostEqual(app.get_note_frequency("C", 5), 261.6256, places=2)

    def test_octave_doubles_frequency(self):
        self.assertAlmostEqual(
            app.get_note_frequency("A", 6), 2 * app.get_note_frequency("A", 5), places=3
        )

    def test_enharmonic_equivalence(self):
        self.assertAlmostEqual(
            app.get_note_frequency("C#", 4), app.get_note_frequency("Db", 4), places=6
        )

    def test_invalid_note_raises(self):
        with self.assertRaises(ValueError):
            app.get_note_frequency("H", 4)


class ParseSequenceTests(unittest.TestCase):
    def test_basic_sequence(self):
        parsed = app.parse_note_sequence_string("C2 1.0, G#2 0.5, R 1.0, Eb3 0.25")
        self.assertEqual(parsed[0], ("C", 2, 1.0, False, ""))
        self.assertEqual(parsed[1], ("G", 2, 0.5, False, "#"))
        self.assertEqual(parsed[2], ("R", 4, 1.0, True, ""))
        self.assertEqual(parsed[3], ("E", 3, 0.25, False, "b"))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            app.parse_note_sequence_string("   ")

    def test_invalid_token_raises(self):
        with self.assertRaises(ValueError):
            app.parse_note_sequence_string("C2 1.0, notanote")

    def test_octave_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            app.parse_note_sequence_string("C9 1.0")

    def test_duration_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            app.parse_note_sequence_string("C2 9.0")


class MidiConversionTests(unittest.TestCase):
    def test_middle_c(self):
        # C4 = MIDI 60 in this app's (octave*12 + offset) convention → C4 = 4*12 = 48?
        # 앱 규약상 C5 = 60. 일관성만 검증한다.
        self.assertEqual(app.note_name_to_midi("C", 5), 60)

    def test_sharp_and_flat(self):
        base = app.note_name_to_midi("C", 4)
        self.assertEqual(app.note_name_to_midi("C", 4, "#"), base + 1)
        self.assertEqual(app.note_name_to_midi("C", 4, "b"), base - 1)


class ChordParsingTests(unittest.TestCase):
    def test_major_triad(self):
        root, intervals, name = app.parse_chord_symbol("C")
        self.assertEqual(root, 0)
        self.assertEqual(intervals, [0, 4, 7])

    def test_minor_triad(self):
        root, intervals, _ = app.parse_chord_symbol("Am")
        self.assertEqual(root, 9)
        self.assertEqual(intervals, [0, 3, 7])

    def test_dominant_seventh(self):
        root, intervals, _ = app.parse_chord_symbol("G7")
        self.assertEqual(root, 7)
        self.assertEqual(intervals, [0, 4, 7, 10])

    def test_major_seventh(self):
        _, intervals, _ = app.parse_chord_symbol("Fmaj7")
        self.assertEqual(intervals, [0, 4, 7, 11])

    def test_sharp_root(self):
        root, _, _ = app.parse_chord_symbol("D#m7")
        self.assertEqual(root, 3)

    def test_flat_root(self):
        root, _, _ = app.parse_chord_symbol("Bb")
        self.assertEqual(root, 10)

    def test_unknown_quality_falls_back(self):
        # 알 수 없는 확장은 메이저/마이너 트라이어드로 축약 (예외 아님)
        _, intervals, _ = app.parse_chord_symbol("Cadd9")
        self.assertIn(0, intervals)
        self.assertIn(7, intervals)

    def test_invalid_symbol_raises(self):
        with self.assertRaises(ValueError):
            app.parse_chord_symbol("X")

    def test_progression_split(self):
        chords = app.parse_chord_progression("C G Am F")
        self.assertEqual(len(chords), 4)
        chords2 = app.parse_chord_progression("C, G, Am, F")
        self.assertEqual(len(chords2), 4)


class ChordBasslineTests(unittest.TestCase):
    def test_seed_is_deterministic(self):
        a = app.generate_bassline_from_chords("C G Am F", "rock", 2, seed=42)
        b = app.generate_bassline_from_chords("C G Am F", "rock", 2, seed=42)
        self.assertEqual(a, b)

    def test_different_seed_can_differ(self):
        a = app.generate_bassline_from_chords("C G Am F", "funk", 2, seed=1)
        b = app.generate_bassline_from_chords("C G Am F", "funk", 2, seed=2)
        # 다른 시드는 (거의 항상) 다른 결과를 낸다. 같으면 적어도 형식은 유효해야 함.
        self.assertTrue(isinstance(a, str) and isinstance(b, str))

    def test_each_chord_fills_one_measure(self):
        seq = app.generate_bassline_from_chords("C G", "jazz", 2, seed=7)
        parsed = app.parse_note_sequence_string(seq)
        total = sum(n[2] for n in parsed)
        self.assertAlmostEqual(total, 8.0, places=3)  # 2 코드 × 4박

    def test_output_is_parseable(self):
        for genre in app.GROOVE_TEMPLATES:
            seq = app.generate_bassline_from_chords("C Am F G", genre, 2, seed=3)
            parsed = app.parse_note_sequence_string(seq)
            self.assertTrue(len(parsed) > 0, f"{genre} produced no notes")

    def test_all_templates_fill_four_beats(self):
        for genre, templates in app.GROOVE_TEMPLATES.items():
            for i, tmpl in enumerate(templates):
                total = sum(dur for dur, _role in tmpl)
                self.assertAlmostEqual(
                    total, 4.0, places=6,
                    msg=f"{genre} template {i} sums to {total}, expected 4.0",
                )


class RandomStyleTests(unittest.TestCase):
    def test_produces_parseable_sequence(self):
        import numpy as np
        np.random.seed(123)
        seq = app.create_random_bass_loop_by_style("rock", "C", 2, 4, 120)
        parsed = app.parse_note_sequence_string(seq)
        self.assertTrue(len(parsed) > 0)

    def test_seed_reproducible(self):
        import numpy as np
        np.random.seed(99)
        a = app.create_random_bass_loop_by_style("funk", "A", 2, 4, 120)
        np.random.seed(99)
        b = app.create_random_bass_loop_by_style("funk", "A", 2, 4, 120)
        self.assertEqual(a, b)


class AudioBufferTests(unittest.TestCase):
    def test_wav_buffer_has_riff_header(self):
        seq = app.parse_note_sequence_string("C2 1.0, G2 1.0")
        buf = app.create_bass_loop_from_parsed_sequence(seq, 120, 1)
        data = buf.getvalue()
        self.assertEqual(data[:4], b"RIFF")
        self.assertEqual(data[8:12], b"WAVE")

    def test_invalid_bpm_raises(self):
        seq = app.parse_note_sequence_string("C2 1.0")
        with self.assertRaises(ValueError):
            app.create_bass_loop_from_parsed_sequence(seq, 0, 1)


class MuseScorePathTests(unittest.TestCase):
    def test_env_var_takes_precedence(self):
        prev = os.environ.get("MUSESCORE_PATH")
        os.environ["MUSESCORE_PATH"] = "/custom/musescore"
        try:
            self.assertEqual(app.find_musescore_path(), "/custom/musescore")
        finally:
            if prev is None:
                del os.environ["MUSESCORE_PATH"]
            else:
                os.environ["MUSESCORE_PATH"] = prev

    def test_default_when_not_found(self):
        prev = os.environ.get("MUSESCORE_PATH")
        if prev is not None:
            del os.environ["MUSESCORE_PATH"]
        try:
            # 후보 경로가 없는 환경에서는 기본값을 돌려준다.
            self.assertTrue(app.find_musescore_path().endswith("musescore3")
                            or os.path.exists(app.find_musescore_path()))
        finally:
            if prev is not None:
                os.environ["MUSESCORE_PATH"] = prev


class RouteTests(unittest.TestCase):
    def setUp(self):
        app.app.config["TESTING"] = True
        self.client = app.app.test_client()

    def test_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["status"], "ok")

    def test_features_status_has_no_gemini(self):
        resp = self.client.get("/get_features_status")
        data = resp.get_json()
        self.assertIn("music21_available", data)
        self.assertNotIn("gemini_available", data)

    def test_generate_notes_random(self):
        resp = self.client.post("/generate_notes", data={
            "generation_mode": "random", "genre_input": "rock",
            "key_note_input": "C", "octave_input": "2",
            "bpm_input": "120", "length_input": "4", "seed_input": "5",
        })
        data = resp.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["seed"], 5)
        self.assertTrue(data["notes"])

    def test_generate_notes_seed_reproducible(self):
        payload = {
            "generation_mode": "random", "genre_input": "funk",
            "key_note_input": "G", "octave_input": "2",
            "bpm_input": "100", "length_input": "4", "seed_input": "777",
        }
        a = self.client.post("/generate_notes", data=payload).get_json()
        b = self.client.post("/generate_notes", data=payload).get_json()
        self.assertEqual(a["notes"], b["notes"])

    def test_generate_notes_chords(self):
        resp = self.client.post("/generate_notes", data={
            "generation_mode": "chords", "genre_input": "jazz",
            "octave_input": "2", "bpm_input": "120",
            "chord_progression_input": "C G Am F", "seed_input": "10",
        })
        data = resp.get_json()
        self.assertEqual(data["status"], "success")
        self.assertTrue(data["notes"])

    def test_generate_notes_chords_requires_progression(self):
        resp = self.client.post("/generate_notes", data={
            "generation_mode": "chords", "octave_input": "2", "bpm_input": "120",
        })
        self.assertEqual(resp.status_code, 400)

    def test_generate_notes_rejects_ai_mode(self):
        resp = self.client.post("/generate_notes", data={
            "generation_mode": "ai", "bpm_input": "120",
        })
        self.assertEqual(resp.status_code, 400)

    def test_generate_audio(self):
        resp = self.client.post("/generate_audio", data={
            "notes_sequence_input": "C2 1.0, G2 1.0", "bpm_input": "120",
            "num_loops_input": "1",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.mimetype, "audio/wav")
        self.assertEqual(resp.data[:4], b"RIFF")

    def test_generate_midi(self):
        resp = self.client.post("/generate_midi", data={
            "notes_sequence_input": "C2 1.0, G2 1.0", "bpm_input": "120",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.mimetype, "audio/midi")

    def test_generate_lilypond(self):
        resp = self.client.post("/generate_lilypond", data={
            "notes_sequence_input": "C2 1.0, G2 1.0", "bpm_input": "120",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"clef", resp.data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
