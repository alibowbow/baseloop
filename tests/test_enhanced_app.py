"""Integration checks for the additive BaseLoop audio/UX asset layer."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from enhanced_app import ASSET_VERSION, app  # noqa: E402


class EnhancedAppInjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()

    def test_homepage_injects_versioned_enhancement_assets(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn(f"/static/baseloop-enhanced.css?v={ASSET_VERSION}", html)
        self.assertIn(f"/static/baseloop-enhanced-core.js?v={ASSET_VERSION}", html)
        self.assertIn(f"/static/baseloop-enhanced-audio.js?v={ASSET_VERSION}", html)
        self.assertIn(f"/static/baseloop-mix-balance.js?v={ASSET_VERSION}", html)
        self.assertIn(f"/static/baseloop-metronome-presence.js?v={ASSET_VERSION}", html)
        self.assertEqual(response.headers.get("X-BaseLoop-Enhancement"), ASSET_VERSION)

    def test_assets_are_injected_only_once(self) -> None:
        response = self.client.get("/")
        html = response.get_data(as_text=True)
        self.assertEqual(html.count('data-baseloop-enhanced="1"'), 5)

    def test_metronome_calibration_loads_after_mix_layer(self) -> None:
        response = self.client.get("/")
        html = response.get_data(as_text=True)
        mix_index = html.index("/static/baseloop-mix-balance.js")
        metronome_index = html.index("/static/baseloop-metronome-presence.js")
        self.assertLess(mix_index, metronome_index)

    def test_static_enhancement_assets_are_served(self) -> None:
        paths = (
            "/static/baseloop-enhanced.css",
            "/static/baseloop-enhanced-core.js",
            "/static/baseloop-enhanced-audio.js",
            "/static/baseloop-mix-balance.js",
            "/static/baseloop-metronome-presence.js",
        )
        for path in paths:
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertGreater(len(response.data), 1000)

    def test_studio_mixer_defaults_to_collapsed_on_all_viewports(self) -> None:
        response = self.client.get("/static/baseloop-enhanced-core.js")
        self.assertEqual(response.status_code, 200)
        javascript = response.get_data(as_text=True)
        self.assertIn("mixerCollapsed: true", javascript)
        self.assertIn(": DEFAULTS.mixerCollapsed", javascript)
        sync_start = javascript.index("function syncMixerMode()")
        sync_end = javascript.index("function tapTempo()", sync_start)
        sync_mixer = javascript[sync_start:sync_end]
        self.assertNotIn("matchMedia", sync_mixer)
        self.assertIn("panel.classList.toggle('is-collapsed', collapsed)", sync_mixer)

    def test_mix_layer_uses_beatbox_drums(self) -> None:
        response = self.client.get("/static/baseloop-mix-balance.js")
        self.assertEqual(response.status_code, 200)
        javascript = response.get_data(as_text=True)
        self.assertIn("BEATBOX_REFERENCE", javascript)
        self.assertIn("createBeatboxDrumKit", javascript)
        self.assertIn("kickClick", javascript)
        self.assertIn("snareHeadLow", javascript)
        self.assertIn("Tone.MetalSynth", javascript)

    def test_metronome_layer_restores_presence_without_touching_drum_mix(self) -> None:
        response = self.client.get("/static/baseloop-metronome-presence.js")
        self.assertEqual(response.status_code, 200)
        javascript = response.get_data(as_text=True)
        self.assertIn("const CLICK_MAKEUP = 1.12", javascript)
        self.assertIn("const TONE_VOLUME_DB = -9", javascript)
        self.assertIn("new Tone.NoiseSynth", javascript)
        self.assertIn("accent ? 'A6' : 'E6'", javascript)
        self.assertNotIn("makeDrumKit =", javascript)
        self.assertNotIn("makeBassBus =", javascript)


if __name__ == "__main__":
    unittest.main()
