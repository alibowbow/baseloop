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
        self.assertEqual(response.headers.get("X-BaseLoop-Enhancement"), ASSET_VERSION)

    def test_assets_are_injected_only_once(self) -> None:
        response = self.client.get("/")
        html = response.get_data(as_text=True)
        self.assertEqual(html.count('data-baseloop-enhanced="1"'), 4)

    def test_static_enhancement_assets_are_served(self) -> None:
        paths = (
            "/static/baseloop-enhanced.css",
            "/static/baseloop-enhanced-core.js",
            "/static/baseloop-enhanced-audio.js",
            "/static/baseloop-mix-balance.js",
        )
        for path in paths:
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertGreater(len(response.data), 1000)


if __name__ == "__main__":
    unittest.main()
