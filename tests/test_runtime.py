import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from buyer_show.runtime import InstanceLock, app_data_dir, load_settings, save_settings


class DesktopRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="buyer_show_runtime_")
        self.addCleanup(self.temp.cleanup)
        self.env = patch.dict(os.environ, {"LOCALAPPDATA": self.temp.name})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_settings_roundtrip_does_not_store_links_or_credentials(self):
        settings = {"output": str(Path(self.temp.name) / "中文目录"), "max_scrolls": 123,
                    "channel": "chrome", "cookies": "secret", "links": ["private"]}
        save_settings(settings)
        restored = load_settings()
        self.assertEqual(restored["max_scrolls"], 123)
        self.assertEqual(restored["output"], settings["output"])
        self.assertEqual(set(restored), {"output", "max_scrolls", "channel"})
        self.assertNotIn("secret", (app_data_dir() / "settings.json").read_text(encoding="utf-8"))

    def test_invalid_settings_fall_back_without_preventing_launch(self):
        app_data_dir().mkdir()
        config = app_data_dir() / "settings.json"
        config.write_text("{invalid", encoding="utf-8")
        self.assertEqual(load_settings()["channel"], "msedge")
        config.write_text(json.dumps({"max_scrolls": True, "channel": "evil", "output": ""}), encoding="utf-8")
        self.assertEqual(load_settings()["max_scrolls"], 200)

    def test_second_instance_cannot_lock_and_release_allows_reopen(self):
        first, second = InstanceLock(), InstanceLock()
        try:
            self.assertTrue(first.acquire())
            self.assertFalse(second.acquire())
            first.release()
            self.assertTrue(second.acquire())
        finally:
            first.release()
            second.release()
