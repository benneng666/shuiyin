import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from buyer_show.storage import atomic_json


class ManifestReplaceTests(unittest.TestCase):
    def test_transient_windows_lock_preserves_old_file_until_complete_commit(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'manifest.json'
            atomic_json(path, {'version': 'old'})
            real_replace, attempts = Path.replace, []
            def replace(stage, target):
                attempts.append(1)
                self.assertEqual(json.loads(path.read_text()), {'version': 'old'})
                self.assertEqual(json.loads(stage.read_text()), {'version': 'new', 'items': list(range(50))})
                if len(attempts) <= 2:
                    raise PermissionError('simulated sharing violation')
                return real_replace(stage, target)
            with patch.object(Path, 'replace', replace), patch('buyer_show.storage.time.sleep'):
                atomic_json(path, {'version': 'new', 'items': list(range(50))})
            self.assertEqual(json.loads(path.read_text())['version'], 'new')

    def test_persistent_lock_is_reported_without_destroying_committed_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'manifest.json'
            atomic_json(path, {'version': 'old'})
            with patch.object(Path, 'replace', side_effect=PermissionError('still locked')) as replace, \
                 patch('buyer_show.storage.time.sleep'):
                with self.assertRaises(PermissionError):
                    atomic_json(path, {'version': 'new'})
            self.assertEqual(replace.call_count, 5)
            self.assertEqual(json.loads(path.read_text()), {'version': 'old'})
