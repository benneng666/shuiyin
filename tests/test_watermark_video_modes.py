"""Integrated video-mode dispatch, rollback, and original/audio preservation."""
import asyncio
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from buyer_show.video import ffmpeg_executable
from buyer_show.watermark_video import repair_video
from buyer_show.watermark_video_transport import QualityUnsupported


class VideoModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.source = cls.root / 'source.mp4'
        subprocess.run([ffmpeg_executable(), '-nostdin', '-v', 'error', '-f', 'lavfi', '-i',
            'color=blue:s=160x120:r=8:d=1,drawbox=x=20:y=20:w=20:h=12:color=white:t=fill',
            '-f', 'lavfi', '-i', 'sine=duration=1', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', str(cls.source)], check=True, capture_output=True, timeout=20,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        cls.boxes = [dict(x=18, y=18, w=24, h=16, start=0, end=1)]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_real_quality_pipeline_preserves_original_audio_frames_and_records_mode(self):
        before = self.source.read_bytes()
        with patch('buyer_show.watermark_ai.ensure_model', side_effect=AssertionError('flat background must not load AI')):
            result = repair_video(self.source, self.boxes, self.root/'quality', method='lama_temporal')
        data = json.loads(Path(result['manifest']).read_text(encoding='utf-8'))
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(data['method'], 'lama_temporal')
        self.assertEqual(data['validation']['frames_decoded'], 8)
        self.assertTrue(data['output_metadata']['has_audio'])
        self.assertTrue(data['quality']['timing_audio'])

    def test_smart_falls_back_only_for_known_capability_limit(self):
        with patch('buyer_show.watermark_video_transport.render_quality', side_effect=QualityUnsupported('sample limit')):
            result = repair_video(self.source, self.boxes, self.root/'fallback', method='smart')
        data = json.loads(Path(result['manifest']).read_text(encoding='utf-8'))
        self.assertEqual(result['method'], 'delogo')
        self.assertEqual(data['requested_method'], 'smart')
        self.assertEqual(data['quality_fallback_reason'], 'sample limit')
        self.assertTrue(any('快速修补' in w for w in result['warnings']))

    def test_quality_failure_or_cancel_never_silently_publishes_fast_result(self):
        for number, error in enumerate((ValueError('encoding validation failed'), asyncio.CancelledError())):
            folder = self.root/f'failed-{number}'
            with patch('buyer_show.watermark_video_transport.render_quality', side_effect=error):
                with self.assertRaises(type(error)):
                    repair_video(self.source, self.boxes, folder, method='smart')
            records = list(folder.glob('video-*/processing.json'))
            self.assertEqual(len(records), 1)
            data = json.loads(records[0].read_text(encoding='utf-8'))
            self.assertEqual(data['status'], 'cancelled' if number else 'failed')
            self.assertIsNone(data['output'])
            self.assertFalse(list(folder.rglob('result*.mp4')))

    def test_explicit_quality_never_silently_changes_to_fast(self):
        with patch('buyer_show.watermark_video_transport.render_quality', side_effect=QualityUnsupported('limit')):
            with self.assertRaises(QualityUnsupported):
                repair_video(self.source, self.boxes, self.root/'explicit-limit', method='lama_temporal')

    def test_unknown_mode_rejected_before_io(self):
        with self.assertRaisesRegex(ValueError, '未知视频修复方式'):
            repair_video('missing.mp4', [], method='not-a-mode')
