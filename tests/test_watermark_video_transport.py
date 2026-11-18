"""Real local FFmpeg timing, audio, masking and cancellation regressions."""
import asyncio
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from buyer_show.video import ffmpeg_executable
from buyer_show.watermark_video import probe_video, extract_frame
from buyer_show.watermark_video_transport import (render_quality, verify_timing_audio,
    QualityUnsupported, _video_times, _spatial_groups, _frames)


class TransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='video-transport-tests-')
        cls.root = Path(cls.temp.name)
        cls.source = cls.root / 'source.mp4'
        cls.command(['-f', 'lavfi', '-i', 'color=c=black:s=160x120:r=10:d=2',
            '-f', 'lavfi', '-i', 'sine=frequency=440:duration=2',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(cls.source)])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @classmethod
    def command(cls, args):
        result = subprocess.run([ffmpeg_executable(), '-nostdin', '-hide_banner', '-v', 'error',
            '-threads', '2', *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW, timeout=60)
        if result.returncode:
            raise AssertionError(result.stderr.decode(errors='replace'))

    def render(self, source, name, boxes=None, processor=None):
        out = self.root / name
        info = probe_video(source)
        return out, render_quality(source, out, boxes or [dict(x=12, y=12, w=20, h=16, start=.5, end=1)],
            info, processor or (lambda rgb, t: np.full_like(rgb, 255)))

    def test_cfr_half_open_intervals_force_outside_box_original(self):
        seen = []
        def fill_everything(rgb, t):
            seen.append(t)
            rgb[:] = 255  # In-place misbehaving callback must not alter outside approved boxes.
            return rgb
        boxes = [dict(x=12, y=12, w=20, h=16, start=.5, end=1),
                 dict(x=100, y=80, w=16, h=16, start=1.2, end=1.5)]
        out, result = self.render(self.source, 'cfr.mp4', boxes, fill_everything)
        self.assertEqual(result['frames_processed'], 20)
        self.assertEqual(len(result['spatial_rois']), 2)
        self.assertTrue(result['timing_audio']['all_audio_packets_preserved'])
        self.assertEqual(seen, [.5, .6, .7, .8, .9, 1.2, 1.3, 1.4])
        for seconds, first, second in [(0, False, False), (.5, True, False), (.9, True, False),
                                       (1, False, False), (1.2, False, True), (1.5, False, False)]:
            frame = np.asarray(extract_frame(out, seconds))
            self.assertLess(int(frame[50, 60, 0]), 20)
            self.assertGreater(int(frame[18, 20, 0]), 230) if first else self.assertLess(int(frame[18, 20, 0]), 20)
            self.assertGreater(int(frame[86, 108, 0]), 230) if second else self.assertLess(int(frame[86, 108, 0]), 20)

    def test_vfr_preserves_every_original_frame_time(self):
        source = self.root / 'vfr.mp4'
        self.command(['-i', str(self.source), '-vf', 'select=if(lt(t\\,1)\\,1\\,not(mod(n\\,3)))',
            '-fps_mode', 'passthrough', '-c:v', 'libx264', '-c:a', 'copy', str(source)])
        times = _video_times(source, lambda: False)
        self.assertGreater(len(set(b-a for a,b in zip(times,times[1:]))), 1)
        out, result = self.render(source, 'vfr-result.mp4')
        self.assertEqual(result['frames_processed'], len(times))
        self.assertLessEqual(result['timing_audio']['video_relative_pts_max_error_seconds'], .000002)

    def test_delayed_audio_offset_and_multiple_audio_streams_preserved(self):
        source = self.root / 'delayed.mp4'
        self.command(['-f', 'lavfi', '-i', 'color=c=black:s=160x120:r=10:d=2',
            '-itsoffset', '0.4', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=1.5',
            '-itsoffset', '0.2', '-f', 'lavfi', '-i', 'sine=frequency=660:duration=1.7',
            '-map', '0:v', '-map', '1:a', '-map', '2:a', '-c:v', 'libx264', '-c:a', 'aac', str(source)])
        _, result = self.render(source, 'delayed-result.mp4')
        audio = result['timing_audio']['audio_streams']
        self.assertEqual(len(audio), 2)
        self.assertGreater(audio[0]['first_seconds'], .3)
        self.assertTrue(result['timing_audio']['audio_video_offset_preserved'])

    def test_no_audio(self):
        source = self.root / 'silent.mp4'
        self.command(['-i', str(self.source), '-an', '-c:v', 'copy', str(source)])
        _, result = self.render(source, 'silent-result.mp4')
        self.assertEqual(result['timing_audio']['audio_streams'], [])

    def test_fractional_frame_rate_patch_matches_every_frame_and_odd_coordinates(self):
        source = self.root / 'ntsc.mp4'
        self.command(['-f', 'lavfi', '-i', 'color=c=black:s=160x120:r=30000/1001:d=2',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(source)])
        count = 0
        def alternating(rgb, timestamp):
            nonlocal count
            result = np.full_like(rgb, 255 if count % 2 else 0)
            count += 1
            return result
        boxes = [dict(x=13, y=13, w=21, h=17, start=0, end=2)]
        out, report = self.render(source, 'ntsc-result.mp4', boxes, alternating)
        frames = _frames(out, (120,160), lambda:False)
        try:
            for n, frame in enumerate(frames):
                self.assertLess(abs(int(frame[20,20,0]) - (255 if n % 2 else 0)), 20)
                self.assertLess(int(frame[12,20,0]), 35)
        finally:
            frames.close()
        self.assertEqual(report['frames_processed'], count)

    def test_video_starts_after_audio_retains_offset(self):
        source = self.root / 'late-video.mp4'
        self.command(['-itsoffset', '0.4', '-f', 'lavfi', '-i', 'color=c=black:s=160x120:r=10:d=2',
            '-f', 'lavfi', '-i', 'sine=frequency=440:duration=2.4',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(source)])
        _, result = self.render(source, 'late-video-result.mp4')
        self.assertTrue(result['timing_audio']['audio_video_offset_preserved'])

    def test_nonzero_container_start_allows_only_actual_timebase_quantization(self):
        source = self.root / 'nonzero-start.mp4'
        self.command(['-i', str(self.source), '-c', 'copy', '-output_ts_offset', '5', str(source)])
        self.assertGreater(probe_video(source)['input_start_seconds'], 4)
        _, result = self.render(source, 'nonzero-start-result.mp4')
        checks = result['timing_audio']
        self.assertLessEqual(checks['video_relative_pts_max_error_seconds'], .000002)
        self.assertLess(checks['video_start_quantization_tolerance_seconds'], .0001)
        self.assertLessEqual(checks['video_start_offset_error_seconds'],
                             checks['video_start_quantization_tolerance_seconds'])
        self.assertTrue(checks['all_audio_packets_preserved'])

    def test_audio_video_offset_change_larger_than_tick_is_rejected(self):
        shifted = self.root / 'desynchronized.mp4'
        self.command(['-itsoffset', '0.01', '-i', str(self.source), '-i', str(self.source),
            '-map', '0:v', '-map', '1:a', '-c', 'copy', str(shifted)])
        with self.assertRaisesRegex(ValueError, '偏移'):
            verify_timing_audio(self.source, shifted)

    def test_encoding_phase_cancel_and_bad_callback_do_not_leave_output(self):
        stopped = False
        def emit(message):
            nonlocal stopped
            if '正在按原始' in message:
                stopped = True
        output = self.root/'encoding-cancel.mp4'
        with self.assertRaises(asyncio.CancelledError):
            render_quality(self.source, output, [dict(x=12,y=12,w=20,h=16)], probe_video(self.source),
                           lambda rgb,t:rgb, should_stop=lambda:stopped, emit=emit)
        self.assertFalse(output.exists())
        with self.assertRaisesRegex(ValueError, 'RGB'):
            self.render(self.source, 'invalid-callback.mp4', processor=lambda rgb,t:None)
        self.assertFalse(list(self.root.glob('video-patches-*')))

    def test_existing_output_is_preserved(self):
        output = self.root/'preexisting.mp4'
        output.write_bytes(b'original unrelated file')
        with self.assertRaises(FileExistsError):
            self.render(self.source, output.name)
        self.assertEqual(output.read_bytes(), b'original unrelated file')

    def test_prepare_receives_real_pts_and_process_object_is_supported(self):
        class Processor:
            diagnostics = {'control': 'identity'}
            def prepare(self, source, info, times, *, should_stop, emit):
                self.times = times
                self.seen = []
            def process(self, rgb, timestamp):
                self.seen.append(timestamp)
                return rgb
        processor = Processor()
        _, evidence = self.render(self.source, 'prepared.mp4', processor=processor)
        self.assertEqual(processor.times, [n / 10 for n in range(20)])
        self.assertEqual(processor.seen, [.5,.6,.7,.8,.9])
        self.assertEqual(evidence['algorithm'], {'control':'identity'})

    def test_callback_cancel_reaps_decoder_and_cleans_temporary_frames(self):
        out = self.root / 'cancelled.mp4'
        def cancel(rgb, timestamp):
            raise asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            self.render(self.source, out.name, processor=cancel)
        self.assertFalse(out.exists())
        self.assertFalse(list(self.root.glob('video-patches-*')))

    def test_preflight_limit_and_spatial_group_merging(self):
        info = {**probe_video(self.source), 'duration': 181}
        with self.assertRaises(QualityUnsupported):
            render_quality(self.source, self.root/'unsupported.mp4', [], info, lambda x,t:x)
        groups = _spatial_groups([dict(x=0,y=0,w=10,h=10),dict(x=8,y=8,w=10,h=10),dict(x=90,y=90,w=5,h=5)])
        self.assertEqual(groups, [(0,0,18,18),(90,90,5,5)])

    def test_corrupted_frame_count_rejected_by_verifier(self):
        short = self.root / 'short.mp4'
        self.command(['-i', str(self.source), '-t', '1', '-c', 'copy', str(short)])
        with self.assertRaisesRegex(ValueError, '帧数'):
            verify_timing_audio(self.source, short)


if __name__ == '__main__':
    unittest.main()
