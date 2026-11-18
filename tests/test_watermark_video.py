"""Local synthetic CPU video tests, no platform/network/model download."""
import asyncio
import hashlib
import json
import re
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from buyer_show.video import ffmpeg_executable
from buyer_show.watermark_video import (_probe, _regions, _run, detect_video,
    extract_frame, probe_video, repair_video)


class WatermarkVideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="buyer_show_watermark_video_")
        cls.root = Path(cls.temp.name)
        cls.source = cls.root / "原视频 带音轨.mp4"
        cls.exe = ffmpeg_executable()
        cls.command(["-f", "lavfi", "-i", "color=c=blue:s=320x240:r=12:d=3,drawbox=x=20:y=20:w=24:h=16:color=white:t=fill",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "10", "-c:a", "aac", str(cls.source)])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @classmethod
    def command(cls, args):
        return subprocess.run([cls.exe, "-nostdin", "-v", "error", *args],
            check=True, capture_output=True, timeout=20,
            **({"creationflags": subprocess.CREATE_NO_WINDOW} if hasattr(subprocess, "CREATE_NO_WINDOW") else {}))

    def box(self, **changes):
        return dict(x=18, y=18, w=28, h=20, start=1, end=2, **changes)

    def test_probe_and_rgb_frame(self):
        info = probe_video(self.source)
        self.assertEqual((info["width"], info["height"]), (320, 240))
        self.assertAlmostEqual(info["duration"], 3, delta=0.1)
        self.assertEqual(info["fps"], 12)
        self.assertTrue(info["has_audio"])
        frame = extract_frame(self.source, 0.25)
        self.assertEqual(frame.mode, "RGB")
        self.assertEqual(frame.size, (320, 240))
        self.assertGreater(frame.getpixel((30, 25))[0], 240)

    def test_repair_preserves_audio_duration_size_source_and_half_open_range(self):
        before = self.source.read_bytes()
        result = repair_video(self.source, [self.box()])
        output = Path(result["output"])
        manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(manifest["source_sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["validation"]["audio_validation"], "full_decode")
        self.assertEqual(manifest["validation"]["frames_decoded"], 36)
        info = probe_video(output)
        self.assertEqual((info["width"], info["height"]), (320, 240))
        self.assertAlmostEqual(info["duration"], 3, delta=0.1)
        self.assertTrue(info["has_audio"])
        for seconds, removed in ((0.25, False), (1.25, True), (2.0, False), (2.25, False)):
            pixel = extract_frame(output, seconds).getpixel((30, 25))
            self.assertLess(pixel[0], 50) if removed else self.assertGreater(pixel[0], 230)
        hashes = [self.command(["-i", str(path), "-map", "0:a:0", "-c:a", "copy", "-f", "hash", "-"]).stdout
                  for path in (self.source, output)]
        self.assertEqual(hashes[0], hashes[1])

    def test_detection_is_bounded_deduplicated_and_temporal(self):
        mocked = {"width": 320, "height": 240, "regions": [
            {"x": 18, "y": 18, "w": 28, "h": 20, "label": "平台", "confidence": 0.9},
            {"x": 18, "y": 18, "w": 28, "h": 20, "label": "平台", "confidence": 0.9}], "warnings": []}
        with patch("buyer_show.watermark.detect_image", return_value=mocked) as detector:
            result = detect_video(self.source)
        self.assertEqual(detector.call_count, 6)
        self.assertEqual(len(result["regions"]), 1)
        self.assertEqual(result["regions"][0]["start"], 0)
        self.assertEqual(result["regions"][0]["end"], 3)
        self.assertEqual(result["tracking"], "sampled_segments_only")

    def test_bad_large_and_invalid_time_regions_rejected(self):
        info = probe_video(self.source)
        for bad in (dict(x=-1, y=2, w=20, h=10), dict(x=10, y=10, w=200, h=100),
                    dict(x=10, y=10, w=-3, h=10), dict(x=10.5, y=10, w=10, h=10),
                    dict(x=310, y=10, w=11, h=10), dict(x=10, y=230, w=10, h=11),
                    dict(x=10, y=10, w=10, h=10, start=2, end=1),
                    dict(x=10, y=10, w=10, h=10, end=4)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                _regions([bad], info)
        self.assertEqual(len(_regions([self.box(), self.box()], info)), 1)

    def test_edge_regions_are_accepted_exactly_inside_display_bounds(self):
        info = probe_video(self.source)
        for box in (dict(x=0, y=0, w=20, h=10), dict(x=300, y=230, w=20, h=10),
                    dict(x=0, y=230, w=20, h=10), dict(x=300, y=0, w=20, h=10)):
            with self.subTest(box=box):
                self.assertEqual({k: _regions([box], info)[0][k] for k in box}, box)

    def test_merchant_profile_forwarding_and_adjacent_jitter_merges_with_union(self):
        frames = []
        for x, y in ((18, 18), (19, 19), (20, 18), (19, 20), (18, 19), (19, 18)):
            frames.append({"regions": [{"x": x, "y": y, "w": 28, "h": 20,
                                       "label": "测试商家", "confidence": .9}], "warnings": []})
        with patch("buyer_show.watermark.detect_image", side_effect=frames) as detector:
            result = detect_video(self.source, include_corner_text=True,
                                  merchant_terms="测试商家;商标", detection_profile="product")
        self.assertEqual(len(result["regions"]), 1)
        region = result["regions"][0]
        self.assertEqual([region[k] for k in ("x", "y", "w", "h", "start", "end")], [18, 18, 30, 22, 0, 3])
        for call in detector.call_args_list:
            self.assertIs(call.kwargs["include_corner_text"], True)
            self.assertEqual(call.kwargs["merchant_terms"], "测试商家;商标")
            self.assertEqual(call.kwargs["detection_profile"], "product")

    def test_missing_sample_and_scene_change_stop_merging(self):
        found = {"regions": [{"x": 18, "y": 18, "w": 28, "h": 20,
                               "label": "测试商家", "confidence": .9}], "warnings": []}
        absent = {"regions": [], "warnings": []}
        with patch("buyer_show.watermark.detect_image", side_effect=[found, found, absent, found, found, found]):
            result = detect_video(self.source)
        self.assertEqual([(r["start"], r["end"]) for r in result["regions"]], [(0, 1), (1.5, 3)])
        frames = [Image.new("RGB", (320, 240), color) for color in ("blue", "blue", "blue", "red", "red", "red")]
        with patch("buyer_show.watermark_video._frame", side_effect=frames), \
                patch("buyer_show.watermark.detect_image", return_value=found):
            result = detect_video(self.source)
        self.assertEqual([(r["start"], r["end"]) for r in result["regions"]], [(0, 1.5), (1.5, 3)])
        self.assertEqual(result["scene_change_samples"], 1)
        self.assertEqual(result["tracking"], "sampled_segments_only")

    def test_adaptive_sampling_is_bounded_and_long_intervals_are_not_joined(self):
        info = probe_video(self.source)
        found = {"regions": [{"x": 18, "y": 18, "w": 28, "h": 20,
                               "label": "测试商家", "confidence": .9}], "warnings": []}
        for duration, count in ((3, 6), (40, 10), (120, 24), (1800, 24)):
            with self.subTest(duration=duration), \
                    patch("buyer_show.watermark_video._probe", return_value={**info, "duration": duration}), \
                    patch("buyer_show.watermark_video._frame", return_value=Image.new("RGB", (320, 240), "blue")), \
                    patch("buyer_show.watermark.detect_image", return_value=found) as detector:
                result = detect_video(self.source)
                self.assertEqual(result["sample_count"], count)
                self.assertEqual(detector.call_count, count)
                self.assertLessEqual(len(result["regions"]), 128)
                self.assertTrue(all(r["end"] - r["start"] <= max(30, duration / count) for r in result["regions"]))
                if duration == 1800:
                    self.assertEqual(len(result["regions"]), 24)

    def test_fractional_duration_candidates_remain_repairable_without_float_overshoot(self):
        info = {**probe_video(self.source), "duration": 6.01}
        found = {"regions": [{"x": 18, "y": 18, "w": 28, "h": 20,
                               "label": "商家", "confidence": .9}], "warnings": []}
        with patch("buyer_show.watermark_video._probe", return_value=info), \
                patch("buyer_show.watermark_video._frame", return_value=Image.new("RGB", (320, 240), "blue")), \
                patch("buyer_show.watermark.detect_image", return_value=found):
            result = detect_video(self.source)
        self.assertEqual(result["regions"][-1]["end"], info["duration"])
        self.assertEqual(len(_regions(result["regions"], info)), 1)

    def test_actual_edge_repair_restores_size_audio_timing_and_preserves_original(self):
        source = self.root / "added_edge_overlay.mp4"
        self.command(["-i", str(self.source), "-vf",
            "drawbox=x=0:y=80:w=24:h=16:color=white:t=fill,drawbox=x=296:y=224:w=24:h=16:color=white:t=fill",
            "-c:v", "libx264", "-crf", "10", "-c:a", "copy", str(source)])
        before = source.read_bytes()
        result = repair_video(source, [dict(x=0, y=78, w=28, h=20, start=1, end=2),
                                       dict(x=292, y=220, w=28, h=20, start=1, end=2)])
        output = Path(result["output"])
        manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["edge_strategy"], "mirror_pad_then_delogo_crop")
        self.assertEqual(manifest["internal_padding_pixels"], 4)
        self.assertEqual(source.read_bytes(), before)
        info = probe_video(output)
        self.assertEqual((info["width"], info["height"]), (320, 240))
        self.assertAlmostEqual(info["duration"], 3, delta=.1)
        self.assertTrue(info["has_audio"])
        self.assertEqual(manifest["validation"]["frames_decoded"], 36)
        self.assertEqual(manifest["validation"]["audio_validation"], "full_decode")
        # Edge interpolation has less context; require an actual change in the
        # enabled interval, not perfect reconstruction of the hidden blue pixels.
        self.assertGreater(extract_frame(output, .25).getpixel((18, 86))[0], 230)
        self.assertLess(extract_frame(output, 1.25).getpixel((18, 86))[0], 180)
        self.assertGreater(extract_frame(output, 2.0).getpixel((18, 86))[0], 230)
        hashes = [self.command(["-i", str(path), "-map", "0:a:0", "-c:a", "copy", "-f", "hash", "-"]).stdout
                  for path in (source, output)]
        self.assertEqual(hashes[0], hashes[1])

    def test_rotation_preview_is_display_oriented_but_repair_rejects(self):
        rotated = self.root / "rotated.mp4"
        self.command(["-display_rotation", "90", "-i", str(self.source), "-map", "0", "-c", "copy", str(rotated)])
        info = probe_video(rotated)
        self.assertEqual((info["width"], info["height"]), (240, 320))
        self.assertEqual(extract_frame(rotated, 0.25).size, (240, 320))
        with self.assertRaisesRegex(ValueError, "旋转"):
            repair_video(rotated, [self.box()])

    def test_alpha_odd_size_non_square_pixels_and_high_depth_rejected(self):
        original = probe_video(self.source)
        for modification, reason in (({"has_alpha": True}, "透明"), ({"width": 319}, "偶数"),
                ({"sample_aspect_ratio": [2, 1]}, "非方形"), ({"pixel_format": "yuv420p10le"}, "8 位")):
            with self.subTest(reason=reason), patch("buyer_show.watermark_video._probe", return_value={**original, **modification}):
                with self.assertRaisesRegex(ValueError, reason):
                    repair_video(self.source, [self.box()])

    def test_network_input_and_invalid_frame_time_rejected(self):
        for path in ("https://example.test/a.mp4", r"\\server\share\a.mp4"):
            with self.assertRaises(ValueError):
                probe_video(path)
        for seconds in (-1, 5, float("nan")):
            with self.assertRaises(ValueError):
                extract_frame(self.source, seconds)

    def test_actual_process_cancellation_reaps_child(self):
        started = time.monotonic()
        with self.assertRaises(asyncio.CancelledError):
            _run(["-v", "error", "-stream_loop", "-1", "-re", "-i", str(self.source),
                  "-f", "null", "-"], should_stop=lambda: time.monotonic() - started > 0.2)
        self.assertLess(time.monotonic() - started, 3)

    def test_repair_cancellation_records_cancelled_and_no_final_file(self):
        stopped = False

        def emit(message):
            nonlocal stopped
            if "开始 CPU" in message:
                stopped = True

        root = self.root / "cancelled-jobs"
        with self.assertRaises(asyncio.CancelledError):
            repair_video(self.source, [self.box()], root, should_stop=lambda: stopped, emit=emit)
        paths = list(root.glob("*/processing.json"))
        self.assertEqual(len(paths), 1)
        self.assertEqual(json.loads(paths[0].read_text(encoding="utf-8"))["status"], "cancelled")
        self.assertFalse(list(root.glob("*/result*.mp4")))

    def test_source_modified_during_processing_does_not_publish(self):
        source = self.root / "externally_modified.mp4"
        source.write_bytes(self.source.read_bytes())
        target = self.root / "modified-jobs"

        def mutate(message):
            if "完整解码" in message:
                with source.open("ab") as handle:
                    handle.write(b"external change")

        with self.assertRaisesRegex(ValueError, "其他程序修改"):
            repair_video(source, [self.box()], target, emit=mutate)
        manifest = next(target.glob("*/processing.json"))
        self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["status"], "failed")
        self.assertFalse(list(target.glob("*/result*.mp4")))

    def test_vfr_frame_timestamps_survive_repair(self):
        source = self.root / "variable_rate.mp4"
        self.command(["-i", str(self.source), "-vf", "select=if(lt(t\\,1)\\,1\\,not(mod(n\\,3)))",
                      "-fps_mode", "passthrough", "-c:v", "libx264", "-c:a", "copy", str(source)])
        result = repair_video(source, [self.box()])

        def timestamps(path):
            _, _, diagnostic = _run(["-v", "info", "-i", str(path), "-map", "0:V:0",
                "-vf", "showinfo", "-an", "-fps_mode", "passthrough", "-f", "null", "-"])
            return [float(x) for x in re.findall(r"\bpts_time:([\d.-]+)", diagnostic)]

        original, processed = timestamps(source), timestamps(result["output"])
        self.assertEqual(len(original), len(processed))
        self.assertGreater(len(set(round(b - a, 3) for a, b in zip(original, original[1:]))), 1)
        for a, b in zip(original, processed):
            self.assertAlmostEqual(a, b, places=4)


if __name__ == "__main__":
    unittest.main()
