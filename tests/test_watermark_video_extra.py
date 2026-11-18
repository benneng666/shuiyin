"""Known-merchant video enhancement, with local encoded audio controls."""
import asyncio
import hashlib
import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from buyer_show import watermark_video as video
from buyer_show.watermark_batch_video import classify_video_detection, plan_video, _visual
from buyer_show.video import ffmpeg_executable


def region(**changes):
    return dict(x=20, y=20, w=80, h=24, label="浅色店名增强候选：ACME",
                text="ACME", confidence=.98, kind="enhanced_overlay",
                brand_only_line=True, **changes)


def extra(regions=None, **changes):
    return {"regions": regions or [], "warnings": [], "scan_complete": True,
            "evidence": {"ocr_calls": 1}, **changes}


class VideoExtraIntegrationTests(unittest.TestCase):
    def test_no_known_brand_or_platform_profile_does_not_call_extra(self):
        base = {"regions": [], "scan_complete": True}
        with patch("buyer_show.watermark_text_extra.detect_extra_text") as detector:
            for terms, profile in [("", "product"), ("   ", "product"), ("ACME", "platform")]:
                self.assertIs(video._enhance_known_text("unused.png", base,
                    merchant_terms=terms, detection_profile=profile), base)
        detector.assert_not_called()

    def test_passes_base_and_cancel_progress_without_second_full_frame_scan(self):
        base = {"regions": [], "text_regions": [], "scan_complete": True}
        stop, emit = lambda: False, lambda message: None
        with patch("buyer_show.watermark_text_extra.detect_extra_text", return_value=extra([region()])) as detector:
            result = video._enhance_known_text("frame.png", base, merchant_terms="ACME",
                detection_profile="product", should_stop=stop, emit=emit)
        self.assertIs(detector.call_args.kwargs["base_detection"], base)
        self.assertIs(detector.call_args.kwargs["should_stop"], stop)
        self.assertIs(detector.call_args.kwargs["emit"], emit)
        self.assertNotIn("roi_hints", detector.call_args.kwargs)
        self.assertEqual(result["regions"][0]["extra_text_kind"], "enhanced_overlay")
        self.assertTrue(result["scan_complete"])
        self.assertEqual(base["regions"], [])

    def test_same_text_overlap_keeps_original_footprint_and_does_not_double_track(self):
        original = dict(x=22, y=22, w=76, h=20, label="商家关键词候选：ACME", confidence=.99)
        base = {"regions": [original], "scan_complete": True}
        with patch("buyer_show.watermark_text_extra.detect_extra_text", return_value=extra([region()])):
            result = video._enhance_known_text("frame.png", base, merchant_terms="ACME", detection_profile="product")
        self.assertEqual(result["regions"], [original])

    def test_duplicate_uncertain_text_cannot_disappear_behind_a_base_box(self):
        base = {"regions": [region()], "scan_complete": True}
        uncertain = {**region(), "kind": "uncertain_brand", "brand_only_line": False}
        with patch("buyer_show.watermark_text_extra.detect_extra_text", return_value=extra([uncertain])):
            result = video._enhance_known_text("frame.png", base, merchant_terms="ACME", detection_profile="product")
        self.assertEqual(len(result["regions"]), 1)
        self.assertEqual(result["regions"][0]["extra_text_kind"], "uncertain_brand")

    def test_incomplete_extra_scan_or_base_text_cap_propagates(self):
        for base, enhanced in [({"regions": [], "scan_complete": True}, extra(scan_complete=False)),
            ({"regions": [], "scan_complete": True, "text_regions_truncated": True}, extra())]:
            with patch("buyer_show.watermark_text_extra.detect_extra_text", return_value=enhanced):
                result = video._enhance_known_text("frame.png", base, merchant_terms="ACME", detection_profile="product")
            self.assertFalse(result["scan_complete"])

    def test_cancel_before_extra_has_no_detection(self):
        with patch("buyer_show.watermark_text_extra.detect_extra_text") as detector:
            with self.assertRaises(asyncio.CancelledError):
                video._enhance_known_text("frame.png", {}, merchant_terms="ACME",
                    detection_profile="product", should_stop=lambda: True)
        detector.assert_not_called()

    def test_low_score_and_embedded_or_uncertain_text_never_auto_approve(self):
        for kind, score, expected in [("enhanced_overlay", .85, "low_text_confidence"),
            ("embedded_brand", .99, "enhanced_text_unconfirmed"),
            ("uncertain_brand", .99, "enhanced_text_unconfirmed")]:
            candidate = {**region(), "confidence": score, "extra_text_kind": kind,
                         "extra_text_brand_only": kind == "enhanced_overlay"}
            detection = {"width": 640, "height": 400, "duration": 4.0,
                "samples": [{"seconds": n * 3.9 / 7, "regions": [candidate], "scan_complete": True} for n in range(8)],
                "regions": [candidate], "scan_complete": True, "repair_supported": True}
            result = classify_video_detection(detection, merchant_terms="ACME", visual_evidence={0:{
                "screen_locked_ink_with_changing_background": True, "simple_background": True}})
            self.assertEqual(result["status"], "needs_review")
            self.assertEqual(result["approved_regions"], [])
            self.assertIn(expected, [r["code"] for r in result["review_reasons"]])

    def test_relative_gray_strokes_need_shape_and_changing_background(self):
        patches = []
        for level in (140, 155, 170, 155, 140, 125, 110, 125):
            pixels = np.full((60, 140), level, np.uint8)
            pixels[20:38, 30:33] = 205
            pixels[20:38, 40:43] = 205
            pixels[20:38, 50:53] = 205
            patches.append(pixels)
        result = _visual(patches, (20, 10, 100, 40))
        self.assertEqual(result["stable_ink_fraction"], 0)
        self.assertTrue(result["relative_stroke_evidence"])
        self.assertTrue(result["screen_locked_ink_with_changing_background"])
        self.assertFalse(_visual([patches[0]]*8, (20, 10, 100, 40))["screen_locked_ink_with_changing_background"])
        # One opaque patch is not a pattern of repeated text strokes.
        for pixels in patches:
            pixels[20:38, 30:65] = 205
        self.assertFalse(_visual(patches, (20, 10, 100, 40))["relative_stroke_evidence"])


class VideoExtraEncodedControls(unittest.TestCase):
    """Real MP4/audio decoding; OCR is deterministic to isolate integration."""
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="buyer_show_extra_video_")
        cls.root = Path(cls.temp.name)
        cls.exe = ffmpeg_executable()
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 24)
        cls.paths = {}
        for kind in ("faint", "moving", "subtitle"):
            frames = cls.root / kind
            frames.mkdir()
            for n in range(16):
                level = round(160 + 15 * math.sin(n / 16 * math.tau))
                image = Image.new("RGB", (640, 400), (level,) * 3)
                x, y = (250, 330) if kind == "subtitle" else (20 + n * 3, 20) if kind == "moving" else (20, 20)
                ImageDraw.Draw(image).text((x, y), "ACME", font=font, fill=(210,)*3)
                image.save(frames / f"{n:03}.png")
                image.close()
            target = cls.root / (kind + ".mp4")
            cls.command(["-framerate", "8", "-i", str(frames / "%03d.png"),
                "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:v", "libx264",
                "-crf", "10", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(target)])
            cls.paths[kind] = target

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @classmethod
    def command(cls, args):
        return subprocess.run([cls.exe, "-nostdin", "-v", "error", *args], check=True,
            capture_output=True, timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    @staticmethod
    def enhanced_detector(path, merchant_terms, **kwargs):
        with Image.open(path) as opened:
            pixels = np.asarray(opened.convert("L"))
        ys, xs = np.nonzero(pixels > 195)
        found = []
        if len(xs):
            found = [{**region(), "x": int(xs.min())-2, "y": int(ys.min())-2,
                "w": int(xs.max()-xs.min())+5, "h": int(ys.max()-ys.min())+5, "confidence": .98}]
        return extra(found)

    def test_real_faint_can_pass_but_moving_and_subtitle_stay_reviewable(self):
        for name, expected_code in [("faint", None),
            ("moving", "temporal_gap_or_motion"), ("subtitle", "not_small_corner")]:
            source = self.paths[name]
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            with patch("buyer_show.watermark_batch_video.detect_image", return_value={
                    "regions": [], "text_regions": [], "scan_complete": True}) as base, \
                    patch("buyer_show.watermark_text_extra.detect_extra_text", side_effect=self.enhanced_detector) as enhanced:
                result = plan_video(source, merchant_terms="ACME")
            self.assertTrue(result["regions"])
            if expected_code:
                self.assertEqual(result["status"], "needs_review")
                self.assertEqual(result["approved_regions"], [])
                self.assertIn(expected_code, [r["code"] for r in result["review_reasons"]])
            else:
                self.assertEqual(result["status"], "approved", result["review_reasons"])
                self.assertTrue(result["approved_regions"])
                self.assertTrue(result["evidence"]["tracks"][0]["visual"]["relative_stroke_evidence"])
            self.assertEqual(enhanced.call_count, 8)
            self.assertTrue(all(c.kwargs["return_text_regions"] for c in base.call_args_list))
            self.assertTrue(result["detection"]["has_audio"])
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)

    def test_manual_video_candidates_merge_per_time_and_audio_repair_is_unchanged(self):
        source = self.paths["faint"]
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        with patch("buyer_show.watermark.detect_image", return_value={"regions": [], "text_regions": [], "scan_complete": True}), \
                patch("buyer_show.watermark_text_extra.detect_extra_text", side_effect=self.enhanced_detector):
            found = video.detect_video(source, merchant_terms="ACME", detection_profile="product")
        self.assertEqual(len(found["regions"]), 1)
        self.assertEqual(len(found["extra_text_samples"]), 6)
        self.assertEqual((found["regions"][0]["start"], found["regions"][0]["end"]), (0, 2))
        # Manual confirmation is deliberately distinct from an automatic plan.
        repaired = video.repair_video(source, found["regions"], self.root / "processed")
        manifest = json.loads(Path(repaired["manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["validation"]["audio_validation"], "full_decode")
        hashes = [self.command(["-i", str(p), "-map", "0:a:0", "-c:a", "copy", "-f", "hash", "-"]).stdout
                  for p in (source, repaired["output"])]
        self.assertEqual(hashes[0], hashes[1])
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()
