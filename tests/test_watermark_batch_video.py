"""Policy and real local FFmpeg fixtures; no browser, network or CUDA."""
import asyncio
import copy
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

from buyer_show.watermark_batch_video import classify_video_detection, plan_video, _visual
from buyer_show.watermark_video import repair_video, probe_video
from buyer_show.video import ffmpeg_executable


def candidate(**updates):
    return {"x": 20, "y": 20, "w": 100, "h": 30,
            "label": "商家关键词候选：ACME", "confidence": .98, **updates}


def detection():
    rows = [{"seconds": index * 3.95 / 7, "regions": [candidate()], "scan_complete": True} for index in range(8)]
    return {"width": 640, "height": 400, "duration": 4.0, "samples": rows,
            "regions": [{**candidate(), "start": 0, "end": 4}], "scan_complete": True,
            "repair_supported": True, "warnings": []}


VISUAL = {0: {"screen_locked_ink_with_changing_background": True, "simple_background": True}}


class BatchVideoPolicyTests(unittest.TestCase):
    def test_explicit_stationary_corner_with_visual_evidence_is_approved(self):
        result = classify_video_detection(detection(), merchant_terms="ACME", visual_evidence=VISUAL)
        self.assertEqual(result["status"], "approved")
        self.assertEqual(len(result["approved_regions"]), 1)
        self.assertEqual((result["approved_regions"][0]["start"], result["approved_regions"][0]["end"]), (0, 4))
        self.assertEqual(result["evidence"]["tracking"], "sampled_segments_only")

    def test_clear_shop_marker_does_not_require_supplied_keyword(self):
        value = detection()
        for sample in value["samples"]:
            sample["regions"][0]["label"] = "店铺标记候选：ACME official store"
        result = classify_video_detection(value, visual_evidence=VISUAL)
        self.assertEqual(result["status"], "approved")

    def test_ocr_score_or_keyword_alone_never_approves(self):
        result = classify_video_detection(detection(), merchant_terms="ACME")
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["approved_regions"], [])
        self.assertIn("overlay_evidence_insufficient", [r["code"] for r in result["review_reasons"]])

    def test_motion_missing_sample_subtitle_large_low_score_and_edge_stay_reviewable(self):
        changes = {
            "motion": lambda v: v["samples"][4]["regions"][0].update(x=28),
            "gap": lambda v: v["samples"][4].update(regions=[]),
            "subtitle": lambda v: [s["regions"][0].update(x=220, y=330) for s in v["samples"]],
            "large": lambda v: [s["regions"][0].update(w=260, h=70) for s in v["samples"]],
            "score": lambda v: v["samples"][4]["regions"][0].update(confidence=.85),
            "edge": lambda v: [s["regions"][0].update(x=0) for s in v["samples"]],
            "copy": lambda v: [s["regions"][0].update(label="商家关键词候选：ACME 限时包邮") for s in v["samples"]],
            "physical": lambda v: [s["regions"][0].update(label="商家关键词候选：ACME 实物吊牌") for s in v["samples"]],
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                value = detection()
                change(value)
                result = classify_video_detection(value, merchant_terms="ACME", visual_evidence=VISUAL)
                self.assertEqual(result["status"], "needs_review")
                self.assertEqual(result["approved_regions"], [])

    def test_no_detection_is_not_a_watermark_free_claim_and_incomplete_scan_is_review(self):
        value = detection()
        value["regions"] = []
        for sample in value["samples"]:
            sample["regions"] = []
        result = classify_video_detection(value)
        self.assertEqual(result["status"], "not_detected")
        self.assertTrue(any("不代表" in w for w in result["warnings"]))
        value["scan_complete"] = False
        self.assertEqual(classify_video_detection(value)["status"], "needs_review")

    def test_long_sparse_or_merged_only_detection_cannot_be_auto_approved(self):
        value = detection()
        value["duration"] = 40
        for index, sample in enumerate(value["samples"]):
            sample["seconds"] = index * 39.95 / 7
        self.assertEqual(classify_video_detection(value, merchant_terms="ACME", visual_evidence=VISUAL)["status"], "needs_review")
        del value["samples"]
        self.assertEqual(classify_video_detection(value, merchant_terms="ACME", visual_evidence=VISUAL)["status"], "needs_review")

    def test_static_sign_visual_evidence_is_rejected(self):
        patch = np.full((60, 140), 110, np.uint8)
        patch[20:35, 30:100:4] = 255
        result = _visual([patch] * 8, (20, 10, 100, 40))
        self.assertFalse(result["screen_locked_ink_with_changing_background"])


class BatchVideoRealFilesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="buyer_show_batch_video_test_")
        cls.root = Path(cls.temp.name)
        cls.exe = ffmpeg_executable()
        font_path = Path("C:/Windows/Fonts/arial.ttf")
        font = ImageFont.truetype(str(font_path), 26) if font_path.is_file() else ImageFont.load_default(size=26)
        cls.paths = {}
        for name in ("overlay", "unmarked", "moving", "subtitle", "physical"):
            directory = cls.root / name
            directory.mkdir()
            for index in range(24):
                level = round(110 + 45 * math.sin(index / 24 * math.tau))
                frame = Image.new("RGB", (640, 400), (level, level, level))
                painter = ImageDraw.Draw(frame)
                if name != "unmarked":
                    x, y = ((220, 330) if name == "subtitle" else (20 + index * 2, 20) if name == "moving" else (20, 20))
                    if name == "physical":
                        painter.rectangle((5, 5, 170, 80), fill=(110, 110, 110))
                    painter.text((x, y), "ACME", fill="white", font=font)
                frame.save(directory / f"{index:03}.png")
                frame.close()
            target = cls.root / (name + ".mp4")
            cls.command(["-framerate", "8", "-i", str(directory / "%03d.png"),
                         "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
                         "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", "-c:a", "aac",
                         "-shortest", str(target)])
            cls.paths[name] = target

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @classmethod
    def command(cls, args):
        return subprocess.run([cls.exe, "-nostdin", "-v", "error", *args], check=True,
                              capture_output=True, timeout=30,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    @staticmethod
    def synthetic_detector(path, **kwargs):
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("L"))
        ys, xs = np.nonzero(pixels > 230)
        regions = []
        if xs.size:
            regions = [candidate(x=int(xs.min())-2, y=int(ys.min())-2,
                                 w=int(xs.max()-xs.min())+5, h=int(ys.max()-ys.min())+5)]
        return {"regions": regions, "warnings": [], "scan_complete": True}

    def test_real_encoded_videos_approve_static_overlay_reject_motion_subtitle_and_sign(self):
        expected = {"overlay": "approved", "unmarked": "not_detected", "moving": "needs_review",
                    "subtitle": "needs_review", "physical": "needs_review"}
        with patch("buyer_show.watermark_batch_video.detect_image", side_effect=self.synthetic_detector) as detector:
            for name, status in expected.items():
                with self.subTest(name=name):
                    before = hashlib.sha256(self.paths[name].read_bytes()).hexdigest()
                    result = plan_video(self.paths[name], merchant_terms="ACME")
                    self.assertEqual(result["status"], status, result["review_reasons"])
                    self.assertEqual(result["source_sha256"], before)
                    self.assertEqual(hashlib.sha256(self.paths[name].read_bytes()).hexdigest(), before)
                    self.assertEqual(result["detection"]["sample_count"], 8)
            self.assertTrue(all(call.kwargs["detection_profile"] == "product" for call in detector.call_args_list))
            self.assertTrue(all(call.kwargs["include_corner_text"] is False for call in detector.call_args_list))

    def test_approved_plan_uses_existing_repair_audio_and_full_decode_validation(self):
        source = self.paths["overlay"]
        with patch("buyer_show.watermark_batch_video.detect_image", side_effect=self.synthetic_detector):
            plan = plan_video(source, merchant_terms="ACME")
        self.assertEqual(plan["status"], "approved")
        result = repair_video(source, plan["approved_regions"], self.root / "processed")
        manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["validation"]["frames_decoded"], 24)
        self.assertEqual(manifest["validation"]["audio_validation"], "full_decode")
        self.assertAlmostEqual(probe_video(result["output"])["duration"], 3, delta=.02)
        hashes = [self.command(["-i", str(p), "-map", "0:a:0", "-c:a", "copy", "-f", "hash", "-"]).stdout
                  for p in (source, result["output"])]
        self.assertEqual(hashes[0], hashes[1])

    def test_cancel_before_frame_has_no_detection_or_outputs(self):
        with patch("buyer_show.watermark_batch_video.detect_image") as detector:
            with self.assertRaises(asyncio.CancelledError):
                plan_video(self.paths["overlay"], should_stop=lambda: True)
        detector.assert_not_called()


if __name__ == "__main__":
    unittest.main()
