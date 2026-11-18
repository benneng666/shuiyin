"""Image repair invariants and conservative OCR filtering; no UI automation."""

import asyncio
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from buyer_show import watermark as wm


def box(x, y, w, h, text, confidence=.96):
    return [[[x, y], [x + w, y], [x + w, y + h], [x, y + h]], text, confidence]


class RegionTests(unittest.TestCase):
    def test_round_outwards_deduplicate_and_ignore_image_timeline(self):
        result = wm.validate_regions([{"x": 1.2, "y": 3.4, "w": 5.2, "h": 6.1,
                "label": "候选", "confidence": .92, "start": "ignored", "end": -10},
                {"x": 1, "y": 3, "w": 6, "h": 7}], 100, 100)
        self.assertEqual(result, [{"x": 1, "y": 3, "w": 6, "h": 7, "label": "候选", "confidence": .92}])

    def test_coverage_is_union_not_sum_and_forty_percent_is_allowed(self):
        regions = [{"x": 0, "y": 0, "w": 30, "h": 100}, {"x": 10, "y": 0, "w": 30, "h": 100}]
        self.assertEqual(len(wm.validate_regions(regions, 100, 100)), 2)
        regions[1]["w"] = 31
        with self.assertRaisesRegex(ValueError, "40%"):
            wm.validate_regions(regions, 100, 100)

    def test_invalid_boxes_are_rejected_instead_of_clipped(self):
        valid = {"x": 0, "y": 0, "w": 10, "h": 10}
        bad = [None, {}, {**valid, "x": -1}, {**valid, "y": -0.01}, {**valid, "w": 0},
               {**valid, "h": -1}, {**valid, "x": 99}, {**valid, "x": float("nan")},
               {**valid, "w": float("inf")}, {**valid, "w": "10"}, {**valid, "y": True},
               {**valid, "confidence": 1.1}]
        for region in bad:
            with self.subTest(region=region), self.assertRaises(ValueError):
                wm.validate_regions([region], 100, 100)
        for width, height in [(0, 100), (100, -1), (True, 100), (100.5, 100)]:
            with self.assertRaises(ValueError):
                wm.validate_regions([valid], width, height)

    def test_empty_is_editable_but_region_count_is_bounded(self):
        self.assertEqual(wm.validate_regions([], 20, 20), [])
        with self.assertRaises(ValueError):
            wm.validate_regions([{"x": 0, "y": 0, "w": 1, "h": 1}] * 257, 100, 100)


class ImageRepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source.png"
        self.regions = [{"x": 75, "y": 65, "w": 32, "h": 12, "label": "审核区域"}]
        self.original = np.full((100, 140, 4), 255, dtype=np.uint8)
        self.original[69:74, 80:102, :3] = 0  # Synthetic small overlay on a flat background.
        Image.fromarray(self.original).save(self.source)
        self.before = self.source.read_bytes()

    def tearDown(self):
        self.temp.cleanup()

    def decoded(self, path):
        with Image.open(path) as image:
            return np.asarray(ImageOps.exif_transpose(image).convert("RGBA")).copy()

    def check_outputs(self, result, baseline):
        repaired = self.decoded(result["output"])
        mask = np.asarray(Image.open(result["mask"]).convert("L"))
        self.assertTrue(np.array_equal(repaired[mask == 0], baseline[mask == 0]))
        self.assertTrue(np.array_equal(repaired[:, :, 3], baseline[:, :, 3]))
        manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["source_sha256"], hashlib.sha256(self.before).hexdigest())
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["algorithm_version"], wm.ALGORITHM_VERSION)
        self.assertEqual(manifest["regions"], wm.validate_regions(self.regions, result["width"], result["height"]))
        self.assertEqual(manifest["mask_area_pixels"], np.count_nonzero(mask))
        self.assertEqual(self.source.read_bytes(), self.before)
        self.assertEqual(set(Path(result["output"]).parent.iterdir()), {Path(result[k]) for k in ("output", "mask", "manifest")})
        return repaired, mask, manifest

    def test_synthetic_watermark_repaired_without_changing_mask_exterior(self):
        result = wm.repair_image(self.source, self.regions)
        repaired, mask, _ = self.check_outputs(result, self.original)
        before_error = np.mean((255 - self.original[mask != 0, :3].astype(float)) ** 2)
        after_error = np.mean((255 - repaired[mask != 0, :3].astype(float)) ** 2)
        self.assertLess(after_error, before_error / 10)
        self.assertEqual(Path(result["output"]).parent.parent, self.root / "processed")

    def test_alpha_and_invisible_rgb_outside_mask_are_preserved(self):
        self.original[:, :, 3] = np.arange(140, dtype=np.uint8)[None, :]
        self.original[0, :30, :3] = [7, 19, 222]
        Image.fromarray(self.original).save(self.source)
        self.before = self.source.read_bytes()
        result = wm.repair_image(self.source, self.regions)
        _, _, manifest = self.check_outputs(result, self.original)
        self.assertTrue(any("透明度" in warning for warning in manifest["warnings"]))

    def test_repeated_runs_and_explicit_source_directory_never_overwrite(self):
        sentinel = self.root / "result.png"
        sentinel.write_bytes(b"existing unrelated output")
        first = wm.repair_image(self.source, self.regions, output_root=self.root)
        second = wm.repair_image(self.source, self.regions, output_root=self.root)
        self.assertNotEqual(first["output"], second["output"])
        self.assertEqual(sentinel.read_bytes(), b"existing unrelated output")
        self.assertEqual(self.source.read_bytes(), self.before)

    def test_exif_orientation_coordinates_and_png_orientation(self):
        self.source = self.root / "rotated.jpg"
        pixels = np.arange(80 * 40 * 3, dtype=np.uint8).reshape(40, 80, 3)
        exif = Image.Exif()
        exif[274] = 6
        Image.fromarray(pixels).save(self.source, exif=exif)
        self.before = self.source.read_bytes()
        self.regions = [{"x": 3, "y": 4, "w": 8, "h": 6}]
        baseline = self.decoded(self.source)
        result = wm.repair_image(self.source, self.regions)
        self.assertEqual((result["width"], result["height"]), (40, 80))
        self.check_outputs(result, baseline)
        with Image.open(result["output"]) as opened:
            self.assertIn(opened.getexif().get(274), (None, 1))

    def test_palette_transparency_is_preserved_as_rgba(self):
        palette = Image.new("P", (140, 100))
        palette.putpalette([0, 0, 0, 240, 120, 30] + [0] * (768 - 6))
        ImageDraw.Draw(palette).rectangle((20, 20, 110, 80), fill=1)
        palette.info["transparency"] = 0
        palette.save(self.source)
        self.before = self.source.read_bytes()
        result = wm.repair_image(self.source, self.regions)
        self.check_outputs(result, self.decoded(self.source))

    def test_animation_invalid_image_high_precision_and_empty_mask_rejected(self):
        animated = self.root / "animated.gif"
        Image.new("RGB", (20, 20), "red").save(animated, save_all=True,
            append_images=[Image.new("RGB", (20, 20), "blue")], loop=0)
        for action in (lambda: wm.repair_image(animated, [{"x": 0, "y": 0, "w": 2, "h": 2}]),
                       lambda: wm.detect_image(animated)):
            with self.assertRaisesRegex(ValueError, "动画|多帧"):
                action()
        bad = self.root / "bad.png"
        bad.write_bytes(b"not an image secret=value")
        with self.assertRaisesRegex(ValueError, "无法读取"):
            wm.repair_image(bad, self.regions)
        high = self.root / "precision.png"
        Image.fromarray(np.ones((10, 10), dtype=np.uint16) * 5000).save(high)
        with self.assertRaisesRegex(ValueError, "色彩模式"):
            wm.repair_image(high, [{"x": 0, "y": 0, "w": 2, "h": 2}])
        with self.assertRaisesRegex(ValueError, "至少"):
            wm.repair_image(self.source, [])
        self.assertFalse((self.root / "processed").exists())

    def test_cancellation_before_work_and_after_native_repair(self):
        with self.assertRaises(asyncio.CancelledError):
            wm.repair_image(self.source, self.regions, should_stop=lambda: True)
        import cv2
        real = cv2.inpaint
        cancelled = False
        def wrapped(*args, **kwargs):
            nonlocal cancelled
            value = real(*args, **kwargs)
            cancelled = True
            return value
        with mock.patch.object(cv2, "inpaint", side_effect=wrapped), self.assertRaises(asyncio.CancelledError):
            wm.repair_image(self.source, self.regions, should_stop=lambda: cancelled, method='telea')
        self.assertFalse((self.root / "processed").exists())
        self.assertEqual(self.source.read_bytes(), self.before)

    def test_cancel_before_publish_removes_staging_success_manifest(self):
        out = self.root / "processed"
        def stop():
            return out.exists() and any(out.glob(".image-repair-*/processing.json"))
        with self.assertRaises(asyncio.CancelledError):
            wm.repair_image(self.source, self.regions, should_stop=stop)
        self.assertEqual(list(out.iterdir()), [])
        self.assertEqual(self.source.read_bytes(), self.before)

    def test_write_or_commit_failure_never_leaves_success_marker(self):
        real_save = Image.Image.save
        def save(image, fp, *args, **kwargs):
            if Path(fp).name == "result.png":
                raise OSError("private signature should not appear")
            return real_save(image, fp, *args, **kwargs)
        with mock.patch.object(Image.Image, "save", save), self.assertRaises(RuntimeError) as error:
            wm.repair_image(self.source, self.regions)
        self.assertNotIn("private signature", str(error.exception))
        self.assertEqual(list((self.root / "processed").iterdir()), [])
        with mock.patch.object(wm.os, "rename", side_effect=PermissionError("secret")), self.assertRaises(RuntimeError):
            wm.repair_image(self.source, self.regions)
        self.assertEqual(list((self.root / "processed").iterdir()), [])
        self.assertEqual(self.source.read_bytes(), self.before)

    def test_logging_callback_failure_does_not_corrupt_committed_result(self):
        def broken(_message):
            raise RuntimeError("UI closed")
        result = wm.repair_image(self.source, self.regions, emit=broken)
        self.assertTrue(Path(result["manifest"]).exists())


class DetectionTests(unittest.TestCase):
    def test_platform_and_corner_account_candidates_not_all_product_text(self):
        rows = [box(820, 720, 100, 30, "小红书"), box(800, 760, 160, 20, "@buyer_123"),
                box(5, 5, 100, 30, "RED"), box(100, 350, 200, 40, "纯棉舒适透气"),
                box(820, 5, 150, 30, "天猫官方旗舰店"), box(830, 100, 130, 30, "RED DRESS"),
                box(400, 300, 130, 30, "@center_user"), box(5, 700, 180, 30, "客服 a@shop.com"),
                box(10, 750, 100, 30, "抖音", .2)]
        result = wm._candidate_regions(rows, 1000, 800)
        self.assertEqual(len(result), 3)
        self.assertTrue(any("小红书" in r["label"] for r in result))
        self.assertTrue(any("@buyer_123" in r["label"] for r in result))
        self.assertTrue(all(isinstance(r[k], int) for r in result for k in ("x", "y", "w", "h")))

    def test_detection_normalizes_orientation_uses_matte_and_warns_no_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "image.png"
            exif = Image.Exif()
            exif[274] = 6
            Image.new("RGBA", (80, 40), (255, 0, 0, 0)).save(source, exif=exif)
            original = source.read_bytes()
            def engine(image):
                self.assertEqual(image.shape, (80, 40, 3))
                self.assertTrue(np.all(image == 255))
                return None, None
            with mock.patch.object(wm, "_get_ocr", return_value=engine):
                result = wm.detect_image(source)
            self.assertEqual((result["width"], result["height"]), (40, 80))
            self.assertEqual(result["regions"], [])
            self.assertTrue(any("不是水印概率" in s for s in result["warnings"]))
            self.assertEqual(source.read_bytes(), original)

    def test_cpu_provider_and_thread_configuration_is_explicit(self):
        fake = mock.Mock()
        fake.RapidOCR.return_value = "engine"
        with mock.patch.object(wm, "_OCR_ENGINE", None), mock.patch.dict(sys.modules, {"rapidocr_onnxruntime": fake}):
            self.assertEqual(wm._get_ocr(), "engine")
        kwargs = fake.RapidOCR.call_args.kwargs
        for name in ("det", "cls", "rec"):
            self.assertFalse(kwargs[name + "_use_cuda"])
            self.assertFalse(kwargs[name + "_use_dml"])
        self.assertEqual(kwargs["intra_op_num_threads"], 2)
        self.assertEqual(kwargs["inter_op_num_threads"], 1)

    def test_detection_cancellation_and_safe_error(self):
        with self.assertRaises(asyncio.CancelledError):
            wm.detect_image("unused", should_stop=lambda: True)
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "image.png"
            Image.new("RGB", (30, 30)).save(source)
            def broken(_image):
                raise RuntimeError("private data")
            with mock.patch.object(wm, "_get_ocr", return_value=broken), self.assertRaises(RuntimeError) as error:
                wm.detect_image(source)
            self.assertNotIn("private data", str(error.exception))

    def test_module_import_does_not_initialize_native_or_ocr_dependencies(self):
        check = subprocess.run([sys.executable, "-c", "import sys; import buyer_show.watermark; "
                 "assert not any(n in sys.modules for n in ('cv2','numpy','onnxruntime','rapidocr_onnxruntime'))"],
                 capture_output=True, timeout=15)
        self.assertEqual(check.returncode, 0, check.stderr.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    unittest.main()
