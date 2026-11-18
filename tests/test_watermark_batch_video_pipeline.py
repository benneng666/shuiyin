"""Runner-to-real-FFmpeg regression; OCR policy has separate actual-clip QA."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from buyer_show.watermark_batch import run_batch
from buyer_show.watermark_video import repair_video
from buyer_show.video import ffmpeg_executable


class BatchVideoPipelineTests(unittest.TestCase):
    def test_approved_video_routes_through_real_repair_and_verified_classified_export(self):
        with tempfile.TemporaryDirectory(prefix="buyer_show_video_pipeline_") as directory:
            root = Path(directory)
            source = root / "control.mp4"
            exe = ffmpeg_executable()

            def ffmpeg(arguments):
                return subprocess.run([exe, "-nostdin", "-v", "error", *arguments],
                                      check=True, capture_output=True, timeout=30,
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

            ffmpeg(["-f", "lavfi", "-i",
                    "color=c=blue:s=320x240:r=12:d=3,drawbox=x=20:y=20:w=48:h=16:color=white:t=fill",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(source)])
            original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            region = {"x": 18, "y": 18, "w": 52, "h": 20, "start": 0, "end": 3,
                      "label": "合成控制框", "confidence": .99}
            # Only OCR/policy is replaced. Encoding, full video/audio validation,
            # journalling and the classified independent copy are all real.
            decision = {"status": "approved", "source_sha256": original_hash,
                        "regions": [region], "approved_regions": [region], "review_reasons": [],
                        "evidence": {"tracks": [{"approved": True, "region": region}]}}
            item = {"path": str(source), "media_type": "video", "source_types": ["product_video"],
                    "product_key": "control:123", "sha256": original_hash}
            with patch("buyer_show.watermark_batch.plan_video", return_value=decision) as plan, \
                    patch("buyer_show.watermark_batch.repair_video", wraps=repair_video) as repair, \
                    patch("buyer_show.watermark_batch.scan_image") as scan, \
                    patch("buyer_show.watermark_batch.repair_image") as image_repair:
                summary = run_batch([item], output_root=root / "derived", merchant_terms="ACME")
            self.assertEqual(plan.call_count, 1)
            self.assertEqual(plan.call_args.kwargs["merchant_terms"], "ACME")
            repair.assert_called_once()
            scan.assert_not_called()
            image_repair.assert_not_called()
            self.assertEqual(summary["status"], "completed")
            self.assertEqual((summary["counts"]["processed"], summary["counts"]["failed"],
                              summary["counts"]["exported_files"]), (1, 0, 1))
            row = summary["items"][0]
            self.assertEqual(row["repair_method"], "smart")
            self.assertEqual(row["exports"][0]["category"], "商品视频")
            exported = Path(row["exports"][0]["path"])
            processing = json.loads(Path(row["manifest"]).read_text(encoding="utf-8"))
            expected = processing["validation"]["sha256"]
            self.assertEqual(hashlib.sha256(exported.read_bytes()).hexdigest(), expected)
            self.assertEqual(row["exports"][0]["sha256"], expected)
            self.assertEqual(processing["validation"]["frames_decoded"], 36)
            self.assertEqual(processing["validation"]["audio_validation"], "full_decode")
            self.assertEqual(processing["source_sha256"], original_hash)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), original_hash)
            hashes = [ffmpeg(["-i", str(path), "-map", "0:a:0", "-c:a", "copy", "-f", "hash", "-"]).stdout
                      for path in (source, exported)]
            self.assertEqual(hashes[0], hashes[1])
            self.assertTrue(Path(summary["html"]).is_file())
            saved = json.loads(Path(summary["manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(saved["items"][0]["status"], "processed")


if __name__ == "__main__":
    unittest.main()
