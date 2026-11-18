"""Windowed executable entry. --self-check validates the bundle without a browser."""
from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
import sys
import subprocess


def self_check(path: Path, *, include_ai: bool = False) -> int:
    result = {"ok": False, "frozen": bool(getattr(sys, "frozen", False)), "browser_launched": False}
    try:
        import certifi
        import httpx
        import imageio_ffmpeg
        import tkinter
        from PIL import Image
        from playwright.async_api import async_playwright
        from playwright._impl._driver import compute_driver_executable
        from buyer_show import __version__
        from buyer_show import gui, engine
        from buyer_show.links import classify_url
        import cv2
        import numpy as np
        from rapidocr_onnxruntime import RapidOCR
        from buyer_show import watermark_ai

        # This check imports the optional backend but never downloads weights.
        model = watermark_ai.model_status()
        result["ai_backend_imported"] = callable(watermark_ai.inpaint_rgb)
        result["ai_model_cached"] = model["installed"]
        result["ai_inference_requested"] = include_ai

        node, driver = compute_driver_executable()
        ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
        if not Path(node).is_file() or not Path(driver).is_file():
            raise RuntimeError("missing_driver")
        if not Path(certifi.where()).is_file():
            raise RuntimeError("missing_certificates")
        if result["frozen"]:
            bundle = Path(sys._MEIPASS).resolve()
            if not Path(node).resolve().is_relative_to(bundle):
                raise RuntimeError("driver_outside_bundle")
            if not ffmpeg.is_relative_to(bundle):
                raise RuntimeError("ffmpeg_outside_bundle")
        video_check = subprocess.run([str(ffmpeg), "-version"], capture_output=True, timeout=15,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        result["video_runtime"] = video_check.returncode == 0 and b"ffmpeg version" in video_check.stdout
        detector = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)
        detector(np.full((96, 256, 3), 255, dtype=np.uint8))
        test_mask = np.zeros((32, 32), dtype=np.uint8)
        test_mask[12:16, 12:16] = 255
        repaired = cv2.inpaint(np.zeros((32, 32, 3), dtype=np.uint8), test_mask, 3, cv2.INPAINT_TELEA)
        result["watermark_runtime"] = repaired.shape == (32, 32, 3)
        result.update(version=__version__, tcl=tkinter.Tcl().eval("info patchlevel"),
                      httpx=httpx.__version__, playwright_driver_files=True,
                      ca_certificates=True, gui_imported=gui.ValidatorWindow is not None,
                      adapters=sorted(engine.ADAPTERS))
        output = io.BytesIO()
        Image.new("RGB", (4, 4), "navy").save(output, "PNG")
        with Image.open(io.BytesIO(output.getvalue())) as image:
            image.load()
            result["image_codec"] = image.size == (4, 4)
        result["product_links"] = classify_url("https://item.taobao.com/item.htm?id=123").product_id == "123"
        # Exercise the same spawn entry point used by the watermark window.
        # This catches frozen-app recursion and missing child-process modules.
        import multiprocessing
        import tempfile
        import time
        from buyer_show.watermark_ui import worker_main
        with tempfile.TemporaryDirectory(prefix="buyer-show-check-") as directory:
            source = Path(directory) / "sample.png"
            Image.new("RGB", (64, 48), "navy").save(source)
            context = multiprocessing.get_context("spawn")

            def check_worker(job, timeout=20):
                events, stop = context.Queue(), context.Event()
                worker = context.Process(target=worker_main, args=(job, events, stop))
                try:
                    worker.start()
                    deadline = time.monotonic() + timeout
                    while True:
                        kind, payload = events.get(timeout=max(.1, deadline - time.monotonic()))
                        if kind in {"result", "error", "cancelled"}:
                            break
                        if time.monotonic() >= deadline:
                            raise TimeoutError("watermark_worker_timeout")
                    worker.join(timeout=5)
                    if kind != "result" or worker.exitcode != 0:
                        raise RuntimeError("watermark_worker_failed")
                    return payload
                finally:
                    stop.set()
                    if worker.is_alive():
                        worker.terminate()
                        worker.join(timeout=5)
                    events.close()
                    events.join_thread()

            preview = check_worker({"action": "preview", "source": str(source), "seconds": 0})
            result["watermark_worker"] = preview["meta"]["width"] == 64
            detected = check_worker({"action": "detect", "source": str(source),
                "detection_profile": "product", "merchant_terms": "样例商家", "include_corner_text": False})
            result["watermark_merchant_worker"] = detected["width"] == 64 and detected["regions"] == []
            batch = check_worker({"action": "batch", "items": [{"path": str(source), "media_type": "image",
                                  "source_types": ["product_main"], "product_key": "selfcheck:1"}],
                                  "output_root": str(Path(directory) / "batch"), "image_method": "smart"}, timeout=30)
            result["watermark_batch_worker"] = (batch["status"] == "completed" and batch["counts"]["unchanged"] == 1
                and Path(batch["items"][0]["output"]).read_bytes() == source.read_bytes()
                and Path(batch["manifest"]).is_file())
            video_source = Path(directory) / "sample-video.mp4"
            subprocess.run([str(ffmpeg), "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                "color=c=blue:s=320x240:r=8:d=1,drawbox=x=20:y=20:w=24:h=16:color=white:t=fill",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-c:a", "aac", str(video_source)], check=True,
                capture_output=True, timeout=20, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            video_original = video_source.read_bytes()
            video_result = check_worker({"action": "repair", "source": str(video_source),
                "video_method": "lama_temporal", "regions": [{"x": 18, "y": 18, "w": 28, "h": 20,
                "start": 0, "end": 1}], "seconds": 0}, timeout=90)
            video_manifest = json.loads(Path(video_result["manifest"]).read_text(encoding="utf-8"))
            result["watermark_video_quality_worker"] = (
                video_result.get("method") == "lama_temporal"
                and video_manifest["validation"]["frames_decoded"] == 8
                and video_manifest["output_metadata"]["has_audio"]
                and bool(video_manifest.get("quality"))
                and video_source.read_bytes() == video_original)
            pixels = np.full((48, 64, 3), (0, 0, 128), dtype=np.uint8)
            pixels[18:24, 20:28] = 255
            Image.fromarray(pixels).save(source)
            original = source.read_bytes()
            repaired = check_worker({"action": "repair", "source": str(source), "method": "auto",
                "regions": [{"x": 16, "y": 14, "w": 18, "h": 16}], "seconds": 0})
            with Image.open(repaired["output"]) as checked:
                result["watermark_repair_worker"] = (
                    checked.size == (64, 48) and checked.convert("RGB").getpixel((22, 20)) == (0, 0, 128)
                    and source.read_bytes() == original)
            if include_ai:
                # Explicit diagnostic only, with a verified local model. It
                # must fail instead of initiating a model download.
                if not watermark_ai._verify(watermark_ai.model_path(), lambda: False):
                    raise RuntimeError("verified_local_ai_model_required")
                ai_result = check_worker({"action": "repair", "source": str(source), "method": "lama",
                    "regions": [{"x": 16, "y": 14, "w": 18, "h": 16}], "seconds": 0}, timeout=120)
                with Image.open(ai_result["output"]) as checked:
                    rgba = np.asarray(checked.convert("RGBA"))
                    with Image.open(source) as original_image:
                        expected = np.asarray(original_image.convert("RGBA"))
                    exterior = np.ones((48, 64), dtype=bool)
                    exterior[14:30, 16:34] = False
                    manifest = json.loads(Path(ai_result["manifest"]).read_text(encoding="utf-8"))
                    result["watermark_ai_worker"] = (
                        checked.size == (64, 48) and np.array_equal(rgba[exterior], expected[exterior])
                        and np.array_equal(rgba[:, :, 3], expected[:, :, 3])
                        and source.read_bytes() == original and manifest["requested_method"] == "lama")

        async def check_driver():
            async with async_playwright() as pw:
                return pw.chromium.name == "chromium"

        result["driver_process_started"] = asyncio.run(check_driver())
        result["ok"] = all(result[k] for k in ("image_codec", "product_links", "driver_process_started", "video_runtime",
            "watermark_runtime", "watermark_worker", "watermark_merchant_worker", "watermark_repair_worker",
            "ai_backend_imported", "watermark_batch_worker", "watermark_video_quality_worker")) and (not include_ai or result.get("watermark_ai_worker") is True)
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        # Dependency initialization only: do not include browser/profile data.
        result["error"] = str(exc)[:500]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if result["ok"] else 1


def main():
    if len(sys.argv) == 3 and sys.argv[1] in {"--self-check", "--self-check-ai"}:
        return self_check(Path(sys.argv[2]).resolve(), include_ai=sys.argv[1] == "--self-check-ai")
    from tkinter import messagebox
    from buyer_show.runtime import InstanceLock, app_data_dir, setup_logging
    lock = InstanceLock()
    try:
        if not lock.acquire():
            messagebox.showinfo("软件已运行", "买家秀采集助手已在运行，请切换到现有窗口。")
            return 0
        logger = setup_logging()
        from buyer_show import __version__
        logger.info("Desktop started version=%s", __version__)
        from buyer_show.gui import main as gui_main
        gui_main()
        logger.info("Desktop closed")
        return 0
    except Exception as exc:
        # Avoid logging exception strings containing cookies, signatures or URLs.
        try:
            logger = setup_logging()
            logger.error("Desktop failure: %s", type(exc).__name__)
        except Exception:
            pass
        messagebox.showerror("启动未完成", "软件未能正常启动（" + type(exc).__name__ + "）。\n请检查解压后的文件是否完整以及本地目录权限。\n日志目录：" + str(app_data_dir() / "logs"))
        return 1
    finally:
        lock.release()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(main())
