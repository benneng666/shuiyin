"""Mask geometry, worker lifecycle and GUI coordination without opening Tk."""
import asyncio
import io
from pathlib import Path
import queue
import tempfile
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from buyer_show.gui import ValidatorWindow
from buyer_show.watermark_ui import (
    CanvasTransform, WorkerController, WatermarkWindow, discover_media,
    validated_region, worker_main, region_active_at,
    _read_preview,
    lama_model_notice,
)


class GeometryTests(unittest.TestCase):
    def test_actual_size_long_image_scroll_coordinates_and_crop(self):
        transform = CanvasTransform.actual_size(1000, 10000, 400, 300)
        self.assertEqual(transform.scale, 1)
        self.assertEqual(transform.visible_crop(200, 7200, 400, 300), (200, 7200, 600, 7500))
        region = transform.rectangle((220, 7230), (280, 7290))
        self.assertEqual((region["x"], region["y"], region["w"], region["h"]), (220, 7230, 60, 60))
        self.assertEqual(transform.box(region), (220, 7230, 280, 7290))
        centered = CanvasTransform.actual_size(100, 50, 400, 300)
        self.assertEqual(centered.point(175, 140), (25, 15))
        self.assertEqual(centered.visible_crop(0, 0, 400, 300), (0, 0, 100, 50))

    def test_letterboxed_preview_maps_back_to_original_coordinates(self):
        fit = CanvasTransform.fit(2000, 1000, 600, 500)
        self.assertEqual((fit.scale, fit.left, fit.top), (0.3, 0, 100))
        self.assertEqual(fit.point(300, 250), (1000, 500))
        region = fit.rectangle((450, 340), (150, 160))
        self.assertEqual({key: region[key] for key in ("x", "y", "w", "h")},
                         {"x": 500, "y": 200, "w": 1000, "h": 600})
        self.assertEqual(fit.box(region), (150, 160, 450, 340))

    def test_drag_clips_letterbox_and_rejects_zero_area(self):
        fit = CanvasTransform.fit(100, 50, 200, 200)
        region = fit.rectangle((-20, -20), (250, 250))
        self.assertEqual((region["x"], region["y"], region["w"], region["h"]), (0, 0, 100, 50))
        self.assertIsNone(fit.rectangle((10, 0), (20, 20)))

    def test_pixels_must_fit_image_instead_of_silently_clipping_edits(self):
        for edits in ({"x": -1}, {"w": 0}, {"x": 95}, {"x": "1.2"}, {"w": "nan"}):
            row = {"x": 10, "y": 10, "w": 20, "h": 20, **edits}
            with self.subTest(edits=edits), self.assertRaises(ValueError):
                validated_region(row, 100, 100)

    def test_video_time_ranges_and_image_removal_of_time_fields(self):
        region = {"x": "2", "y": "3", "w": "20", "h": "30", "start": "1.25", "end": "4.5"}
        result = validated_region(region, 100, 100, 5.0)
        self.assertEqual((result["x"], result["start"], result["end"]), (2, 1.25, 4.5))
        for start, end in ((-1, 2), (2, 2), (4, 3), (0, 6), (0, "nan")):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                validated_region({**region, "start": start, "end": end}, 100, 100, 5.0)
        self.assertNotIn("start", validated_region(region, 100, 100))

    def test_only_current_video_interval_is_visible_at_boundaries(self):
        first, next_region = {"start": 0, "end": 2.5}, {"start": 2.5, "end": 5}
        self.assertTrue(region_active_at(first, 0))
        self.assertTrue(region_active_at(first, 2.499))
        self.assertFalse(region_active_at(first, 2.5))
        self.assertTrue(region_active_at(next_region, 2.5))
        self.assertFalse(region_active_at(next_region, 5))
        self.assertTrue(region_active_at({"x": 1}, 0))
        self.assertFalse(region_active_at({"start": "invalid", "end": 4}, 1))


class WorkerTests(unittest.TestCase):
    def test_batch_forwards_review_only_scope_for_mixed_source_item(self):
        module = ModuleType('buyer_show.watermark_batch')
        module.run_batch = Mock(return_value={'status': 'completed', 'items': [], 'counts': {}})
        item = {'path': 'mixed.png', 'source_types': ['product_main', 'review_initial']}
        with patch.dict('sys.modules', {'buyer_show.watermark_batch': module}):
            worker_main({'action': 'batch', 'items': [item], 'include_products': False, 'include_reviews': True},
                        queue.Queue(), threading.Event())
        self.assertFalse(module.run_batch.call_args.kwargs['include_products'])
        self.assertTrue(module.run_batch.call_args.kwargs['include_reviews'])
        self.assertEqual(module.run_batch.call_args.args[0][0]['source_types'], ['product_main', 'review_initial'])
        self.assertEqual(module.run_batch.call_args.kwargs['video_method'], 'smart')

    def test_manual_video_mode_reaches_video_backend_without_image_mode(self):
        events = queue.Queue()
        module = ModuleType('buyer_show.watermark_video')
        module.repair_video = Mock(return_value={'output': 'result.mp4', 'method': 'lama_temporal'})
        with patch.dict('sys.modules', {'buyer_show.watermark_video': module}), \
             patch('buyer_show.watermark_ui._read_preview', return_value={'preview': b'preview'}):
            worker_main({'action': 'repair', 'source': 'source.mp4', 'regions': [],
                         'video_method': 'lama_temporal', 'method': 'telea'}, events, threading.Event())
        self.assertEqual(module.repair_video.call_args.kwargs['method'], 'lama_temporal')
        self.assertEqual(events.get_nowait()[0], 'result')

    def test_real_spawn_batch_worker_copies_unmarked_local_image(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'plain.png'
            Image.new('RGB', (256, 256), 'white').save(source)
            before = source.read_bytes()
            controller = WorkerController()
            messages = []
            try:
                controller.start({'action': 'batch', 'items': [{'path': str(source), 'media_type': 'image', 'source_types': []}],
                                  'include_local': True, 'image_method': 'auto', 'output_root': str(Path(folder)/'processed')})
                deadline = time.monotonic() + 60
                while controller.process.is_alive() and time.monotonic() < deadline:
                    messages.extend(controller.drain())
                    time.sleep(.05)
                self.assertFalse(controller.process.is_alive(), 'Real batch worker did not finish')
                messages.extend(controller.drain(limit=None))
                controller.reap()
                results = [value for kind, value in messages if kind == 'result']
                self.assertEqual(len(results), 1, messages)
                result = results[0]
                self.assertEqual(result['counts']['unchanged'], 1)
                self.assertEqual(result['counts']['processed'], 0)
                self.assertTrue(Path(result['manifest']).is_file())
                self.assertEqual(Path(result['items'][0]['output']).read_bytes(), before)
                self.assertEqual(source.read_bytes(), before)
                self.assertTrue(any(kind == 'batch_progress' for kind, _ in messages))
                self.assertFalse(any(kind == 'error' for kind, _ in messages))
            finally:
                if controller.busy:
                    controller.cancel()
                    end = time.monotonic() + 30
                    while controller.process.is_alive() and time.monotonic() < end:
                        controller.drain()
                        time.sleep(.05)
                    controller.drain(limit=None)
                    controller.reap()

    def test_batch_uses_one_runner_call_and_returns_cancelled_partial_results_without_previews(self):
        events, stop = queue.Queue(), threading.Event()
        module = ModuleType('buyer_show.watermark_batch')
        items = [{'path': 'one.png', 'product_key': 'taobao:1'}, {'path': 'two.png', 'product_key': 'taobao:2'}]
        def batch(rows, **options):
            self.assertEqual(rows, items)
            self.assertEqual(options['image_method'], 'smart')
            self.assertFalse(options['include_reviews'])
            self.assertEqual(options['detection_profile'], 'product')
            options['on_event']({'path': 'one.png', 'index': 1, 'total': 2, 'status': 'unchanged', 'counts': {'unchanged': 1}})
            stop.set()
            self.assertTrue(options['should_stop']())
            return {'status': 'cancelled', 'batch_dir': 'batch', 'counts': {'unchanged': 1, 'cancelled': 1}, 'items': []}
        module.run_batch = Mock(side_effect=batch)
        with patch.dict('sys.modules', {'buyer_show.watermark_batch': module}), \
             patch('buyer_show.watermark_ui._read_preview') as preview:
            worker_main({'action': 'batch', 'items': items}, events, stop)
        module.run_batch.assert_called_once()
        preview.assert_not_called()
        self.assertEqual([kind for kind, _ in events.queue], ['batch_progress', 'result'])
        self.assertEqual(events.queue[-1][1]['status'], 'cancelled')

    def test_final_drain_keeps_batch_result_after_many_progress_messages(self):
        controller = WorkerController(Mock())
        controller.events = queue.Queue()
        for index in range(350):
            controller.events.put(('batch_progress', {'index': index}))
        controller.events.put(('result', {'batch_dir': 'saved'}))
        first = controller.drain()
        rest = controller.drain(limit=None)
        self.assertEqual(len(first), 100)
        self.assertEqual(len(rest), 251)
        self.assertEqual(rest[-1], ('result', {'batch_dir': 'saved'}))

    def test_pre_cancelled_batch_reaches_runner_to_persist_cancellation_summary(self):
        events, stop = queue.Queue(), threading.Event()
        stop.set()
        module = ModuleType('buyer_show.watermark_batch')
        def cancelled(items, **options):
            self.assertTrue(options['should_stop']())
            return {'status': 'cancelled', 'counts': {'cancelled': len(items)}, 'items': []}
        module.run_batch = Mock(side_effect=cancelled)
        with patch.dict('sys.modules', {'buyer_show.watermark_batch': module}):
            worker_main({'action': 'batch', 'items': [{'path': 'one.png'}]}, events, stop)
        self.assertEqual(events.get_nowait(), ('result', {'status': 'cancelled', 'counts': {'cancelled': 1}, 'items': []}))

    def test_lama_model_notice_only_reads_cache_status(self):
        from buyer_show import watermark_ai
        with patch.object(watermark_ai, 'ensure_model') as download, \
             patch.object(watermark_ai, '_get_session') as load, \
             patch.object(watermark_ai, 'model_status') as status:
            status.return_value = {'installed': False, 'size_bytes': 0}
            self.assertIn('未下载', lama_model_notice())
            self.assertIn('208 MB', lama_model_notice())
            status.return_value = {'installed': False, 'size_bytes': 8000}
            self.assertIn('重新下载', lama_model_notice())
            status.return_value = {'installed': True, 'size_bytes': 208044816}
            self.assertIn('已缓存', lama_model_notice())
            self.assertIn('处理前校验', lama_model_notice())
            status.side_effect = PermissionError()
            self.assertIn('暂不可读', lama_model_notice())
            download.assert_not_called()
            load.assert_not_called()

    def test_lama_worker_progress_and_cancellation_use_existing_channel(self):
        events, stop = queue.Queue(), threading.Event()
        module = ModuleType('buyer_show.watermark')
        def repair(source, regions, *, output_root, should_stop, emit, method):
            self.assertEqual(method, 'lama')
            self.assertFalse(should_stop())
            emit('下载 AI 模型：已下 12.0 / 208.0 MB，可取消。')
            stop.set()
            self.assertTrue(should_stop())
            raise asyncio.CancelledError()
        module.repair_image = repair
        with patch.dict('sys.modules', {'buyer_show.watermark': module}):
            worker_main({'action': 'repair', 'source': 'image.png', 'regions': [], 'method': 'lama'}, events, stop)
        self.assertEqual(list(events.queue), [('progress', '下载 AI 模型：已下 12.0 / 208.0 MB，可取消。'), ('cancelled', None)])

    def test_lama_failure_preserves_actionable_backend_error_without_fake_result(self):
        events = queue.Queue()
        module = ModuleType('buyer_show.watermark')
        module.repair_image = Mock(side_effect=RuntimeError('AI 模型完整性校验失败，已拒绝加载；请重试下载。'))
        with patch.dict('sys.modules', {'buyer_show.watermark': module}), \
             patch('buyer_show.watermark_ui._read_preview') as preview:
            worker_main({'action': 'repair', 'source': 'image.png', 'regions': [], 'method': 'lama'}, events, threading.Event())
        preview.assert_not_called()
        self.assertEqual(list(events.queue), [('error', 'AI 模型完整性校验失败，已拒绝加载；请重试下载。')])

    def test_full_resolution_preview_preserves_long_image_pixels_separately_from_thumbnail(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "detail.png"
            image = Image.new("RGB", (800, 5000), "blue")
            image.putpixel((450, 4200), (247, 31, 92))
            image.save(path)
            result = _read_preview(path, full_resolution=True)
            with Image.open(io.BytesIO(result["preview"])) as thumbnail:
                self.assertEqual(thumbnail.size, (160, 1000))
            with Image.open(io.BytesIO(result["full_preview"])) as full:
                self.assertEqual(full.size, (800, 5000))
                self.assertEqual(full.getpixel((450, 4200)), (247, 31, 92))

    def test_worker_forwards_merchant_detection_options_to_images_and_videos(self):
        for filename, name, function in (("image.png", "buyer_show.watermark", "detect_image"),
                                         ("video.mp4", "buyer_show.watermark_video", "detect_video")):
            with self.subTest(filename=filename):
                module = ModuleType(name)
                detector = Mock(return_value={"width": 100, "height": 100, "regions": []})
                setattr(module, function, detector)
                events, stop = queue.Queue(), threading.Event()
                with patch.dict("sys.modules", {name: module}):
                    worker_main({"action": "detect", "source": filename, "include_corner_text": True,
                                 "merchant_terms": "店名,品牌", "detection_profile": "product"}, events, stop)
                kwargs = detector.call_args.kwargs
                self.assertEqual(kwargs["merchant_terms"], "店名,品牌")
                self.assertEqual(kwargs["detection_profile"], "product")
                self.assertTrue(kwargs["include_corner_text"])
                self.assertFalse(kwargs["should_stop"]())
                self.assertEqual(events.get_nowait()[0], "result")

    def test_image_method_forwarded_but_video_keeps_its_own_default_method(self):
        for filename, name, function in (("image.png", "buyer_show.watermark", "repair_image"),
                                         ("video.mp4", "buyer_show.watermark_video", "repair_video")):
            with self.subTest(filename=filename):
                module = ModuleType(name)
                repair = Mock(return_value={"output": "result.png"})
                setattr(module, function, repair)
                events = queue.Queue()
                with patch.dict("sys.modules", {name: module}), \
                     patch("buyer_show.watermark_ui._read_preview", return_value={"preview": b"small", "full_preview": b"full"}):
                    worker_main({"action": "repair", "source": filename, "regions": [], "method": "lama",
                                 "full_resolution": True}, events, threading.Event())
                if filename.endswith("png"):
                    self.assertEqual(repair.call_args.kwargs["method"], "lama")
                else:
                    self.assertEqual(repair.call_args.kwargs["method"], "delogo")
                self.assertEqual(events.get_nowait()[1]["full_after"], b"full")

    def test_directory_import_excludes_processed_at_every_depth(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("one.jpg", "sub/two.mp4", "processed/out.jpg", "sub/PrOcEsSeD/out.png", "note.txt"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            self.assertEqual({Path(x).name for x in discover_media(root)}, {"one.jpg", "two.mp4"})
            self.assertEqual(discover_media(root / "processed"), [])
            with self.assertRaises(asyncio.CancelledError):
                discover_media(root, should_stop=lambda: True)

    def test_preview_runs_without_ui_and_reports_original_size(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.png"
            Image.new("RGB", (1800, 900), "blue").save(path)
            events = queue.Queue()
            worker_main({"action": "preview", "source": str(path), "seconds": 0}, events, threading.Event())
            kind, result = events.get_nowait()
            self.assertEqual(kind, "result")
            self.assertEqual(result["meta"], {"width": 1800, "height": 900})
            with Image.open(io.BytesIO(result["preview"])) as preview:
                self.assertEqual(preview.size, (1400, 700))

    def test_worker_preserves_cancelled_state_and_forwards_stop_to_detector(self):
        events, stop = queue.Queue(), threading.Event()
        module = ModuleType("buyer_show.watermark")
        def detect(path, *, should_stop, emit):
            self.assertFalse(should_stop())
            emit("正在识别")
            stop.set()
            raise asyncio.CancelledError()
        module.detect_image = detect
        with patch.dict("sys.modules", {"buyer_show.watermark": module}):
            worker_main({"action": "detect", "source": "image.png"}, events, stop)
        self.assertEqual(list(events.queue), [("progress", "正在识别"), ("cancelled", None)])

    def test_repair_success_survives_failure_to_preview_result(self):
        events = queue.Queue()
        module = ModuleType("buyer_show.watermark")
        module.repair_image = Mock(return_value={"output": "output.png", "manifest": "manifest.json"})
        with patch.dict("sys.modules", {"buyer_show.watermark": module}), \
             patch("buyer_show.watermark_ui._read_preview", side_effect=OSError()):
            worker_main({"action": "repair", "source": "image.png", "regions": [{"x": 1}]}, events, threading.Event())
        kind, result = events.get_nowait()
        self.assertEqual(kind, "result")
        self.assertEqual(result["output"], "output.png")
        self.assertIn("已另存", result["warnings"][0])

    def test_pre_cancelled_job_does_not_read_source(self):
        events, stop = queue.Queue(), threading.Event()
        stop.set()
        with patch("buyer_show.watermark_ui._read_preview") as preview:
            worker_main({"action": "preview", "source": "absent.png"}, events, stop)
        preview.assert_not_called()
        self.assertEqual(events.get_nowait(), ("cancelled", None))

    def test_controller_only_reaps_after_exit_and_cancel_does_not_terminate(self):
        context = Mock()
        context.Process.return_value.is_alive.return_value = True
        controller = WorkerController(context)
        controller.start({"action": "preview", "source": "a.png"})
        with self.assertRaises(RuntimeError):
            controller.start({"action": "preview", "source": "b.png"})
        controller.cancel()
        context.Event.return_value.set.assert_called_once()
        context.Process.return_value.terminate.assert_not_called()
        self.assertFalse(controller.reap())
        context.Process.return_value.is_alive.return_value = False
        self.assertTrue(controller.reap())
        context.Process.return_value.join.assert_called_once_with(timeout=0)
        context.Queue.return_value.close.assert_called_once()
        self.assertFalse(controller.busy)


class WindowCoordinationTests(unittest.TestCase):
    def test_main_close_waits_for_watermark_cleanup(self):
        window = ValidatorWindow.__new__(ValidatorWindow)
        window.root, window._remember_settings = Mock(), Mock()
        window.running = False
        window.watermark_window = Mock(closed=False)
        window._close()
        window.root.destroy.assert_not_called()
        callback = window.watermark_window.request_close.call_args.kwargs["on_closed"]
        window.watermark_window.closed = True
        callback()
        window.root.destroy.assert_called_once()

    def test_main_window_uses_selected_product_manifest_directory(self):
        window = ValidatorWindow.__new__(ValidatorWindow)
        window.running, window.watermark_window = False, None
        window.root, window.output, window.tasks = Mock(), Mock(), Mock()
        window.output.get.return_value = "downloads"
        window.tasks.selection.return_value = ["2"]
        window.task_rows = {2: {"manifest": str(Path("downloads") / "xiaohongshu" / "123" / "manifest.json")}}
        with patch("buyer_show.watermark_ui.open_window") as open_window:
            window._open_watermark()
        self.assertEqual(open_window.call_args.args[1], Path("downloads/xiaohongshu/123"))

    def test_close_requests_cancellation_and_keeps_ui_until_reaped(self):
        window = WatermarkWindow.__new__(WatermarkWindow)
        window.closed, window.closing = False, False
        window.close_callbacks = []
        window.controller = Mock(busy=True)
        window.window, window.cancel_button, window.status = Mock(), Mock(), Mock()
        callback = Mock()
        window.request_close(callback)
        window.controller.cancel.assert_called_once()
        window.window.destroy.assert_not_called()
        window.controller.busy = False
        window.on_closed = Mock()
        window._finish_close()
        callback.assert_called_once()
        window.on_closed.assert_called_once()

    def test_new_preview_job_after_scan_does_not_get_false_crash_warning(self):
        window = WatermarkWindow.__new__(WatermarkWindow)
        process = Mock(exitcode=0)
        process.is_alive.return_value = False
        window.controller = Mock(busy=True, job={"action": "scan"}, process=process)
        window.controller.drain.side_effect = [[("result", {"paths": []})], []]
        window.closed = window.closing = window.terminal_received = False
        window.window, window.status, window._warn, window._idle = Mock(), Mock(), Mock(), Mock()
        def result_starts_next_worker(job, result):
            window.terminal_received = False
        window._handle_result = Mock(side_effect=result_starts_next_worker)
        window._poll()
        window._warn.assert_not_called()

    def test_detect_keeps_hand_edited_masks_and_rejects_wrong_size(self):
        window = WatermarkWindow.__new__(WatermarkWindow)
        path = "image.png"
        region = {"x": 1, "y": 2, "w": 3, "h": 4, "label": "手动区域"}
        window.items = {path: {"meta": {"width": 100, "height": 100}, "regions": [region]}}
        window._warn, window._sync_regions, window.status = Mock(), Mock(), Mock()
        window._handle_result({"source": path, "action": "detect"},
                              {"width": 101, "height": 100, "regions": []})
        self.assertEqual(window.items[path]["regions"], [region])
        window._handle_result({"source": path, "action": "detect"},
                              {"width": 100, "height": 100, "regions": [{"x": 10, "y": 10, "w": 10, "h": 10}]})
        self.assertEqual(len(window.items[path]["regions"]), 2)

    def test_redetect_replaces_old_auto_candidates_but_preserves_manual_edits(self):
        window = WatermarkWindow.__new__(WatermarkWindow)
        manual = {"x": 1, "y": 2, "w": 3, "h": 4, "_ui_origin": "manual"}
        old_auto = {"x": 20, "y": 20, "w": 10, "h": 10, "_ui_origin": "automatic"}
        path = "image.png"
        window.items = {path: {"meta": {"width": 100, "height": 100}, "regions": [manual, old_auto], "result": {"output": "old.png"}}}
        window.selected_region = 0
        window._warn, window._sync_regions, window.status = Mock(), Mock(), Mock()
        result = {"width": 100, "height": 100, "regions": [{"x": 40, "y": 40, "w": 5, "h": 5}]}
        job = {"source": path, "action": "detect", "merchant_terms": "新品牌", "detection_profile": "product"}
        window._handle_result(job, result)
        regions = window.items[path]["regions"]
        self.assertEqual(len(regions), 2)
        self.assertIs(regions[0], manual)
        self.assertEqual(regions[1]["x"], 40)
        self.assertEqual(window.selected_region, 0)
        self.assertTrue(window.items[path]["result_stale"])
        window._handle_result(job, result)
        self.assertEqual(len(window.items[path]["regions"]), 2)
        window._handle_result(job, {"width": 100, "height": 100, "regions": []})
        self.assertEqual(window.items[path]["regions"], [manual])


if __name__ == "__main__":
    unittest.main()
