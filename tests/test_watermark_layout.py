"""Instantiate real Tk widgets without showing or controlling user windows."""
from pathlib import Path
import hashlib
import json
import tempfile
import time
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from PIL import Image

from buyer_show.watermark_ui import WatermarkWindow, load_task_catalog


def make_manifest(root, colors=("red", "blue")):
    (root / "images").mkdir(parents=True)
    data = {"schema_version": 1, "product_id": "fixture", "complete": False,
            "stats": {"reviews_read": 0}, "reason": "测试文件，不是实际采集验收", "images": {}, "videos": {},
            "records": [], "product_media": []}
    for index, color in enumerate(colors):
        temporary = root / "sample.png"
        Image.new("RGB", (200, 100), color).save(temporary)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        relative = f"images/{digest}.png"
        temporary.replace(root / relative)
        url = f"https://example.invalid/{index}.png"
        data["images"][url] = {"status": "downloaded", "file": relative, "sha256": digest}
        data["product_media"].append({"url": url, "media_type": "image", "source_type": "product_main"})
        # Category copies and processed outputs are not new queue items.
        for folder in ("分类素材/主图", "processed"):
            (root / folder).mkdir(parents=True, exist_ok=True)
            (root / folder / f"{index}.png").write_bytes((root / relative).read_bytes())
    path = root / "manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def pump_until(root, predicate, timeout=20):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        root.update()
        if predicate():
            return
        time.sleep(.02)
    raise AssertionError("Tk / worker flow did not finish in time")


class LayoutSmokeTests(unittest.TestCase):
    def test_batch_scope_ignores_ui_masks_and_filter_navigation_is_lazy(self):
        root = tk.Tk()
        root.withdraw()
        window = None
        try:
            with tempfile.TemporaryDirectory() as directory:
                paths = []
                for index in range(4):
                    path = Path(directory) / f'{index}.png'
                    Image.new('RGB', (200, 100), 'blue').save(path)
                    paths.append(str(path.resolve()))
                window = WatermarkWindow(root, directory)
                window.window.withdraw()
                window._add_files([{'path': path, 'media_type': 'image', 'product_key': f'taobao:{index}',
                    'platform': 'taobao', 'product_id': str(index), 'manifest_path': str(Path(directory)/'manifest.json'),
                    'source_types': [["product_main"],["product_detail"],["review_initial"],[]][index]}
                    for index, path in enumerate(paths)], preview_first=False)
                window.items[paths[0]]['regions'] = [{'x': 1, 'y': 2, 'w': 3, 'h': 4, '_ui_origin': 'automatic'}]
                self.assertEqual(len(window._batch_items()), 2)
                with patch.object(window, '_start') as start:
                    window.batch_button.invoke()
                    job = start.call_args.args[0]
                    self.assertEqual(job['action'], 'batch')
                    self.assertEqual(job['image_method'], 'smart')
                    self.assertEqual(len(job['items']), 2)
                    self.assertTrue(all('regions' not in row and 'preview' not in row for row in job['items']))
                window.batch_reviews.set(True)
                self.assertEqual(len(window._batch_items()), 3)
                window.batch_products.set(False)
                self.assertEqual([row['path'] for row in window._batch_items()], [paths[2]])
                with patch.object(window, '_start') as start:
                    window._batch_start()
                    self.assertFalse(start.call_args.args[0]['include_products'])
                    self.assertTrue(start.call_args.args[0]['include_reviews'])
                window.batch_local.set(True)
                self.assertEqual(len(window._batch_items()), 2)
                result = {'batch_dir': directory, 'status': 'partial', 'counts': {'processed': 1, 'needs_review': 1, 'failed': 1},
                    'items': [{'path': paths[0], 'status': 'processed', 'output': paths[0]},
                              {'path': paths[1], 'status': 'needs_review', 'output': paths[1],
                               'partially_processed': True,
                               'proposed_regions': [{'x': 10, 'y': 10, 'w': 20, 'h': 10}]},
                              {'path': paths[2], 'status': 'failed', 'reason': '合成测试失败'}]}
                with patch.object(window, '_refresh_preview') as preview:
                    window._batch_finished(result)
                    self.assertLessEqual(preview.call_count, 1)
                self.assertTrue(all(window.items[path]['preview'] is None for path in paths))
                self.assertEqual(window.items[paths[1]]['state'], '部分处理 · 待确认')
                self.assertEqual(window.items[paths[0]]['regions'][0]['x'], 1)
                window.queue_filter.set('待确认 / 失败')
                with patch.object(window, '_refresh_preview') as preview:
                    window._change_filter()
                    self.assertEqual(window.visible_files, paths[1:3])
                    self.assertEqual(window.current, paths[1])
                    window._navigate(1)
                    self.assertEqual(window.current, paths[2])
                    window._navigate(-1)
                    self.assertEqual(window.current, paths[1])
                window.items[paths[1]]['meta'] = {'width': 200, 'height': 100}
                window._batch_candidates(window.items[paths[1]])
                self.assertEqual(window.items[paths[1]]['regions'][0]['x'], 10)
                # Detection can reach N/N before repairs start; the progress
                # bar is driven by terminal counts and must not jump backwards.
                window._batch_update({'stage': 'detect', 'index': 4, 'total': 4, 'counts': {'unchanged': 0}})
                self.assertEqual(float(window.batch_progress['value']), 0)
                self.assertIn('检测 4 / 4', window.batch_summary.get())
                window._batch_update({'stage': 'repair', 'index': 1, 'total': 4, 'counts': {'unchanged': 1}})
                self.assertEqual(float(window.batch_progress['value']), 1)
                self.assertIn('处理 1 / 4', window.batch_summary.get())
                window.request_close()
        finally:
            if window is not None and not window.closed:
                window.request_close()
            root.destroy()

    def test_catalog_keeps_identical_bytes_separate_for_different_products(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = make_manifest(root/'one', ('red',)), make_manifest(root/'two', ('red',))
            for path, product_id in ((first, '123'), (second, '456')):
                data = json.loads(path.read_text(encoding='utf-8'))
                data.update(platform='taobao', product_id=product_id)
                path.write_text(json.dumps(data), encoding='utf-8')
            items = load_task_catalog(root, [str(first), str(second)])['items']
            self.assertEqual(len(items), 2)
            self.assertEqual({row['product_key'] for row in items}, {'taobao:123', 'taobao:456'})
            self.assertEqual(items[0]['sha256'], items[1]['sha256'])
            self.assertTrue(all(Path(row['manifest_path']).is_file() for row in items))

    def test_open_and_select_ai_only_read_status_without_download_or_loading(self):
        from buyer_show import watermark_ai
        root = tk.Tk()
        root.withdraw()
        window = None
        try:
            with patch.object(watermark_ai, 'ensure_model') as download, \
                 patch.object(watermark_ai, '_get_session') as load, \
                 patch.object(watermark_ai, 'model_status', return_value={'installed': False, 'size_bytes': 0}):
                window = WatermarkWindow(root, Path('.'))
                window.window.withdraw()
                root.update_idletasks()
                self.assertEqual(window.repair_method.get(), '自动（按背景选择）')
                self.assertIn('AI纹理修复（CPU）', window.method_combo['values'])
                self.assertIn('未下载', window.model_notice.get())
                window.repair_method.set('AI纹理修复（CPU）')
                self.assertIn('208 MB', window.repair_limit.get())
                self.assertIn('CPU', window.repair_limit.get())
                self.assertIn('取消会等待', window.repair_limit.get())
                self.assertFalse(window.busy)
                download.assert_not_called()
                load.assert_not_called()
                window.request_close()
        finally:
            if window is not None and not window.closed:
                window.request_close()
            root.destroy()

    def test_delete_or_clear_selected_region_then_switch_files_never_restores_stale_index(self):
        root = tk.Tk()
        root.withdraw()
        window = None
        try:
            with tempfile.TemporaryDirectory() as directory:
                manifest = make_manifest(Path(directory) / "task")
                window = WatermarkWindow(root, manifest.parent, manifests=[str(manifest)], autoload=True)
                window.window.withdraw()
                pump_until(root, lambda: len(window.files) == 2 and not window.busy and bool(window._item()["preview"]))
                first = window.current
                for action in ("delete", "clear"):
                    with self.subTest(action=action):
                        window.items[first]["regions"] = [
                            {"x": 5, "y": 5, "w": 10, "h": 10},
                            {"x": 30, "y": 30, "w": 10, "h": 10}]
                        window.selected_region = 1
                        window._sync_regions()
                        window._navigate(1)
                        pump_until(root, lambda: not window.busy)
                        window._navigate(-1)
                        self.assertEqual(window.selected_region, 1)
                        (window._delete_region if action == "delete" else window._clear_regions)()
                        self.assertIsNone(window.items[first]["selected_region"])
                        window._navigate(1)
                        window._navigate(-1)
                        self.assertIsNone(window.selected_region)
                        with patch.object(window, "_start") as start:
                            window._detect()
                            self.assertEqual(start.call_args.args[0]["action"], "detect")
                            if action == "delete":
                                window._repair()
                                self.assertEqual(start.call_args.args[0]["action"], "repair")
                        window._navigate(1)
                        window._navigate(-1)
                        # Old in-memory states are also sanitized on restore.
                        window.items[first]["selected_region"] = 9
                        window.selected_region = 9
                        self.assertTrue(window._apply_region())
                        self.assertIsNone(window.selected_region)
                window.request_close()
        finally:
            if window is not None and not window.closed:
                window.request_close()
                pump_until(root, lambda: window.closed)
            root.destroy()

    def test_merchant_settings_and_repair_method_jobs_from_real_widgets(self):
        root = tk.Tk()
        root.withdraw()
        window = None
        try:
            window = WatermarkWindow(root, Path("."))
            window.window.withdraw()
            root.update_idletasks()
            path = "fixture.png"
            window.current = path
            window.items[path] = {"meta": {"width": 200, "height": 100}, "regions": [], "detect_attempted": False}
            self.assertEqual(window.detection_profile.get(), "商家水印（商品）")
            self.assertFalse(window.auto_detect.get())
            window.merchant_terms.set("示例店铺,示例品牌")
            with patch.object(window, "_start") as start:
                window._detect()
                job = start.call_args.args[0]
                self.assertEqual(job["detection_profile"], "product")
                self.assertEqual(job["merchant_terms"], "示例店铺,示例品牌")
                self.assertTrue(job["include_corner_text"])
            window.items[path]["detect_attempted"] = True
            window.items[path]["detect_options"] = window._detection_options()
            window.detection_profile.set("平台水印")
            self.assertFalse(window.items[path]["detect_attempted"])
            window.items[path]["regions"] = [{"x": 5, "y": 6, "w": 20, "h": 10, "_ui_origin": "automatic"}]
            window.selected_region = 0
            window._sync_regions()
            window.edits["x"].set("7")
            self.assertTrue(window._apply_region())
            self.assertEqual(window.items[path]["regions"][0]["_ui_origin"], "manual")
            window.repair_method.set("NS")
            with patch.object(window, "_start") as start:
                window._repair()
                self.assertEqual(start.call_args.args[0]["method"], "ns")
                self.assertNotIn("_ui_origin", start.call_args.args[0]["regions"][0])
            window.repair_method.set("AI纹理修复（CPU）")
            with patch.object(window, "_start") as start:
                window._repair()
                self.assertEqual(start.call_args.args[0]["method"], "lama")
                self.assertEqual(start.call_args.args[0]["source"], path)
            # Editing the seconds input without refreshing must not compare a
            # new output frame against the old, still displayed source frame.
            video_path = "fixture.mp4"
            window.current = video_path
            window.selected_region = None
            window.items[video_path] = {"meta": {"width": 200, "height": 100, "duration": 10.0},
                "regions": [{"x": 0, "y": 0, "w": 20, "h": 10, "start": 0, "end": 10}],
                "preview_seconds": 1.25}
            window.seconds.set("6.5")
            window._show_dimensions()
            self.assertEqual(str(window.method_combo['state']), 'disabled')
            self.assertIn('视频使用', window.model_notice.get())
            self.assertNotIn('AI 图片修复', window.repair_limit.get())
            with patch.object(window, "_start") as start:
                window._repair()
                self.assertEqual(start.call_args.args[0]["seconds"], 1.25)
                self.assertNotIn("method", start.call_args.args[0])
                self.assertEqual(start.call_args.args[0]["regions"][0]["x"], 0)
                self.assertEqual(start.call_args.args[0]["regions"][0]["y"], 0)
            self.assertEqual(window.seconds.get(), "1.25")
            self.assertIn("当前显示的 1.25 秒", window.warning.get())
            window.request_close()
        finally:
            if window is not None and not window.closed:
                window.request_close()
            root.destroy()

    def test_actual_size_scroll_sync_and_long_image_drawing_preserve_original_coordinates(self):
        root = tk.Tk()
        root.withdraw()
        window = None
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "long-detail.png"
                Image.new("RGB", (900, 6000), "#abcdee").save(path)
                window = WatermarkWindow(root, Path(directory))
                window.window.geometry("1220x700+10000+10000")
                root.update()
                window._add_files([str(path)])
                pump_until(root, lambda: not window.busy and bool(window._item()["preview"]))
                window.zoom.set("100%")
                window._change_zoom()
                pump_until(root, lambda: not window.busy and "preview" in window.full_images)
                window._item()["after"] = window._item()["preview"]
                window.full_images["after"] = window.full_images["preview"].copy()
                for fraction in (0, .5, 1):
                    window._scroll_preview("y", "moveto", fraction)
                    window._scroll_preview("x", "moveto", fraction)
                    self.assertEqual(window.before_canvas.canvasx(0), window.after_canvas.canvasx(0))
                    self.assertEqual(window.before_canvas.canvasy(0), window.after_canvas.canvasy(0))
                window._scroll_preview("y", "moveto", .7)
                window._scroll_preview("x", "moveto", .2)
                root.update()
                self.assertEqual(window.transform.scale, 1)
                self.assertEqual(window.before_canvas.canvasx(0), window.after_canvas.canvasx(0))
                self.assertEqual(window.before_canvas.canvasy(0), window.after_canvas.canvasy(0))
                x, y = window.before_canvas.canvasx(20), window.before_canvas.canvasy(30)
                self.assertGreater(y, 4000)
                window._press(SimpleNamespace(x=20, y=30))
                window._motion(SimpleNamespace(x=80, y=90))
                window._release(SimpleNamespace(x=80, y=90))
                region = window._item()["regions"][0]
                self.assertEqual((region["x"], region["y"], region["w"], region["h"]), (int(x), int(y), 60, 60))
                self.assertEqual(region["_ui_origin"], "manual")
                self.assertLessEqual(window.photos["preview"].height(), window.before_canvas.winfo_height() + 1)
                self.assertLessEqual(window.photos["preview"].width(), window.before_canvas.winfo_width() + 1)
                previous_y = window.before_canvas.canvasy(0)
                window.zoom.set("适应窗口")
                window._change_zoom()
                window.zoom.set("100%")
                window._change_zoom()
                self.assertEqual(window.before_canvas.canvasy(0), previous_y)
                window.request_close()
                self.assertEqual(window.full_images, {})
        finally:
            if window is not None and not window.closed:
                window.request_close()
                pump_until(root, lambda: window.closed)
            root.destroy()

    def test_controls_and_status_fit_small_desktop_without_covering_preview(self):
        root = tk.Tk()
        root.withdraw()
        try:
            window = WatermarkWindow(root, Path("."))
            # Map our own test window outside the desktop to exercise real Tk
            # geometry without touching the user's foreground application.
            window.window.geometry("1220x700+10000+10000")
            for method in ('自动（按背景选择）', 'AI纹理修复（CPU）'):
                window.repair_method.set(method)
                root.update()
                for widget in (window.repair_button, window.previous_button, window.next_button,
                               window.model_label, window.status_label):
                    top = widget.winfo_rooty() - window.window.winfo_rooty()
                    self.assertGreaterEqual(top, 0)
                    self.assertLessEqual(top + widget.winfo_height(), 700)
                    self.assertTrue(widget.winfo_ismapped())
                self.assertGreaterEqual(window.before_canvas.winfo_width(), 250)
                self.assertGreaterEqual(window.before_canvas.winfo_height(), 250)
            window.request_close()
        finally:
            root.destroy()

    def test_real_widgets_construct_and_close_without_worker(self):
        root = tk.Tk()
        root.withdraw()
        try:
            with tempfile.TemporaryDirectory() as directory:
                window = WatermarkWindow(root, Path(directory))
                window.window.withdraw()
                root.update_idletasks()
                self.assertGreater(window.before_canvas.winfo_reqwidth(), 0)
                self.assertFalse(window.busy)
                self.assertEqual(window.files, [])
                window.request_close()
                self.assertTrue(window.closed)
        finally:
            root.destroy()

    def test_catalog_deduplicates_originals_across_manifests_and_latest_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = make_manifest(root / "one")
            second = make_manifest(root / "two", ("red", "green"))
            (root / "batch-latest.json").write_text(json.dumps([{"manifest": str(first)}, {"manifest": str(second)}]), encoding="utf-8")
            catalog = load_task_catalog(root)
            self.assertEqual(len(catalog["items"]), 3)
            self.assertEqual(len(catalog["summaries"]), 2)
            self.assertTrue(all(Path(item["path"]).parent.name == "images" for item in catalog["items"]))

    def test_real_tk_autoload_spawn_preview_navigation_edit_and_shortcuts(self):
        root = tk.Tk()
        root.withdraw()
        window = None
        try:
            with tempfile.TemporaryDirectory() as directory:
                manifest = make_manifest(Path(directory) / "task")
                window = WatermarkWindow(root, manifest.parent, manifests=[str(manifest)], autoload=True)
                window.window.withdraw()
                self.assertFalse(window.auto_detect.get())
                pump_until(root, lambda: len(window.files) == 2 and not window.busy and bool(window._item()["preview"]))
                first = window.current
                self.assertIn("第 1 / 2", window.counter.get())
                self.assertIn("图片", window.file_list.get(0))
                self.assertIn("未完成", window.context_notice.get())
                window._item()["regions"] = [{"x": 2, "y": 3, "w": 10, "h": 10, "label": "手动"}]
                window.selected_region = 0
                window._sync_regions()
                window.edits["x"].set("6")
                window.next_button.invoke()
                self.assertTrue(window.busy)
                pending_target = window.current
                window._navigate(-1)
                self.assertEqual(window.current, pending_target)
                pump_until(root, lambda: not window.busy and window.current != first and bool(window._item()["preview"]))
                self.assertEqual(window.items[first]["regions"][0]["x"], 6)
                window.previous_button.invoke()
                root.update()
                self.assertEqual(window.current, first)
                self.assertEqual(window.edits["x"].get(), "6")
                editor = window.time_entry
                self.assertIsNone(window._shortcut(SimpleNamespace(widget=editor, keysym="Right", state=0)))
                self.assertEqual(window.current, first)
                self.assertEqual(window._shortcut(SimpleNamespace(widget=window.before_canvas, keysym="d", state=0)), "break")
                self.assertNotEqual(window.current, first)
                window._navigate(-1)
                window._shortcut(SimpleNamespace(widget=window.before_canvas, keysym="Delete", state=0))
                self.assertEqual(window.items[first]["regions"], [])
                window.request_close()
                pump_until(root, lambda: window.closed)
        finally:
            if window is not None and not window.closed:
                window.request_close()
                pump_until(root, lambda: window.closed)
            root.destroy()


if __name__ == '__main__':
    unittest.main()
