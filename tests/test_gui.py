"""Media reports, event handling and a hidden real-Tk stage display; no browser."""
import queue
import sys
import tempfile
import threading
import tkinter as tk
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

from buyer_show.gui import ValidatorWindow, _media_counts


class GuiMediaTests(unittest.TestCase):
    def test_real_hidden_tk_shows_pipeline_stages_with_distinct_saved_counts(self):
        from buyer_show.links import classify_url
        root = tk.Tk()
        root.withdraw()
        try:
            with tempfile.TemporaryDirectory() as folder:
                window = ValidatorWindow(root, settings={"output": folder, "channel": "msedge", "max_scrolls": 40},
                                         profiles=Path(folder) / "profiles")
                window._initialize_tasks([classify_url("https://item.taobao.com/item.htm?id=123")])
                window.tasks.selection_set("0")
                self.assertEqual(window.tasks.heading("saved", "text"), "已存图片")
                for stage, label, saved in (("read_reviews", "读取评价", 0),
                                            ("download_reviews", "下载买家秀", 0),
                                            ("organize_categories", "整理分类", 45),
                                            ("read_product", "读取商品素材", 45),
                                            ("download_product", "下载商品素材", 45)):
                    window._task_event({"type": "task", "index": 0, "status": "running", "stage": stage,
                                        "stats": {"image_urls": 45, "video_urls": 1,
                                                  "unique_images": saved, "unique_videos": int(saved > 0)}})
                    root.update_idletasks()
                    values = window.tasks.item("0", "values")
                    self.assertEqual(values[2], label)
                    self.assertEqual(int(values[3]), saved)
                    self.assertIn(label, window.status.get())
                    self.assertIn("图片 45，视频 1（不是保存数量）", window.task_detail.get())
                    self.assertIn(f"已保存：图片 {saved}", window.task_detail.get())
        finally:
            root.destroy()

    def test_live_stage_survives_worker_queue_and_distinguishes_found_from_saved(self):
        window = ValidatorWindow.__new__(ValidatorWindow)
        window.events = queue.Queue()
        window.tasks, window.task_detail, window.status = Mock(), Mock(), Mock()
        window.tasks.selection.return_value = ("0",)
        window.stop_event = threading.Event()
        window.task_rows = {0: {"platform": "taobao", "product_id": "123", "status": "running", "stats": {}}}
        event = {"type": "task", "index": 0, "status": "running", "stage": "download_reviews",
                 "stats": {"image_urls": 45, "video_urls": 1, "reviews_read": 62, "unique_images": 0, "unique_videos": 0},
                 "download_failure_summary": "未保存：视频 1 个"}
        window._queue_task_event(event)
        kind, copied = window.events.get_nowait()
        self.assertEqual(copied["stage"], "download_reviews")
        self.assertEqual(copied["download_failure_summary"], "未保存：视频 1 个")
        window._task_event(copied)
        values = window.tasks.item.call_args.kwargs["values"]
        self.assertEqual(values[2:5], ("下载买家秀", 0, 0))
        self.assertIn("图片 45，视频 1（不是保存数量）", window.status.set.call_args.args[0])
        self.assertIn("已保存：图片 0，视频 0", window.status.set.call_args.args[0])
        event.update(stage="read_product", stats={**event["stats"], "unique_images": 45, "unique_videos": 1})
        window._task_event(event)
        self.assertEqual(window.tasks.item.call_args.kwargs["values"][2:5], ("读取商品素材", 45, 1))
        # A stale stage must never override a terminal/blocked status.
        window._task_event({"type": "task", "index": 0, "status": "partial"})
        self.assertEqual(window.tasks.item.call_args.kwargs["values"][2], "未完成")

    def test_legacy_image_report_keeps_counts_and_defaults_video_to_zero(self):
        values = ValidatorWindow._row_values({
            "platform": "taobao", "product_id": "123", "status": "partial",
            "stats": {"unique_images": 7, "downloads_failed": 2},
        })
        self.assertEqual(values, ("淘宝 / 天猫", "123", "未完成", 7, 0, 2))

    def test_video_failures_are_added_without_counting_successful_urls_as_unique_files(self):
        values = ValidatorWindow._row_values({
            "platform": "douyin", "product_id": "456", "status": "partial_download",
            "stats": {"unique_images": 3, "unique_videos": 2, "video_urls": 9,
                      "video_downloads_succeeded": 6, "downloads_failed": 1, "videos_failed": 3},
        })
        self.assertEqual(values[3:], (3, 2, 4))

    def test_absent_or_invalid_video_stats_do_not_break_old_reports(self):
        self.assertEqual(_media_counts(None)["videos"], 0)
        for value in (-1, "5", True, None):
            with self.subTest(value=value):
                self.assertEqual(_media_counts({"unique_videos": value})["videos"], 0)

    def test_worker_event_copies_video_stats_before_queueing(self):
        window = ValidatorWindow.__new__(ValidatorWindow)
        window.events = queue.Queue()
        event = {"type": "task", "index": 0, "stats": {"unique_videos": 2, "videos_failed": 1}}
        window._queue_task_event(event)
        event["stats"]["unique_videos"] = 99
        kind, payload = window.events.get_nowait()
        self.assertEqual(kind, "task")
        self.assertEqual(payload["stats"], {"unique_videos": 2, "videos_failed": 1})

    def test_worker_summary_accepts_mixed_current_and_legacy_reports(self):
        window = ValidatorWindow.__new__(ValidatorWindow)
        window.events = queue.Queue()
        window.stop_event = threading.Event()
        window.pending_lock = threading.Lock()
        window.pending = None
        window.profiles = Path("unused-profiles")
        reports = [
            {"complete": True, "stats": {"unique_images": 4, "unique_videos": 2}},
            {"complete": False, "stats": {"unique_images": 1, "downloads_failed": 1}},
        ]

        async def fake_batch(*args, **kwargs):
            return reports

        engine = ModuleType("buyer_show.engine")
        engine.run_batch = fake_batch
        with patch.dict(sys.modules, {"buyer_show.engine": engine}):
            window._run_worker([], Path("unused-output"), 40)
        events = list(window.events.queue)
        summary = next(payload for kind, payload in events if kind == "done")
        self.assertEqual(summary, {"failed": False, "jobs": 2, "completed": 1,
                                   "images": 5, "videos": 2, "downloads_failed": 1})

    def test_finished_status_shows_both_media_types_even_for_old_summary(self):
        window = ValidatorWindow.__new__(ValidatorWindow)
        window.events = queue.Queue()
        window.stop_event = threading.Event()
        window.running = True
        window.done_received = True
        window.worker = Mock()
        window.worker.is_alive.return_value = False
        window.last_summary = {"failed": False, "completed": 1}
        window.task_rows = {0: {"stats": {"unique_images": 8, "unique_videos": 2, "videos_failed": 1}}}
        window._set_idle = Mock()
        window._finish_pending_rows = Mock()
        window.status = Mock()
        window.prompt = Mock()
        window.root = Mock()
        window.close_requested = False
        window._poll()
        message = window.status.set.call_args.args[0]
        self.assertIn("图片 8，视频 2", message)
        self.assertIn("下载失败 1", message)
        self.assertIn("1/1", message)
        self.assertFalse(window.running)

    def test_task_details_keep_review_and_product_media_categories_visible(self):
        window = ValidatorWindow.__new__(ValidatorWindow)
        window.tasks = Mock()
        window.tasks.selection.return_value = ("0",)
        window.task_detail = Mock()
        window.task_rows = {0: {"reason": "评价入口尚未支持", "stats": {
            "unique_images": 5, "unique_videos": 1, "reviews_read": 0, "product_media_count": 6,
        }}}
        window._select_task()
        detail = window.task_detail.set.call_args.args[0]
        self.assertIn("已读取评价 0，商品素材记录 6", detail)
        self.assertIn("评价入口尚未支持", detail)
        self.assertIn("来源见清单", detail)

    def test_download_failure_reason_stays_visible_and_clears_after_successful_retry(self):
        window = ValidatorWindow.__new__(ValidatorWindow)
        window.tasks = Mock()
        window.tasks.selection.return_value = ("0",)
        window.task_detail = Mock()
        window.task_rows = {0: {"status": "partial", "reason": "评价状态" * 100, "stats": {}}}
        window._task_event({"type": "task", "index": 0,
                            "download_failure_summary": "未保存：初评视频 1 个（HTTP 490）"})
        self.assertIn("初评视频 1 个（HTTP 490）", window.task_detail.set.call_args.args[0])
        window._task_event({"type": "task", "index": 0, "download_failure_summary": ""})
        self.assertNotIn("HTTP 490", window.task_detail.set.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
