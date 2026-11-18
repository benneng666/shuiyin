"""Integration using a fake browser and synthetic bytes; never opens a real browser."""
import asyncio
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from PIL import Image

from buyer_show.engine import (_cancellable_collect, run_batch, _ensure_live_page,
                               _target_closed)
from buyer_show.links import classify_url


class FakePage:
    def __init__(self):
        self.url = "about:blank"
        self.closed = False
        self.visits = []

    async def goto(self, url, **kwargs):
        self.visits.append(url)
        self.url = "https://www.douyin.com/video/123" if "v.douyin.com" in url else url

    async def wait_for_timeout(self, timeout):
        pass

    async def close(self):
        self.closed = True

    def is_closed(self):
        return self.closed


class FakeBrowser:
    def __init__(self):
        self.pages = []

    async def new_page(self):
        page = FakePage()
        self.pages.append(page)
        return page

    async def close(self):
        pass


class FakePlaywright:
    def __init__(self):
        self.browser = FakeBrowser()
        self.chromium = SimpleNamespace(launch_persistent_context=AsyncMock(return_value=self.browser))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class EngineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="buyer_show_engine_")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.fake = FakePlaywright()
        patcher = patch("playwright.async_api.async_playwright", return_value=self.fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.product_collector = AsyncMock(return_value={"status": "completed", "complete": True,
                                                         "media": [], "reason": "测试商品素材终点"})
        product_patch = patch("buyer_show.engine.collect_product_media", self.product_collector)
        product_patch.start()
        self.addCleanup(product_patch.stop)

    async def test_short_video_does_not_reach_product_adapter(self):
        collector = AsyncMock()
        with patch("buyer_show.engine.ADAPTERS", {"douyin": SimpleNamespace(collect=collector)}):
            result = await run_batch([classify_url("https://v.douyin.com/fixture/")], self.root / "out", self.root / "profiles", emit=lambda _: None)
        collector.assert_not_awaited()
        self.assertEqual(result[0]["status"], "unsupported_link")
        self.assertFalse(result[0]["complete"])
        self.assertEqual(len(result), 1)

    async def _run_short_link(self):
        with patch("buyer_show.engine.ADAPTERS", {"douyin": SimpleNamespace(collect=AsyncMock())}):
            return await run_batch([classify_url("https://v.douyin.com/fixture/")],
                                   self.root / "out", self.root / "profiles", emit=lambda _: None)

    async def test_default_launch_keeps_machine_proxy_settings(self):
        with patch.dict(os.environ, {"BUYER_SHOW_DIRECT_CONNECTION": "0"}), \
             patch("buyer_show.engine.httpx.AsyncClient") as client_cls:
            await self._run_short_link()
        launch = self.fake.chromium.launch_persistent_context.await_args.kwargs
        self.assertNotIn("args", launch)
        client_cls.assert_called_once_with(timeout=30)

    async def test_direct_connection_bypasses_dead_local_proxy(self):
        with patch.dict(os.environ, {"BUYER_SHOW_DIRECT_CONNECTION": "1"}), \
             patch("buyer_show.engine.httpx.AsyncClient") as client_cls:
            await self._run_short_link()
        launch = self.fake.chromium.launch_persistent_context.await_args.kwargs
        self.assertEqual(launch["args"], ["--no-proxy-server"])
        client_cls.assert_called_once_with(timeout=30, trust_env=False)

    async def test_full_core_pipeline_preserves_append_and_photos(self):
        url = "https://item.taobao.com/item.htm?id=123"
        record = {"platform": "taobao", "product_id": "123", "review_id": "r1", "content": "测试",
                  "review_type": "append", "images": ["https://img.alicdn.com/test.png"]}
        collector = AsyncMock(return_value={"status": "completed", "complete": True, "records": [record], "reason": "测试末页"})
        img = io.BytesIO()
        Image.new("RGB", (20, 30), "navy").save(img, "PNG")
        with patch("buyer_show.engine.ADAPTERS", {"taobao": SimpleNamespace(collect=collector)}), patch("buyer_show.engine.ImageFetcher", return_value=AsyncMock(return_value=img.getvalue())):
            result = await run_batch([classify_url(url)], self.root / "out", self.root / "profiles", emit=lambda _: None)
        self.assertTrue(result[0]["complete"])
        self.assertEqual(result[0]["stats"]["unique_images"], 1)
        self.assertTrue(Path(result[0]["manifest"]).is_file())
        self.assertTrue(self.fake.browser.pages[0].closed)

    async def test_manual_login_retry_requires_human_continue(self):
        url = "https://item.taobao.com/item.htm?id=123"
        blocker = {"status": "login_required", "complete": False, "records": [], "reason": "请登录"}
        partial = {"status": "partial", "complete": False, "records": [], "reason": "没有末页"}
        collector = AsyncMock(side_effect=[blocker, partial])
        ask = AsyncMock(return_value=True)
        events = []
        with patch("buyer_show.engine.ADAPTERS", {"taobao": SimpleNamespace(collect=collector)}):
            result = await run_batch([classify_url(url)], self.root / "out", self.root / "profiles", emit=lambda _: None, ask=ask, on_event=events.append)
        ask.assert_awaited_once()
        self.assertEqual(collector.await_count, 2)
        self.assertFalse(result[0]["complete"])
        self.assertIn("login.taobao.com", self.fake.browser.pages[0].visits[0])
        statuses = [event["status"] for event in events]
        self.assertEqual([value for i, value in enumerate(statuses) if not i or value != statuses[i-1]],
                         ["running", "waiting_user", "running", "partial"])
        self.assertTrue(all(event["type"] == "task" and event["index"] == 0 for event in events))
        self.assertEqual(events[-1]["stats"], result[0]["stats"])

    async def test_declining_manual_login_keeps_blocked_status(self):
        collector = AsyncMock(return_value={"status": "login_required", "complete": False, "records": [], "reason": "请登录"})
        with patch("buyer_show.engine.ADAPTERS", {"taobao": SimpleNamespace(collect=collector)}):
            result = await run_batch([classify_url("https://item.taobao.com/item.htm?id=123")], self.root / "out", self.root / "profiles", emit=lambda _: None, ask=AsyncMock(return_value=False))
        self.assertEqual(collector.await_count, 1)
        self.assertEqual(result[0]["status"], "login_required")

    async def test_product_images_saved_when_reviews_are_unsupported(self):
        import json
        collector = AsyncMock(return_value={"status": "unsupported_review_entry", "complete": False,
                                             "records": [], "reason": "评价未支持"})
        image_url = "https://img.alicdn.com/product-main.png"
        self.product_collector.return_value = {"status": "partial", "complete": False,
            "media": [{"url": image_url, "media_type": "image", "source_type": "product_main"}],
            "reason": "已取得轮播图；详情尚未完整"}
        content = io.BytesIO()
        Image.new("RGB", (30, 40), "teal").save(content, "PNG")
        with patch("buyer_show.engine.ADAPTERS", {"taobao": SimpleNamespace(collect=collector)}), \
             patch("buyer_show.engine.ImageFetcher", return_value=AsyncMock(return_value=content.getvalue())):
            reports = await run_batch([classify_url("https://detail.tmall.com/item.htm?id=123")],
                                      self.root / "out", self.root / "profiles", emit=lambda _: None)
        report = reports[0]
        self.assertFalse(report["complete"])
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["stats"]["unique_images"], 1)
        self.assertEqual(report["stats"]["reviews_read"], 0)
        manifest = json.loads(Path(report["manifest"]).read_text(encoding="utf-8"))
        self.assertFalse(manifest["scopes"]["reviews"]["complete"])
        self.assertEqual(manifest["product_media"][0]["source_type"], "product_main")

    async def test_product_login_can_be_handled_without_repeating_review_prompt(self):
        collector = AsyncMock(return_value={"status": "partial", "complete": False,
                                             "records": [], "reason": "评价未到底"})
        self.product_collector.side_effect = [
            {"status": "login_required", "complete": False, "media": [], "reason": "商品素材要求登录"},
            {"status": "completed", "complete": True, "media": [], "reason": "商品素材已加载"},
        ]
        ask = AsyncMock(return_value=True)
        with patch("buyer_show.engine.ADAPTERS", {"taobao": SimpleNamespace(collect=collector)}):
            reports = await run_batch([classify_url("https://detail.tmall.com/item.htm?id=123")],
                                      self.root / "out", self.root / "profiles", emit=lambda _: None, ask=ask)
        ask.assert_awaited_once()
        self.assertEqual(self.product_collector.await_count, 2)
        self.assertFalse(reports[0]["complete"])
        self.assertEqual(reports[0]["scopes"]["product"]["status"], "completed")

    async def test_manual_retry_retains_pre_login_records(self):
        row = {"platform": "taobao", "product_id": "123", "review_id": "before",
               "review_type": "initial", "content": "第一批", "images": []}
        after = dict(row, review_id="after", content="第二批")
        collector = AsyncMock(side_effect=[
            {"status": "login_required", "complete": False, "records": [row], "reason": "会话过期"},
            {"status": "completed", "complete": True, "records": [after], "reason": "末页"}])
        with patch("buyer_show.engine.ADAPTERS", {"taobao": SimpleNamespace(collect=collector)}):
            result = await run_batch([classify_url("https://item.taobao.com/item.htm?id=123")], self.root / "out", self.root / "profiles", emit=lambda _: None, ask=AsyncMock(return_value=True))
        self.assertEqual(result[0]["stats"]["reviews_read"], 2)

    async def test_short_link_login_can_continue_to_product(self):
        page = FakePage()
        short_visits = []

        async def goto(url, **kwargs):
            if url == "https://m.tb.cn/fixture":
                short_visits.append(True)
                page.url = "https://login.taobao.com/member/login.jhtml" if len(short_visits) == 1 else "https://item.taobao.com/item.htm?id=123"
            else:
                page.url = url

        page.goto = goto
        self.fake.browser.new_page = AsyncMock(return_value=page)
        collector = AsyncMock(return_value={"status": "partial", "complete": False, "records": [], "reason": "未到末页"})
        ask = AsyncMock(return_value=True)
        with patch("buyer_show.engine.ADAPTERS", {"taobao": SimpleNamespace(collect=collector)}):
            result = await run_batch([classify_url("https://m.tb.cn/fixture")], self.root / "out", self.root / "profiles", emit=lambda _: None, ask=ask)
        ask.assert_awaited_once()
        self.assertEqual(result[0]["product_id"], "123")
        self.assertEqual(len(short_visits), 2)

    async def test_advertising_link_is_resolved_to_tmall_before_collection(self):
        ad = "https://click.simba.taobao.com/cc_im?s=123&e=test%2Bsig"
        page = FakePage()

        async def goto(url, **kwargs):
            page.url = "https://detail.tmall.com/item.htm?id=950216355977" if url == ad else url

        page.goto = goto
        self.fake.browser.new_page = AsyncMock(return_value=page)
        collector = AsyncMock(return_value={"status": "partial", "complete": False,
                                             "records": [], "reason": "待实页验证"})
        with patch("buyer_show.engine.ADAPTERS", {"taobao": SimpleNamespace(collect=collector)}):
            reports = await run_batch([classify_url(ad)], self.root / "out", self.root / "profiles", emit=lambda _: None)
        self.assertEqual(reports[0]["product_id"], "950216355977", reports[0])
        self.assertEqual(collector.call_args.args[1], "https://detail.tmall.com/item.htm?id=950216355977")

    async def test_short_link_and_review_login_prompt_at_most_once_per_product(self):
        page = FakePage()
        visits = []

        async def goto(url, **kwargs):
            if url == "https://m.tb.cn/fixture":
                visits.append(url)
                page.url = ("https://login.taobao.com/member/login.jhtml" if len(visits) == 1
                            else "https://detail.tmall.com/item.htm?id=123")
            else:
                page.url = url

        page.goto = goto
        self.fake.browser.new_page = AsyncMock(return_value=page)
        collector = AsyncMock(return_value={"status": "login_required", "complete": False,
                                             "records": [], "reason": "评价登录仍未确认"})
        self.product_collector.return_value = {"status": "login_required", "complete": False,
                                              "media": [], "reason": "商品登录仍未确认"}
        ask = AsyncMock(return_value=True)
        with patch("buyer_show.engine.ADAPTERS", {"taobao": SimpleNamespace(collect=collector)}):
            reports = await run_batch([classify_url("https://m.tb.cn/fixture")],
                                      self.root / "out", self.root / "profiles", emit=lambda _: None, ask=ask)
        ask.assert_awaited_once()
        self.assertEqual(reports[0]["status"], "login_required")
        self.assertFalse(reports[0]["complete"])

    async def test_advertising_link_wrong_destination_cannot_be_collected(self):
        page = FakePage()

        async def goto(url, **kwargs):
            page.url = "https://www.douyin.com/video/123"

        page.goto = goto
        self.fake.browser.new_page = AsyncMock(return_value=page)
        collector = AsyncMock()
        with patch("buyer_show.engine.ADAPTERS", {"taobao": SimpleNamespace(collect=collector)}):
            reports = await run_batch([classify_url("https://click.simba.taobao.com/cc_im?e=test")],
                                      self.root / "out", self.root / "profiles", emit=lambda _: None)
        collector.assert_not_awaited()
        self.product_collector.assert_not_awaited()
        self.assertEqual(reports[0]["status"], "unsupported_link")

    async def test_user_closed_browser_does_not_lose_batch_report(self):
        page = FakePage()
        page.close = AsyncMock(side_effect=RuntimeError("closed"))
        self.fake.browser.new_page = AsyncMock(return_value=page)
        self.fake.browser.close = AsyncMock(side_effect=RuntimeError("closed"))
        collector = AsyncMock(return_value={"status": "partial", "complete": False, "records": [], "reason": "未到末页"})
        with patch("buyer_show.engine.ADAPTERS", {"taobao": SimpleNamespace(collect=collector)}):
            result = await run_batch([classify_url("https://item.taobao.com/item.htm?id=123")], self.root / "out", self.root / "profiles", emit=lambda _: None)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["status"], "partial")
        self.assertEqual(len(list((self.root / "out").glob("batch-*.json"))), 1)

    async def test_collector_error_finishes_manifest_as_error(self):
        import json
        collector = AsyncMock(side_effect=RuntimeError("closed page"))
        with patch("buyer_show.engine.ADAPTERS", {"taobao": SimpleNamespace(collect=collector)}):
            result = await run_batch([classify_url("https://item.taobao.com/item.htm?id=123")], self.root / "out", self.root / "profiles", emit=lambda _: None)
        manifest = json.loads(Path(result[0]["manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "error")
        self.assertFalse(manifest["complete"])

    async def test_reviews_are_downloaded_and_classified_before_product_error(self):
        image_url = "https://img.alicdn.com/early.png"
        record = {"platform": "taobao", "product_id": "123", "review_id": "r1",
                  "review_type": "initial", "content": "合成测试", "images": [image_url]}
        collector = AsyncMock(return_value={"status": "completed", "complete": True, "records": [record], "reason": "测试评价末页"})
        payload = io.BytesIO()
        Image.new("RGB", (20, 30), "navy").save(payload, "PNG")
        path = self.root / "out/taobao/123/manifest.json"
        observed = []
        async def product(*args, **kwargs):
            manifest = json.loads(path.read_text(encoding="utf-8"))
            observed.append(manifest)
            self.assertEqual(manifest["images"][image_url]["status"], "downloaded")
            self.assertTrue((path.parent / manifest["images"][image_url]["file"]).is_file())
            self.assertGreater(manifest["stats"]["category_files"], 0)
            self.assertFalse(manifest["complete"])
            self.assertEqual(manifest["stage"], "read_product")
            raise RuntimeError("synthetic product collector failure")
        self.product_collector.side_effect = product
        events = []
        with patch("buyer_show.engine.ADAPTERS", {"taobao": SimpleNamespace(collect=collector)}), \
             patch("buyer_show.engine.ImageFetcher", return_value=AsyncMock(return_value=payload.getvalue())):
            reports = await run_batch([classify_url("https://item.taobao.com/item.htm?id=123")],
                                     self.root / "out", self.root / "profiles", emit=lambda _: None, on_event=events.append)
        self.assertEqual(len(observed), 1)
        self.assertEqual(reports[0]["status"], "error")
        self.assertEqual(reports[0]["stats"]["unique_images"], 1)
        self.assertTrue(reports[0]["scopes"]["reviews"]["complete"])
        self.assertEqual(reports[0]["scopes"]["product"]["status"], "error")
        first_download = next(e for e in events if e.get("stage") == "download_reviews")
        self.assertEqual(first_download["stats"]["image_urls"], 1)
        self.assertEqual(first_download["stats"]["unique_images"], 0)
        before_product = next(e for e in events if e.get("stage") == "read_product")
        self.assertEqual(before_product["stats"]["unique_images"], 1)
        self.assertGreater(before_product["stats"]["category_files"], 0)

    async def test_second_scope_does_not_retry_failed_media_but_new_run_does(self):
        first, second = "https://img.alicdn.com/failed.png", "https://img.alicdn.com/product.png"
        video = "https://cloud.video.taobao.com/vod/a.mp4"
        record = {"platform": "taobao", "product_id": "123", "review_id": "r1",
                  "review_type": "initial", "content": "合成测试", "images": [first], "videos": [video]}
        collector = AsyncMock(return_value={"status": "completed", "complete": True, "records": [record], "reason": "测试末页"})
        self.product_collector.return_value = {"status": "completed", "complete": True, "reason": "测试终点", "media": [
            {"url": first, "media_type": "image", "source_type": "product_detail"},
            {"url": second, "media_type": "image", "source_type": "product_main"},
            {"url": video, "media_type": "video", "source_type": "product_video"}]}
        payload = io.BytesIO()
        Image.new("RGB", (20, 30), "green").save(payload, "PNG")
        async def image_fetch(url):
            if url == first:
                raise ValueError("synthetic failed image")
            return payload.getvalue()
        images = AsyncMock(side_effect=image_fetch)
        videos = AsyncMock(side_effect=ValueError("synthetic failed video"))
        with patch("buyer_show.engine.ADAPTERS", {"taobao": SimpleNamespace(collect=collector)}), \
             patch("buyer_show.engine.ImageFetcher", return_value=images), \
             patch("buyer_show.engine.VideoFetcher", return_value=SimpleNamespace(fetch=videos)):
            for attempt in (1, 2):
                reports = await run_batch([classify_url("https://item.taobao.com/item.htm?id=123")],
                                         self.root / "out", self.root / "profiles", emit=lambda _: None)
                self.assertEqual(sum(call.args[0] == first for call in images.await_args_list), attempt)
                self.assertEqual(videos.await_count, attempt)
                self.assertEqual(reports[0]["stats"]["unique_images"], 1)
                self.assertEqual(reports[0]["stats"]["downloads_failed"], 1)
                self.assertEqual(reports[0]["stats"]["videos_failed"], 1)
                self.assertFalse(reports[0]["complete"])
        self.assertEqual(sum(call.args[0] == second for call in images.await_args_list), 1)

    async def test_cancel_during_product_keeps_review_file_and_early_categories(self):
        stopped = False
        image_url = "https://img.alicdn.com/saved.png"
        record = {"platform": "taobao", "product_id": "123", "review_id": "r1",
                  "review_type": "append", "content": "测试", "images": [image_url]}
        collector = AsyncMock(return_value={"status": "completed", "complete": True, "records": [record], "reason": "测试末页"})
        payload = io.BytesIO()
        Image.new("RGB", (20, 30), "blue").save(payload, "PNG")
        async def product(*args, **kwargs):
            nonlocal stopped
            stopped = True
            await asyncio.sleep(20)
        self.product_collector.side_effect = product
        with patch("buyer_show.engine.ADAPTERS", {"taobao": SimpleNamespace(collect=collector)}), \
             patch("buyer_show.engine.ImageFetcher", return_value=AsyncMock(return_value=payload.getvalue())):
            reports = await run_batch([classify_url("https://item.taobao.com/item.htm?id=123")],
                                     self.root / "out", self.root / "profiles", emit=lambda _: None,
                                     should_stop=lambda: stopped)
        report = reports[0]
        self.assertEqual(report["status"], "cancelled", report.get("reason"))
        self.assertEqual(report["stats"]["unique_images"], 1)
        path = Path(report["manifest"])
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual((path.parent / manifest["images"][image_url]["file"]).read_bytes(), payload.getvalue())
        self.assertTrue(any((path.parent / manifest["classification"]["root"]).rglob("*.png")))

    async def test_cancellation_consumes_adapter_snapshot(self):
        calls = []

        async def collect(*args, **kwargs):
            try:
                await asyncio.sleep(20)
            except asyncio.CancelledError:
                return {"status": "cancelled", "complete": False, "records": [{"review_id": "saved"}]}

        def stop():
            calls.append(True)
            return len(calls) > 1

        result = await _cancellable_collect(SimpleNamespace(collect=collect), None, "fixture", 40, stop)
        self.assertEqual(result["records"][0]["review_id"], "saved")
        self.assertFalse(result["complete"])

    async def _run_taobao_task(self, product_mock):
        collector = AsyncMock(return_value={"status": "completed", "complete": True,
                                            "records": [], "reason": "测试末页"})
        with patch("buyer_show.engine.ADAPTERS", {"taobao": SimpleNamespace(collect=collector)}), \
             patch("buyer_show.engine.collect_product_media", product_mock):
            return await run_batch([classify_url("https://item.taobao.com/item.htm?id=123")],
                                   self.root / "out", self.root / "profiles", emit=lambda _: None)

    async def test_ensure_live_page_keeps_live_page_and_reopens_closed_one(self):
        browser = FakeBrowser()
        live = await browser.new_page()
        emit = lambda _: None
        self.assertIs(await _ensure_live_page(browser, live, "https://example.test", emit), live)
        live.closed = True
        reopened = await _ensure_live_page(browser, live, "https://example.test", emit)
        self.assertIsNot(reopened, live)
        self.assertIn("https://example.test", reopened.visits)
        self.assertFalse(reopened.is_closed())

    def test_target_closed_detects_only_navigation_target_errors(self):
        self.assertTrue(_target_closed({"diagnostics": {"errors": ["navigation:TargetClosedError"]}}))
        self.assertFalse(_target_closed({"diagnostics": {"errors": ["gallery_video_wait:TimeoutError"]}}))
        self.assertFalse(_target_closed({}))

    async def test_closed_review_page_is_reopened_before_product_stage(self):
        product_mock = AsyncMock(return_value={"status": "completed", "complete": True,
                                              "media": [], "reason": "商品素材已读取", "diagnostics": {}})

        def close_page(*_args, **_kwargs):
            self.fake.browser.pages[0].closed = True
            return {"complete": True}

        with patch("buyer_show.engine.ProductStore.refresh_categories", side_effect=close_page):
            result = await self._run_taobao_task(product_mock)
        self.assertEqual(len(self.fake.browser.pages), 2)
        self.assertEqual(product_mock.await_count, 1)
        self.assertIs(product_mock.await_args.args[0], self.fake.browser.pages[1])
        self.assertTrue(result[0]["complete"])

    async def test_product_target_closed_error_is_retried_once(self):
        first = {"status": "error", "complete": False, "media": [], "reason": "商品页面未成功加载",
                 "diagnostics": {"errors": ["navigation:TargetClosedError"]}}
        second = {"status": "completed", "complete": True, "media": [], "reason": "商品素材已读取",
                  "diagnostics": {}}
        product_mock = AsyncMock(side_effect=[first, second])
        result = await self._run_taobao_task(product_mock)
        self.assertEqual(product_mock.await_count, 2)
        self.assertEqual(result[0]["scopes"]["product"]["details"]["recovered_after"],
                         "target_closed_retry")
        self.assertTrue(result[0]["complete"])

    async def test_cancellation_runs_collector_cleanup(self):
        cleaned = []

        async def collect(*args, **kwargs):
            try:
                await asyncio.sleep(20)
            finally:
                cleaned.append(True)

        calls = []

        def stop():
            calls.append(True)
            return len(calls) > 1

        result = await _cancellable_collect(SimpleNamespace(collect=collect), None, "fixture", 40, stop)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(cleaned, [True])


if __name__ == "__main__":
    unittest.main()
