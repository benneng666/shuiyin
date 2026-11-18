"""Synthetic images and mock HTTP. No live-platform claim is made by these tests."""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
from PIL import Image

from buyer_show.storage import ImageFetcher, ProductStore, image_url_allowed, validate_image

SOURCE = "https://item.taobao.com/item.htm?id=123"
A = "https://img.alicdn.com/a.png"
B = "https://img.alicdn.com/b.png"


def png(size=(16, 24)):
    out = io.BytesIO()
    Image.new("RGB", size, "coral").save(out, "PNG")
    return out.getvalue()


def row(rid="r1", images=None, kind="initial"):
    return {"platform": "taobao", "product_id": "123", "review_id": rid,
            "review_type": kind, "content": "合成测试评价", "images": images if images is not None else [A]}


COMPLETE = {"status": "completed", "complete": True, "reason": "合成测试末页", "diagnostics": {}}


class StorageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="buyer_show_tests_")
        self.addCleanup(self.temp.cleanup)
        self.store = ProductStore(Path(self.temp.name), "taobao", "123", SOURCE)

    async def test_duplicates_keep_all_review_associations_and_one_file(self):
        self.store.merge([row(images=[A, B]), row("r2"), row(images=[B], kind="append")])
        fetch = AsyncMock(return_value=png())
        await self.store.download(fetch)
        report = self.store.finish(COMPLETE)
        self.assertTrue(report["complete"])
        self.assertEqual(report["stats"]["reviews_read"], 2)
        self.assertEqual(report["stats"]["review_entries"], 3)
        self.assertEqual(report["stats"]["image_associations"], 4)
        self.assertEqual(report["stats"]["unique_images"], 1)
        self.assertEqual(len(list((self.store.root / "images").glob("*.png"))), 1)

    async def test_resume_checks_file_and_repairs_corruption(self):
        self.store.merge([row()])
        fetch = AsyncMock(return_value=png())
        await self.store.download(fetch)
        resumed = ProductStore(Path(self.temp.name), "taobao", "123", SOURCE)
        fetch.reset_mock()
        await resumed.download(fetch)
        fetch.assert_not_awaited()
        saved = resumed.root / resumed.data["images"][A]["file"]
        saved.write_bytes(b"corrupt")
        await resumed.download(fetch)
        fetch.assert_awaited_once()
        self.assertEqual(saved.read_bytes(), png())

    async def test_new_one_pixel_image_is_excluded_and_original_retained(self):
        self.store.merge([row()])
        payload = png((1, 1))
        await self.store.download(AsyncMock(return_value=payload))
        item = self.store.data["images"][A]
        self.assertEqual(item["status"], "excluded")
        self.assertEqual(item["exclusion_reason"], "one_pixel_placeholder")
        self.assertEqual((item["width"], item["height"]), (1, 1))
        self.assertEqual((self.store.root / item["file"]).read_bytes(), payload)
        self.assertEqual(item["sha256"], validate_image(payload)["sha256"])
        report = self.store.finish(COMPLETE)
        self.assertTrue(report["complete"])
        self.assertEqual(report["stats"]["images_excluded"], 1)
        for key in ("downloads_succeeded", "downloads_failed", "downloads_pending", "unique_images"):
            self.assertEqual(report["stats"][key], 0)
        self.assertEqual(report["download_failures"], [])
        self.assertEqual(report["download_failure_summary"], "")
        resumed = ProductStore(Path(self.temp.name), "taobao", "123", SOURCE)
        fetch = AsyncMock()
        await resumed.download(fetch)
        fetch.assert_not_awaited()
        self.assertEqual(resumed.data["images"][A]["status"], "excluded")

    async def test_resume_migrates_old_downloaded_one_pixel_from_decoded_file(self):
        self.store.merge([row()])
        payload = png((1, 1))
        info = validate_image(payload)
        relative = "images/" + info["sha256"] + info.pop("extension")
        saved = self.store.root / relative
        saved.parent.mkdir()
        saved.write_bytes(payload)
        # Deliberately incorrect old dimensions must not evade the decoded check.
        self.store.data["images"][A].update(status="downloaded", file=relative, **info)
        self.store.data["images"][A].update(width=800, height=800)
        self.store.checkpoint()
        resumed = ProductStore(Path(self.temp.name), "taobao", "123", SOURCE)
        fetch = AsyncMock()
        await resumed.download(fetch)
        fetch.assert_not_awaited()
        item = resumed.data["images"][A]
        self.assertEqual(item["status"], "excluded")
        self.assertEqual((item["width"], item["height"]), (1, 1))
        self.assertEqual(item["file"], relative)
        self.assertEqual(saved.read_bytes(), payload)
        self.assertTrue(resumed.finish(COMPLETE)["complete"])
        persisted = json.loads(resumed.path.read_text(encoding="utf-8"))
        self.assertEqual(persisted["images"][A]["exclusion_reason"], "one_pixel_placeholder")

    async def test_small_non_placeholder_images_are_still_successful(self):
        urls = [A, B, "https://img.alicdn.com/c.png"]
        sizes = [(1, 2), (2, 1), (2, 2)]
        self.store.merge([row(images=urls)])
        await self.store.download(AsyncMock(side_effect=[png(size) for size in sizes]))
        for url, size in zip(urls, sizes):
            item = self.store.data["images"][url]
            self.assertEqual(item["status"], "downloaded")
            self.assertEqual((item["width"], item["height"]), size)
            self.assertNotIn("exclusion_reason", item)
        report = self.store.finish(COMPLETE)
        self.assertTrue(report["complete"])
        self.assertEqual(report["stats"]["downloads_succeeded"], 3)
        self.assertEqual(report["stats"]["images_excluded"], 0)

    async def test_html_response_is_failed_and_can_retry(self):
        self.store.merge([row()])
        await self.store.download(AsyncMock(return_value=b"<html>Please log in</html>"))
        report = self.store.finish(COMPLETE)
        self.assertFalse(report["complete"])
        self.assertEqual(report["status"], "partial_download")
        self.assertEqual(report["stats"]["downloads_failed"], 1)
        await self.store.download(AsyncMock(return_value=png()))
        self.assertTrue(self.store.finish(COMPLETE)["complete"])

    async def test_saved_images_do_not_mean_complete_collection(self):
        self.store.merge([row()])
        await self.store.download(AsyncMock(return_value=png()))
        report = self.store.finish({"status": "partial", "complete": False})
        self.assertFalse(report["complete"])
        self.assertEqual(report["stats"]["downloads_succeeded"], 1)

    async def test_rerun_immediately_revokes_old_complete_status(self):
        self.store.merge([row()])
        await self.store.download(AsyncMock(return_value=png()))
        self.store.finish(COMPLETE)
        saved = self.store.root / self.store.data["images"][A]["file"]
        saved.unlink()
        resumed = ProductStore(Path(self.temp.name), "taobao", "123", SOURCE)
        resumed.begin()
        persisted = json.loads(resumed.path.read_text(encoding="utf-8"))
        self.assertFalse(persisted["complete"])
        self.assertEqual(persisted["status"], "running")

    async def test_stop_retains_files_and_pending_entries(self):
        self.store.merge([row(images=[A, B])])
        calls = []

        async def fetch(url):
            calls.append(url)
            return png()

        await self.store.download(fetch, should_stop=lambda: bool(calls))
        report = self.store.finish(COMPLETE, cancelled=True)
        self.assertEqual(report["status"], "cancelled")
        self.assertEqual(report["stats"]["downloads_succeeded"], 1)
        self.assertEqual(report["stats"]["downloads_pending"], 1)

    async def test_cross_product_and_non_review_rejected(self):
        other = row()
        other["product_id"] = "999"
        with self.assertRaises(ValueError):
            self.store.merge([other])
        invalid = row()
        invalid["review_type"] = "product_gallery"
        with self.assertRaises(ValueError):
            self.store.merge([invalid])

    async def test_bad_host_is_never_fetched(self):
        self.store.merge([row(images=["https://alicdn.com.evil.test/photo.jpg"])])
        fetch = AsyncMock(return_value=png())
        await self.store.download(fetch)
        fetch.assert_not_awaited()
        self.assertFalse(self.store.finish(COMPLETE)["complete"])

    async def test_redirect_to_other_domain_is_rejected(self):
        calls = []

        def handler(req):
            calls.append(str(req.url))
            return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaises(ValueError):
                await ImageFetcher(client, "taobao", SOURCE)(A)
        self.assertEqual(calls, [A])

    async def test_real_http_pipeline_with_mock_transport(self):
        def handler(req):
            self.assertEqual(req.headers.get("referer"), SOURCE)
            self.assertNotIn("cookie", req.headers)
            return httpx.Response(200, content=png(), headers={"content-type": "image/png"})

        self.store.merge([row()])
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await self.store.download(ImageFetcher(client, "taobao", SOURCE))
        self.store.finish(COMPLETE)
        persisted = json.loads(self.store.path.read_text(encoding="utf-8"))
        self.assertTrue(persisted["complete"])
        self.assertEqual(persisted["images"][A]["height"], 24)

    def test_host_and_identity_boundaries(self):
        for url in ("file:///a.png", "https://user:pass@img.alicdn.com/a.png", "https://img.alicdn.com:8080/a.png"):
            self.assertFalse(image_url_allowed(url, "taobao"))
        with self.assertRaises(ValueError):
            ProductStore(Path(self.temp.name), "taobao", "../escape", SOURCE)

    async def test_transient_connection_failure_retries_then_succeeds(self):
        calls = []

        def handler(request):
            calls.append(request)
            if len(calls) == 1:
                raise httpx.RemoteProtocolError("server disconnected")
            return httpx.Response(200, content=png())

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            self.assertEqual(await ImageFetcher(client, "taobao", SOURCE)(A), png())
        self.assertEqual(len(calls), 2)

    async def test_404_and_429_are_not_automatically_retried(self):
        for status in (404, 429):
            calls = []

            def handler(request):
                calls.append(request)
                return httpx.Response(status, headers={"Retry-After": "120"})

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                with self.assertRaises(httpx.HTTPStatusError):
                    await ImageFetcher(client, "taobao", SOURCE)(A)
            self.assertEqual(len(calls), 1)

    async def test_shared_client_cookies_and_auth_are_never_sent(self):
        def handler(request):
            self.assertNotIn("cookie", request.headers)
            self.assertNotIn("authorization", request.headers)
            self.assertNotIn("proxy-authorization", request.headers)
            return httpx.Response(200, content=png())

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler),
                cookies={"unrelated": "value"}, auth=httpx.BasicAuth("user", "secret"),
                headers={"Authorization": "Bearer unrelated", "Proxy-Authorization": "secret"}) as client:
            self.assertEqual(await ImageFetcher(client, "taobao", SOURCE)(A), png())

    async def test_server_5xx_retries_are_bounded_and_bad_payload_errors_do_not_retry(self):
        calls = []

        def unavailable(request):
            calls.append(request)
            return httpx.Response(503)

        async with httpx.AsyncClient(transport=httpx.MockTransport(unavailable)) as client:
            with self.assertRaises(httpx.HTTPStatusError):
                await ImageFetcher(client, "taobao", SOURCE)(A)
        self.assertEqual(len(calls), 3)
        calls.clear()

        def malformed(request):
            calls.append(request)
            raise httpx.DecodingError("malformed content encoding")

        async with httpx.AsyncClient(transport=httpx.MockTransport(malformed)) as client:
            with self.assertRaises(httpx.DecodingError):
                await ImageFetcher(client, "taobao", SOURCE)(A)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
