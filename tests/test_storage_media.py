"""Media association, migration, cancellation and incomplete-download contracts."""
import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from buyer_show.storage import MAX_VIDEO_SOURCES, ProductStore
from buyer_show.video import UnsupportedVideoError, VideoError, VideoFetcher

SOURCE = "https://item.taobao.com/item.htm?id=123"
A = "https://cloud.video.taobao.com/vod/a.mp4"
B = "https://cloud.video.taobao.com/vod/b.mp4"
IMAGE = "https://img.alicdn.com/a.png"
COMPLETE = {"status": "completed", "complete": True}
PAYLOAD = b"synthetic mock video; real decode is separately tested in test_video.py"
INFO = {"sha256": hashlib.sha256(PAYLOAD).hexdigest(), "bytes": len(PAYLOAD),
        "format": "MP4", "extension": ".mp4", "frames_decoded": 4, "validation": "ffmpeg_full_decode"}


def review(kind="initial", videos=None):
    return {"platform": "taobao", "product_id": "123", "review_id": "r1",
            "review_type": kind, "content": "合成测试", "images": [], "videos": videos or [A]}


async def fetch(url, path, should_stop):
    path.write_bytes(PAYLOAD)


async def validate(path, should_stop):
    if path.read_bytes() != PAYLOAD:
        from buyer_show.video import VideoError
        raise VideoError("video_decode_failed")
    return dict(INFO)


class MediaStorageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="buyer_show_media_")
        self.addCleanup(self.temp.cleanup)
        self.store = ProductStore(Path(self.temp.name), "taobao", "123", SOURCE)
        self.validator = patch("buyer_show.storage.validate_video", side_effect=validate)
        self.validator.start()
        self.addCleanup(self.validator.stop)

    async def test_product_and_review_associations_are_separate_sha_deduplicated(self):
        self.store.merge([review(), review("append", [B])])
        self.store.merge_product_media([
            {"url": A, "media_type": "video", "source_type": "product_video"},
            {"url": A, "media_type": "video", "source_type": "product_detail"},
            {"url": A, "media_type": "video", "source_type": "product_video"}])
        await self.store.download_videos(fetch)
        report = self.store.finish(COMPLETE)
        self.assertTrue(report["complete"])
        self.assertEqual(report["stats"]["reviews_read"], 1)
        self.assertEqual(report["stats"]["review_entries"], 2)
        self.assertEqual(report["stats"]["product_media_count"], 2)
        self.assertEqual(report["stats"]["video_downloads_succeeded"], 2)
        self.assertEqual(report["stats"]["unique_videos"], 1)
        self.assertEqual(report["stats"]["downloads_succeeded"], 0)
        self.assertEqual(len(list((self.store.root / "videos").glob("*.mp4"))), 1)
        self.assertEqual(self.store.data["videos"][A]["file"], self.store.data["videos"][B]["file"])

    async def test_product_images_join_image_queue_without_reviews(self):
        self.store.merge_product_media([
            {"url": IMAGE, "media_type": "image", "source_type": "product_main"},
            {"url": IMAGE, "media_type": "image", "source_type": "product_detail"}])
        self.assertEqual(self.store.stats()["reviews_read"], 0)
        self.assertEqual(self.store.stats()["product_media_count"], 2)
        self.assertEqual(self.store.stats()["image_urls"], 1)

    async def test_schema_one_image_manifest_migrates_additively(self):
        self.store.data.pop("videos")
        self.store.data.pop("product_media")
        self.store.data["records"] = [{"review_id": "r1", "review_type": "initial", "content": "", "images": []}]
        self.store.path.write_text(json.dumps(self.store.data), encoding="utf-8")
        resumed = ProductStore(Path(self.temp.name), "taobao", "123", SOURCE)
        self.assertEqual(resumed.data["schema_version"], 1)
        self.assertEqual(resumed.data["records"][0]["videos"], [])
        resumed.merge([review()])
        self.assertEqual(resumed.stats()["video_urls"], 1)

    async def test_video_pending_or_unsupported_prevents_complete(self):
        self.store.merge([review()])
        self.assertFalse(self.store.finish(COMPLETE)["complete"])
        with patch("buyer_show.storage.validate_video", side_effect=UnsupportedVideoError("video_decoder_unavailable")):
            await self.store.download_videos(fetch)
        report = self.store.finish(COMPLETE)
        self.assertEqual(report["status"], "partial_download")
        self.assertEqual(report["stats"]["videos_failed"], 1)
        self.assertEqual(report["stats"]["downloads_failed"], 0)
        self.assertEqual(self.store.data["videos"][A]["error"], "video_decoder_unavailable")
        self.assertFalse(list((self.store.root / "videos").glob("*.part")))

    async def test_resume_revalidates_and_repairs_corrupt_original(self):
        self.store.merge([review()])
        await self.store.download_videos(fetch)
        resumed = ProductStore(Path(self.temp.name), "taobao", "123", SOURCE)
        downloader = AsyncMock(side_effect=fetch)
        await resumed.download_videos(downloader)
        downloader.assert_not_awaited()
        saved = resumed.root / resumed.data["videos"][A]["file"]
        saved.write_bytes(b"corrupt")
        await resumed.download_videos(downloader)
        downloader.assert_awaited_once()
        self.assertEqual(saved.read_bytes(), PAYLOAD)

    async def test_cancel_keeps_completed_associations_and_pending_url(self):
        self.store.merge([review(videos=[A, B])])
        waiting = asyncio.Event()

        async def waiting_fetch(url, path, should_stop):
            path.write_bytes(PAYLOAD)
            if url == B:
                waiting.set()
                await asyncio.sleep(60)

        task = asyncio.create_task(self.store.download_videos(waiting_fetch))
        await asyncio.wait_for(waiting.wait(), 2)
        task.cancel()
        await asyncio.wait_for(task, 2)
        report = self.store.finish(COMPLETE, cancelled=True)
        self.assertEqual(report["stats"]["unique_videos"], 1)
        self.assertEqual(report["stats"]["videos_pending"], 1)
        self.assertEqual(report["status"], "cancelled")
        self.assertFalse(list((self.store.root / "videos").glob("*.part")))

    async def test_cancel_during_old_video_revalidation_preserves_saved_record_and_file(self):
        self.store.merge([review()])
        await self.store.download_videos(fetch)
        before = json.loads(json.dumps(self.store.data["videos"][A]))
        saved = self.store.root / before["file"]
        downloader = AsyncMock(side_effect=fetch)
        with patch("buyer_show.storage.validate_video", side_effect=asyncio.CancelledError()):
            await self.store.download_videos(downloader)
        downloader.assert_not_awaited()
        self.assertEqual(self.store.data["videos"][A], before)
        self.assertEqual(hashlib.sha256(saved.read_bytes()).hexdigest(), before["sha256"])
        report = self.store.finish(COMPLETE, cancelled=True)
        self.assertEqual(report["status"], "cancelled")
        self.assertFalse(report["complete"])
        self.assertEqual(report["stats"]["unique_videos"], 1)

    async def test_cancel_after_invalid_old_video_enters_new_download_leaves_pending(self):
        self.store.merge([review()])
        await self.store.download_videos(fetch)
        saved = self.store.root / self.store.data["videos"][A]["file"]
        saved.write_bytes(b"corrupt")
        downloader = AsyncMock(side_effect=asyncio.CancelledError())
        await self.store.download_videos(downloader)
        downloader.assert_awaited_once()
        self.assertEqual(self.store.data["videos"][A]["status"], "pending")
        self.assertEqual(self.store.data["videos"][A]["error_code"], "cancelled")
        self.assertFalse(self.store.finish(COMPLETE, cancelled=True)["complete"])

    async def test_unsafe_video_never_reaches_fetcher(self):
        self.store.merge([review(videos=["blob:https://item.taobao.com/id"])])
        downloader = AsyncMock()
        await self.store.download_videos(downloader)
        downloader.assert_not_awaited()
        self.assertEqual(self.store.stats()["videos_unsupported"], 1)

    async def test_manifest_path_traversal_does_not_read_external_file(self):
        self.store.merge([review()])
        self.store.data["videos"][A].update(status="downloaded", file="../outside.mp4", sha256=INFO["sha256"])
        downloader = AsyncMock(side_effect=fetch)
        await self.store.download_videos(downloader)
        downloader.assert_awaited_once()
        self.assertTrue(self.store.data["videos"][A]["file"].startswith("videos/"))

    async def test_native_fallback_saves_one_video_and_resume_reuses_selected_source(self):
        row = review()
        row["video_sources"] = {A: [A, B]}
        self.store.merge([row])
        calls = []

        async def fallback(url, path, should_stop):
            calls.append(url)
            if url == A:
                response = httpx.Response(490, request=httpx.Request("GET", url))
                response.raise_for_status()
            await fetch(url, path, should_stop)

        await self.store.download_videos(fallback)
        item = self.store.data["videos"][A]
        self.assertEqual(calls, [A, B])
        self.assertEqual(item["selected_source"], B)
        self.assertEqual(item["source_candidates"], [A, B])
        self.assertEqual(item["source_attempts"][0]["http_status"], 490)
        self.assertEqual(item["source_attempts"][1]["status"], "downloaded")
        self.assertEqual(self.store.stats()["video_urls"], 1)
        self.assertEqual(self.store.stats()["video_associations"], 1)
        self.assertEqual(self.store.stats()["video_downloads_succeeded"], 1)
        self.assertNotIn(B, self.store.data["videos"])
        report = self.store.finish(COMPLETE)
        self.assertTrue(report["complete"])
        self.assertEqual(report["download_failures"], [])
        self.assertEqual(report["download_failure_summary"], "")
        resumed = ProductStore(Path(self.temp.name), "taobao", "123", SOURCE)
        downloader = AsyncMock()
        await resumed.download_videos(downloader)
        downloader.assert_not_awaited()
        self.assertEqual(resumed.data["videos"][A]["selected_source"], B)
        self.assertEqual(resumed.data["videos"][A]["source_attempts"], item["source_attempts"])

    async def test_sources_only_bind_to_row_canonical_and_are_bounded_deduplicated(self):
        row = review()
        row["video_sources"] = {A: [B, B, None, 2] + [f"https://cloud.video.taobao.com/{i}.mp4" for i in range(30)],
                                "https://cloud.video.taobao.com/unrelated.mp4": [B]}
        self.store.merge([row])
        self.assertEqual(list(self.store.data["videos"]), [A])
        candidates = self.store.data["videos"][A]["source_candidates"]
        self.assertEqual(candidates[:2], [A, B])
        self.assertEqual(len(candidates), MAX_VIDEO_SOURCES)
        self.store.merge([review()])  # A later record without alternatives must retain them.
        self.assertEqual(self.store.data["videos"][A]["source_candidates"], candidates)

    async def test_bad_candidate_not_requested_and_decode_failure_tries_native_alternate(self):
        row = review()
        bad = "https://cloud.video.taobao.com.evil.test/secret.mp4?token=private"
        row["video_sources"] = {A: [A, bad, B]}
        self.store.merge([row])
        calls = []

        async def invalid_then_valid(url, path, should_stop):
            calls.append(url)
            path.write_bytes(b"not a video" if url == A else PAYLOAD)

        await self.store.download_videos(invalid_then_valid)
        self.assertEqual(calls, [A, B])
        item = self.store.data["videos"][A]
        self.assertEqual(item["selected_source"], B)
        self.assertEqual(item["source_attempts"][0]["error_code"], "video_decode_failed")
        self.assertEqual(item["source_attempts"][1]["error_code"], "unrecognized_video_host")
        self.assertNotIn("private", json.dumps(item["source_attempts"]))
        self.assertFalse(list((self.store.root / "videos").glob("*.part")))

    async def test_cancel_during_fallback_preserves_attempt_and_removes_temporary(self):
        row = review()
        third = "https://cloud.video.taobao.com/vod/third.mp4"
        row["video_sources"] = {A: [A, B, third]}
        self.store.merge([row])
        waiting = asyncio.Event()
        calls = []

        async def stalled(url, path, should_stop):
            calls.append(url)
            if url == A:
                raise VideoError("video_decode_failed")
            path.write_bytes(b"partial")
            waiting.set()
            await asyncio.sleep(60)

        task = asyncio.create_task(self.store.download_videos(stalled))
        await asyncio.wait_for(waiting.wait(), 2)
        task.cancel()
        await asyncio.wait_for(task, 2)
        item = self.store.data["videos"][A]
        self.assertEqual(calls, [A, B])
        self.assertEqual(item["status"], "pending")
        self.assertEqual(item["source_attempts"][-1]["status"], "cancelled")
        self.assertNotIn("selected_source", item)
        self.assertFalse(list((self.store.root / "videos").glob("*.part")))

    async def test_429_stops_immediate_alternate_attempts(self):
        row = review()
        row["video_sources"] = {A: [A, B]}
        self.store.merge([row])
        calls = []

        async def limited(url, path, should_stop):
            calls.append(url)
            httpx.Response(429, request=httpx.Request("GET", url)).raise_for_status()

        await self.store.download_videos(limited)
        self.assertEqual(calls, [A])
        self.assertEqual(self.store.data["videos"][A]["http_status"], 429)
        self.assertIn("限流", self.store.finish(COMPLETE)["download_failure_summary"])

    async def test_alternate_redirect_is_checked_and_shared_client_auth_is_not_sent(self):
        row = review()
        row["video_sources"] = {A: [A, B]}
        self.store.merge([row])
        calls = []

        def handler(request):
            calls.append(str(request.url))
            for header in ("authorization", "cookie", "proxy-authorization"):
                self.assertNotIn(header, request.headers)
            if str(request.url) == A:
                return httpx.Response(490)
            return httpx.Response(302, headers={"location": "https://cloud.video.taobao.com.evil.test/private"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler),
                cookies={"unrelated": "secret"}, headers={"Authorization": "Bearer unrelated"},
                auth=httpx.BasicAuth("unrelated", "secret")) as client:
            downloader = VideoFetcher(client, "taobao", SOURCE)
            await self.store.download_videos(downloader.fetch)
        self.assertEqual(calls, [A, B])
        item = self.store.data["videos"][A]
        self.assertNotIn("selected_source", item)
        self.assertEqual(item["http_status"], 490)
        self.assertEqual(item["source_attempts"][1]["error_code"], "unrecognized_video_host")
        self.assertFalse(list((self.store.root / "videos").glob("*.part")))

    async def test_failure_report_groups_sources_http_and_never_copies_urls_or_errors(self):
        row = review()
        row["video_sources"] = {A: [A, "https://evil.test/secret?token=hidden"]}
        self.store.merge([row, review("append")])
        self.store.merge_product_media([
            {"url": IMAGE, "media_type": "image", "source_type": "product_main"},
            {"url": IMAGE, "media_type": "image", "source_type": "product_detail"},
            {"url": A, "media_type": "video", "source_type": "product_video"}])

        async def denied(url, path, should_stop):
            httpx.Response(490, request=httpx.Request("GET", url)).raise_for_status()

        await self.store.download_videos(denied)
        self.store.data["images"][IMAGE].update(status="failed", http_status=420, error="private https://secret.test/?token=hidden")
        report = self.store.finish(COMPLETE)
        groups = report["download_failures"]
        self.assertEqual(len(groups), 5)
        self.assertEqual({g["source_label"] for g in groups}, {"商品主图", "详情图", "商品视频", "初评买家秀", "追评买家秀"})
        self.assertEqual({g["http_status"] for g in groups}, {420, 490})
        self.assertTrue(all(g["count"] == 1 for g in groups))
        self.assertTrue(report["download_failure_summary"].startswith("2 项素材未保存"))
        safe = json.dumps(groups, ensure_ascii=False) + report["download_failure_summary"]
        for secret in ("https://", "token", "private", "hidden", "secret.test"):
            self.assertNotIn(secret, safe)
        on_disk = json.loads(self.store.path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["download_failures"], groups)
        self.assertEqual(on_disk["download_failure_summary"], report["download_failure_summary"])

    async def test_unknown_exception_text_is_not_recorded_or_emitted(self):
        self.store.merge([review()])
        messages = []

        async def failure(url, path, should_stop):
            raise RuntimeError("private https://example.test/?token=hidden")

        await self.store.download_videos(failure, emit=messages.append)
        item = self.store.data["videos"][A]
        self.assertEqual(item["error_code"], "download_error")
        report = self.store.finish(COMPLETE)
        safe = json.dumps(item["source_attempts"]) + "".join(messages) + report["download_failure_summary"]
        self.assertNotIn("private", safe)
        self.assertNotIn("hidden", safe)


if __name__ == "__main__":
    unittest.main()
