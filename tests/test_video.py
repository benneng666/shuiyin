"""Synthetic media + mock HTTP only: no live platform or account is accessed."""
import asyncio
import hashlib
import io
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from PIL import Image

from buyer_show.video import (VideoError, UnsupportedVideoError, VideoFetcher,
    _InputMetadata, ffmpeg_executable, validate_video, video_url_allowed)

SOURCE = "https://item.taobao.com/item.htm?id=123"
URL = "https://cloud.video.taobao.com/vod/example.mp4"


class SlowStream(httpx.AsyncByteStream):
    def __init__(self):
        self.started = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        yield b"x" * 65536
        self.started.set()
        await asyncio.sleep(60)

    async def aclose(self):
        self.closed = True


class VideoDownloadTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="buyer_show_video_")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "video.part"

    async def test_original_bytes_and_anonymous_headers(self):
        payload = b"original source bytes"

        def handler(request):
            self.assertNotIn("cookie", request.headers)
            self.assertNotIn("authorization", request.headers)
            self.assertEqual(request.headers["referer"], SOURCE)
            return httpx.Response(200, content=payload)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler),
                headers={"Authorization": "Bearer unrelated"}, cookies={"unrelated": "cookie"},
                auth=httpx.BasicAuth("unrelated", "secret")) as client:
            await VideoFetcher(client, "taobao", SOURCE).fetch(URL, self.path)
        self.assertEqual(self.path.read_bytes(), payload)

    async def test_redirect_is_rechecked_before_network(self):
        calls = []

        def handler(request):
            calls.append(str(request.url))
            return httpx.Response(302, headers={"location": "https://cloud.video.taobao.com.evil.test/x"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaisesRegex(UnsupportedVideoError, "unrecognized_video_host"):
                await VideoFetcher(client, "taobao", SOURCE).fetch(URL, self.path)
        self.assertEqual(calls, [URL])
        self.assertFalse(self.path.exists())

    async def test_retry_transient_error_but_not_403(self):
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(503 if len(calls) == 1 else 200, content=b"source")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await VideoFetcher(client, "taobao", SOURCE).fetch(URL, self.path)
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.path.read_bytes(), b"source")
        self.path.unlink()
        calls.clear()

        def denied(request):
            calls.append(request)
            return httpx.Response(403)

        async with httpx.AsyncClient(transport=httpx.MockTransport(denied)) as client:
            with self.assertRaises(httpx.HTTPStatusError):
                await VideoFetcher(client, "taobao", SOURCE).fetch(URL, self.path)
        self.assertEqual(len(calls), 1)
        self.assertFalse(self.path.exists())

    async def test_rejects_cover_playlist_partial_and_oversized_body(self):
        cases = [
            (httpx.Response(200, content=b"png", headers={"content-type": "image/png"}), UnsupportedVideoError),
            (httpx.Response(200, content=b"#EXTM3U", headers={"content-type": "application/vnd.apple.mpegurl"}), UnsupportedVideoError),
            (httpx.Response(206, content=b"range"), VideoError),
            (httpx.Response(200, content=b"123456789"), VideoError),
        ]
        for response, error in cases:
            with self.subTest(status=response.status_code, headers=response.headers):
                async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as client:
                    with self.assertRaises(error):
                        await VideoFetcher(client, "taobao", SOURCE, max_bytes=8).fetch(URL, self.path)
                self.assertFalse(self.path.exists())

    async def test_rate_limit_permanent_refusals_and_decoding_errors_do_not_retry(self):
        for status in (404, 420, 429, 490):
            calls = []

            def handler(request):
                calls.append(request)
                return httpx.Response(status, headers={"retry-after": "120"})

            with self.subTest(status=status):
                async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                    with self.assertRaises(httpx.HTTPStatusError):
                        await VideoFetcher(client, "taobao", SOURCE, attempts=3).fetch(URL, self.path)
                self.assertEqual(len(calls), 1)
                self.assertFalse(self.path.exists())

        calls = []

        def broken_encoding(request):
            calls.append(request)
            raise httpx.DecodingError("invalid response encoding", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(broken_encoding)) as client:
            with self.assertRaises(httpx.DecodingError):
                await VideoFetcher(client, "taobao", SOURCE, attempts=3).fetch(URL, self.path)
        self.assertEqual(len(calls), 1)
        self.assertFalse(self.path.exists())

    async def test_existing_destination_is_never_overwritten(self):
        self.path.write_bytes(b"user content")
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: self.fail("no request"))) as client:
            with self.assertRaises(FileExistsError):
                await VideoFetcher(client, "taobao", SOURCE).fetch(URL, self.path)
        self.assertEqual(self.path.read_bytes(), b"user content")

    async def test_blob_and_hls_are_unsupported_without_http(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: self.fail("no request"))) as client:
            for url in ("blob:https://item.taobao.com/id", URL.replace(".mp4", ".m3u8")):
                with self.assertRaises(UnsupportedVideoError):
                    await VideoFetcher(client, "taobao", SOURCE).fetch(url, self.path)
        self.assertFalse(self.path.exists())

    async def test_cancel_waiting_body_closes_response_and_removes_part(self):
        stream = SlowStream()
        stopped = False
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda _: httpx.Response(200, stream=stream))) as client:
            task = asyncio.create_task(VideoFetcher(client, "taobao", SOURCE).fetch(
                URL, self.path, lambda: stopped))
            await asyncio.wait_for(stream.started.wait(), 2)
            stopped = True
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
        self.assertTrue(stream.closed)
        self.assertFalse(self.path.exists())

    async def test_task_cancel_and_total_timeout_remove_partial_file(self):
        for mode in ("cancel", "timeout"):
            stream = SlowStream()
            async with httpx.AsyncClient(transport=httpx.MockTransport(
                    lambda _: httpx.Response(200, stream=stream))) as client:
                fetcher = VideoFetcher(client, "taobao", SOURCE, timeout=0.25 if mode == "timeout" else 20)
                task = asyncio.create_task(fetcher.fetch(URL, self.path))
                await asyncio.wait_for(stream.started.wait(), 2)
                if mode == "cancel":
                    task.cancel()
                with self.assertRaises(asyncio.CancelledError if mode == "cancel" else TimeoutError):
                    await task
            self.assertFalse(self.path.exists())
            self.assertTrue(stream.closed)

    def test_domain_boundaries_and_platform_separation(self):
        self.assertTrue(video_url_allowed(URL, "taobao"))
        self.assertTrue(video_url_allowed("https://v9-v2-mps-cdn.douyinvod.com/x", "douyin"))
        for url in ("file:///local.mp4", "http://127.0.0.1/a.mp4", "https://alicdn.com.evil.test/x",
                    "https://@cloud.video.taobao.com/a.mp4",
                    "https://u:p@cloud.video.taobao.com/a", "https://cloud.video.taobao.com:8080/a"):
            self.assertFalse(video_url_allowed(url, "taobao"))
        self.assertFalse(video_url_allowed(URL, "douyin"))

    def test_observed_taobao_cloud_video_host_is_exact_and_platform_scoped(self):
        host = "video-sh.cloudvideocdn.taobao.com"
        self.assertTrue(video_url_allowed(f"https://{host}/example.mp4", "taobao"))
        self.assertFalse(video_url_allowed(f"https://{host}/example.mp4", "douyin"))
        for refused in (f"{host}.evil.test", f"evil{host}", f"child.{host}",
                        "cloudvideocdn.taobao.com", "video-bj.cloudvideocdn.taobao.com",
                        f"user:pass@{host}", f"{host}:8080"):
            with self.subTest(host=refused):
                self.assertFalse(video_url_allowed(f"https://{refused}/example.mp4", "taobao"))

    def test_second_measured_taobao_cdn_region_does_not_open_the_region(self):
        """Product 950216355977 exposed video-zb as a real <video> source."""
        host = "video-zb.cloudvideocdn.taobao.com"
        self.assertTrue(video_url_allowed(f"https://{host}/example.mp4", "taobao"))
        self.assertFalse(video_url_allowed(f"https://{host}/example.mp4", "douyin"))
        for refused in (f"{host}.evil.test", f"evil{host}", f"child.{host}",
                        "cloudvideocdn.taobao.com", "video-bj.cloudvideocdn.taobao.com",
                        "video-zj.cloudvideocdn.taobao.com", f"user:pass@{host}", f"{host}:8080"):
            with self.subTest(host=refused):
                self.assertFalse(video_url_allowed(f"https://{refused}/example.mp4", "taobao"))

    async def test_redirect_to_observed_cloud_host_downloads_original_bytes(self):
        target = "https://video-sh.cloudvideocdn.taobao.com/example.mp4"
        calls = []

        def handler(request):
            calls.append(str(request.url))
            if str(request.url) == URL:
                return httpx.Response(302, headers={"location": target})
            self.assertEqual(str(request.url), target)
            return httpx.Response(200, content=b"original video bytes")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await VideoFetcher(client, "taobao", SOURCE).fetch(URL, self.path)
        self.assertEqual(calls, [URL, target])
        self.assertEqual(self.path.read_bytes(), b"original video bytes")

    async def test_observed_cloud_host_cannot_redirect_to_lookalike(self):
        target = "https://video-sh.cloudvideocdn.taobao.com/example.mp4"
        calls = []

        def handler(request):
            calls.append(str(request.url))
            if str(request.url) == URL:
                return httpx.Response(302, headers={"location": target})
            return httpx.Response(302, headers={
                "location": "https://video-sh.cloudvideocdn.taobao.com.evil.test/example.mp4"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaisesRegex(UnsupportedVideoError, "unrecognized_video_host"):
                await VideoFetcher(client, "taobao", SOURCE).fetch(URL, self.path)
        self.assertEqual(calls, [URL, target])
        self.assertFalse(self.path.exists())

    def test_metadata_excludes_attached_cover_and_output_streams(self):
        meta = _InputMetadata()
        for line in ["[info] Input #0, mov, from 'example.mp4':",
                     "[info] Duration: 00:01:02.50, start: 0.0, bitrate: 123 kb/s",
                     "[info] Stream #0:0: Video: mjpeg, yuvj420p, 600x600, 90k tbr (attached pic)",
                     "[info] Stream #0:1: Video: h264, yuv420p, 1920x1080 [SAR 1:1], 30 fps",
                     "[info] Stream #0:2: Audio: aac, 48000 Hz, stereo, fltp, 128 kb/s",
                     "[info] Stream mapping:",
                     "[info] Stream #0:0: Audio: pcm_s16le, 48000 Hz, stereo, s16"]:
            meta.consume(line)
        self.assertEqual(meta.video["width"], 1920)
        self.assertEqual(meta.video["height"], 1080)
        self.assertEqual(meta.video["fps"], 30)
        self.assertEqual(meta.duration, 62.5)
        self.assertEqual(len(meta.audio), 1)
        self.assertEqual(meta.audio[0]["codec"], "aac")


class VideoDecodeTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.exe = ffmpeg_executable()
        except UnsupportedVideoError as exc:
            raise unittest.SkipTest("imageio-ffmpeg runtime not yet installed") from exc

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="buyer_show_decode_")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "原始 小视频.mp4"
        command = [self.exe, "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                   "color=c=red:s=32x24:r=8:d=0.5", "-f", "lavfi", "-i",
                   "sine=frequency=440:duration=0.5", "-c:v", "mpeg4", "-c:a", "aac", str(self.path)]
        subprocess.run(command, check=True, capture_output=True, timeout=15,
                       **({"creationflags": subprocess.CREATE_NO_WINDOW} if hasattr(subprocess, "CREATE_NO_WINDOW") else {}))

    async def test_actual_full_decode_preserves_source_bytes(self):
        original = self.path.read_bytes()
        result = await validate_video(self.path)
        self.assertGreaterEqual(result["frames_decoded"], 4)
        self.assertEqual(result["validation"], "ffmpeg_full_decode")
        self.assertEqual(result["sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(original, self.path.read_bytes())
        self.assertEqual((result["width"], result["height"]), (32, 24))
        self.assertGreater(result["duration_seconds"], 0)
        self.assertTrue(result["audio_present"])
        self.assertEqual(result["audio_stream_count"], 1)
        self.assertEqual(result["audio_validation"], "full_decode")
        self.assertEqual(result["audio_streams"][0]["codec"], "aac")

    async def test_image_or_playlist_renamed_mp4_is_not_video(self):
        out = io.BytesIO()
        Image.new("RGB", (16, 16), "red").save(out, "PNG")
        for payload in (out.getvalue(), b"#EXTM3U\nhttps://private.invalid/segment.ts", b"<html>login</html>",
                        b"\x00\x00\x00\x18ftypavif" + b"\x00" * 20):
            self.path.write_bytes(payload)
            with self.assertRaises(UnsupportedVideoError):
                await validate_video(self.path)

    async def test_broken_container_fails_decode(self):
        self.path.write_bytes(self.path.read_bytes()[:40])
        with self.assertRaisesRegex(VideoError, "video_decode_failed"):
            await validate_video(self.path)

    async def test_missing_decoder_is_explicit(self):
        with patch("buyer_show.video.ffmpeg_executable", side_effect=UnsupportedVideoError("video_decoder_unavailable")):
            with self.assertRaisesRegex(UnsupportedVideoError, "video_decoder_unavailable"):
                await validate_video(self.path)

    async def test_audio_with_attached_cover_is_not_a_video(self):
        cover = self.path.with_name("cover.jpg")
        Image.new("RGB", (32, 32), "green").save(cover)
        only_audio = self.path.with_name("audio_with_cover.mp4")
        command = [self.exe, "-nostdin", "-v", "error", "-i", str(self.path), "-i", str(cover),
                   "-map", "0:a:0", "-map", "1:v:0", "-c", "copy", "-disposition:v:0",
                   "attached_pic", str(only_audio)]
        subprocess.run(command, check=True, capture_output=True, timeout=15,
                       **({"creationflags": subprocess.CREATE_NO_WINDOW} if hasattr(subprocess, "CREATE_NO_WINDOW") else {}))
        with self.assertRaisesRegex(VideoError, "video_decode_failed"):
            await validate_video(only_audio)

    async def test_actual_decoder_cancellation_reaps_child_process(self):
        original_spawn = asyncio.create_subprocess_exec
        children = []

        async def slow_spawn(*args, **kwargs):
            args = list(args)
            pos = args.index("-i")
            args[pos:pos] = ["-stream_loop", "-1", "-re"]  # Bounded by our cancellation, not clip length.
            child = await original_spawn(*args, **kwargs)
            children.append(child)
            return child

        start = time.monotonic()
        with patch("buyer_show.video.asyncio.create_subprocess_exec", side_effect=slow_spawn):
            with self.assertRaises(asyncio.CancelledError):
                await validate_video(self.path, should_stop=lambda: time.monotonic() - start > 0.15)
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].returncode)

    async def test_original_video_download_to_product_store_end_to_end(self):
        from buyer_show.storage import ProductStore
        original = self.path.read_bytes()
        store = ProductStore(Path(self.temp.name) / "output", "taobao", "123", SOURCE)
        store.merge_product_media([{"url": URL, "media_type": "video", "source_type": "product_video"}])
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda _: httpx.Response(200, content=original, headers={"content-type": "video/mp4"}))) as client:
            await store.download_videos(VideoFetcher(client, "taobao", SOURCE).fetch)
        report = store.finish({"status": "completed", "complete": True})
        self.assertTrue(report["complete"])
        self.assertEqual(report["stats"]["unique_videos"], 1)
        self.assertEqual(report["stats"]["reviews_read"], 0)
        saved = store.data["videos"][URL]
        self.assertEqual(saved["audio_stream_count"], 1)
        self.assertEqual((store.root / saved["file"]).read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
