"""Anonymous, bounded media downloads and local full-video decode validation.

Only progressive MP4/MOV and WebM/Matroska are supported. No browser cookies,
HLS/DASH playlists, DRM, or blob extraction is implemented here.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

import httpx

# Provenance, checked 2026-10-29 (domains are matched at label boundaries):
# cloud.video.taobao.com: https://developer.alibaba.com/docs/api.htm?apiId=74174
# alicdn.com video: https://help.aliyun.com/zh/vod/developer-reference/faq-about-apsaravideo-player-sdk
# Douyin measured video CDN entries only, not the project's speculative siblings:
# https://github.com/Evil0ctal/Douyin_TikTok_Download_API/blob/main/src/dtk/media/domains.py
VIDEO_DOMAINS = {"taobao": ("cloud.video.taobao.com", "alicdn.com"),
                 "douyin": ("douyinvod.com", "zjcdn.com"), "xiaohongshu": ()}
# Observed in real product pages' <video> sources, admitted one measured host
# at a time (never all taobao.com hosts or speculative CDN regions):
#   2026-10-30, product 1012082682965: video-sh.cloudvideocdn.taobao.com
#   2026-11-10, product 950216355977: video-zb.cloudvideocdn.taobao.com
#     verified HTTP 200, content-type video/mp4, ISO BMFF `ftypisom` header.
# Unmeasured regions (e.g. video-bj...) and lookalike hosts stay rejected.
VIDEO_EXACT_HOSTS = {"taobao": ("video-sh.cloudvideocdn.taobao.com",
                                "video-zb.cloudvideocdn.taobao.com")}
MAX_VIDEO_BYTES = 256 * 1024 * 1024
DOWNLOAD_TIMEOUT = 240
VALIDATION_TIMEOUT = 120
VIDEO_EXTENSIONS = {"MP4": ".mp4", "MOV": ".mov", "WEBM": ".webm", "MATROSKA": ".mkv"}


class VideoError(ValueError):
    """A stable non-sensitive error code suitable for the manifest."""


class UnsupportedVideoError(VideoError):
    pass


def video_url_allowed(url: str, platform: str) -> bool:
    try:
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        return (parsed.scheme in {"http", "https"} and "@" not in parsed.netloc
                and parsed.port in {None, 80, 443}
                and (host in VIDEO_EXACT_HOSTS.get(platform, ())
                     or any(host == d or host.endswith("." + d) for d in VIDEO_DOMAINS.get(platform, ()))))
    except (TypeError, ValueError):
        return False


def check_video_url(url: str, platform: str):
    if isinstance(url, str) and url.startswith("blob:"):
        raise UnsupportedVideoError("blob_video_unsupported")
    if not video_url_allowed(url, platform):
        raise UnsupportedVideoError("unrecognized_video_host")
    suffix = Path(unquote(urlsplit(url).path)).suffix.lower()
    if suffix in {".m3u8", ".m3u", ".mpd"}:
        raise UnsupportedVideoError("streaming_playlist_unsupported")


def _check_stop(should_stop):
    if should_stop():
        raise asyncio.CancelledError()


async def _interruptible(awaitable, should_stop):
    task = asyncio.ensure_future(awaitable)
    try:
        while not task.done():
            _check_stop(should_stop)
            await asyncio.wait({task}, timeout=0.1)
        return await task
    finally:
        if not task.done():
            task.cancel()
        try:
            await task
        except (Exception, asyncio.CancelledError):
            pass


def _container_kind(header: bytes) -> str:
    stripped = header.lstrip(b"\xef\xbb\xbf \r\n\t")
    if stripped.startswith(b"#EXTM3U") or b"<MPD" in stripped[:256]:
        raise UnsupportedVideoError("streaming_playlist_unsupported")
    if len(header) >= 12 and header[4:8] == b"ftyp":
        if header[8:12] in {b"avif", b"avis", b"mif1", b"msf1", b"heic", b"heix", b"hevc", b"hevx"}:
            raise UnsupportedVideoError("image_container_not_video")
        return "MOV" if header[8:12] == b"qt  " else "MP4"
    if header.startswith(b"\x1a\x45\xdf\xa3"):
        return "WEBM" if b"webm" in header[:4096].lower() else "MATROSKA"
    raise UnsupportedVideoError("unsupported_video_container")


def ffmpeg_executable() -> str:
    """imageio-ffmpeg wheels bundle ffmpeg; ffprobe is deliberately not used."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError) as exc:
        raise UnsupportedVideoError("video_decoder_unavailable") from exc


class _InputMetadata:
    """Read ffmpeg's input header, not filename extensions or a cover URL."""

    def __init__(self):
        self.active = False
        self.done = False
        self.video = {}
        self.audio = []
        self.duration = None

    def consume(self, line):
        if "Input #0," in line:
            self.active = True
        if "Stream mapping:" in line or "Output #0," in line:
            self.done = True
        if not self.active or self.done:
            return
        duration = re.search(r"Duration: (\d+):(\d{2}):(\d{2}(?:\.\d+)?)", line)
        if duration:
            self.duration = int(duration[1]) * 3600 + int(duration[2]) * 60 + float(duration[3])
        if not re.search(r"Stream #0:\d+", line):
            return
        if "Video:" in line and "attached pic" not in line and not self.video:
            size = re.search(r", (\d{1,6})x(\d{1,6})(?:[ ,\[]|$)", line)
            codec = re.search(r"Video: ([A-Za-z0-9_]+)", line)
            fps = re.search(r", ([\d.]+) fps", line)
            if size:
                self.video = {"width": int(size[1]), "height": int(size[2]),
                              "video_codec": codec[1] if codec else None,
                              "fps": float(fps[1]) if fps else None}
        if "Audio:" in line:
            codec = re.search(r"Audio: ([A-Za-z0-9_]+)", line)
            sample_rate = re.search(r", (\d+) Hz", line)
            layout = re.search(r"Hz, ([^,]+)", line)
            self.audio.append({"codec": codec[1] if codec else None,
                "sample_rate": int(sample_rate[1]) if sample_rate else None,
                "channel_layout": layout[1].strip()[:80] if layout else None})


async def validate_video(path: Path, should_stop=lambda: False, *, timeout=VALIDATION_TIMEOUT) -> dict:
    """Decode real video + available audio to null; preserve original file bytes."""
    path = Path(path)
    _check_stop(should_stop)
    size = path.stat().st_size
    if not size or size > MAX_VIDEO_BYTES:
        raise VideoError("empty_or_oversized_video")
    with path.open("rb") as handle:
        kind = _container_kind(handle.read(4096))
    exe = ffmpeg_executable()
    # The container and protocols are restricted before handing untrusted bytes
    # to ffmpeg. A downloaded playlist cannot trigger hidden network requests.
    command = [exe, "-nostdin", "-hide_banner", "-v", "level+info", "-xerror",
               "-err_detect", "explode", "-protocol_whitelist", "file",
               "-format_whitelist", "mov,matroska,webm", "-threads", "2", "-i", str(path.resolve()),
               # Uppercase V excludes attached pictures/thumbnails/cover art.
               "-map", "0:V:0", "-map", "0:a?", "-sn", "-dn", "-threads", "2",
               "-progress", "pipe:1", "-nostats", "-f", "null", "-"]
    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    try:
        proc = await asyncio.create_subprocess_exec(*command, stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, **kwargs)
    except (OSError, NotImplementedError) as exc:
        raise UnsupportedVideoError("video_decoder_unavailable") from exc
    frames, decoded_seconds, has_errors = 0, None, False
    metadata = _InputMetadata()

    async def read_progress():
        nonlocal frames, decoded_seconds
        while line := await proc.stdout.readline():
            if line.startswith(b"frame="):
                try:
                    frames = max(frames, int(line.split(b"=", 1)[1].strip()))
                except ValueError:
                    pass
            if line.startswith(b"out_time_us="):
                try:
                    decoded_seconds = int(line.split(b"=", 1)[1].strip()) / 1_000_000
                except ValueError:
                    pass

    async def read_errors():
        nonlocal has_errors
        pending = b""
        while chunk := await proc.stderr.read(8192):
            pending += chunk
            lines = pending.split(b"\n")
            pending = lines.pop()[-16384:]
            for raw in lines:
                line = raw[:16384].decode("utf-8", errors="replace")
                if "[error]" in line or "[fatal]" in line:
                    has_errors = True
                metadata.consume(line)
        if pending:
            line = pending.decode("utf-8", errors="replace")
            has_errors = has_errors or "[error]" in line or "[fatal]" in line
            metadata.consume(line)

    readers = [asyncio.create_task(read_progress()), asyncio.create_task(read_errors())]
    try:
        async with asyncio.timeout(timeout):
            await _interruptible(proc.wait(), should_stop)
            await asyncio.gather(*readers)
        if proc.returncode != 0 or has_errors or frames < 1 or not metadata.video:
            raise VideoError("video_decode_failed")
    except TimeoutError as exc:
        raise VideoError("video_validation_timeout") from exc
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        for task in readers:
            if not task.done():
                task.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            _check_stop(should_stop)
            digest.update(chunk)
            await asyncio.sleep(0)
    _check_stop(should_stop)
    return {"sha256": digest.hexdigest(), "bytes": size, "format": kind,
            "extension": VIDEO_EXTENSIONS[kind], "frames_decoded": frames,
            "validation": "ffmpeg_full_decode", **metadata.video,
            "duration_seconds": metadata.duration, "decoded_duration_seconds": decoded_seconds,
            "audio_present": bool(metadata.audio), "audio_streams": metadata.audio,
            "audio_stream_count": len(metadata.audio),
            "audio_validation": "full_decode" if metadata.audio else "no_audio_stream"}


class VideoFetcher:
    """Download original bytes to a new temporary file. Storage validates them.

    A supplied httpx client may have unrelated cookies/auth; neither is sent.
    Every redirect is checked, and no browser session data is read.
    """

    def __init__(self, client: httpx.AsyncClient, platform: str, referer: str, *,
                 max_bytes=MAX_VIDEO_BYTES, attempts=2, timeout=DOWNLOAD_TIMEOUT):
        self.client, self.platform, self.referer = client, platform, referer
        self.max_bytes = min(max_bytes, MAX_VIDEO_BYTES)
        self.attempts = max(1, min(attempts, 3))
        self.timeout = timeout

    async def fetch(self, url: str, path: Path, should_stop=lambda: False):
        path = Path(path)
        if path.exists():
            raise FileExistsError("video_destination_already_exists")
        check_video_url(url, self.platform)
        owned = False
        try:
            async with asyncio.timeout(self.timeout):
                for attempt in range(self.attempts):
                    _check_stop(should_stop)
                    try:
                        # Reserve only our own new temporary file, never an existing file.
                        with path.open("xb") as handle:
                            owned = True
                            await self._transfer(url, handle, should_stop)
                        return
                    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                        if owned:
                            path.unlink(missing_ok=True)
                            owned = False
                        # Rate limiting and permanent refusals must not cause an
                        # immediate retry. Match the image downloader's bounded
                        # retry policy for transient transport/server failures.
                        retryable = (isinstance(exc, (httpx.ConnectError, httpx.ReadError,
                                     httpx.WriteError, httpx.RemoteProtocolError, httpx.TimeoutException))
                                     or isinstance(exc, httpx.HTTPStatusError)
                                     and exc.response.status_code in {500, 502, 503, 504})
                        if not retryable or attempt + 1 >= self.attempts:
                            raise
                        await _interruptible(asyncio.sleep(0.5 * (attempt + 1)), should_stop)
        except BaseException:
            if owned:
                path.unlink(missing_ok=True)
            raise

    async def __call__(self, url: str, path: Path, should_stop=lambda: False):
        return await self.fetch(url, path, should_stop)

    async def _transfer(self, url, handle, should_stop):
        for _ in range(5):
            _check_stop(should_stop)
            check_video_url(url, self.platform)
            request = self.client.build_request("GET", url, headers={"Referer": self.referer,
                "Accept-Encoding": "identity"}, timeout=httpx.Timeout(20.0, connect=10.0))
            for name in ("cookie", "authorization", "proxy-authorization"):
                request.headers.pop(name, None)
            response = await _interruptible(self.client.send(request, stream=True,
                                      follow_redirects=False, auth=None), should_stop)
            try:
                _check_stop(should_stop)
                if response.is_redirect:
                    target = response.headers.get("location")
                    if not target:
                        raise VideoError("redirect_without_location")
                    url = str(response.url.join(target))
                    continue
                response.raise_for_status()
                if response.status_code != 200:
                    raise VideoError("incomplete_video_response")
                media_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                if "mpegurl" in media_type or "dash+xml" in media_type:
                    raise UnsupportedVideoError("streaming_playlist_unsupported")
                if media_type.startswith(("image/", "text/")) or media_type in {"application/json", "application/xml"}:
                    raise UnsupportedVideoError("non_video_response")
                expected = response.headers.get("content-length")
                try:
                    expected = int(expected) if expected is not None else None
                except ValueError as exc:
                    raise VideoError("invalid_content_length") from exc
                if expected is not None and (expected <= 0 or expected > self.max_bytes):
                    raise VideoError("empty_or_oversized_video")
                if response.headers.get("content-encoding", "identity").lower() not in {"", "identity"}:
                    raise UnsupportedVideoError("encoded_video_response_unsupported")
                size = 0
                iterator = response.aiter_bytes(chunk_size=64 * 1024).__aiter__()
                while True:
                    try:
                        chunk = await _interruptible(iterator.__anext__(), should_stop)
                    except StopAsyncIteration:
                        break
                    _check_stop(should_stop)
                    size += len(chunk)
                    if size > self.max_bytes:
                        raise VideoError("video_size_limit_exceeded")
                    handle.write(chunk)
                if not size or (expected is not None and expected != size):
                    raise VideoError("incomplete_video_response")
                return
            finally:
                await response.aclose()
        raise VideoError("too_many_redirects")
