"""Verified media files and resumable, product-scoped manifests."""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from PIL import Image

from .video import UnsupportedVideoError, VideoError, check_video_url, validate_video

IMAGE_DOMAINS = {
    "taobao": ("alicdn.com", "taobaocdn.com"),
    "douyin": ("douyinpic.com", "ecombdimg.com", "byteimg.com", "ibytedtos.com", "pstatp.com", "toutiaoimg.com"),
    "xiaohongshu": ("xhscdn.com",),
}
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_VIDEO_SOURCES = 4
EXTENSIONS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif", "BMP": ".bmp", "AVIF": ".avif"}

# Only our own fixed codes may reach reports/logs. Exception messages can
# contain signed URLs, server response bodies or authentication information.
MEDIA_ERROR_REASONS = {
    "unrecognized_image_host": "图片来源域名未获支持",
    "unrecognized_video_host": "视频来源域名未获支持",
    "blob_video_unsupported": "临时 blob 视频地址暂不支持下载",
    "streaming_playlist_unsupported": "分段流视频暂不支持下载",
    "video_decoder_unavailable": "视频解码组件不可用",
    "video_decode_failed": "视频未通过完整解码校验",
    "video_validation_timeout": "视频解码校验超时",
    "non_video_response": "服务端返回的内容不是视频",
    "image_container_not_video": "返回的是图片文件，不是视频",
    "unsupported_video_container": "视频封装格式暂不支持",
    "encoded_video_response_unsupported": "视频响应编码暂不支持",
    "empty_or_oversized_video": "视频为空或超过大小上限",
    "video_size_limit_exceeded": "视频超过大小上限",
    "incomplete_video_response": "视频响应不完整",
    "invalid_content_length": "服务端响应长度无效",
    "image_exceeds_32MiB": "图片超过大小上限",
    "empty_or_oversized_image": "图片为空或超过大小上限",
    "unsupported_image_format": "图片格式暂不支持",
    "redirect_without_location": "服务端重定向缺少目标地址",
    "too_many_redirects": "服务端重定向次数过多",
    "network_timeout": "网络请求超时",
    "network_error": "网络连接异常",
    "invalid_image": "图片未通过内容校验",
    "file_error": "本地文件读写失败",
    "unsupported_media": "该素材暂不支持下载",
    "download_error": "下载或内容校验失败",
    "cancelled": "下载已取消",
}


def _failure_code(exc):
    if isinstance(exc, httpx.HTTPStatusError):
        return {"error_code": "http_error", "http_status": exc.response.status_code}
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        code = "network_timeout"
    elif isinstance(exc, httpx.RequestError):
        code = "network_error"
    elif isinstance(exc, ValueError) and str(exc) in MEDIA_ERROR_REASONS:
        code = str(exc)
    elif isinstance(exc, UnsupportedVideoError):
        code = "unsupported_media"
    elif isinstance(exc, Image.UnidentifiedImageError):
        code = "invalid_image"
    elif isinstance(exc, OSError):
        code = "file_error"
    else:
        code = "download_error"
    return {"error_code": code}


def _video_sources(canonical, *groups):
    """Bound native alternatives, keeping a single canonical association."""
    result = [canonical]
    for group in groups:
        if not isinstance(group, list):
            continue
        for value in group[:32]:
            if isinstance(value, str) and value and len(value) <= 8192 and value not in result:
                result.append(value)
                if len(result) == MAX_VIDEO_SOURCES:
                    return result
    return result


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    # Windows may briefly deny replacing a recently read manifest. Keep the
    # previous committed file intact and retry the same complete temporary file.
    for attempt in range(5):
        try:
            temp.replace(path)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(.05 * (attempt + 1))


def image_url_allowed(url: str, platform: str) -> bool:
    try:
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        return (parsed.scheme in {"http", "https"} and not parsed.username and not parsed.password
                and parsed.port in {None, 80, 443}
                and any(host == d or host.endswith("." + d) for d in IMAGE_DOMAINS.get(platform, ())))
    except (ValueError, TypeError):
        return False


def validate_image(data: bytes) -> dict:
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("empty_or_oversized_image")
    with Image.open(io.BytesIO(data)) as img:
        if img.format not in EXTENSIONS:
            raise ValueError("unsupported_image_format")
        kind, size = img.format, img.size
        img.verify()
    with Image.open(io.BytesIO(data)) as img:
        img.load()  # A header alone, or a truncated response, is not success.
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
            "width": size[0], "height": size[1], "format": kind,
            "extension": EXTENSIONS[kind]}


class ImageFetcher:
    """Anonymous CDN GETs. Browser login cookies are never exported here."""

    def __init__(self, client: httpx.AsyncClient, platform: str, referer: str):
        self.client, self.platform, self.referer = client, platform, referer

    async def __call__(self, url: str) -> bytes:
        # A dropped keep-alive connection should not require restarting a whole
        # product. 429 is deliberately not retried automatically.
        async with asyncio.timeout(90):
            for attempt in range(3):
                try:
                    return await self._fetch_once(url)
                except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                    transient = (isinstance(exc, (httpx.ConnectError, httpx.ReadError, httpx.WriteError,
                        httpx.RemoteProtocolError, httpx.TimeoutException)) or
                        isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in {500, 502, 503, 504})
                    if not transient or attempt == 2:
                        raise
                    await asyncio.sleep(0.3 * (attempt + 1))

    async def _fetch_once(self, url: str) -> bytes:
        for _ in range(5):
            if not image_url_allowed(url, self.platform):
                raise ValueError("unrecognized_image_host")
            request = self.client.build_request("GET", url, headers={"Referer": self.referer},
                                               timeout=httpx.Timeout(20, connect=10))
            for name in ("cookie", "authorization", "proxy-authorization"):
                request.headers.pop(name, None)
            res = await self.client.send(request, follow_redirects=False, stream=True, auth=None)
            try:
                if res.is_redirect:
                    target = res.headers.get("location")
                    if not target:
                        raise ValueError("redirect_without_location")
                    url = str(res.url.join(target))
                    continue
                res.raise_for_status()
                chunks, size = [], 0
                async for chunk in res.aiter_bytes():
                    size += len(chunk)
                    if size > MAX_IMAGE_BYTES:
                        raise ValueError("image_exceeds_32MiB")
                    chunks.append(chunk)
                return b"".join(chunks)
            finally:
                await res.aclose()
        raise ValueError("too_many_redirects")


class ProductStore:
    def __init__(self, output: Path, platform: str, product_id: str, source_url: str):
        if platform not in IMAGE_DOMAINS or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", product_id):
            raise ValueError("invalid_product_identity")
        self.root = Path(output) / platform / product_id
        self.path = self.root / "manifest.json"
        self.root.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("platform") != platform or data.get("product_id") != product_id:
                raise ValueError("manifest_product_mismatch")
            self.data = data
        else:
            self.data = {"schema_version": 1, "platform": platform, "product_id": product_id,
                         "source_urls": [], "created_at": now(), "records": [], "images": {}, "attempts": []}
        # Additive migration: schema-1 image-only manifests remain readable.
        self.data.setdefault("videos", {})
        self.data.setdefault("product_media", [])
        for record in self.data["records"]:
            record.setdefault("videos", [])
        if source_url not in self.data["source_urls"]:
            self.data["source_urls"].append(source_url)

    def checkpoint(self):
        self.data["updated_at"] = now()
        self.data["stats"] = self.stats()
        atomic_json(self.path, self.data)

    def begin(self):
        # Prior downloaded flags have not yet been revalidated in this attempt.
        # An interrupted rerun must never leave an old complete=true claim.
        self.data.update(status="running", complete=False, collected_complete=False,
                         reason="本次任务正在进行，完整性尚未重新确认。")
        self.checkpoint()

    def merge(self, records):
        lookup = {(str(r["review_id"]), r["review_type"]): r for r in self.data["records"]}
        for row in records:
            if row.get("platform") != self.data["platform"] or str(row.get("product_id")) != self.data["product_id"]:
                raise ValueError("review_product_mismatch")
            if row.get("review_type") not in {"initial", "append"} or not row.get("review_id"):
                raise ValueError("invalid_review_identity")
            key = (str(row["review_id"]), row["review_type"])
            target = lookup.setdefault(key, {"review_id": key[0], "review_type": key[1], "content": "", "images": [], "videos": []})
            content = str(row.get("content", ""))
            if content and content not in target["content"]:
                target["content"] = "\n".join(filter(None, [target["content"], content]))
            for field in ("images", "videos"):
                values = row.get(field, [])
                if not isinstance(values, list):
                    raise ValueError("invalid_media_list")
                for url in values:
                    if not isinstance(url, str) or not url:
                        raise ValueError("invalid_media_url")
                    if url not in target[field]:
                        target[field].append(url)
                    item = self.data[field].setdefault(url, {"status": "pending"})
                    if field == "videos":
                        sources = row.get("video_sources", {})
                        candidates = sources.get(url, []) if isinstance(sources, dict) else []
                        # Only keys already present in this row's videos can
                        # contribute candidates. Alternatives never create a
                        # second logical video or review association.
                        item["source_candidates"] = _video_sources(url, item.get("source_candidates"), candidates)
        self.data["records"] = list(lookup.values())
        self._mark_categories_stale()
        self.checkpoint()

    def _mark_categories_stale(self):
        if "classification" in self.data:
            self.data["classification"].update(complete=False, needs_refresh=True)

    def merge_product_media(self, items):
        """Product gallery/detail associations are not fabricated reviews."""
        known = {(m["url"], m["media_type"], m["source_type"]) for m in self.data["product_media"]}
        for media in items:
            kind, source = media.get("media_type"), media.get("source_type")
            url = media.get("url")
            if kind not in {"image", "video"} or source not in {"product_main", "product_detail", "product_video"}:
                raise ValueError("invalid_product_media_type")
            if not isinstance(url, str) or not url:
                raise ValueError("invalid_media_url")
            if source == "product_video" and kind != "video":
                raise ValueError("product_video_requires_video")
            key = (url, kind, source)
            if key not in known:
                self.data["product_media"].append({"url": url, "media_type": kind, "source_type": source})
                known.add(key)
            self.data["images" if kind == "image" else "videos"].setdefault(url, {"status": "pending"})
        self._mark_categories_stale()
        self.checkpoint()

    def _existing_valid(self, item):
        if item.get("status") not in {"downloaded", "excluded"}:
            return False
        relative = str(item.get("file", ""))
        # Paths in a previous manifest must also stay inside this product folder.
        if not re.fullmatch(r"images/[a-f0-9]{64}\.(?:jpg|png|webp|gif|bmp|avif)", relative):
            return False
        try:
            info = validate_image((self.root / relative).read_bytes())
            if info["sha256"] != item.get("sha256"):
                return False
            # Inspect the decoded file, not old manifest dimensions. Keep the
            # original bytes even when a legacy successful item is a 1px pixel.
            excluded = info["width"] == 1 and info["height"] == 1
            status = "excluded" if excluded else "downloaded"
            if item.get("status") != status:
                self._mark_categories_stale()
            info.pop("extension")
            item.update(status=status, **info)
            if excluded:
                item["exclusion_reason"] = "one_pixel_placeholder"
            else:
                item.pop("exclusion_reason", None)
            return True
        except (OSError, ValueError, Image.DecompressionBombError):
            return False

    async def download(self, fetch, emit=lambda message: None, should_stop=lambda: False, skip_urls=()):
        images = self.data["images"]
        for index, (url, item) in enumerate(images.items(), 1):
            if should_stop():
                break
            if url in skip_urls:
                continue
            if self._existing_valid(item):
                continue
            try:
                if not image_url_allowed(url, self.data["platform"]):
                    raise ValueError("unrecognized_image_host")
                payload = await fetch(url)
                info = validate_image(payload)
                relative = "images/" + info["sha256"] + info.pop("extension")
                path = self.root / relative
                path.parent.mkdir(exist_ok=True)
                # Duplicate image bytes use one file; every review association remains.
                if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
                    temp = path.with_suffix(path.suffix + ".part")
                    temp.write_bytes(payload)
                    temp.replace(path)
                item.clear()
                if info["width"] == 1 and info["height"] == 1:
                    item.update(status="excluded", exclusion_reason="one_pixel_placeholder", file=relative, **info)
                    emit(f"图片 {index}/{len(images)} 为 1×1 占位像素，已排除出素材队列并保留原件")
                else:
                    item.update(status="downloaded", file=relative, **info)
                    emit(f"图片 {index}/{len(images)} 已校验保存（{info['width']}×{info['height']}）")
            except Exception as exc:
                item.clear()
                item.update(status="failed", error=type(exc).__name__, **_failure_code(exc))
                emit(f"图片 {index}/{len(images)} 下载失败，已记录，可重新运行重试")
            self.checkpoint()
        self.checkpoint()

    async def _existing_video_valid(self, item, should_stop):
        if item.get("status") != "downloaded":
            return False
        relative = str(item.get("file", ""))
        if not re.fullmatch(r"videos/[a-f0-9]{64}\.(?:mp4|mov|webm|mkv)", relative):
            return False
        try:
            info = await validate_video(self.root / relative, should_stop)
            return info["sha256"] == item.get("sha256")
        except UnsupportedVideoError:
            raise  # Missing decoder is a truthful unsupported result, not success.
        except (OSError, VideoError):
            return False

    async def download_videos(self, fetch, emit=lambda message: None, should_stop=lambda: False, skip_urls=()):
        """Try native sources for one video; a source is successful only after decode."""
        videos = self.data["videos"]
        for index, (url, item) in enumerate(videos.items(), 1):
            if should_stop():
                break
            if url in skip_urls:
                continue
            previous_download = dict(item) if item.get("status") == "downloaded" else None
            new_download_started = False
            candidates = _video_sources(url, item.get("source_candidates"))
            item["source_candidates"] = candidates
            attempt = None
            try:
                if await self._existing_video_valid(item, should_stop):
                    item.setdefault("selected_source", url)
                    continue
                new_download_started = True
                item.clear()
                item.update(status="pending", source_candidates=candidates, source_attempts=[])
                failures = []
                for candidate_index, candidate in enumerate(candidates):
                    if should_stop():
                        raise asyncio.CancelledError()
                    attempt = {"candidate_index": candidate_index, "status": "pending"}
                    item["source_attempts"].append(attempt)
                    try:
                        # Check even with custom fetchers; VideoFetcher separately
                        # checks every redirect before making the next request.
                        check_video_url(candidate, self.data["platform"])
                        info = await self._save_video_candidate(fetch, candidate, should_stop)
                        attempt["status"] = "downloaded"
                        item.update(status="downloaded", selected_source=candidate, **info)
                        emit(f"视频 {index}/{len(videos)} 已通过完整解码校验并保存原文件")
                        break
                    except Exception as exc:
                        failure = _failure_code(exc)
                        failure["status"] = "unsupported" if isinstance(exc, UnsupportedVideoError) else "failed"
                        attempt.update(failure)
                        failures.append(failure)
                        self.checkpoint()
                        # A missing local decoder cannot be fixed by another URL.
                        # Rate limiting also stops immediate alternate requests.
                        if failure["error_code"] == "video_decoder_unavailable" or failure.get("http_status") == 429:
                            break
                if item["status"] != "downloaded":
                    # Preserve all attempts; prefer the actual server refusal to
                    # an unsupported subsequent candidate in the short report.
                    failure = next((f for f in failures if "http_status" in f), failures[-1])
                    item.update(failure, error=failure["error_code"])
                    emit(f"视频 {index}/{len(videos)} 未保存成功，已记录各候选源的安全错误代码")
            except asyncio.CancelledError:
                # Previously saved files remain intact; this URL can be retried.
                if previous_download is not None and not new_download_started:
                    # Cancelling revalidation does not erase an already saved
                    # file's history. The enclosing run still finishes cancelled.
                    item.clear()
                    item.update(previous_download)
                else:
                    if attempt is not None and attempt["status"] == "pending":
                        attempt.update(status="cancelled", error_code="cancelled")
                    item.update(status="pending", error="cancelled", error_code="cancelled")
                self.checkpoint()
                break
            except Exception as exc:
                item.update(status="unsupported" if isinstance(exc, UnsupportedVideoError) else "failed",
                            **_failure_code(exc))
                item["error"] = item["error_code"]
                emit(f"视频 {index}/{len(videos)} 未保存成功，已记录错误代码")
            self.checkpoint()
        self.checkpoint()

    async def _save_video_candidate(self, fetch, url, should_stop):
        directory = self.root / "videos"
        directory.mkdir(exist_ok=True)
        temporary = directory / (".download-" + uuid4().hex + ".part")
        try:
            await fetch(url, temporary, should_stop)
            if should_stop():
                raise asyncio.CancelledError()
            info = await validate_video(temporary, should_stop)
            relative = "videos/" + info["sha256"] + info.pop("extension")
            final = self.root / relative
            # All sources of the logical video share one SHA-addressed file;
            # original bytes and audio are retained without re-encoding.
            replace = not final.exists()
            if not replace:
                digest = hashlib.sha256()
                with final.open("rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        if should_stop():
                            raise asyncio.CancelledError()
                        digest.update(chunk)
                        await asyncio.sleep(0)
                replace = digest.hexdigest() != info["sha256"]
            if should_stop():
                raise asyncio.CancelledError()
            if replace:
                temporary.replace(final)
            return {"file": relative, **info}
        finally:
            temporary.unlink(missing_ok=True)

    def refresh_categories(self, should_stop=lambda: False, emit=lambda message: None):
        """Copy verified originals into source-aware folders without replacing edits."""
        from .media_catalog import refresh_categories
        result = refresh_categories(self.root, self.data, should_stop, emit)
        self.data["classification"]["updated_at"] = now()
        if not result["complete"]:
            self.data["complete"] = False
            if self.data.get("status") == "completed":
                self.data.update(status="partial", reason="原件已保存，素材分类存在未完成项，请查看分类记录。")
        self.checkpoint()
        return result

    def stats(self):
        rows, images = self.data["records"], self.data["images"]
        videos = self.data["videos"]
        successful = [v for v in images.values() if v.get("status") == "downloaded"]
        videos_successful = [v for v in videos.values() if v.get("status") == "downloaded"]
        categories = self.data.get("classification", {}).get("summary", {})
        return {"reviews_read": len({r["review_id"] for r in rows}), "review_entries": len(rows),
                "reviews_with_images": len({r["review_id"] for r in rows if r["images"]}),
                "image_associations": sum(len(r["images"]) for r in rows),
                "image_urls": len(images), "unique_images": len({v["sha256"] for v in successful}),
                "downloads_succeeded": len(successful),
                "images_excluded": sum(v.get("status") == "excluded" for v in images.values()),
                "downloads_failed": sum(v.get("status") == "failed" for v in images.values()),
                "downloads_pending": sum(v.get("status") == "pending" for v in images.values()),
                "reviews_with_videos": len({r["review_id"] for r in rows if r.get("videos")}),
                "video_associations": sum(len(r.get("videos", [])) for r in rows),
                "video_urls": len(videos),
                "unique_videos": len({v["sha256"] for v in videos_successful}),
                "video_downloads_succeeded": len(videos_successful),
                "videos_failed": sum(v.get("status") in {"failed", "unsupported"} for v in videos.values()),
                "videos_pending": sum(v.get("status") == "pending" for v in videos.values()),
                "videos_unsupported": sum(v.get("status") == "unsupported" for v in videos.values()),
                "product_media_count": len(self.data["product_media"]),
                "product_image_associations": sum(m["media_type"] == "image" for m in self.data["product_media"]),
                "product_video_associations": sum(m["media_type"] == "video" for m in self.data["product_media"]),
                "category_files": categories.get("created", 0) + categories.get("reused", 0),
                "categories_failed": categories.get("failed", 0),
                "categories_pending": categories.get("pending", 0),
                "category_user_edits_preserved": categories.get("preserved_user_modified", 0)}

    def _download_failures(self):
        """Summarize logical media by provenance without URLs or exception text.

        One file can belong to several provenance groups; within each group it
        is counted once, independently of review duplicates or source attempts.
        """
        associations = {}
        labels = {"product_main": "商品主图", "product_detail": "详情图", "product_video": "商品视频",
                  "initial": "初评买家秀", "append": "追评买家秀", "unknown": "未分类素材"}
        for media in self.data["product_media"]:
            source = media.get("source_type")
            if source in {"product_main", "product_detail", "product_video"}:
                associations.setdefault((media.get("media_type"), media.get("url")), set()).add((source, ""))
        for row in self.data["records"]:
            kind = row.get("review_type")
            if kind in {"initial", "append"}:
                for field, media_type in (("images", "image"), ("videos", "video")):
                    for url in row.get(field, []):
                        associations.setdefault((media_type, url), set()).add(("review", kind))
        groups, failed_count, pending_count = {}, 0, 0
        for field, media_type in (("images", "image"), ("videos", "video")):
            for url, item in self.data[field].items():
                state = item.get("status")
                if state == "pending":
                    pending_count += 1
                if state not in {"failed", "unsupported"}:
                    continue
                failed_count += 1
                code = item.get("error_code", item.get("error", ""))
                status = item.get("http_status")
                status = status if isinstance(status, int) and not isinstance(status, bool) and 100 <= status <= 599 else None
                if status is not None:
                    code = "http_error"
                    if status == 429:
                        reason = "服务端限流，未立即重试"
                    elif status in {404, 410}:
                        reason = "资源不存在或已不可用"
                    elif status >= 500:
                        reason = "服务端暂时异常"
                    else:
                        reason = "服务器拒绝下载"
                    reason += f"（HTTP {status}）"
                else:
                    if code in {"RemoteProtocolError", "ConnectError", "ReadError", "WriteError"}:
                        code = "network_error"
                    elif code in {"TimeoutError", "ReadTimeout", "ConnectTimeout", "WriteTimeout", "PoolTimeout"}:
                        code = "network_timeout"
                    elif code == "UnidentifiedImageError":
                        code = "invalid_image"
                    if code not in MEDIA_ERROR_REASONS:
                        code = "unsupported_media" if state == "unsupported" else "download_error"
                    reason = MEDIA_ERROR_REASONS[code]
                sources = associations.get((media_type, url), {("unknown", "")})
                for source, review_type in sorted(sources):
                    key = (source, review_type, media_type, state, code, status)
                    if key not in groups:
                        groups[key] = {"source_type": source, "review_type": review_type or None,
                            "source_label": labels[review_type or source], "media_type": media_type,
                            "status": state, "error_code": code, "http_status": status, "count": 0, "reason": reason}
                    groups[key]["count"] += 1
        failures = list(groups.values())
        parts = []
        for group in failures[:6]:
            image = group["media_type"] == "image"
            label = group["source_label"]
            if group["source_type"] in {"review", "unknown"}:
                label += "图片" if image else "视频"
            elif not image and group["source_type"] != "product_video":
                label = "商品详情视频" if group["source_type"] == "product_detail" else "商品主图视频"
            parts.append(f"{label} {group['count']}{'张' if image else '段'}：{group['reason']}")
        if len(failures) > 6:
            parts.append(f"另有 {len(failures) - 6} 类失败，详见清单")
        summary = (f"{failed_count} 项素材未保存。" + "；".join(parts)) if failed_count else ""
        if pending_count:
            summary += ("；" if summary else "") + f"另有 {pending_count} 项素材待下载或校验"
        return failures, summary

    def finish(self, collection: dict, cancelled=False):
        stats = self.stats()
        download_failures, download_failure_summary = self._download_failures()
        collected_complete = collection.get("complete") is True and collection.get("status") == "completed"
        complete = (collected_complete and stats["downloads_failed"] == 0 and stats["downloads_pending"] == 0
                    and stats["videos_failed"] == 0 and stats["videos_pending"] == 0 and not cancelled
                    and self.data.get("classification", {}).get("complete", True))
        status = "completed" if complete else ("cancelled" if cancelled else collection.get("status", "partial"))
        if status == "completed" and not complete:
            status = "partial_download"
        self.data.update(status=status, complete=complete, collected_complete=collected_complete,
                         reason=collection.get("reason", ""), diagnostics=collection.get("diagnostics", {}),
                         download_failures=download_failures, download_failure_summary=download_failure_summary)
        self.data["attempts"].append({"at": now(), "collection_status": collection.get("status"),
                                      "collected_complete": collected_complete, "status": status, "stats": stats})
        self.checkpoint()
        return {"platform": self.data["platform"], "product_id": self.data["product_id"],
                "status": status, "complete": complete, "stats": stats, "manifest": str(self.path.resolve()),
                "reason": self.data["reason"], "download_failures": download_failures,
                "download_failure_summary": download_failure_summary}
