"""Small URL guards shared by media parsers; no network access."""

from __future__ import annotations

import html
import re
from urllib.parse import urlsplit


def media_url(value, media_type="image") -> str:
    """Keep observed HTTP URLs byte-for-byte apart from HTML escaping.

    A URL's extension cannot prove its content, but obvious image/poster URLs
    cannot be video payloads. The downloader must still validate every file.
    """
    if not isinstance(value, str):
        return ""
    value = html.unescape(value.strip())
    if value.startswith("//"):
        value = "https:" + value
    try:
        parsed = urlsplit(value)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password
                or re.search(r"[\x00-\x20\x7f]", value)):
            return ""
        if media_type == "video" and re.search(
            r"\.(?:jpe?g|png|webp|gif|avif|svg|bmp|ico)(?:$|[/_!])", parsed.path, re.I
        ):
            return ""
    except (TypeError, ValueError):
        return ""
    return value


def video_urls(values) -> list[str]:
    """Normalize a known review-local URL list, never recurse or read covers."""
    if not isinstance(values, list):
        return []
    return list(dict.fromkeys(url for value in values if (url := media_url(value, "video"))))
