"""Conservative Douyin *product review* adapter.

Selector reference (read, not executed, 2026-10-29):
https://github.com/yufeiyang111/douyin-mall-comments/blob/main/scripts/scraper.py
The upstream selectors describe the desktop product popup, not a verified mobile
share-page API. This adapter deliberately reports unsupported pages instead of
collecting arbitrary product, avatar, or video-comment images.

parse_reviews accepts explicit product-review lists (reviews/review_list), the
upstream CSV-style rows (imgs JSON array), or this collector's DOM snapshot.
This is a normalization contract, NOT a claim about an undocumented JSON API.
URLs are preserved as displayed; neither original resolution nor completeness is
inferred from a URL or from a stalled scroll.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import parse_qs, urlsplit

from ._media import video_urls


SELECTOR_SOURCE = "https://github.com/yufeiyang111/douyin-mall-comments/blob/main/scripts/scraper.py"
REVIEW_ITEM = ".Ez3MC6d7"
REVIEW_IMAGE = ".qZmIVPuL img"
_DOM_SCHEMA = "douyin_product_reviews_dom_v1"


def product_id_from_url(url: str) -> str:
    """Read a product identifier only from a supported platform/product route."""
    try:
        parsed = urlsplit(url)
    except (TypeError, ValueError):
        return ""
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or not any(
        host == domain or host.endswith("." + domain)
        for domain in ("jinritemai.com", "douyin.com")
    ):
        return ""
    if re.search(r"/(video|note|user|search)(/|$)", parsed.path):
        return ""
    route = re.search(r"/product/(\d+)(?:/|$)", parsed.path)
    if route:
        return route.group(1)
    if parsed.path.rstrip("/") not in (
        "/ecommerce/trade/detail/index.html", "/views/product/item2",
    ):
        return ""
    query = parse_qs(parsed.query)
    for key in ("id", "product_id"):
        value = query.get(key, [""])[0]
        if value.isdigit():
            return value
    return ""


def _url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = html.unescape(value.strip())
    if value.startswith("//"):
        value = "https:" + value
    try:
        parsed = urlsplit(value)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.username or parsed.password):
            return ""
    except ValueError:
        return ""
    # A secondary guard; the primary protection is review-local image selection.
    if re.search(r"avatar|regular_customer|(?:^|[/_.-])(?:icon|emoji|sticker)(?:[/_.-]|$)",
                 parsed.path, re.I):
        return ""
    return value


def _images(row: Mapping[str, Any]) -> list[str]:
    value = row.get("imgs", row.get("images", []))
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return []
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for image in value:
        if isinstance(image, Mapping):
            # Only a direct image object. Never recurse into user/product/reply.
            image = image.get("url", image.get("src", ""))
        candidate = _url(image)
        if candidate and candidate not in result:
            result.append(candidate)
    return result


def _videos(row: Mapping[str, Any]) -> list[str]:
    # video_urls is the upstream product-review CSV field; videos is our DOM
    # normalization field. Do not use poster/cover/video_id as media payloads.
    value = row.get("videos", row.get("video_urls", []))
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return []
    return video_urls(value)


def _rows(payload: Any, inherited_id: str = ""):
    """Walk only explicit envelope keys, never arbitrary nested dictionaries."""
    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, Mapping):
                yield row, inherited_id
        return
    if not isinstance(payload, Mapping):
        return
    if any(key in payload for key in ("aweme_id", "aweme_detail", "aweme_list")):
        return
    parent_id = str(payload.get("product_id") or inherited_id)
    for key in ("reviews", "review_list"):
        if isinstance(payload.get(key), list):
            yield from _rows(payload[key], parent_id)
    # The documented upstream CSV shape is also accepted as a single row.
    if any(key in payload for key in ("imgs", "images", "videos", "video_urls")) and (
        "content" in payload or "review_id" in payload
    ) and not any(key in payload for key in ("reviews", "review_list")):
        yield payload, parent_id
    if isinstance(payload.get("data"), Mapping):
        yield from _rows(payload["data"], parent_id)


def parse_reviews(payload: Any, source_url: str) -> list[dict]:
    """Normalize review-owned images, preserving signed URLs and review grouping.

    A product link supplies the product ID. A resolved short link can instead
    supply product_id in a normalized product-review envelope. Video comment
    objects (aweme_id/cid/image_list) are deliberately rejected.
    """
    if not isinstance(source_url, str):
        return []
    if isinstance(payload, (str, bytes, bytearray)):
        try:
            payload = json.loads(payload)
        except (ValueError, TypeError, UnicodeError):
            return []
    try:
        source = urlsplit(source_url)
    except (TypeError, ValueError):
        return []
    if re.search(r"/(video|note|user|search)(/|$)", source.path):
        return []
    source_id = product_id_from_url(source_url)
    source_host = (source.hostname or "").lower()
    allowed_short = source.scheme in ("http", "https") and source_host == "v.douyin.com"
    if not source_id and not allowed_short:
        return []
    records: dict[tuple[str, str, str], dict] = {}
    for row, inherited_id in _rows(payload, source_id):
        if any(key in row for key in ("aweme_id", "cid", "image_list")):
            continue
        product_id = str(row.get("product_id") or inherited_id or source_id)
        if not product_id.isdigit() or (source_id and product_id != source_id):
            continue
        if "content" not in row and "review_id" not in row and not _videos(row):
            continue
        content = row.get("content", "")
        if not isinstance(content, str):
            continue
        # The parent evaluation owns both initial and appended content. Derive
        # its fallback ID once, so changing/adding an append does not relink it.
        parent_id = str(row.get("parent_review_id") or row.get("review_id") or "")
        if not parent_id:
            fingerprint = [product_id, row.get("nickname", ""), row.get("time", ""),
                           row.get("spec", ""), content, _images(row), _videos(row)]
            digest = hashlib.sha256(json.dumps(fingerprint, ensure_ascii=False,
                                               sort_keys=True).encode()).hexdigest()
            parent_id = "synthetic:" + digest[:24]
        variants = [(row, "append" if row.get("review_type") == "append" else "initial")]
        if isinstance(row.get("append_review"), Mapping):
            variants.append((row["append_review"], "append"))
        for review, review_type in variants:
            content = review.get("content", "")
            if not isinstance(content, str):
                continue
            images = _images(review)
            videos = _videos(review)
            review_id = parent_id
            key = product_id, review_id, review_type
            if key in records:
                records[key]["images"] = list(dict.fromkeys(records[key]["images"] + images))
                records[key]["videos"] = list(dict.fromkeys(records[key]["videos"] + videos))
                continue
            records[key] = {"platform": "douyin", "product_id": product_id,
                            "review_id": review_id, "review_type": review_type,
                            "content": content.strip(), "images": images, "videos": videos}
    return list(records.values())


# Only the verified review-item and review-media selectors are used for images.
_SNAPSHOT_JS = r"""() => {
  const visible = e => !!(e && e.getClientRects().length && getComputedStyle(e).visibility !== 'hidden');
  // Text belonging to a buyer (including their replies) cannot classify the UI.
  const uiParts = [];
  if (document.body) {
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const node = walker.currentNode;
      const parent = node.parentElement;
      if (visible(parent) && !parent.closest('.Ez3MC6d7') && !parent.closest('script, style, noscript')) {
        const value = (node.textContent || '').trim();
        if (value) uiParts.push(value);
      }
    }
  }
  const items = [...document.querySelectorAll('.Ez3MC6d7')].filter(visible);
  const first = items[0];
  let root = first?.closest('.eRfaxg7R') || null;
  if (!root) {
    for (let p = first?.parentElement; p && p !== document.body && p !== document.documentElement; p = p.parentElement) {
      if (/(auto|scroll|overlay)/.test(getComputedStyle(p).overflowY) && p.clientHeight > 100) { root = p; break; }
    }
  }
  if (!root) root = [...document.querySelectorAll('.eRfaxg7R')].find(visible) || null;
  const read = (e, s) => e.querySelector(s)?.innerText?.trim() || '';
  let unresolvedVideos = 0, ambiguousAppends = 0;
  const reviews = items.map(item => {
    // Upstream has no independently verified appended-review container. A
    // visible append marker makes attribution ambiguous: retain the row but
    // never attach all its media to the initial review as a guessed fallback.
    const appendAmbiguous = [...item.querySelectorAll('*')].some(e => visible(e) && !e.children.length
      && /^(?:追评|追加评价|用户追评|\d+天后追评|购买\d+天后追评)$/.test((e.innerText || '').trim()));
    if (appendAmbiguous) ambiguousAppends++;
    const players = [...item.querySelectorAll('video')].filter(v => v.closest('.Ez3MC6d7') === item);
    const urls = v => [v.currentSrc, v.getAttribute('src'), v.getAttribute('data-src'),
      ...[...v.querySelectorAll('source')].flatMap(s => [s.getAttribute('src'), s.getAttribute('data-src')])]
      .filter(u => /^(https?:)?\/\//i.test(u || ''));
    const comparable = u => (u || '').replace(/^\/\//, 'https://');
    const posters = new Set(players.map(v => comparable(v.getAttribute('poster'))).filter(Boolean));
    unresolvedVideos += players.filter(v => !urls(v).length).length;
    return {
    review_id: item.getAttribute('data-review-id') || item.getAttribute('data-comment-id') || '',
    content: read(item, '.mgNuPdjB'), nickname: read(item, '.j6DkqvoQ'),
    time: read(item, '.Xug6qnCc'), spec: read(item, '.bFXVK94u'),
    videos: appendAmbiguous ? [] : [...new Set(players.flatMap(urls))],
    imgs: appendAmbiguous ? [] : [...item.querySelectorAll('.qZmIVPuL img')].filter(img => img.closest('.Ez3MC6d7') === item && !img.closest('video'))
      .map(img => [img.currentSrc, img.getAttribute('src'), img.getAttribute('data-src')]
        .find(url => /^(https?:)?\/\//i.test(url || '')) || '').filter(url => !posters.has(comparable(url)))
    };
  });
  const exactMarker = (scope, values) => !!scope && [...scope.querySelectorAll('*')].some(e =>
    visible(e) && !e.closest('.Ez3MC6d7') && !e.children.length && values.includes((e.innerText || '').trim()));
  const end = items.length > 0 && exactMarker(root, ['没有更多评价了', '没有更多评价', '全部评价已加载', '没有更多了', '已展示全部评价']);
  const empty = exactMarker(root, ['暂无商品评价', '暂无评价', '还没有评价']);
  return {schema: 'douyin_product_reviews_dom_v1', reviews, end, empty,
    review_scope_found: !!root, unresolved_videos: unresolvedVideos, ambiguous_appends: ambiguousAppends,
    ui_text: uiParts.join('\n').slice(0, 30000),
    can_scroll: !!root && root.scrollHeight > root.clientHeight + 2,
    at_bottom: !!root && root.scrollTop + root.clientHeight >= root.scrollHeight - 4};
}"""

_SCROLL_JS = r"""() => {
  const item = document.querySelector('.Ez3MC6d7');
  for (let p = item?.parentElement; p && p !== document.body && p !== document.documentElement; p = p.parentElement) {
    if (/(auto|scroll|overlay)/.test(getComputedStyle(p).overflowY) && p.scrollHeight > p.clientHeight + 2) {
      p.scrollBy(0, Math.max(400, p.clientHeight * .8)); return true;
    }
  }
  const root = document.querySelector('.eRfaxg7R');
  if (root && root.scrollHeight > root.clientHeight + 2) { root.scrollBy(0, 700); return true; }
  return false;
}"""


def _blocking_state(text: str) -> tuple[str, str] | None:
    checks = (
        ("verification_required", r"请完成安全验证|完成验证后继续|请完成以下验证|拖动滑块|点击完成验证", "页面要求人工验证"),
        ("login_required", r"请先登录|登录后查看(?:商品)?评价|登录后继续|请登录后", "页面要求登录后继续"),
        ("product_unavailable", r"商品已下架|商品不存在|商品已失效|该商品已删除|商品已售罄|(?:^|\s)已抢光(?:\s|$)", "页面显示商品下架、失效或售罄"),
        ("app_required", r"请在抖音\s*(?:APP|App|app|客户端)\s*内|打开抖音查看(?:商品)?评价|下载抖音查看评价|仅支持在抖音", "评价入口要求在抖音 App 中打开"),
    )
    for status, pattern, reason in checks:
        if re.search(pattern, text):
            return status, reason
    return None


async def collect(page, source_url: str, max_scrolls: int = 40) -> dict:
    """Collect the product reviews the supplied browser session can actually see.

    No browser is launched, no cookies are exported, and challenges are not
    solved. The caller can log in manually and rerun. An explicit end marker
    inside the review scrolling container is required for complete=True.
    """
    diagnostics: dict[str, Any] = {"source_url": source_url, "selector_source": SELECTOR_SOURCE,
        "selector_validation": "upstream desktop popup, not live-verified for this link",
        "image_quality": "displayed URL; original resolution unverified", "scrolls": 0,
        "end_marker_seen": False, "errors": []}
    diagnostics["unresolved_video_rows"] = 0
    diagnostics["append_dom_schema_verified"] = False
    diagnostics["ambiguous_append_rows"] = 0
    saved: dict[tuple[str, str, str], dict] = {}

    def result(status: str, reason: str, complete: bool = False) -> dict:
        diagnostics["review_count"] = len(saved)
        diagnostics["image_count"] = sum(len(r["images"]) for r in saved.values())
        diagnostics["video_count"] = sum(len(r["videos"]) for r in saved.values())
        if complete and (diagnostics["unresolved_video_rows"] or diagnostics["ambiguous_append_rows"]):
            status, complete = "partial", False
            reason = "评价已到末尾，但仍有未解析视频或无法确定初评/追评归属的媒体，未确认完整性"
        return {"status": status, "records": list(saved.values()), "reason": reason,
                "complete": complete, "diagnostics": diagnostics}

    try:
        source = urlsplit(source_url)
    except (TypeError, ValueError):
        return result("unsupported_link", "不是有效商品链接")
    short = source.scheme in ("http", "https") and source.hostname == "v.douyin.com"
    if not product_id_from_url(source_url) and not short:
        return result("unsupported_link", "只接受抖音商品详情或待解析分享短链，不采集视频评论")
    if getattr(page, "url", "") != source_url:
        try:
            await page.goto(source_url, wait_until="domcontentloaded", timeout=30000)
        except asyncio.CancelledError:
            return result("cancelled", "用户取消采集，保留已读取的评价和图片", False)
        except Exception as exc:
            diagnostics["errors"].append("navigation:" + type(exc).__name__)
    try:
        await page.wait_for_timeout(1500)
        resolved_url = getattr(page, "url", "")
        source_id = product_id_from_url(source_url)
        product_id = product_id_from_url(resolved_url)
        diagnostics["resolved_url"] = resolved_url
        diagnostics["product_identity_verified"] = False
        if re.search(r"/(video|note|user|search)(/|$)", urlsplit(resolved_url).path):
            return result("unsupported_link", "分享链接跳转至视频或账号，不能作为商品评价入口")
        if source_id and product_id and source_id != product_id:
            return result("product_mismatch", "跳转后的商品 ID 与输入商品不一致，拒绝收集")
        snapshot = await page.evaluate(_SNAPSHOT_JS)
        blocker = _blocking_state(snapshot.get("ui_text", ""))
        if blocker and not snapshot.get("reviews"):
            return result(*blocker)
        if not product_id:
            return result("unsupported_page", "当前页面没有可确认的商品 ID，不能沿用输入链接的商品身份")
        diagnostics["product_identity_verified"] = True
        # Ordinary UI clicks only; never force-click through an overlay.
        candidates = [page.locator(".xgTWV_ga").filter(has_text="商品评价").first]
        candidates.extend(page.get_by_text(label).first for label in (
            re.compile(r"^商品评价(?:[（(].*[）)])?$"), re.compile(r"^查看全部评价$")))
        for locator in candidates:
            try:
                if await locator.count() and await locator.is_visible():
                    await locator.click(timeout=2500)
                    await page.wait_for_timeout(1000)
                    break
            except Exception as exc:
                diagnostics["errors"].append("open_reviews:" + type(exc).__name__)

        unchanged = 0
        previous = None
        for turn in range(max(0, int(max_scrolls)) + 1):
            current_url = getattr(page, "url", "")
            current_id = product_id_from_url(current_url)
            if current_id != product_id:
                diagnostics["resolved_url"] = current_url
                diagnostics["product_identity_verified"] = False
                changed_snapshot = await page.evaluate(_SNAPSHOT_JS)
                blocker = _blocking_state(changed_snapshot.get("ui_text", ""))
                if not current_id and blocker:
                    return result(*blocker)
                return result("product_mismatch", "评价浏览期间商品身份改变或丢失，停止收集")
            snapshot = await page.evaluate(_SNAPSHOT_JS)
            # Navigation may occur while the snapshot promise is pending.
            if product_id_from_url(getattr(page, "url", "")) != product_id:
                diagnostics["resolved_url"] = getattr(page, "url", "")
                diagnostics["product_identity_verified"] = False
                return result("product_mismatch", "读取评价时页面已离开原商品，丢弃该次结果")
            snapshot["product_id"] = product_id
            for record in parse_reviews(snapshot, source_url):
                key = record["product_id"], record["review_id"], record["review_type"]
                if key in saved:
                    saved[key]["images"] = list(dict.fromkeys(saved[key]["images"] + record["images"]))
                    saved[key]["videos"] = list(dict.fromkeys(saved[key]["videos"] + record["videos"]))
                else:
                    saved[key] = record
            diagnostics["review_scope_found"] = bool(snapshot.get("review_scope_found"))
            diagnostics["unresolved_video_rows"] = max(diagnostics["unresolved_video_rows"],
                                                        int(snapshot.get("unresolved_videos") or 0))
            diagnostics["ambiguous_append_rows"] = max(diagnostics["ambiguous_append_rows"],
                                                        int(snapshot.get("ambiguous_appends") or 0))
            blocker = _blocking_state(snapshot.get("ui_text", ""))
            if blocker and (blocker[0] in ("login_required", "verification_required")
                            or not snapshot.get("reviews")):
                return result(*blocker)
            if snapshot.get("empty") and not saved:
                return result("completed", "商品评价容器明确显示暂无评价", True)
            if snapshot.get("end") and snapshot.get("at_bottom") and saved:
                diagnostics["end_marker_seen"] = True
                return result("completed", "评价容器显示明确末页标记；范围为当前会话可见评价", True)
            signature = json.dumps(snapshot.get("reviews", []), ensure_ascii=False, sort_keys=True)
            unchanged = unchanged + 1 if signature == previous else 0
            previous = signature
            if unchanged >= 3:
                return result("partial" if saved else "unsupported_page",
                              "连续滚动未出现新评价，未见明确末页，不能认定采集完成")
            if turn >= max(0, int(max_scrolls)):
                return result("partial" if saved else "unsupported_page",
                              "达到滚动上限，未见明确评价末页")
            scrolled = await page.evaluate(_SCROLL_JS)
            if not scrolled:
                body = snapshot.get("ui_text", "")
                if not saved and re.search(r"打开抖音|抖音APP|抖音 App", body):
                    return result("unsupported_page", "页面提供打开抖音入口，但未发现已适配评价容器；不足以证明评价仅限 App")
                return result("partial" if saved else "unsupported_page",
                              "未找到可滚动的商品评价容器；当前页面结构尚未适配")
            diagnostics["scrolls"] += 1
            await page.wait_for_timeout(900)
    except asyncio.CancelledError:
        # The engine requests a cancellation snapshot; do not discard records
        # already observed before an awaited scroll/read was interrupted.
        return result("cancelled", "用户取消采集，保留已读取的评价和图片", False)
    except Exception as exc:
        diagnostics["errors"].append("collect:" + type(exc).__name__)
        return result("error", "读取商品评价失败；详见诊断，未宣称完整")
    return result("partial", "采集停止，未确认评价末页")
