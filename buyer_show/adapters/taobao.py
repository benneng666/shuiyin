"""Conservative Taobao/Tmall review adapter; no browser or login is created here.

Only observed review responses and review-card DOM are accepted. The parser's
legacy schema is documented at https://gist.github.com/xuexingdong/47baa6ed4b88ac4cf622905293cb9718
DOM selector reference (not copied implementation):
https://github.com/l010306/Unified-Review-Scraper/blob/main/core/config.py
The current 6.0 photo/video fields and drawer were verified on 2026-10-30.
Folded evaluations and appended-review video fields remain unverified.
"""
from __future__ import annotations

import asyncio
import hashlib
import html
import json
import re
from urllib.parse import parse_qs, urlsplit

from ._media import video_urls
from ..video import video_url_allowed


_DOM_SCHEMA = "taobao_product_reviews_dom_v1"
# Confirmed in both live product diagnostics (2026-10-30). The product page
# also has a non-scrollable summary list outside this drawer.
_ALL_REVIEWS_CONTAINER = '[class*="Drawer--"] [class*="comments--"]'
_FOLDED_NOTICE = '[class*="foldInfoContent--"]'
_FOLD_EXPAND_TEXT = re.compile(r"展开|查看折叠评价")


def _product_id(url):
    try:
        u = urlsplit(url)
        if u.hostname not in {"item.taobao.com", "detail.tmall.com", "h5.m.taobao.com", "m.tb.cn", "new.m.taobao.com"}:
            return ""
        value = parse_qs(u.query).get("id", [""])[0]
        return value if value.isdigit() else ""
    except (TypeError, ValueError):
        return ""


def _decode(payload):
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", errors="replace")
    if isinstance(payload, str):
        if len(payload) > 10_000_000:
            return {}
        value = payload.strip()
        wrapped = re.fullmatch(r"[\w.$]+\s*\((.*)\)\s*;?", value, re.S)
        if wrapped:
            value = wrapped.group(1)
        try:
            payload = json.loads(value)
        except (ValueError, RecursionError):
            return {}
    return payload if isinstance(payload, dict) else {}


def _review_lists(payload):
    # A finite envelope allowlist, never a recursive search for arbitrary images.
    for path in ((), ("rateDetail",), ("data",), ("data", "rateDetail")):
        node = payload
        for key in path:
            node = node.get(key, {}) if isinstance(node, dict) else {}
        if isinstance(node, dict):
            for key in ("rateList", "rate_list", "comments"):
                if isinstance(node.get(key), list):
                    yield node[key]


def _images(review):
    result = []
    # feedPicPathList is verified in the live 6.0 detail-list response on
    # 2026-10-30. Do not inspect headPicUrl/shareCover or arbitrary nested data.
    for field in ("feedPicPathList", "appendFeedPicPathList", "pics", "photos", "images"):
        values = review.get(field, [])
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, dict):
                value = next((value[k] for k in ("originalUrl", "full", "url", "picUrl", "picPath", "imageUrl", "path")
                              if isinstance(value.get(k), str) and value[k]), "")
            if not isinstance(value, str):
                continue
            value = html.unescape(value.strip())
            if value.startswith("//"):
                value = "https:" + value
            try:
                u = urlsplit(value)
                host = u.hostname or ""
                if u.scheme not in {"http", "https"} or u.username or u.password:
                    continue
                if not any(host == d or host.endswith("." + d) for d in ("alicdn.com", "taobaocdn.com")):
                    continue
                if not u.path or re.search(r"/(?:avatar|icons?)/|(?:blank|spacer|spaceball)\.(?:gif|png)", u.path, re.I):
                    continue
            except ValueError:
                continue
            # Preserve observed URLs, including underscores and signed queries.
            if value not in result:
                result.append(value)
    return result


def _text(row):
    for key in ("rateContent", "content", "feedback", "appendedFeedback", "review_text"):
        if isinstance(row.get(key), str):
            return html.unescape(row[key]).strip()
    return ""


def _network_video_candidates(row):
    """6.0 initial-review video object, observed and downloaded on 2026-10-30.

    coverUrl is a picture. Both media fields describe the same video. Keep all
    actual legal candidates in priority order, never manufacture a new URL.
    """
    video = row.get("video")
    if not isinstance(video, dict):
        return []
    return [url for url in video_urls([video.get("cloudVideoUrl"), video.get("sourceVideoUrl")])
            if video_url_allowed(url, "taobao")]


def _network_videos(row):
    return _network_video_candidates(row)[:1]


def _merge_video_sources(previous, candidate):
    for canonical, urls in candidate.get("video_sources", {}).items():
        sources = previous.setdefault("video_sources", {})
        sources[canonical] = list(dict.fromkeys(sources.get(canonical, [canonical]) + urls))


def _image_identity(url):
    """Identity comparison only; never rewrite a download URL.

    The live DOM renders Alibaba's resized variants of feedPicPathList. Match
    the same CDN object before its image-resize suffix, keeping native API URLs.
    """
    try:
        u = urlsplit(url)
        path = re.sub(r"(\.(?:jpe?g|png|webp))(?:_[^/]*)$", r"\1", u.path, flags=re.I)
        return u.hostname, path
    except (TypeError, ValueError):
        return None


def _network_match(row, records):
    """Only merge a DOM row with one unambiguous matching native evaluation."""
    images = {_image_identity(u) for u in row["images"]}
    candidates = [other for other in records if other["review_type"] == row["review_type"]
                  and other["content"] == row["content"]
                  and images.issubset({_image_identity(u) for u in other["images"]})
                  and set(row["videos"]).issubset(other["videos"])]
    return candidates[0] if len(candidates) == 1 else None


def _id(row, fallback="", videos=None):
    for key in ("id", "rateId", "rate_id", "commentId", "reviewId"):
        if isinstance(row.get(key), (str, int)) and str(row[key]).strip():
            return str(row[key])
    if fallback:
        return fallback
    fingerprint = {k: row.get(k) for k in ("rateDate", "date", "userId", "displayUserNick", "meta")}
    fingerprint.update(content=_text(row), images=_images(row))
    if videos:
        fingerprint["videos"] = videos
    return "synthetic:" + hashlib.sha256(json.dumps(fingerprint, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:24]


def parse_reviews(payload, source_url) -> list[dict]:
    """Extract explicit review envelopes, not detail/carousel/avatar images.

    Records without image URLs are retained for counting reviewed text-only rows.
    Source must be a resolved Taobao/Tmall product URL. An append record keeps the
    parent review_id, with review_type distinguishing it from the initial review.
    """
    product = _product_id(source_url)
    payload = _decode(payload)
    # Arbitrary videos arrays remain a DOM normalization field. The observed
    # 6.0 response uses video.cloudVideoUrl/sourceVideoUrl, never its coverUrl.
    dom = payload.get("schema") == _DOM_SCHEMA
    videos = lambda row: video_urls(row.get("videos", [])) if dom else _network_videos(row)
    if not product:
        return []
    data = payload.get("data", {})
    for node in (payload, payload.get("rateDetail", {}), data,
                 data.get("rateDetail", {}) if isinstance(data, dict) else {}):
        if isinstance(node, dict):
            ident = node.get("itemId", node.get("item_id"))
            if ident is not None and str(ident) != product:
                return []
    records = {}
    for rows in _review_lists(payload):
        for row in rows:
            if not isinstance(row, dict):
                continue
            ident = row.get("itemId", row.get("item_id", row.get("auctionNumId")))
            if ident is not None and str(ident) != product:
                continue
            if not any(k in row for k in ("rateContent", "content", "feedback", "review_text", "pics", "photos", "images", "feedPicPathList", "appendComment")) and not videos(row):
                continue
            rid = _id(row, videos=videos(row))
            parts = [("initial", row)]
            for key in ("appendedFeed", "appendComment", "appendComments", "append"):
                value = row.get(key)
                for appended in value if isinstance(value, list) else [value]:
                    if isinstance(appended, dict) and (_text(appended) or _images(appended) or videos(appended)):
                        parts.append(("append", appended))
            for kind, part in parts:
                key = (rid, kind)
                candidate = {"platform": "taobao", "product_id": product,
                             "review_id": rid, "review_type": kind,
                             "content": _text(part), "images": _images(part), "videos": videos(part)}
                alternatives = [] if dom else _network_video_candidates(part)
                if len(alternatives) > 1:
                    candidate["video_sources"] = {alternatives[0]: alternatives}
                if key in records:
                    previous = records[key]
                    previous["images"] = list(dict.fromkeys(previous["images"] + candidate["images"]))
                    previous["videos"] = list(dict.fromkeys(previous["videos"] + candidate["videos"]))
                    _merge_video_sources(previous, candidate)
                    if candidate["content"] and candidate["content"] not in previous["content"]:
                        previous["content"] = "\n".join(filter(None, [previous["content"], candidate["content"]]))
                else:
                    records[key] = candidate
    return list(records.values())


def _response_product_id(request):
    """Read only item IDs from the request; never log headers, body or queries."""
    try:
        q = parse_qs(urlsplit(request.url).query)
        for key in ("itemId", "item_id", "auctionNumId", "auction_id"):
            if q.get(key):
                return str(q[key][0])
        for value in (q.get("data", [""])[0], request.post_data or ""):
            data = _decode(value)
            for key in ("itemId", "item_id", "auctionNumId", "auction_id"):
                if key in data:
                    return str(data[key])
    except (ValueError, AttributeError):
        pass
    return ""


def _review_endpoint(url):
    try:
        u = urlsplit(url)
        host = u.hostname or ""
        if not any(host == d or host.endswith("." + d) for d in ("taobao.com", "tmall.com")):
            return False
        return (u.path in {"/list_detail_rate.htm", "/feedRateList.htm"}
                or bool(re.search(r"/mtop\.[\w.]*rate\.[\w.]*list[\w.]*/", u.path, re.I)))
    except ValueError:
        return False


_DOM = r"""() => {
 const visible = e => !!(e && e.getClientRects().length && getComputedStyle(e).visibility !== 'hidden');
 const appendSel = '[class*="append--"], .tm-rate-append';
 const textSel = '[class*="content--"], [class*="commentContent--"], .tm-rate-content';
 const imgSel = '[class*="album--"] [class*="photo--"] img, [class*="imageWrapper--"] img, .comment-image img, .tm-m-photos img';
 // The page summary and the open all-reviews drawer coexist in the DOM.
 // Prefer the drawer so its cards/scrolling cannot be confused with summary.
 const drawer = [...document.querySelectorAll('[class*="Drawer--"]')].find(e => visible(e) && e.querySelector('div[class*="Comment--"]'));
 return [...(drawer || document).querySelectorAll('div[class*="Comment--"], .rate-grid .rate-item')].filter(visible).map(card => {
   const app = card.querySelector(appendSel);
   const belongs = (e, root) => e.closest('div[class*="Comment--"], .rate-grid .rate-item') === card
     && (root !== card || !e.closest(appendSel));
   const media = root => {
     const players = [...root.querySelectorAll('video')].filter(e => belongs(e, root));
     const urls = v => [v.currentSrc, v.getAttribute('src'), v.getAttribute('data-src'),
       ...[...v.querySelectorAll('source')].flatMap(s => [s.getAttribute('src'), s.getAttribute('data-src')])]
       .filter(u => /^(https?:)?\/\//i.test(u || ''));
     const comparable = u => (u || '').replace(/^\/\//, 'https://');
     const posters = new Set(players.map(v => comparable(v.getAttribute('poster'))).filter(Boolean));
     const pics = [...root.querySelectorAll(imgSel)].filter(e => belongs(e, root) && !e.closest('video')
       && !e.closest('[class*="photo--"]')?.querySelector('[class*="playerIcon--"]'))
       .map(e => e.currentSrc || e.getAttribute('data-src') || e.getAttribute('src') || '')
       .filter(u => u && !posters.has(comparable(u)));
     return {pics, videos:[...new Set(players.flatMap(urls))],
       unresolved_videos: players.filter(v => !urls(v).length).length
         + [...root.querySelectorAll('[class*="playerIcon--"]')].filter(e => belongs(e, root)).length};
   };
   const first = media(card), appended = app ? media(app) : null;
   const content = [...card.querySelectorAll(textSel)].find(e => !e.closest(appendSel));
   return {id: card.getAttribute('data-rate-id') || card.getAttribute('data-comment-id') || '',
     content: content ? content.textContent.trim() : '', ...first,
     displayUserNick: card.querySelector('[class*="userName--"]')?.textContent || '',
     meta: card.querySelector('[class*="meta--"], [class*="commentMeta--"]')?.textContent || '',
     appendComment: app ? {content: app.querySelector(textSel)?.textContent || '', ...appended} : null};
 });
}"""


_SCROLL_REVIEWS = r"""() => {
 const visible = e => !!(e && e.getClientRects().length && getComputedStyle(e).visibility !== 'hidden');
 const drawer = [...document.querySelectorAll('[class*="Drawer--"]')].find(e => visible(e) && e.querySelector('div[class*="Comment--"]'));
 let root = drawer?.querySelector('[class*="comments--"]');
 if (!root) {
   const cards = [...document.querySelectorAll('div[class*="Comment--"], .rate-grid .rate-item')].filter(visible);
   for (let p=cards[cards.length-1]?.parentElement; p && p!==document.body; p=p.parentElement) {
     if (p.scrollHeight > p.clientHeight + 40 && /auto|scroll/.test(getComputedStyle(p).overflowY)) { root=p; break; }
   }
 }
 if (root) {
   const before=root.scrollTop;
   // Every card in this page is in the observed DOM/response. Moving to its
   // bottom activates the platform's ordinary next-page loading, not an API replay.
   root.scrollTo(0,root.scrollHeight);
   return {scope:'review_container',before,after:root.scrollTop,height:root.scrollHeight,
     at_bottom:root.scrollTop+root.clientHeight>=root.scrollHeight-4,
     moved:Math.abs(root.scrollTop-before)>2};
 }
 return {scope:'not_found',moved:false,at_bottom:false};
}"""


async def _visible(locator):
    for i in range(min(await locator.count(), 20)):
        target = locator.nth(i)
        if await target.is_visible():
            return target
    return None


async def _folded_entry(page):
    """Distinguish the observed informational footer from an explicit control.

    Live screenshots show '已折叠 N 条…', not an invitation to expand. Do not
    click that text merely because a count exists. A future supported control
    must have explicit expansion text and button/link semantics in this scope.
    """
    notice = await _visible(page.locator(_FOLDED_NOTICE).filter(
        has_text=re.compile(r"已折叠\s*\d+\s*条|展开|查看折叠评价")))
    if not notice:
        return None, None
    self_control = await notice.evaluate("""el => ({
      semantic:el.matches('button, a[href], [role="button"], [role="link"]'),
      text:(el.innerText || '').trim()
    })""")
    candidate = notice if self_control.get("semantic") and _FOLD_EXPAND_TEXT.search(self_control.get("text", "")) else None
    if candidate is None:
        candidate = await _visible(notice.locator('button, a[href], [role="button"], [role="link"]').filter(has_text=_FOLD_EXPAND_TEXT))
    if candidate is not None and (not await candidate.is_enabled() or await candidate.get_attribute("aria-disabled") == "true"):
        candidate = None
    return notice, candidate


async def _page_problem(page):
    host = urlsplit(page.url).hostname or ""
    if host in {"login.taobao.com", "login.tmall.com", "login.m.taobao.com"}:
        return "login_required", "页面要求登录；请在软件浏览器中登录后重新运行。"
    if "_____tmd_____" in page.url or host == "sec.taobao.com":
        return "verification_required", "平台要求安全验证，已停止自动操作。"
    if await _visible(page.locator('#baxia-dialog, iframe[src*="baxia"], iframe[src*="captcha"]')):
        return "verification_required", "检测到可见验证窗口，已停止自动操作。"
    for pattern, status, reason in (
        (r"^(?:扫码登录|密码登录|请先登录|登录后查看)$", "login_required", "检测到登录要求。"),
        (r"^(?:访问受限|访问过于频繁|操作过于频繁|账号异常|网络拥挤，请稍后重试)[！!。]?$", "access_restricted", "平台限制访问。"),
        (r"^(?:商品已下架|宝贝已下架|商品不存在|宝贝不存在|很抱歉，您查看的宝贝不存在)[！!。]?$", "unavailable", "商品不可用或已下架。"),
    ):
        if await _visible(page.get_by_text(re.compile(pattern))):
            return status, reason
    return None


async def collect(page, source_url, max_scrolls=40) -> dict:
    """Collect only what the supplied live page observes; never solve challenges.

    Caller owns browser lifecycle and login. A stalled scroll is partial, never
    complete. Explicit review-scoped UI end markers are required for completion.
    """
    diagnostics = {"adapter": "taobao", "network_responses": 0, "unmatched_responses": 0,
                   "response_errors": 0, "scrolls": 0, "dom_rows": 0,
                   "unresolved_video_rows": 0, "video_network_schema_verified": False,
                   "network_pages": [], "append_video_schema_live_validated": False,
                   "review_scopes": {"normal": {"pages": [], "end_observed": False},
                                     "folded": {"pages": [], "end_observed": False}},
                   "folded_review_count": 0, "folded_expand_attempted": False,
                   "folded_reviews_opened": False, "folded_notice_seen": False,
                   "folded_expand_control_seen": False, "folded_ui_state": "not_observed",
                   "all_reviews_clicked": False, "all_reviews_opened": False,
                   "all_reviews_view_evidence": None,
                   "completion_evidence": None, "schema_live_validated": False}
    product = _product_id(source_url)
    parse_url = source_url
    pending = []
    network_records, dom_records = {}, {}

    def merge(store, rows):
        for row in rows:
            key = (row["review_id"], row["review_type"])
            if key in store:
                store[key]["images"] = list(dict.fromkeys(store[key]["images"] + row["images"]))
                store[key]["videos"] = list(dict.fromkeys(store[key]["videos"] + row["videos"]))
                _merge_video_sources(store[key], row)
            else:
                store[key] = row

    def result(status, reason, complete=False):
        rows = dict(network_records)
        signatures = {(r["review_type"], r["content"], tuple(sorted(r["images"])), tuple(sorted(r["videos"])))
                      for r in network_records.values()}
        unmatched_dom = 0
        unmatched_dom_media = 0
        for key, row in dom_records.items():
            if key in rows:
                rows[key] = dict(rows[key], images=list(dict.fromkeys(rows[key]["images"] + row["images"])),
                                 videos=list(dict.fromkeys(rows[key]["videos"] + row["videos"])))
            elif _network_match(row, network_records.values()):
                # Keep original image URLs and native parent/append identities.
                # A UI thumbnail must not become a second downloaded review.
                continue
            elif (row["review_type"], row["content"], tuple(sorted(row["images"])), tuple(sorted(row["videos"]))) not in signatures:
                if network_records:
                    unmatched_dom += 1
                    unmatched_dom_media += len(row["images"]) + len(row["videos"])
                # Current v6 carries native IDs plus original photo/video URLs.
                # Anonymous DOM snapshots must not become invented extra buyers
                # when their resized media/text cannot be uniquely reconciled.
                # Keep the uncertainty visible; DOM-only/legacy pages still use
                # the fallback and native-ID DOM records remain mergeable above.
                if not (diagnostics["schema_live_validated"] and row["review_id"].startswith("synthetic:")):
                    rows[key] = row
        rows = list(rows.values())
        diagnostics["record_source"] = "network_and_dom" if network_records and dom_records else ("network" if network_records else "dom")
        diagnostics["unmatched_dom_records"] = unmatched_dom
        diagnostics["unmatched_dom_media"] = unmatched_dom_media
        diagnostics["dom_snapshot_policy"] = "native_v6_records_preferred" if diagnostics["schema_live_validated"] else "dom_fallback"
        if complete and unmatched_dom:
            status, complete = "partial", False
            reason = "评价已到末尾，但页面与接口记录尚未完全匹配；已保留原生评价，未把匿名页面快照重复计数。"
        if complete and diagnostics["folded_review_count"] and not diagnostics["review_scopes"]["folded"]["end_observed"]:
            status, complete = "partial", False
            reason = "普通评价已到末尾，但平台另有折叠评价，尚未确认折叠列表已读取完毕。"
        diagnostics["synthetic_ids"] = sum(r["review_id"].startswith("synthetic:") for r in rows)
        diagnostics["video_count"] = sum(len(r["videos"]) for r in rows)
        if complete and diagnostics["unresolved_video_rows"]:
            status, complete = "partial", False
            reason = "评价已到末尾，但部分视频尚未提供可下载地址，未确认完整性。"
        if complete and network_records and not dom_records:
            status, complete = "partial", False
            reason = "已读取评价接口，但尚未核对评价卡片中的视频，未确认媒体完整性。"
        if (not complete and diagnostics["folded_ui_state"] == "notice_only"
                and diagnostics["folded_review_count"]
                and not diagnostics["review_scopes"]["folded"]["end_observed"]):
            # The platform shows a folded count but no verifiable expand
            # control. Mark the gap as platform-unreachable, not a missed read.
            unreachable = int(diagnostics["folded_review_count"])
            diagnostics["folded_unreachable"] = {
                "count": unreachable,
                "reason": "platform_notice_only_no_expand_control",
                "saved_review_records": len(rows),
                "platform_expand_control_seen": False,
            }
            reason += (f" 已保存评价记录 {len(rows)} 条；平台不可达 {unreachable} 条折叠评价"
                       "（页面仅显示折叠提示，未提供可确认的展开控件，非软件漏采）。")
        return {"status": status, "records": rows, "reason": reason,
                "complete": complete, "diagnostics": diagnostics}

    def response_seen(response):
        if _review_endpoint(response.url):
            if len(pending) < 100:
                pending.append(response)
            else:
                diagnostics["response_errors"] += 1

    async def drain():
        while pending:
            response = pending.pop(0)
            if not product or _response_product_id(response.request) != product:
                diagnostics["unmatched_responses"] += 1
                continue
            diagnostics["network_responses"] += 1
            if response.status in (401, 403, 429):
                return ("login_required" if response.status == 401 else "access_restricted", "评价接口拒绝访问。")
            if response.status >= 400:
                diagnostics["response_errors"] += 1
                continue
            try:
                payload = _decode(await asyncio.wait_for(response.text(), timeout=5))
                raw_ret = payload.get("ret", [])
                ret = raw_ret if isinstance(raw_ret, str) else " ".join(map(str, raw_ret))
                if "SESSION_EXPIRED" in ret or "NEED_LOGIN" in ret:
                    return "login_required", "评价接口报告登录已过期。"
                if "RGV587" in ret or "USER_VALIDATE" in ret:
                    return "verification_required", "评价接口要求安全验证。"
                if not list(_review_lists(payload)):
                    diagnostics["response_errors"] += 1
                if "/mtop.taobao.rate.detaillist.get/6.0/" in urlsplit(response.url).path:
                    diagnostics["schema_live_validated"] = True
                    diagnostics["video_network_schema_verified"] = True
                    request_data = _decode(parse_qs(urlsplit(response.request.url).query).get("data", [""])[0])
                    detail = payload.get("data", {})
                    diagnostics["network_pages"].append({
                        "page": request_data.get("pageNo"), "folded": request_data.get("foldFlag"),
                        "has_next": detail.get("hasNext"), "rows": len(detail.get("rateList", []))})
                    # A normal-list end never completes the separate folded list.
                    # Keep a contiguous page ledger for each observed request scope.
                    scope_name = {"0": "normal", "1": "folded"}.get(str(request_data.get("foldFlag")))
                    if scope_name:
                        scope = diagnostics["review_scopes"][scope_name]
                        page_number = str(request_data.get("pageNo", ""))
                        if page_number.isdigit() and 0 < int(page_number) <= 10000:
                            number = int(page_number)
                            scope["pages"] = sorted(set(scope["pages"] + [number]))
                            if number >= scope.get("last_page", 0):
                                scope["last_page"] = number
                                scope["has_next"] = str(detail.get("hasNext", "")).lower()
                            scope["end_observed"] = (scope.get("has_next") == "false"
                                and scope["pages"] == list(range(1, scope["last_page"] + 1)))
                        if scope_name == "folded":
                            diagnostics["folded_reviews_opened"] = True
                    fold_count = str(detail.get("foldCount", ""))
                    if fold_count.isdigit():
                        diagnostics["folded_review_count"] = max(diagnostics["folded_review_count"], int(fold_count))
                merge(network_records, parse_reviews(payload, parse_url))
            except Exception:
                # Do not expose exception messages: libraries may include URLs.
                diagnostics["response_errors"] += 1
        return None

    if not isinstance(max_scrolls, int) or not 1 <= max_scrolls <= 1000:
        return result("error", "max_scrolls 必须为 1 至 1000 的整数。")
    page.on("response", response_seen)
    try:
        diagnostics["stage"] = "navigation"
        try:
            await page.goto(source_url, wait_until="domcontentloaded", timeout=45000)
        except Exception as exc:
            # One ordinary reload only for a named transport failure, never for
            # login/verification. Do not log the exception's potentially signed URL.
            transport = re.search(r"net::(?:ERR_CONNECTION_[A-Z_]+|ERR_NETWORK_CHANGED|ERR_HTTP2_PROTOCOL_ERROR|ERR_TIMED_OUT)", str(exc))
            if not transport or await _page_problem(page):
                raise
            diagnostics["navigation_retry"] = transport.group(0)
            await page.wait_for_timeout(800)
            await page.goto(source_url, wait_until="domcontentloaded", timeout=45000)
        await page.wait_for_timeout(1500)
        problem = await _page_problem(page)
        if problem:
            return result(*problem)
        diagnostics["stage"] = "wait_review_entry"
        all_locator = page.get_by_text("查看全部评价", exact=True)
        if hasattr(all_locator, "wait_for"):
            try:
                await all_locator.first.wait_for(state="visible", timeout=12000)
            except Exception:
                pass
        problem = await _page_problem(page)
        if problem:
            return result(*problem)
        resolved = _product_id(page.url)
        location = urlsplit(page.url)
        diagnostics["page_location"] = {"host": location.hostname, "path": location.path, "product_id": resolved}
        if not product:
            product, parse_url = resolved, page.url
        if not product or product != resolved:
            return result("unsupported", "未能确认目标商品 ID，或页面跳到了其他商品。")
        tab = await _visible(page.get_by_text(re.compile(r"^用户评价(?:[·\s（(].*)?$")))
        if tab:
            await tab.click(timeout=5000)
            await page.wait_for_timeout(1000)
        diagnostics["stage"] = "open_all_reviews"
        all_reviews = await _visible(all_locator)
        if all_reviews:
            # Preserve already visible summary media if opening the full list
            # instead prompts login or fails. The snapshot is still ID-bound.
            problem = await drain()
            if problem:
                return result(*problem)
            if _product_id(page.url) != product:
                return result("product_mismatch", "打开评价前页面已离开目标商品。")
            summary_cards = await page.evaluate(_DOM)
            if _product_id(page.url) != product:
                return result("product_mismatch", "读取评价摘要时页面身份改变，已丢弃该次结果。")
            diagnostics["summary_rows"] = len(summary_cards)
            diagnostics["dom_rows"] = max(diagnostics["dom_rows"], len(summary_cards))
            merge(dom_records, parse_reviews({"schema": _DOM_SCHEMA, "comments": summary_cards}, parse_url))
            await all_reviews.click(timeout=5000)
            diagnostics["all_reviews_clicked"] = True
            problem = await _page_problem(page) or await drain()
            if problem:
                return result(*problem)
            # A successful click is not evidence that the drawer opened. Wait
            # passively once for the observed container; no re-clicks, navigation
            # retries or scrolling through a login/verification prompt.
            full_list = page.locator(_ALL_REVIEWS_CONTAINER)
            if not await _visible(full_list) and hasattr(full_list, "wait_for"):
                try:
                    await full_list.first.wait_for(state="visible", timeout=8000)
                except Exception as exc:
                    diagnostics["review_entry_wait_error"] = type(exc).__name__
            problem = await _page_problem(page) or await drain()
            if problem:
                return result(*problem)
            if _product_id(page.url) != product:
                return result("product_mismatch", "打开评价后页面已离开目标商品。")
            diagnostics["all_reviews_opened"] = bool(await _visible(full_list))
            if diagnostics["all_reviews_opened"]:
                diagnostics["all_reviews_view_evidence"] = "visible_drawer_review_container"
            else:
                return result("partial", "已点击全部评价入口，但未确认完整评价列表打开；已保留取得的评价摘要，不能认定只有这些评价。")
        stall = 0
        last_size = (-1, -1)
        last_scroll_moved = False
        for step in range(max_scrolls):
            diagnostics["stage"] = "collect_reviews"
            diagnostics["scrolls"] = step
            problem = await _page_problem(page) or await drain()
            if problem:
                return result(*problem)
            if _product_id(page.url) != product:
                return result("product_mismatch", "当前页面已离开目标商品，停止收集评价。")
            cards = await page.evaluate(_DOM)
            if _product_id(page.url) != product:
                return result("product_mismatch", "读取评价时页面身份改变，已丢弃该次页面结果。")
            diagnostics["dom_rows"] = max(diagnostics["dom_rows"], len(cards))
            current_unresolved = sum(
                int(bool(c.get("unresolved_videos")))
                + int(bool((c.get("appendComment") or {}).get("unresolved_videos")))
                for c in cards if isinstance(c, dict))
            parsed_dom = parse_reviews({"schema": _DOM_SCHEMA, "comments": cards}, parse_url)
            resolved_from_network = sum(bool((match or {}).get("videos")) for row in parsed_dom
                                        if (match := _network_match(row, network_records.values())))
            diagnostics["unresolved_video_rows"] = max(0, current_unresolved - resolved_from_network)
            merge(dom_records, parsed_dom)
            notice, fold = await _folded_entry(page)
            if notice:
                diagnostics["folded_notice_seen"] = True
                diagnostics["folded_expand_control_seen"] = bool(fold)
                diagnostics["folded_ui_state"] = "expand_control" if fold else "notice_only"
            if fold and not diagnostics["folded_expand_attempted"]:
                problem = await _page_problem(page) or await drain()
                if problem:
                    return result(*problem)
                if _product_id(page.url) != product:
                    return result("product_mismatch", "展开折叠评价前页面已离开目标商品。")
                diagnostics["folded_expand_attempted"] = True
                await fold.click(timeout=5000)
                await page.wait_for_timeout(1200)
                stall = 0
                continue
            if notice and not fold and diagnostics["review_scopes"]["normal"]["end_observed"]:
                return result("partial", "普通评价已到末页；平台显示另有折叠评价，但当前页面未发现可确认的展开控件，未认定全部完成。")
            # Only end text inside a review section is evidence. Generic page
            # footer text could instead belong to recommendations.
            scope = page.locator('[class*="Comments--"], [class*="commentList--"], .rate-grid')
            end = await _visible(scope.get_by_text(re.compile(r"^(?:没有更多评价了|暂无更多评价|已显示全部评价|全部评价已加载|暂无评价)[！!。]?$")))
            if end and await end.evaluate("el => !!el.closest('div[class*=\"Comment--\"], .rate-item')"):
                end = None  # A buyer's text is not a platform completion signal.
            if end:
                diagnostics["completion_evidence"] = "visible_review_section_end_marker"
                end_text = (await end.inner_text()).strip()
                if not diagnostics["all_reviews_opened"] and end_text.rstrip("！!。") != "暂无评价":
                    return result("partial", "评价区域显示结束文字，但尚未确认进入全部评价视图。")
                if not network_records and not dom_records and end_text.rstrip("！!。") != "暂无评价":
                    return result("partial", "观察到结束文字，但未能解析评价记录；不能认定没有评价。")
                if diagnostics["response_errors"] or diagnostics["unmatched_responses"]:
                    return result("partial", "已到评价列表末尾，但存在未解析或未匹配的接口响应。")
                if network_records and dom_records:
                    api_images = {u for r in network_records.values() for u in r["images"]}
                    if any(u not in api_images for r in dom_records.values() for u in r["images"]):
                        return result("partial", "页面与接口图片范围不一致，需要人工核对。")
                return result("completed", "已观察到评价区域的明确结束标记。", True)
            size = (len(network_records) + len(dom_records), sum(len(r["images"]) + len(r["videos"]) for r in [*network_records.values(), *dom_records.values()]))
            stall = stall + 1 if size == last_size and not last_scroll_moved else 0
            last_size = size
            if stall >= 4:
                if diagnostics["folded_ui_state"] == "notice_only":
                    return result("partial", "已保留取得的可见评价；平台显示折叠评价提示，但当前页面未发现可确认的展开控件。")
                if diagnostics["folded_review_count"] and not diagnostics["review_scopes"]["folded"]["end_observed"]:
                    return result("partial", "普通评价停止新增；平台显示另有折叠评价，但未观察到折叠列表完整加载，已保留取得部分。")
                return result("partial", "连续加载未发现新增，尚未确认已读取全部可见评价。")
            next_page = await _visible(scope.get_by_text("下一页", exact=True))
            if next_page and await next_page.is_enabled() and await next_page.get_attribute("aria-disabled") != "true":
                await next_page.click(timeout=5000)
                last_scroll_moved = True
            else:
                scroll = await page.evaluate(_SCROLL_REVIEWS)
                last_scroll_moved = bool(scroll and scroll.get("moved"))
                if isinstance(scroll, dict):
                    diagnostics["last_scroll"] = scroll
            await page.wait_for_timeout(1500)
        problem = await drain()
        if problem:
            return result(*problem)
        return result("partial", "达到加载次数上限，完整性尚未确认。")
    except asyncio.CancelledError:
        return result("cancelled", "用户停止采集；已保留提取到的评价，完整性尚未确认。", False)
    except Exception as exc:
        diagnostics["error_type"] = type(exc).__name__
        try:
            problem = await _page_problem(page)
            if problem:
                return result(*problem)
        except Exception:
            pass
        return result("partial" if network_records or dom_records else "error", "页面操作失败；已保留提取结果，未确认完整性。")
    finally:
        try:
            page.remove_listener("response", response_seen)
        except Exception:
            # A concurrently closed page must not erase the collected snapshot.
            diagnostics["listener_cleanup_failed"] = True
