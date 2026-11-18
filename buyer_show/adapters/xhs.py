"""Conservative diagnostic adapter for Xiaohongshu *commerce* reviews.

The anonymous H5 detail API was checked on 2026-10-29. It supplied product
metadata, carouselH5 and graphicDetailsV4, but no review records. An unboxing
entry in its template was not backed by template_data. No verified public
review response schema or pagination contract has been established.

Consequently this adapter must never turn product photos or social-note
comments into buyer reviews, or report an empty result as complete.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qs, urlsplit, urlunsplit


PLATFORM = "xiaohongshu"
EVIDENCE_DATE = "2026-10-29"
_ID = re.compile(r"^[0-9a-fA-F]{24}$")
_PRODUCT_PATH = re.compile(r"^/(?:goods-detail|goods)/([0-9a-fA-F]{24})/?$")
_HOSTS = {"xiaohongshu.com", "www.xiaohongshu.com", "mall.xiaohongshu.com"}
_SHORT_HOSTS = {"xhslink.com", "www.xhslink.com", "xhslink.cn", "www.xhslink.cn"}


def _product_id(url: str) -> str | None:
    """Accept a commerce identifier only, never a social-note identifier."""
    try:
        parsed = urlsplit(url)
    except (TypeError, ValueError):
        return None
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in _HOSTS:
        return None
    if parsed.username or parsed.password:
        return None
    match = _PRODUCT_PATH.fullmatch(parsed.path)
    if match:
        return match.group(1).lower()
    # This is the read-only public detail URL verified during research.
    if (parsed.hostname == "mall.xiaohongshu.com"
            and parsed.path == "/api/store/jpd/edith/detail/h5/toc"):
        ids = parse_qs(parsed.query).get("item_id", [])
        if len(ids) == 1 and _ID.fullmatch(ids[0]):
            return ids[0].lower()
    return None


def _safe_location(url: str) -> str:
    """Do not include share signatures, login redirects or other query values."""
    try:
        parsed = urlsplit(url)
        return urlunsplit((parsed.scheme, parsed.hostname or "", parsed.path, "", ""))
    except (TypeError, ValueError):
        return ""


def parse_reviews(payload: Any, source_url: str) -> list[dict]:
    """Return no records until an actual commerce-review schema is verified.

    In particular, fields called images, pictures, comments, or unboxing are
    not sufficient evidence of product-to-review-to-image provenance. The
    collector reports this limitation explicitly instead of reporting success.
    """
    return []


async def collect(page: Any, source_url: str, max_scrolls: int = 40) -> dict:
    """Open a commerce link and diagnose it without clicking or logging in.

    An already-open matching product can be reused. max_scrolls exists
    for the shared adapter interface; scrolling cannot establish a missing
    review schema, so this adapter deliberately performs no pagination.
    """
    product_id = _product_id(source_url)
    current_url = str(getattr(page, "url", "") or "")
    resolved_product_id = _product_id(current_url)
    try:
        source = urlsplit(source_url)
        is_short_link = (source.scheme in {"http", "https"}
                         and source.hostname in _SHORT_HOSTS
                         and not source.username and not source.password)
    except (TypeError, ValueError):
        is_short_link = False
    diagnostics: dict[str, Any] = {
        "adapter": "xiaohongshu_commerce_reviews",
        "evidence_date": EVIDENCE_DATE,
        "source_location": _safe_location(source_url),
        "page_location": _safe_location(current_url),
        "product_id": product_id,
        "review_schema_verified": False,
        "pagination_end_observed": False,
        "scrolls_performed": 0,
    }

    def result(status: str, reason: str) -> dict:
        return {
            "status": status,
            "records": [],
            "reason": reason,
            "complete": False,
            "diagnostics": diagnostics,
        }

    if product_id is None and not is_short_link:
        return result(
            "unsupported_review_entry",
            "尚未确认这是小红书商城商品链接；笔记、笔记评论和店铺链接不属于商品评价。",
        )
    if not product_id or resolved_product_id != product_id:
        try:
            response = await page.goto(source_url, wait_until="domcontentloaded", timeout=30000)
            if response is not None:
                diagnostics["navigation_http_status"] = getattr(response, "status", None)
        except Exception as exc:
            diagnostics["navigation_error"] = type(exc).__name__
            diagnostics["page_location"] = _safe_location(str(getattr(page, "url", "") or ""))
            return result("unsupported_review_entry", "商品页面未能完成打开，尚未取得商品评价；请检查网络或手动验证页面。")
        current_url = str(getattr(page, "url", "") or "")
        resolved_product_id = _product_id(current_url)
        diagnostics["page_location"] = _safe_location(current_url)
    if product_id is None and is_short_link:
        product_id = resolved_product_id
        diagnostics["product_id"] = product_id
    if product_id is None:
        return result("unsupported_review_entry", "分享短链未解析为小红书商城商品页，不能按笔记或主图采集。")
    if resolved_product_id and resolved_product_id != product_id:
        diagnostics["product_identity_mismatch"] = True
        return result("unsupported_review_entry", "当前页面与输入商品不一致，未采集任何图片。")

    visible_text = ""
    if resolved_product_id == product_id:
        try:
            visible_text = await page.locator("body").inner_text(timeout=5000)
        except Exception as exc:
            diagnostics["page_read_error"] = type(exc).__name__

    # Only classify an app requirement when the same product page says so.
    app_messages = [
        "打开小红书App", "打开小红书APP", "打开小红书 APP",
        "请在小红书App", "请在小红书APP", "请在小红书 APP",
        "App内查看", "APP内查看", "App 内查看", "APP 内查看",
    ]
    app_markers = [message for message in app_messages if message in visible_text]
    if app_markers:
        diagnostics["app_prompt_markers"] = app_markers
        return result(
            "app_required",
            "商品页提示在小红书 App 内查看；当前网页适配器未取得可验证的商品评价。",
        )

    diagnostics["visible_review_entry_marker"] = any(
        word in visible_text for word in ("商品评价", "买家评价", "全部评价", "查看评价")
    )
    return result(
        "unsupported_review_entry",
        "尚无经实测确认的小红书商城评价入口、图片字段及分页终点。"
        "商品主图、SKU 图、图文详情图和笔记评论均不计入买家秀；当前结果不是采集成功或评价为空。",
    )
