"""Identify supported commerce links without resolving or rewriting them.

Short URLs are only pending candidates. A caller must classify their final URL
again after navigation and must reject an is_short result at that stage.

The Douyin H5 route is evidenced by:
https://github.com/runyuweng/Douyin_EC_Data (product_detail_h5_url example).
No guessed douyin.com product routes are accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import parse_qs, urlsplit


@dataclass(frozen=True)
class ProductLink:
    platform: str
    product_id: str | None
    url: str
    is_short: bool = False


_TAOBAO_ROUTES = {
    ("item.taobao.com", "/item.htm"),
    ("detail.tmall.com", "/item.htm"),
    ("h5.m.taobao.com", "/awp/core/detail.htm"),
}
_REDIRECT_ROUTES = {("click.simba.taobao.com", "/cc_im"): "taobao"}
_XHS_HOSTS = {"xiaohongshu.com", "www.xiaohongshu.com"}
_SHORT_HOSTS = {
    "m.tb.cn": "taobao",
    "v.douyin.com": "douyin",
    "xhslink.com": "xiaohongshu",
    "xhslink.cn": "xiaohongshu",
}
_DECIMAL_ID = re.compile(r"[1-9][0-9]*\Z")
_XHS_PRODUCT = re.compile(r"/goods-detail/([0-9a-fA-F]{24})/?\Z")
_SOCIAL_PATH = re.compile(r"/(?:video|note|explore|discovery/item)(?:/|$)")
# Chinese sentence punctuation and quote characters delimit shared URLs; URL
# query bytes (including signatures, + and percent escapes) are never decoded.
_URL_IN_TEXT = re.compile(r"https?://[^\s<>\"'`，。；！？、【】《》“”‘’]+", re.I)


def _numeric_id(query: str) -> str:
    values = parse_qs(query, keep_blank_values=True).get("id", [])
    if len(values) != 1 or not _DECIMAL_ID.fullmatch(values[0]):
        raise ValueError("商品链接必须包含唯一且有效的数字 id 参数。")
    return values[0]


def classify_url(url: str) -> ProductLink:
    """Classify one exact supported product URL or pending short URL.

    Raises ValueError for every unsupported, unsafe, or clearly social URL.
    The returned url preserves input query order, encoding, and signatures.
    """
    if not isinstance(url, str):
        raise ValueError("请提供 HTTP 或 HTTPS 商品链接。")
    candidate = url.strip()
    if not candidate or re.search(r"[\s\x00-\x1f\x7f\\]", candidate):
        raise ValueError("链接含空白、控制字符或反斜杠，请重新复制完整商品链接。")
    try:
        parsed = urlsplit(candidate)
        host = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError("商品链接格式无效。") from exc
    if parsed.scheme not in {"http", "https"} or not host:
        raise ValueError("仅接受 HTTP 或 HTTPS 商品链接。")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("商品链接不能包含用户名或密码。")
    expected_port = 443 if parsed.scheme == "https" else 80
    if port is not None and port != expected_port:
        raise ValueError("商品链接不能使用非常规端口。")
    if parsed.netloc.endswith(":"):
        raise ValueError("商品链接端口格式无效。")

    if (host, parsed.path) in _REDIRECT_ROUTES:
        # Advertisement s/skuId parameters are not the destination product ID.
        return ProductLink(_REDIRECT_ROUTES[(host, parsed.path)], None, candidate, is_short=True)
    if host in _SHORT_HOSTS:
        if parsed.path in {"", "/"}:
            raise ValueError("短链缺少分享路径，请复制完整的商品分享链接。")
        return ProductLink(_SHORT_HOSTS[host], None, candidate, is_short=True)

    if host in {"douyin.com", "www.douyin.com", "www.iesdouyin.com"}:
        if (_SOCIAL_PATH.search(parsed.path)
                or "modal_id" in parse_qs(parsed.query, keep_blank_values=True)):
            raise ValueError("这是抖音视频或图文链接；请发送商品详情或商品分享链接，视频评论不属于商品评价。")
    if host in _XHS_HOSTS and _SOCIAL_PATH.search(parsed.path):
        raise ValueError("这是小红书笔记链接；请发送商城商品链接，笔记评论不属于商品评价。")

    if (host, parsed.path) in _TAOBAO_ROUTES:
        return ProductLink("taobao", _numeric_id(parsed.query), candidate)
    if host in _XHS_HOSTS:
        match = _XHS_PRODUCT.fullmatch(parsed.path)
        if match:
            return ProductLink("xiaohongshu", match.group(1).lower(), candidate)
    if (host == "haohuo.jinritemai.com"
            and parsed.path == "/ecommerce/trade/detail/index.html"):
        return ProductLink("douyin", _numeric_id(parsed.query), candidate)

    known_hosts = {host_name for host_name, _ in _TAOBAO_ROUTES} | _XHS_HOSTS | {
        "douyin.com", "www.douyin.com", "www.iesdouyin.com", "haohuo.jinritemai.com",
    }
    if host in known_hosts:
        raise ValueError("该平台链接不是已确认支持的商品详情路径，请复制商品分享链接。")
    raise ValueError("不支持该链接域名；仅接受已确认的淘宝、天猫、小红书商城或抖音商城商品链接。")


def _trim_wrapper(value: str) -> str:
    # Strip only unmatched closing wrappers from prose/Markdown. Balanced
    # characters inside a URL are retained, and query strings are not rebuilt.
    pairs = {")": "(", "]": "[", "}": "{", "）": "（", "」": "「", "』": "『"}
    while value and value[-1] in pairs:
        close = value[-1]
        if value.count(close) <= value.count(pairs[close]):
            break
        value = value[:-1]
    return value


def extract_links(text: str) -> list[ProductLink]:
    """Extract ordered unique links from multiline share text.

    Any invalid URL makes the input fail as a whole, including a video/note URL
    mixed with otherwise valid products. Direct products deduplicate by their
    platform and product ID (the first original URL is retained); short links
    deduplicate only by their exact URL because the target is still unknown.
    """
    if not isinstance(text, str):
        raise ValueError("请粘贴商品链接或商品分享文案。")
    candidates = [_trim_wrapper(match.group(0)) for match in _URL_IN_TEXT.finditer(text)]
    if not candidates:
        raise ValueError("没有找到 HTTP 或 HTTPS 链接，请粘贴商品详情链接或商品分享文案。")
    result: list[ProductLink] = []
    seen: set[tuple[str, str, str | None]] = set()
    for index, candidate in enumerate(candidates, start=1):
        try:
            link = classify_url(candidate)
        except ValueError as exc:
            # Do not echo signed URLs into error dialogs/logs.
            raise ValueError(f"第 {index} 个链接：{exc}") from exc
        key = (link.platform, "short" if link.is_short else "product",
               link.url if link.is_short else link.product_id)
        if key not in seen:
            seen.add(key)
            result.append(link)
    return result
