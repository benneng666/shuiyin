"""Product-owned media, kept separate from buyer review records.

Source evidence read on 2026-10-29 (references, not imported implementations):
* Taobao: JeremyDong22/taobao_mcp, taobao_scraper.py at 4cdeb502, declares
  #picGalleryEle / .picGallery--qY53_w0u and .desc-root.
* Douyin: the official H5 script linked by the actual product HTML declares
  .head-figure__media-view__content (CSS background image) and
  .product-big-img-list__every-img (detail image). Its head figure also contains
  buyer-show, comments, matches, live and 3D cards: those are explicitly excluded.
* XHS: the public TOC response for 6a83ca5ffa5d5c0001a98ca5 was read again
  anonymously with HTTP 200 on 2026-10-29. Both carouselH5.images and
  graphicDetailsV4.images contain direct .url objects; carouselH5.video was null.
  No video object schema is inferred from that null value.

Taobao completion is limited to the currently exposed, materialized product-page
DOM. It requires explicit roots, resolved media nodes, reached product-scope ends,
and repeated stable bottom observations. SKU variants, virtual slides not exposed
by that page and hidden templates are not included. Other platforms stay partial.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections.abc import Mapping
from urllib.parse import parse_qs, urlsplit

from ._media import media_url
from . import douyin, taobao
from ..links import classify_url


TAOBAO_SOURCE = "https://github.com/JeremyDong22/taobao_mcp/blob/4cdeb50297929f76e15704e812a3aa15e050919c/taobao_scraper.py"
DOUYIN_SOURCE = "https://lf3-ecom-toc.jinritemai.com/obj/ecom-toc/ecommerce/gecko/resource/js/index.31f7411b.js"
XHS_SOURCE = "https://github.com/GitBinS/xhs-product-search/blob/main/main.js#L233"
XHS_TOC = "https://mall.xiaohongshu.com/api/store/jpd/edith/detail/h5/toc"


def _product_id(source_url, platform):
    try:
        link = classify_url(source_url)
        if link.platform == platform and not link.is_short:
            return link.product_id or ""
    except (TypeError, ValueError):
        pass
    return ""


def _xhs_response_matches(response_url, product_id):
    try:
        url = urlsplit(response_url)
        return (url.scheme == "https" and url.hostname == "mall.xiaohongshu.com"
                and not url.username and not url.password
                and url.path == "/api/store/jpd/edith/detail/h5/toc"
                and parse_qs(url.query).get("item_id") == [product_id])
    except (TypeError, ValueError):
        return False


def _entry(url, media_type, source_type):
    value = media_url(url, media_type)
    return {"url": value, "media_type": media_type, "source_type": source_type} if value else None


def parse_xhs_product_media(payload, source_url, response_url) -> list[dict]:
    """Read only observed product component paths of an ID-bound TOC response."""
    product = _product_id(source_url, "xiaohongshu")
    if not product or not _xhs_response_matches(response_url, product):
        return []
    if not isinstance(payload, Mapping) or payload.get("success") is not True:
        return []
    data = payload.get("data")
    blocks = data.get("template_data", []) if isinstance(data, Mapping) else []
    if not isinstance(blocks, list):
        return []
    result = {}
    for block in blocks:
        if not isinstance(block, Mapping):
            continue
        carousel = block.get("carouselH5")
        if isinstance(carousel, Mapping) and carousel.get("skuId") not in (None, "", product):
            # The verified sample identifies its requested SKU here. Do not
            # silently cross product/SKU identities when this value disagrees.
            continue
        for name, source_type in (("carouselH5", "product_main"), ("graphicDetailsV4", "product_detail")):
            component = block.get(name)
            images = component.get("images", []) if isinstance(component, Mapping) else []
            if not isinstance(images, list):
                continue
            for image in images:
                if not isinstance(image, Mapping):
                    continue
                # A future video/cover media object is an unknown schema, not a
                # reason to classify its cover .url as a product photograph.
                if any(image.get(key) not in (None, "") for key in ("videoId", "media", "type")):
                    continue
                entry = _entry(image.get("url"), "image", source_type)
                if entry:
                    host = urlsplit(entry["url"]).hostname or ""
                    if host == "xhscdn.com" or host.endswith(".xhscdn.com"):
                        result[(source_type, entry["url"])] = entry
    return list(result.values())


# Native <video> URLs are read only inside the confirmed product media roots.
# Posters and video thumbnail images are not returned as video payloads.
_TAOBAO_PRODUCT_ROOTS = r"""() => {
  const excluded = '[class*="Comment--"], .rate-item, [class*="recommend"], [class*="Recommend"]';
  const roots = selector => {
    const all = [...document.querySelectorAll(selector)].filter(e => e.getClientRects().length
      && getComputedStyle(e).visibility !== 'hidden' && !e.closest(excluded));
    return all.filter(e => !all.some(other => other !== e && other.contains(e)));
  };
  return {main:roots('#picGalleryEle, .picGallery, .picGallery--qY53_w0u'), detail:roots('.desc-root')};
}"""

_TAOBAO_SCROLL_ROOT = r"""roots => {
  const root = roots.detail[0] || roots.main[0];
  for (let e = root && root.parentElement; e && e !== document.body && e !== document.documentElement; e = e.parentElement) {
    if (/(auto|scroll)/.test(getComputedStyle(e).overflowY) && e.scrollHeight > e.clientHeight + 2) return e;
  }
  return document.scrollingElement || document.documentElement || document.body;
}"""

_TAOBAO_SCROLL_SCOPE = r"""() => {
  const roots = (""" + _TAOBAO_PRODUCT_ROOTS + r""")();
  const element = (""" + _TAOBAO_SCROLL_ROOT + r""")(roots);
  if (!element) return {roots, element:null, bottom:null};
  const documentScroll = element === document.scrollingElement || element === document.documentElement || element === document.body;
  const viewportTop = documentScroll ? 0 : element.getBoundingClientRect().top;
  const inContext = root => documentScroll || element.contains(root);
  const bounds = [...roots.main, ...roots.detail].filter(inContext).map(root => {
    const rect = root.getBoundingClientRect();
    return rect.height > 0 ? Math.ceil(rect.bottom - viewportTop + element.scrollTop) : null;
  }).filter(value => value !== null);
  // A missing detail root never authorizes searching the document/recommendations.
  return {roots, element, documentScroll, viewportTop,
    bottom:roots.detail.length && bounds.length ? Math.max(...bounds) : null};
}"""

_PRODUCT_SCROLL = r"""platform => {
  const scope = platform === 'taobao' ? (""" + _TAOBAO_SCROLL_SCOPE + r""")() : null;
  const e = scope ? scope.element : document.scrollingElement;
  if (!e) return {moved:false};
  const before = e.scrollTop;
  if (scope && scope.bottom === null) return {before, after:before, moved:false, reason:'missing_detail_root'};
  const limit = scope ? Math.max(0, scope.bottom - e.clientHeight) : e.scrollHeight - e.clientHeight;
  // Also move back into the product scope if a tab/browser restored a position
  // below it. Recompute every time so materialized lazy content can extend it.
  const target = Math.min(limit, before + Math.max(300, e.clientHeight * .8));
  e.scrollBy(0, target - before);
  return {before, after:e.scrollTop, moved:e.scrollTop !== before, scope_bottom:scope ? scope.bottom : null};
}"""

_PRODUCT_DOM = r"""platform => {
  const visible = e => !!(e && e.getClientRects().length && getComputedStyle(e).visibility !== 'hidden');
  const absolute = value => {
    if (!/^(https?:)?\/\//i.test(value || '')) return '';
    try { return new URL(value, location.href).href; } catch (_) { return ''; }
  };
  const out = [], scopes = new Set();
  const observations = {};
  const taobaoScope = platform === 'taobao' ? (""" + _TAOBAO_SCROLL_SCOPE + r""")() : null;
  const scrollRoot = taobaoScope ? taobaoScope.element : null;
  const documentScroll = scrollRoot && (scrollRoot === document.scrollingElement || scrollRoot === document.documentElement || scrollRoot === document.body);
  const viewportTop = !scrollRoot || documentScroll ? 0 : scrollRoot.getBoundingClientRect().top;
  const roots = selector => {
    const all = [...document.querySelectorAll(selector)];
    return all.filter(e => !all.some(other => other !== e && other.contains(e)));
  };
  let unresolved = 0;
  const exclude = platform === 'douyin'
    ? '.head-figure__buyershow, .head-figure__comment, .head-figure__matchbuy, .head-figure__threeD, .head-figure__live, .shop-recommend, .comment-info, .comment-info-v2, .Ez3MC6d7'
    : '[class*="Comment--"], .rate-item, [class*="recommend"], [class*="Recommend"]';
  const add = (url, kind, source) => { url = absolute(url); if (url) out.push({url, media_type:kind, source_type:source}); };
  const source = e => [e.currentSrc, e.getAttribute('data-src'), e.getAttribute('src')].map(absolute).find(Boolean) || '';
  const resourceDescription = value => {
    const url = absolute(value);
    if (!url) return null;
    const parsed = new URL(url);
    // Diagnostic identity only: never copy expiring authorization query values.
    return {host:parsed.hostname, path:parsed.pathname};
  };
  const collect = (root, sourceType, imageSelector, backgrounds) => {
    if (!visible(root) || root.closest(exclude)) return;
    scopes.add(sourceType);
    const observed = observations[sourceType] || (observations[sourceType] = {
      root_count:0, image_nodes:0, loaded_images:0, loading_images:0, broken_images:0,
      unresolved_images:0, lazy_pending:0, iframe_count:0, expand_controls:0, busy_markers:0,
      video_slots:0, unresolved_video_slots:0, hidden_cover_slots:0,
      root_bounds:[], node_states:[], video_slot_details:[]
    });
    observed.root_count++;
    const rect = root.getBoundingClientRect();
    observed.root_bounds.push({height:Math.ceil(rect.height),
      in_scroll_context:!scrollRoot || documentScroll || scrollRoot.contains(root),
      bottom:Math.ceil(rect.bottom - viewportTop + (scrollRoot ? scrollRoot.scrollTop : 0))});
    observed.iframe_count += root.querySelectorAll('iframe').length;
    observed.busy_markers += [...root.querySelectorAll('[aria-busy="true"], [role="progressbar"]')].filter(visible).length;
    observed.expand_controls += [...root.querySelectorAll('*')].filter(e => {
      if (!visible(e)) return false;
      const text = (e.textContent || '').trim();
      return /^(?:展开全部|展开详情|展开图文详情|查看完整详情|点击查看完整详情|查看更多详情|加载更多|查看更多|查看全部)$/.test(text)
        && (['BUTTON','A'].includes(e.tagName) || e.getAttribute('role') === 'button' || e.tabIndex >= 0 || getComputedStyle(e).cursor === 'pointer');
    }).length;
    const players = [...root.querySelectorAll('video')].filter(v => !v.closest(exclude));
    const posters = new Set(players.map(v => absolute(v.getAttribute('poster'))).filter(Boolean));
    const playerUrls = player => [player.currentSrc, player.getAttribute('src'), player.getAttribute('data-src'),
      ...[...player.querySelectorAll('source')].flatMap(s => [s.getAttribute('src'), s.getAttribute('data-src')])]
      .map(absolute).filter(url => url && !/\.(?:jpe?g|png|webp|gif|avif|svg|bmp|ico)(?:$|[/_!])/i.test(new URL(url).pathname));
    for (const player of players) {
      const urls = playerUrls(player);
      if (platform !== 'taobao' && !urls.length) unresolved++;
      for (const url of new Set(urls)) add(url, 'video', 'product_video');
    }
    if (platform === 'taobao') {
      const slots = [...root.querySelectorAll('[class*="mainPicVideo--"], [class*="videoPicWrap--"]')].filter(e => !e.closest(exclude));
      const outer = slots.filter(e => !slots.some(other => other !== e && other.contains(e)));
      for (const player of players) if (!outer.some(slot => slot.contains(player))) outer.push(player);
      // Live Tmall 2026-11-09: the gallery keeps a hidden cover layer
      // (videoPicWrap--*, opacity:0, z-index:-1, one <img>, no <video>) as a
      // sibling of the real mainPicVideo--* player. It is that player's poster,
      // not an unbound second video entry. Ignore it only while a resolved
      // player exists in this root: a genuinely missing video must still block
      // completion instead of being explained away.
      const hiddenCover = slot => {
        if (slot.tagName === 'VIDEO' || slot.querySelectorAll('video').length) return false;
        const style = getComputedStyle(slot);
        const hidden = style.visibility === 'hidden' || style.display === 'none'
          || parseFloat(style.opacity || '1') === 0 || parseInt(style.zIndex || '0', 10) < 0;
        return hidden && slot.querySelectorAll('img').length > 0;
      };
      const hasResolvedPlayer = players.some(player => playerUrls(player).length > 0);
      const counted = outer.filter(slot => !(hasResolvedPlayer && hiddenCover(slot)));
      observed.hidden_cover_slots += outer.length - counted.length;
      observed.video_slots += counted.length;
      for (const slot of counted) {
        const children = slot.tagName === 'VIDEO' ? [slot] : [...slot.querySelectorAll('video')];
        const resolved = children.filter(player => playerUrls(player).length > 0);
        if (!resolved.length) observed.unresolved_video_slots++;
        if (observed.video_slot_details.length < 20) observed.video_slot_details.push({
          kind:String(slot.className || '').includes('mainPicVideo--') ? 'main_player_container'
            : String(slot.className || '').includes('videoPicWrap--') ? 'video_entry_unbound' : 'native_player',
          tag:slot.tagName, id:slot.getAttribute('id') || '', class_name:String(slot.className || '').slice(0,200),
          aria_controls:slot.getAttribute('aria-controls') || '', data_video_id:slot.getAttribute('data-video-id') || '',
          player_count:children.length, resolved_player_count:resolved.length,
          player_resources:resolved.flatMap(player => playerUrls(player).map(resourceDescription)).slice(0,8),
          poster_resources:children.map(player => resourceDescription(player.getAttribute('poster'))).filter(Boolean).slice(0,4),
          preview_resources:[...slot.querySelectorAll('img')].map(image => resourceDescription(source(image))).filter(Boolean).slice(0,4)
        });
      }
    }
    for (const img of root.querySelectorAll(imageSelector)) {
      if (img.closest(exclude) || img.closest('video') || img.closest('[class*="video"], [class*="Video"]')) continue;
      const url = source(img);
      if (posters.has(url)) continue;
      if (platform !== 'taobao' && img.naturalWidth > 0 && img.naturalWidth < 80) continue;
      const declared = absolute(img.getAttribute('data-src'));
      const src = absolute(img.getAttribute('src'));
      const lazyPending = !!(declared && declared !== url && declared !== src);
      observed.image_nodes++;
      if (!url) observed.unresolved_images++;
      if (!img.complete) observed.loading_images++;
      else if (!(img.naturalWidth > 0 && img.naturalHeight > 0)) observed.broken_images++;
      else observed.loaded_images++;
      if (lazyPending) observed.lazy_pending++;
      observed.node_states.push([url, src, declared, !!img.complete, img.naturalWidth, img.naturalHeight]);
      // Size alone cannot distinguish real content from a placeholder. Preserve
      // observed URLs, and keep unresolved/load evidence separate from success.
      if (url) add(url, 'image', sourceType);
    }
    for (const image of root.querySelectorAll(backgrounds || ':not(*)')) {
      if (image.closest(exclude)) continue;
      const value = getComputedStyle(image).backgroundImage || '';
      const match = /^url\(["']?(.*?)["']?\)$/.exec(value);
      if (match && !posters.has(absolute(match[1]))) add(match[1], 'image', sourceType);
    }
  };
  if (platform === 'taobao') {
    for (const root of taobaoScope.roots.main) collect(root, 'product_main', 'img');
    for (const root of taobaoScope.roots.detail) collect(root, 'product_detail', 'img');
  } else if (platform === 'douyin') {
    for (const root of document.querySelectorAll('.head-figure__media-view')) {
      collect(root, 'product_main', ':not(*)', '.head-figure__media-view__content');
      // H5 displays a cover until its player opens. A cover is not a video URL.
      unresolved += root.querySelectorAll('.head-figure__media-view__video__main-pic').length;
    }
    for (const root of document.querySelectorAll('.product-big-img-list')) collect(root, 'product_detail', 'img.product-big-img-list__every-img');
  }
  const ui = [];
  if (document.body) {
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const p = walker.currentNode.parentElement;
      if (visible(p) && !p.closest(exclude + ', script, style, noscript, .desc-root, .product-big-img-list')) {
        const text = (walker.currentNode.textContent || '').trim();
        if (text) ui.push(text);
      }
    }
  }
  let completion = null;
  if (platform === 'taobao' && scrollRoot) {
    unresolved = Object.values(observations).reduce((n, scope) => n + scope.unresolved_video_slots, 0);
    const detail = observations.product_detail;
    const bottom = scrollRoot.scrollTop + scrollRoot.clientHeight;
    completion = {schema:'taobao_product_dom_scope_v1',
      scroll_context:documentScroll ? 'document' : [scrollRoot.tagName, scrollRoot.id || '', String(scrollRoot.className || '')].join(':').slice(0,200),
      scroll_top:Math.round(scrollRoot.scrollTop), client_height:scrollRoot.clientHeight, scroll_height:scrollRoot.scrollHeight,
      scope_bottom:taobaoScope.bottom,
      at_bottom:scrollRoot.clientHeight > 0 && taobaoScope.bottom !== null && bottom >= taobaoScope.bottom - 4,
      detail_end_reached:!!(detail && detail.root_bounds.length && detail.root_bounds.every(r => r.height > 0 && r.bottom <= bottom + 4)),
      scopes:observations};
  }
  return {schema:'product_media_dom_v1', media:out, scopes:[...scopes], unresolved_videos:unresolved,
    completion, ui_text:ui.join('\n').slice(0,30000)};
}"""


def _snapshot_media(snapshot, platform):
    if not isinstance(snapshot, Mapping) or snapshot.get("schema") != "product_media_dom_v1":
        return []
    result = {}
    for row in snapshot.get("media", []):
        if not isinstance(row, Mapping):
            continue
        kind, source = row.get("media_type"), row.get("source_type")
        if (kind, source) not in {("image", "product_main"), ("image", "product_detail"), ("video", "product_video")}:
            continue
        item = _entry(row.get("url"), kind, source)
        if item:
            host = urlsplit(item["url"]).hostname or ""
            if platform == "taobao" and kind == "image" and not any(
                host == domain or host.endswith("." + domain) for domain in ("alicdn.com", "taobaocdn.com")
            ):
                continue
            result[(source, item["url"])] = item
    return list(result.values())


_COMPLETION_SCOPE = "current_product_page_exposed_dom"
_SCOPE_COUNTERS = (
    "root_count", "image_nodes", "loaded_images", "loading_images", "broken_images",
    "unresolved_images", "lazy_pending", "iframe_count", "expand_controls", "busy_markers",
    "video_slots", "unresolved_video_slots",
)


def _taobao_scope_bottom(observation):
    """Use validated product bounds, never the document's recommendation tail."""
    bounds = [bound for name in ("product_main", "product_detail")
              for bound in observation["scopes"][name]["root_bounds"]
              if bound.get("in_scroll_context", True)]
    return max((bound["bottom"] for bound in bounds), default=0)


def _taobao_scope_signature_observation(observation):
    # Recommendation widgets may grow even while the product is fully stable.
    # Keep the document height in diagnostics, but not in completion evidence.
    return {key: value for key, value in observation.items() if key != "scroll_height"}


def _taobao_completion_observation(snapshot):
    """Require explicit positive evidence; legacy/missing metadata cannot pass."""
    observation = snapshot.get("completion")
    if not isinstance(observation, Mapping) or observation.get("schema") != "taobao_product_dom_scope_v1":
        return False, "missing_dom_observation", {}, None
    evidence = {key: observation.get(key) for key in (
        "scroll_context", "scroll_top", "client_height", "scroll_height", "scope_bottom", "at_bottom", "detail_end_reached")}
    scopes = observation.get("scopes")
    # Keep both scopes in the diagnostic even when the first one blocks. This
    # is an observation record, not a claim that either scope passed validation.
    evidence["scopes"] = {
        name: {key: scope.get(key) for key in _SCOPE_COUNTERS + ("root_bounds", "video_slot_details")}
        for name in ("product_main", "product_detail")
        if isinstance(scopes, Mapping) and isinstance(scope := scopes.get(name), Mapping)
    }
    for key in ("scroll_top", "client_height", "scroll_height"):
        value = observation.get(key)
        if type(value) is not int or value < 0 or key != "scroll_top" and value == 0:
            return False, "invalid_scroll_observation", evidence, None
    if not isinstance(observation.get("scroll_context"), str) or not observation["scroll_context"]:
        return False, "missing_scroll_context", evidence, None
    if not isinstance(scopes, Mapping):
        return False, "missing_product_roots", evidence, None
    for name in ("product_main", "product_detail"):
        scope = scopes.get(name)
        if not isinstance(scope, Mapping):
            return False, "missing_" + name + "_root", evidence, None
        if any(type(scope.get(key)) is not int or scope[key] < 0 for key in _SCOPE_COUNTERS):
            return False, "incomplete_" + name + "_observation", evidence, None
        if scope["root_count"] == 0 or scope["image_nodes"] == 0:
            return False, "empty_or_missing_" + name + "_root", evidence, None
        bounds = scope.get("root_bounds")
        if (not isinstance(bounds, list) or len(bounds) != scope["root_count"] or
                any(not isinstance(r, Mapping) or any(type(r.get(k)) not in (int, float) or not math.isfinite(r[k])
                    for k in ("height", "bottom")) or r["height"] <= 0 for r in bounds)):
            return False, "invalid_" + name + "_bounds", evidence, None
        nodes = scope.get("node_states")
        if not isinstance(nodes, list) or len(nodes) != scope["image_nodes"]:
            return False, "missing_" + name + "_node_states", evidence, None
        for key in ("loading_images", "broken_images", "unresolved_images", "lazy_pending",
                    "iframe_count", "expand_controls", "busy_markers", "unresolved_video_slots"):
            if scope[key]:
                return False, name + "_" + key, evidence, None
        if scope["loaded_images"] != scope["image_nodes"]:
            return False, name + "_images_not_settled", evidence, None
    if type(snapshot.get("unresolved_videos")) is not int or snapshot["unresolved_videos"] != 0:
        return False, "unresolved_video_observation", evidence, None
    scope_bottom = _taobao_scope_bottom(observation)
    if ("scope_bottom" in observation and observation["scope_bottom"] != scope_bottom):
        return False, "invalid_product_scope_bounds", evidence, None
    if (observation.get("at_bottom") is not True or
            observation["scroll_top"] + observation["client_height"] < scope_bottom - 4):
        return False, "product_scope_end_not_reached", evidence, None
    if (observation.get("detail_end_reached") is not True or any(
            r["bottom"] > observation["scroll_top"] + observation["client_height"] + 4
            for r in scopes["product_detail"]["root_bounds"])):
        return False, "detail_end_not_reached", evidence, None
    # If the DOM exposed a resource which the parser rejected, its scope is not
    # fully handled. A recognized URL still has to pass the download validator.
    rows = snapshot.get("media")
    if not isinstance(rows, list) or not rows:
        return False, "no_observed_media", evidence, None
    raw_keys = set()
    for row in rows:
        if not isinstance(row, Mapping) or any(not isinstance(row.get(k), str) for k in ("source_type", "media_type", "url")):
            return False, "invalid_observed_media", evidence, None
        raw_keys.add((row["source_type"], row["media_type"], row["url"]))
    accepted = _snapshot_media(snapshot, "taobao")
    accepted_keys = {(row["source_type"], row["media_type"], row["url"]) for row in accepted}
    if raw_keys != accepted_keys:
        return False, "unhandled_observed_media", evidence, None
    if sum(scopes[name]["video_slots"] for name in ("product_main", "product_detail")) and not any(
            row["media_type"] == "video" for row in accepted):
        return False, "missing_observed_video_url", evidence, None
    for name in ("product_main", "product_detail"):
        if not any(row["source_type"] == name for row in accepted):
            return False, "no_" + name + "_media", evidence, None
    signature = hashlib.sha256(json.dumps(
        {"observation": _taobao_scope_signature_observation(observation), "media": sorted(accepted_keys)}, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")).hexdigest()
    return True, "bottom_ready", evidence, signature


def _taobao_unresolved_end_signature(snapshot, blocker):
    """An observed stable end can stop retries, but can never prove completion.

    Require actual product-scope end positions and complete observations even
    when a pending image, unknown video entry or embedded content blocks the
    completion path. Missing metadata is not a reason to stop scrolling early.
    """
    observation = snapshot.get("completion")
    if (not isinstance(observation, Mapping) or
            observation.get("schema") != "taobao_product_dom_scope_v1" or
            observation.get("at_bottom") is not True or
            observation.get("detail_end_reached") is not True or
            not isinstance(observation.get("scroll_context"), str) or not observation["scroll_context"]):
        return None
    for key in ("scroll_top", "client_height", "scroll_height"):
        value = observation.get(key)
        if type(value) is not int or value < 0 or key != "scroll_top" and value == 0:
            return None
    bottom = observation["scroll_top"] + observation["client_height"]
    scopes = observation.get("scopes")
    if not isinstance(scopes, Mapping):
        return None
    for name in ("product_main", "product_detail"):
        scope = scopes.get(name)
        if (not isinstance(scope, Mapping) or
                any(type(scope.get(key)) is not int or scope[key] < 0 for key in _SCOPE_COUNTERS) or
                scope["root_count"] == 0 or scope["image_nodes"] == 0):
            return None
        bounds, nodes = scope.get("root_bounds"), scope.get("node_states")
        if (not isinstance(bounds, list) or len(bounds) != scope["root_count"] or
                any(not isinstance(r, Mapping) or any(type(r.get(k)) not in (int, float) or not math.isfinite(r[k])
                    for k in ("height", "bottom")) or r["height"] <= 0 for r in bounds) or
                not isinstance(nodes, list) or len(nodes) != scope["image_nodes"]):
            return None
        if name == "product_detail" and any(r["bottom"] > bottom + 4 for r in bounds):
            return None
    scope_bottom = _taobao_scope_bottom(observation)
    if (bottom < scope_bottom - 4 or
            "scope_bottom" in observation and observation["scope_bottom"] != scope_bottom):
        return None
    rows = snapshot.get("media")
    if (not isinstance(rows, list) or not rows or
            any(not isinstance(row, Mapping) or any(not isinstance(row.get(k), str)
                for k in ("source_type", "media_type", "url")) for row in rows) or
            type(snapshot.get("unresolved_videos")) is not int or snapshot["unresolved_videos"] < 0):
        return None
    return hashlib.sha256(json.dumps({
        "observation": _taobao_scope_signature_observation(observation),
        "media": sorted({(row["source_type"], row["media_type"], row["url"]) for row in rows}),
        "unresolved_videos": snapshot["unresolved_videos"], "blocker": blocker,
    }, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _completion_reason(code):
    if "root" in code or "bounds" in code or "node_states" in code or "observation" in code:
        return "主图、详情容器或加载证据尚不完整"
    if code == "product_scope_end_not_reached":
        return "当前商品主图与详情的末尾尚未进入可见范围"
    if code == "detail_end_not_reached":
        return "详情内容末尾尚未进入可见范围"
    if "iframe" in code:
        return "商品容器内仍有未解析的嵌入内容"
    if "expand" in code:
        return "商品容器内仍有需要展开的内容"
    if "video" in code:
        return "商品视频入口的对应关系或实际地址尚未确认"
    if "broken" in code:
        return "页面仍有未成功加载的商品图片"
    if code == "bottom_ready":
        return "当前商品素材区域已到底，仍需连续稳定观察"
    if "media" in code:
        return "部分已显示的商品资源尚未适配"
    return "商品图片或懒加载内容尚未稳定"


def _url_blocker(url):
    try:
        host = urlsplit(url).hostname or ""
        if host in {"login.taobao.com", "login.tmall.com", "login.m.taobao.com", "passport.douyin.com"}:
            return "login_required", "商品页面要求登录，请在软件浏览器中完成后继续。"
        if host == "sec.taobao.com" or "_____tmd_____" in url:
            return "verification_required", "商品页面要求人工安全验证，已停止自动操作。"
    except (TypeError, ValueError):
        pass
    return None


async def collect_product_media(page, source_url, platform, max_scrolls=40) -> dict:
    """Use the supplied session; never create a browser or export credentials."""
    saved = {}
    pending = []
    product = _product_id(source_url, platform)
    diagnostics = {"adapter": "product_media", "platform": platform, "product_id": product,
                   "scrolls": 0, "response_count": 0, "unresolved_videos": 0, "unresolved_videos_peak": 0,
                   "scopes": [], "errors": [], "completeness_verified": False,
                   "source": {"taobao": TAOBAO_SOURCE, "douyin": DOUYIN_SOURCE,
                              "xiaohongshu": XHS_SOURCE}.get(platform, "")}
    if platform == "taobao":
        diagnostics.update(completion_scope=_COMPLETION_SCOPE, stable_bottom_observations=0,
                           scroll_boundary="current_product_roots", missing_detail_observations=0,
                           stable_unresolved_end_observations=0, unresolved_end_evidence=None,
                           completion_evidence=None, completion_blocker="not_observed",
                           excluded_scopes=["other_sku_variants", "unexposed_virtual_gallery_items", "hidden_templates"])

    def result(status, reason, complete=False):
        diagnostics["media_count"] = len(saved)
        diagnostics["completeness_verified"] = bool(complete)
        return {"status": status, "media": list(saved.values()), "complete": bool(complete),
                "reason": reason, "diagnostics": diagnostics}

    def merge(rows):
        for row in rows:
            saved[(row["source_type"], row["media_type"], row["url"])] = row

    def seen(response):
        if _xhs_response_matches(response.url, product):
            if len(pending) < 16:
                pending.append(response)

    async def read_response(response):
        if not _xhs_response_matches(response.url, product):
            return None
        diagnostics["response_count"] += 1
        if response.status in {401, 403, 429}:
            return ("login_required" if response.status == 401 else "access_restricted",
                    "商品接口要求登录或限制访问，已停止。")
        if response.status != 200:
            diagnostics["errors"].append("response_http_error")
            return None
        try:
            payload = await asyncio.wait_for(response.json(), timeout=10)
            # Read the payload only while the visible page is still this item.
            if _product_id(getattr(page, "url", ""), platform) != product:
                return "product_mismatch", "读取商品响应时页面身份改变，已丢弃该响应。"
            merge(parse_xhs_product_media(payload, source_url, response.url))
        except Exception as exc:
            diagnostics["errors"].append("response:" + type(exc).__name__)
        return None

    if not product:
        return result("unsupported_page", "需要已解析且可确认商品 ID 的商品详情链接。")
    if not isinstance(max_scrolls, int) or not 1 <= max_scrolls <= 1000:
        return result("error", "加载次数必须为 1 至 1000 的整数。")
    listening = False
    try:
        if platform == "xiaohongshu":
            page.on("response", seen)
            listening = True
        # A normal navigation closes any review overlay and observes the same
        # current product anew, using the caller's existing browser profile.
        try:
            await page.goto(source_url, wait_until="domcontentloaded", timeout=45000)
        except Exception as exc:
            diagnostics["errors"].append("navigation:" + type(exc).__name__)
            blocker = _url_blocker(getattr(page, "url", ""))
            if blocker:
                return result(*blocker)
            return result("error", "商品页面未成功加载，尚未提取媒体。")
        await page.wait_for_timeout(1200)
        blocker = _url_blocker(getattr(page, "url", ""))
        if blocker:
            return result(*blocker)
        if _product_id(getattr(page, "url", ""), platform) != product:
            return result("product_mismatch", "当前页面不是输入商品，停止提取商品媒体。")
        if platform == "taobao":
            # Live 950216355977 on 2026-10-30: the gallery's mainPicVideo
            # player initializes later than the image thumbnails. Reading once
            # and immediately opening details loses that actual product video.
            gallery_video = page.locator('#picGalleryEle [class*="mainPicVideo--"] video')
            video_slot = page.locator('#picGalleryEle [class*="mainPicVideo--"], #picGalleryEle [class*="videoPicWrap--"]')
            if await video_slot.count() and hasattr(gallery_video, "wait_for"):
                try:
                    await gallery_video.first.wait_for(state="attached", timeout=7000)
                    await page.wait_for_function("""() => [...document.querySelectorAll('#picGalleryEle video')]
                        .some(v => /^(https?:)?\\/\\//.test(v.currentSrc || v.getAttribute('src') || ''))""", timeout=5000)
                except Exception as exc:
                    diagnostics["errors"].append("gallery_video_wait:" + type(exc).__name__)
        if platform == "xiaohongshu":
            ui = await page.evaluate(_PRODUCT_DOM, platform)
            for pattern, status, reason in (
                (r"^(?:请先登录|登录后查看|扫码登录|密码登录)$", "login_required", "商品页面要求登录，已停止。"),
                (r"^(?:请完成安全验证|拖动滑块|点击完成验证|请完成以下验证)$", "verification_required", "商品页面要求人工安全验证，已停止。"),
                (r"^(?:访问受限|访问过于频繁|操作过于频繁)$", "access_restricted", "商品页面限制访问，已停止。"),
                (r"^(?:商品已下架|商品不存在|商品已失效)$", "product_unavailable", "商品页面显示商品不可用。"),
            ):
                if re.search(pattern, ui.get("ui_text", ""), re.M):
                    return result(status, reason)
            while pending:
                blocker = await read_response(pending.pop(0))
                if blocker:
                    return result(*blocker)
            if not saved and not diagnostics["response_count"]:
                # The same read-only endpoint was independently observed. One
                # bounded request via this browser's API context, no cookie IO.
                response = await page.request.get(XHS_TOC + "?version=0.0.5&item_id=" + product, timeout=15000)
                blocker = await read_response(response)
                if blocker:
                    return result(*blocker)
            return result("partial" if saved else "unsupported_page",
                          "已读取商品主图和详情图；商品视频结构及媒体总量尚未确认。" if saved
                          else "未取得已适配的商品媒体响应，不能认定商品没有图片或视频。")

        previous, unchanged = None, 0
        completion_signature, stable_bottom = None, 0
        bottom_observations = []
        unresolved_end_signature, stable_unresolved_end = None, 0
        unresolved_end_observations = []
        missing_detail_observations = 0
        detail_tab_retried = False
        for turn in range(max_scrolls + 1):
            blocker = _url_blocker(getattr(page, "url", ""))
            if blocker:
                return result(*blocker)
            if _product_id(getattr(page, "url", ""), platform) != product:
                return result("product_mismatch", "页面已离开输入商品，停止提取商品媒体。")
            snapshot = await page.evaluate(_PRODUCT_DOM, platform)
            if _product_id(getattr(page, "url", ""), platform) != product:
                return result("product_mismatch", "读取商品媒体时页面身份改变，已丢弃该次结果。")
            if platform == "taobao":
                blocker = await taobao._page_problem(page)
            else:
                blocker = douyin._blocking_state(snapshot.get("ui_text", ""))
            if blocker:
                return result(*blocker)
            merge(_snapshot_media(snapshot, platform))
            diagnostics["scopes"] = list(dict.fromkeys(diagnostics["scopes"] + snapshot.get("scopes", [])))
            diagnostics["unresolved_videos"] = int(snapshot.get("unresolved_videos") or 0)
            diagnostics["unresolved_videos_peak"] = max(diagnostics["unresolved_videos_peak"], diagnostics["unresolved_videos"])
            signature = tuple(saved)
            unchanged = unchanged + 1 if previous == signature else 0
            previous = signature
            if platform == "taobao":
                ready, code, observation, current_signature = _taobao_completion_observation(snapshot)
                diagnostics["completion_blocker"] = code
                diagnostics["last_dom_observation"] = observation
                observed_scopes = observation.get("scopes", {})
                detail_root = observed_scopes.get("product_detail", {})
                missing_detail_observations = (0 if detail_root.get("root_count", 0) > 0
                                               else missing_detail_observations + 1)
                diagnostics["missing_detail_observations"] = missing_detail_observations
                # A cold product page sometimes exposes the detail tab late.
                # Retry the already-adapted detail control once; never guess a
                # new selector or treat a missing platform container as our bug.
                if not detail_tab_retried and missing_detail_observations >= 3:
                    detail_tab_retried = True
                    retry = {"at_turn": turn, "missing_observations": missing_detail_observations}
                    try:
                        tabs = page.locator('.tabTitleItem--z4AoobEz').filter(has_text="详情")
                        visible = bool(await tabs.count()) and await tabs.first.is_visible()
                        retry["control_visible"] = visible
                        if visible:
                            await tabs.first.click(timeout=3000)
                            await page.wait_for_timeout(1200)
                            retry["clicked"] = True
                    except Exception as exc:
                        retry["error"] = type(exc).__name__
                    diagnostics["detail_tab_retry"] = retry
                if ready:
                    stable_bottom = stable_bottom + 1 if current_signature == completion_signature else 1
                    completion_signature = current_signature
                    bottom_observations = (bottom_observations + [{"turn": turn, "signature": current_signature}])[-3:]
                else:
                    stable_bottom, completion_signature, bottom_observations = 0, None, []
                diagnostics["stable_bottom_observations"] = stable_bottom
                pending_signature = None if ready else _taobao_unresolved_end_signature(snapshot, code)
                if pending_signature:
                    stable_unresolved_end = stable_unresolved_end + 1 if pending_signature == unresolved_end_signature else 1
                    unresolved_end_signature = pending_signature
                    unresolved_end_observations = (unresolved_end_observations + [
                        {"turn": turn, "signature": pending_signature, "blocker": code}])[-12:]
                else:
                    stable_unresolved_end, unresolved_end_signature, unresolved_end_observations = 0, None, []
                diagnostics["stable_unresolved_end_observations"] = stable_unresolved_end
                # Hitting a budget is never evidence of completion. A completed
                # path needs all three stable observations before that limit.
                if turn >= max_scrolls:
                    diagnostics["exit_reason"] = "loading_limit"
                    return result("partial" if saved else "unsupported_page",
                                  "已保留读取到的商品素材；达到加载上限，" + _completion_reason(code) + "。")
                if stable_bottom >= 3:
                    diagnostics["exit_reason"] = "exposed_dom_scope_completed"
                    diagnostics["completion_blocker"] = None
                    diagnostics["completion_evidence"] = {
                        "scope": _COMPLETION_SCOPE, "stable_observations": bottom_observations,
                        "minimum_observation_interval_ms": 800, "terminal": observation,
                    }
                    return result("completed", "当前网页可见、已物化的商品素材范围已读取完成；不含其他 SKU 变体、未展露的轮播项或隐藏模板。", complete=True)
                if missing_detail_observations >= 12:
                    diagnostics["exit_reason"] = "detail_root_missing"
                    diagnostics["detail_container_unavailable"] = {
                        "observations": missing_detail_observations,
                        "reason": "platform_detail_root_not_exposed",
                        "document_scroll_height": observation.get("scroll_height"),
                        "exposed_scopes": list(observed_scopes),
                        "detail_tab_retry": diagnostics.get("detail_tab_retry"),
                    }
                    return result("partial" if saved else "unsupported_page",
                                  f"已取得 {len(saved)} 项商品素材；当前页面连续 {missing_detail_observations} 次观察均未暴露商品详情容器"
                                  "（平台未展露，非读取失败），已停止滚动，不能确认素材完整。")
                if stable_unresolved_end >= 12:
                    diagnostics["exit_reason"] = "unresolved_at_stable_end"
                    diagnostics["unresolved_end_evidence"] = {
                        "stable_observations": unresolved_end_observations,
                        "minimum_observation_interval_ms": 800, "terminal": observation,
                    }
                    return result("partial", "已保留读取到的商品素材；当前商品素材区域末尾已连续稳定，但" +
                                  _completion_reason(code) + "，尚不能确认当前页面素材范围完整。")
            elif turn >= max_scrolls or unchanged >= 3:
                return result("partial" if saved else "unsupported_page",
                              "已提取可见商品媒体；达到加载上限或不再新增，媒体完整性尚未确认。" if saved
                              else "未发现有依据的商品主图或详情容器，当前页面尚未适配。")
            if turn == 0 and platform == "taobao":
                tabs = page.locator('.tabTitleItem--z4AoobEz').filter(has_text="详情")
                if await tabs.count() and await tabs.first.is_visible():
                    await tabs.first.click(timeout=3000)
            diagnostics["last_scroll"] = await page.evaluate(_PRODUCT_SCROLL, platform)
            diagnostics["scrolls"] += 1
            await page.wait_for_timeout(800)
    except asyncio.CancelledError:
        return result("cancelled", "用户停止采集，已保留读取到的商品媒体。")
    except Exception as exc:
        diagnostics["errors"].append("collect:" + type(exc).__name__)
        return result("partial" if saved else "error", "读取商品媒体失败；已保留结果，完整性尚未确认。")
    finally:
        if listening:
            try:
                page.remove_listener("response", seen)
            except Exception:
                diagnostics["errors"].append("listener_cleanup_failed")
    return result("partial", "商品媒体采集停止，完整性尚未确认。")
