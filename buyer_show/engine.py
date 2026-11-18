"""Sequential local browser jobs with explicit human login and result evidence."""
from __future__ import annotations

import asyncio
from datetime import datetime
import os
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from .adapters import taobao, xhs, douyin
from .adapters.product_media import collect_product_media
from .links import ProductLink, classify_url
from .storage import ImageFetcher, ProductStore, atomic_json
from .video import VideoFetcher

ADAPTERS = {"taobao": taobao, "xiaohongshu": xhs, "douyin": douyin}
MANUAL_STATES = {"login_required", "verification_required"}
TAOBAO_LOGIN = "https://login.taobao.com/member/login.jhtml?f=top&redirectURL=https%3A%2F%2Fwww.taobao.com%2F"


def _direct_connection() -> bool:
    """Opt-in workaround for a dead local proxy (VPN client stopped, stale env vars).

    Disabled by default: the browser and the downloader both keep using the
    machine's normal proxy configuration.
    """
    return os.environ.get("BUYER_SHOW_DIRECT_CONNECTION") == "1"


async def _ensure_live_page(context, page, url, emit=print, force=False):
    """Give the product stage a usable page, keeping the same browser session.

    Review collection may leave the tab closed (platform popup handling, a
    renderer crash, a user closing the window). Reopening from the persistent
    context preserves the saved login state. Navigation errors stay visible to
    the collector so its own diagnostics still describe the real failure.
    """
    closed = page.is_closed() if hasattr(page, "is_closed") else False
    if not force and not closed:
        return page
    if force and not closed:
        try:
            await page.close()
        except Exception:
            pass
    emit("评价阶段后的商品页不可用，正在同一浏览器会话中重新打开商品页。")
    fresh = await context.new_page()
    try:
        await fresh.goto(url, wait_until="domcontentloaded", timeout=45000)
    except Exception as exc:
        emit("重开商品页后导航未完成：" + type(exc).__name__ + "；采集器将记录原始错误。")
    return fresh


def _target_closed(result):
    errors = (result.get("diagnostics") or {}).get("errors") or []
    return any("TargetClosedError" in str(item) for item in errors)


async def _cancellable_collect(adapter, page, source_url, max_scrolls, should_stop):
    return await _cancellable_job(adapter.collect(page, source_url, max_scrolls=max_scrolls), should_stop, "records")


async def _cancellable_job(coro, should_stop, result_key):
    task = asyncio.create_task(coro)
    try:
        while not task.done():
            if should_stop():
                task.cancel()
                final = (await asyncio.gather(task, return_exceptions=True))[0]
                if isinstance(final, dict):
                    final.update(status="cancelled", complete=False)
                    return final
                return {"status": "cancelled", result_key: [], "complete": False,
                        "reason": "用户停止本次采集；已保存的素材可在下次复用。", "diagnostics": {}}
            await asyncio.wait({task}, timeout=0.25)
        return await task
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def _combined_collection(reviews, product):
    """A successful media subsection must never conceal another missing scope."""
    scopes = {"reviews": reviews, "product": product}
    complete = all(x.get("complete") is True and x.get("status") == "completed" for x in scopes.values())
    status = "completed" if complete else "partial"
    if not complete:
        for blocked in ("cancelled", "login_required", "verification_required", "access_restricted", "error"):
            if any(x.get("status") == blocked for x in scopes.values()):
                status = blocked
                break
    return {"status": status, "complete": complete,
            "reason": "；".join(f"{label}：{value.get('reason') or value.get('status', '未完成')}"
                               for label, value in (("评价", reviews), ("商品素材", product))),
            "diagnostics": {name: {"status": value.get("status"), "complete": value.get("complete") is True,
                                    "details": value.get("diagnostics", {})}
                            for name, value in scopes.items()}}


def _is_login(url, platform):
    parts = urlsplit(url)
    if platform == "taobao":
        return parts.hostname in {"login.taobao.com", "login.m.taobao.com", "login.tmall.com"}
    if platform == "xiaohongshu":
        return parts.hostname in {"www.xiaohongshu.com", "xiaohongshu.com"} and parts.path.startswith("/login")
    if platform == "douyin":
        return parts.hostname in {"www.douyin.com", "douyin.com"} and parts.path.startswith(("/login", "/passport"))
    return False


async def _resolve_short(page, link, ask, should_stop):
    for attempt in range(2):
        await page.goto(link.url, wait_until="domcontentloaded", timeout=45000)
        await page.wait_for_timeout(1500)
        if _is_login(page.url, link.platform):
            if attempt == 0 and ask and not should_stop():
                if link.platform == "taobao":
                    await page.goto(TAOBAO_LOGIN, wait_until="domcontentloaded", timeout=30000)
                if await ask("商品分享短链要求先登录。请在已打开的浏览器中手动登录，完成后继续解析原短链。") and not should_stop():
                    continue
            return None, {"status": "login_required", "reason": "分享短链要求登录，尚未确认商品ID。"}
        try:
            target = classify_url(page.url)
        except ValueError:
            return None, {"status": "unsupported_link", "reason": "短链未跳转至已支持的商品详情页；未采集图片。"}
        if target.is_short or target.platform != link.platform:
            return None, {"status": "unsupported_link", "reason": "短链未解析到同平台商品，未采集图片。"}
        return target, None


async def run_batch(links: list[ProductLink], output: Path, profiles: Path,
                    emit=print, ask=None, max_scrolls=40, channel="msedge", should_stop=None,
                    on_event=None) -> list[dict]:
    """Run user-supplied product links. Nothing runs merely by importing this module.

    Profiles are dedicated to this application. Short links are resolved and
    classified before an adapter inspects product or review media. Social links
    remain outside the product-page collection scope.
    """
    from playwright.async_api import async_playwright

    if not 1 <= max_scrolls <= 1000:
        raise ValueError("加载上限须为 1 至 1000")
    if channel not in {"msedge", "chrome", "chromium"}:
        raise ValueError("浏览器须为 msedge、chrome 或 chromium")
    should_stop = should_stop or (lambda: False)
    output, profiles = Path(output).resolve(), Path(profiles).resolve()
    output.mkdir(parents=True, exist_ok=True)
    profiles.mkdir(parents=True, exist_ok=True)
    batch_path = output / ("batch-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".json")
    results, contexts = [], {}

    def task_event(index, **values):
        if on_event:
            try:
                on_event(dict(type="task", index=index, **values))
            except Exception:
                emit("任务状态显示暂时不可用，采集结果仍会保存。")
    emit("测试版：采集商品主图、详情图、商品视频及评价素材；各平台完整性仍需实页验证，小红书评价尚未支持。")
    direct = _direct_connection()
    http_kwargs = {"timeout": 30, "trust_env": False} if direct else {"timeout": 30}
    if direct:
        emit("已启用直连模式（BUYER_SHOW_DIRECT_CONNECTION=1）：浏览器与下载均绕过本机代理。")
    async with async_playwright() as pw, httpx.AsyncClient(**http_kwargs) as client:
        try:
            for number, link in enumerate(links, 1):
                if should_stop():
                    break
                emit(f"任务 {number}/{len(links)}：{link.platform} / {link.product_id or '待解析短链'}")
                task_event(number - 1, platform=link.platform, product_id=link.product_id, status="running")
                manual_prompted = False
                page, store = None, None
                active_stage, stage_reason = "opening_browser", "正在打开软件独立浏览器。"

                def stage_event(stage, reason):
                    nonlocal active_stage, stage_reason
                    active_stage, stage_reason = stage, reason
                    if store:
                        store.data.update(stage=stage, reason=reason)
                        store.checkpoint()
                    emit(reason)
                    task_event(number - 1, platform=target.platform, product_id=target.product_id,
                               status="running", stage=stage, reason=reason,
                               stats=store.stats() if store else {},
                               manifest=str(store.path.resolve()) if store else None)

                async def task_ask(message):
                    nonlocal manual_prompted
                    manual_prompted = True
                    task_event(number - 1, platform=link.platform, product_id=link.product_id,
                               status="waiting_user", stage=active_stage, reason=message)
                    answer = await ask(message)
                    task_event(number - 1, platform=link.platform, product_id=link.product_id,
                               status="running", stage=active_stage, reason=stage_reason)
                    return answer

                report = {"platform": link.platform, "product_id": link.product_id,
                          "source_url": link.url, "status": "error", "complete": False}
                try:
                    if link.platform not in contexts:
                        kwargs = {"headless": False, "locale": "zh-CN", "viewport": {"width": 1360, "height": 900}}
                        if channel != "chromium":
                            kwargs["channel"] = channel
                        if direct:
                            kwargs["args"] = ["--no-proxy-server"]
                        contexts[link.platform] = await pw.chromium.launch_persistent_context(
                            str(profiles / link.platform), **kwargs)
                    context = contexts[link.platform]
                    page = await context.new_page()
                    target = link
                    if link.is_short:
                        target, failure = await _resolve_short(page, link, task_ask if ask else None, should_stop)
                        if failure:
                            report.update(failure)
                            results.append(report)
                            atomic_json(batch_path, results)
                            emit(report["reason"])
                            task_event(number - 1, **report)
                            continue
                    store = ProductStore(output, target.platform, str(target.product_id), link.url)
                    store.begin()
                    attempted_images, attempted_videos = set(), set()
                    fetch = ImageFetcher(client, target.platform, target.url)
                    videos = VideoFetcher(client, target.platform, target.url)

                    def media_progress(message):
                        emit(message)
                        task_event(number - 1, platform=target.platform, product_id=target.product_id,
                                   status="running", stage=active_stage, reason=stage_reason,
                                   stats=store.stats(), manifest=str(store.path.resolve()))

                    async def save_discovered(stage, reason):
                        # A scope is durable before the next browser collector
                        # starts. Failed URLs are tried again only on a new run.
                        stage_event(stage, reason)
                        await store.download(fetch, emit=media_progress, should_stop=should_stop,
                                             skip_urls=attempted_images)
                        attempted_images.update(url for url, item in store.data["images"].items()
                                                if item.get("status") != "pending")
                        await store.download_videos(videos.fetch, emit=media_progress, should_stop=should_stop,
                                                    skip_urls=attempted_videos)
                        attempted_videos.update(url for url, item in store.data["videos"].items()
                                                if item.get("status") != "pending")
                        media_progress("当前已发现素材的下载步骤已结束；保存数仅统计校验成功的原文件。")
                        stage_event("organize_categories", "正在整理分类；已经保存的原文件可随时查看。")
                        categories = await asyncio.to_thread(store.refresh_categories, should_stop=should_stop)
                        if categories.get("cancelled"):
                            emit("素材分类已暂停，已经保存的原文件和分类文件保留。")
                        elif categories.get("failed"):
                            emit(f"有 {categories['failed']} 项素材分类未完成，原文件已保留；详情见分类记录。")
                        elif categories.get("complete"):
                            emit("素材分类已整理，商品主图、详情图、商品视频与买家秀分别保存。")
                        else:
                            emit("已整理可用素材，尚有分类项目未完成，详情见分类记录。")
                        media_progress("已刷新保存数量和分类记录；尚未结束的读取范围仍标记未完成。")

                    stage_event("read_reviews", "正在读取评价和追评；本阶段结束后立即下载已发现的买家秀图片、视频。")
                    adapter = ADAPTERS[target.platform]
                    collection = await _cancellable_collect(adapter, page, target.url, max_scrolls, should_stop)
                    store.merge(collection["records"])
                    # A human may log in once or handle a challenge on the normal site.
                    # Never loop infinitely on a denied/unchanged platform state.
                    if collection["status"] in MANUAL_STATES and ask and not manual_prompted and not should_stop():
                        if target.platform == "taobao" and collection["status"] == "login_required":
                            try:
                                await page.goto(TAOBAO_LOGIN, wait_until="domcontentloaded", timeout=30000)
                            except Exception:
                                emit("登录页暂未加载完成，可在浏览器中手动打开淘宝首页登录。")
                        proceed = await task_ask(collection["reason"] + " 请在已打开的独立浏览器中手动处理，完成后点击继续；会话仅保存在本机。")
                        if proceed and not should_stop():
                            collection = await _cancellable_collect(adapter, page, target.url, max_scrolls, should_stop)
                            store.merge(collection["records"])
                    product = {"status": "not_started", "complete": False, "media": [],
                               "reason": "商品素材尚未读取。"}
                    store.data["scopes"] = _combined_collection(collection, product)["diagnostics"]
                    await save_discovered("download_reviews", "正在下载买家秀及已有素材；先保存原文件，再继续读取商品素材。")
                    if should_stop():
                        product = {"status": "not_started", "complete": False, "media": [],
                                   "reason": "任务已停止。"}
                    else:
                        stage_event("read_product", "正在读取商品主图、详情图和商品视频；已保存的买家秀可在任务目录查看，推荐区不采集。")

                        async def read_product():
                            return await _cancellable_job(
                                collect_product_media(page, target.url, target.platform, max_scrolls=max_scrolls),
                                should_stop, "media")

                        # Review downloads can outlive the tab. Reopen it before
                        # the product stage instead of failing the whole task.
                        page = await _ensure_live_page(context, page, target.url, emit)
                        product = await read_product()
                        if _target_closed(product) and not should_stop():
                            emit("商品素材读取时页面连接中断，正在重新打开页面并重试一次。")
                            page = await _ensure_live_page(context, page, target.url, emit, force=True)
                            product = await read_product()
                            retry_diag = product.get("diagnostics")
                            if not isinstance(retry_diag, dict):
                                retry_diag = product["diagnostics"] = {}
                            retry_diag["recovered_after"] = "target_closed_retry"
                        store.merge_product_media(product.get("media", []))
                        if product["status"] in MANUAL_STATES and ask and not manual_prompted and not should_stop():
                            if target.platform == "taobao" and product["status"] == "login_required":
                                try:
                                    await page.goto(TAOBAO_LOGIN, wait_until="domcontentloaded", timeout=30000)
                                except Exception:
                                    emit("登录页暂未加载完成，可在软件打开的浏览器中手动进入淘宝首页登录。")
                            proceed = await task_ask(product.get("reason", "商品素材需要人工处理。") +
                                                    " 请在软件打开的独立浏览器中处理，完成后点击继续。")
                            if proceed and not should_stop():
                                product = await _cancellable_job(
                                    collect_product_media(page, target.url, target.platform, max_scrolls=max_scrolls),
                                    should_stop, "media")
                                store.merge_product_media(product.get("media", []))
                    combined = _combined_collection(collection, product)
                    store.data["scopes"] = combined["diagnostics"]
                    await save_discovered("download_product", "正在下载商品素材；本次已经尝试的图片和视频不会立即重复请求。")
                    store.data["stage"] = "finished"
                    report = store.finish(combined, cancelled=should_stop())
                    report["source_url"] = link.url
                    report["scopes"] = combined["diagnostics"]
                    emit(report["reason"])
                    if report.get("download_failure_summary"):
                        emit(report["download_failure_summary"])
                    emit(f"已读取评价 {report['stats']['reviews_read']}；唯一图片 {report['stats']['unique_images']}；"
                         f"唯一视频 {report['stats'].get('unique_videos', 0)}；"
                         f"下载失败 {report['stats']['downloads_failed'] + report['stats'].get('videos_failed', 0)}")
                except Exception as exc:
                    report.update(status="error", complete=False, error_type=type(exc).__name__,
                                  reason=("商品素材读取未完成，已经保存的买家秀和分类文件保留，可在任务目录查看。"
                                          if active_stage == "read_product" else
                                          "任务未完成，已保存的原文件保留。请检查浏览器、网络和软件会话目录。"))
                    emit(report["reason"] + f"（{type(exc).__name__}）")
                    if store:
                        try:
                            store.data["stage"] = "error"
                            if active_stage == "read_product":
                                store.data.setdefault("scopes", {})["product"] = {
                                    "status": "error", "complete": False, "details": {"error_type": type(exc).__name__}}
                            saved_error = store.finish({"status": "error", "complete": False,
                                                        "reason": report["reason"], "diagnostics": {"error_type": type(exc).__name__}})
                            report.update(saved_error)
                            report["scopes"] = store.data.get("scopes", {})
                        except Exception:
                            emit("结果目录无法写入，当前批次仍将记录任务错误。")
                finally:
                    if page:
                        try:
                            if not page.is_closed():
                                await page.close()
                        except Exception:
                            emit("页面已关闭或断开；已取得的结果仍会保存。")
                results.append(report)
                atomic_json(batch_path, results)
                task_event(number - 1, **report)
        finally:
            for context in contexts.values():
                try:
                    await context.close()
                except Exception:
                    emit("浏览器会话已断开；请以本批次记录为准。")
    atomic_json(batch_path, results)
    emit("本批次记录：" + str(batch_path))
    return results
