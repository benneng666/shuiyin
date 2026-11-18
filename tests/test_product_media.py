"""Synthetic parser/session fixtures; no live browser or platform assertion."""

import asyncio
import copy
import unittest
from types import SimpleNamespace

from buyer_show.adapters.product_media import (
    XHS_TOC, _PRODUCT_DOM, _snapshot_media, collect_product_media, parse_xhs_product_media,
    _taobao_completion_observation, _PRODUCT_SCROLL,
)


XHS_ID = "6a83ca5ffa5d5c0001a98ca5"
XHS_URL = "https://www.xiaohongshu.com/goods-detail/" + XHS_ID
XHS_RESPONSE = XHS_TOC + "?version=0.0.5&item_id=" + XHS_ID
TB_URL = "https://detail.tmall.com/item.htm?id=950216355977"
DY_URL = "https://haohuo.jinritemai.com/ecommerce/trade/detail/index.html?id=3761019362773959052"
XHS_IMAGE = "https://mall-i1.xhscdn.com/arkgoods/a?imageView2/2/w/1080&sig=a%2Bb"


def xhs_payload():
    return {"success": True, "data": {"template_data": [{
        "carouselH5": {"skuId": XHS_ID, "images": [{"url": XHS_IMAGE, "type": None, "videoId": None}]},
        "graphicDetailsV4": {"images": [{"url": XHS_IMAGE}]},
        "sellerH5": {"logo": "https://mall-i1.xhscdn.com/seller.jpg"},
        "recommendations": [{"carouselH5": {"images": [{"url": "https://mall-i1.xhscdn.com/other.jpg"}]}}],
        "comments": [{"images": [{"url": "https://mall-i1.xhscdn.com/review.jpg"}]}],
    }]}}


class ProductMediaParserTests(unittest.TestCase):
    def test_xhs_exact_components_source_labels_and_url_preservation(self):
        rows = parse_xhs_product_media(xhs_payload(), XHS_URL, XHS_RESPONSE)
        self.assertEqual(rows, [
            {"url": XHS_IMAGE, "media_type": "image", "source_type": "product_main"},
            {"url": XHS_IMAGE, "media_type": "image", "source_type": "product_detail"},
        ])

    def test_xhs_wrong_product_response_or_sku_is_rejected(self):
        for response in (XHS_RESPONSE.replace(XHS_ID, "1" * 24),
                         XHS_RESPONSE.replace("mall.xiaohongshu.com", "mall.xiaohongshu.com.evil.test"),
                         XHS_RESPONSE.replace("/toc?", "/other?"), XHS_RESPONSE + "&item_id=" + XHS_ID):
            self.assertEqual(parse_xhs_product_media(xhs_payload(), XHS_URL, response), [])
        payload = xhs_payload()
        payload["data"]["template_data"][0]["carouselH5"]["skuId"] = "1" * 24
        self.assertEqual(parse_xhs_product_media(payload, XHS_URL, XHS_RESPONSE), [])

    def test_unknown_video_objects_not_interpreted_as_images_or_video_urls(self):
        payload = xhs_payload()
        block = payload["data"]["template_data"][0]
        block["carouselH5"]["video"] = {"cover": XHS_IMAGE, "url": "https://xhscdn.com/unverified.mp4"}
        block["carouselH5"]["images"] = [
            {"url": XHS_IMAGE, "videoId": "v1"}, {"url": XHS_IMAGE, "type": "video"},
            {"url": "https://evil.test/product.jpg"}, {"cover": XHS_IMAGE},
        ]
        self.assertEqual(parse_xhs_product_media(payload, XHS_URL, XHS_RESPONSE), [
            {"url": XHS_IMAGE, "media_type": "image", "source_type": "product_detail"}])

    def test_malformed_and_noncommerce_payloads_are_empty(self):
        for payload in (None, [], {"success": False}, {"success": True, "data": {"template_data": {}}}):
            self.assertEqual(parse_xhs_product_media(payload, XHS_URL, XHS_RESPONSE), [])
        self.assertEqual(parse_xhs_product_media(xhs_payload(), "https://www.xiaohongshu.com/explore/" + XHS_ID, XHS_RESPONSE), [])

    def test_dom_normalization_rejects_avatars_bad_source_pairs_and_video_covers(self):
        rows = [{"url": "https://img.alicdn.com/a.jpg", "media_type": "image", "source_type": "product_main"},
                {"url": "https://img.alicdn.com/a.jpg", "media_type": "video", "source_type": "product_video"},
                {"url": "https://cloud.video.taobao.com/a.mp4?sig=a%2Bb", "media_type": "video", "source_type": "product_video"},
                {"url": "https://img.alicdn.com/u.jpg", "media_type": "image", "source_type": "avatar"},
                {"url": "https://evil.test/a.jpg", "media_type": "image", "source_type": "product_detail"}]
        parsed = _snapshot_media({"schema": "product_media_dom_v1", "media": rows}, "taobao")
        self.assertEqual(parsed, [rows[0], rows[2]])
        self.assertEqual(_snapshot_media({"media": rows}, "taobao"), [])


class EmptyLocator:
    @property
    def first(self):
        return self

    async def count(self):
        return 0

    def filter(self, **_kwargs):
        return self


class Response:
    def __init__(self, payload=None, status=200, url=XHS_RESPONSE):
        self.url, self.status = url, status
        self.payload = payload if payload is not None else xhs_payload()

    async def json(self):
        return self.payload


class Page:
    def __init__(self, url=TB_URL, snapshots=None, response=None, navigate_error=False, change_url=None, pause=False):
        self.url = "about:blank"
        self.target = url
        self.snapshots = snapshots or [snapshot()]
        self.response = response
        self.navigate_error = navigate_error
        self.change_url = change_url
        self.pause = pause
        self.ready = asyncio.Event()
        self.wait_count = 0
        self.snapshot_count = 0
        self.navigations = 0
        self.listener = None
        self.removed = False
        self.requests = []
        self.request = SimpleNamespace(get=self.get)

    async def get(self, url, **kwargs):
        self.requests.append(url)
        return Response()

    def on(self, event, listener):
        self.listener = listener

    def remove_listener(self, event, listener):
        self.removed = True
        self.listener = None

    async def goto(self, url, **kwargs):
        self.navigations += 1
        self.url = self.target
        if self.navigate_error:
            raise RuntimeError("never put this raw exception/signature in diagnostics")
        if self.response and self.listener:
            self.listener(self.response)

    async def wait_for_timeout(self, milliseconds):
        self.wait_count += 1
        if self.pause and self.wait_count > 1:
            self.ready.set()
            await asyncio.Event().wait()

    async def evaluate(self, script, arg=None):
        if script != _PRODUCT_DOM:
            return None
        result = copy.deepcopy(self.snapshots[min(self.snapshot_count, len(self.snapshots) - 1)])
        self.snapshot_count += 1
        if self.change_url:
            self.url = self.change_url
        return result

    def locator(self, *args, **kwargs):
        return EmptyLocator()

    def get_by_text(self, *args, **kwargs):
        return EmptyLocator()


def snapshot(media=None, **kwargs):
    return {"schema": "product_media_dom_v1", "scopes": ["product_main"], "ui_text": "",
            "media": media if media is not None else [{"url": "https://img.alicdn.com/a.jpg", "media_type": "image", "source_type": "product_main"}],
            **kwargs}


def ready_snapshot(*, at_bottom=True, detail_urls=None):
    """Explicit synthetic DOM evidence; absence of any field is not success."""
    detail_urls = detail_urls or ["https://img.alicdn.com/detail.jpg?sig=a%2Bb"]
    media = [{"url": "https://img.alicdn.com/main.jpg", "media_type": "image", "source_type": "product_main"}]
    media += [{"url": url, "media_type": "image", "source_type": "product_detail"} for url in detail_urls]
    def scope(urls, height, bottom):
        return {"root_count": 1, "image_nodes": len(urls), "loaded_images": len(urls),
                "loading_images": 0, "broken_images": 0, "unresolved_images": 0,
                "lazy_pending": 0, "iframe_count": 0, "expand_controls": 0, "busy_markers": 0,
                "video_slots": 0, "unresolved_video_slots": 0,
                "root_bounds": [{"height": height, "bottom": bottom}],
                "node_states": [[url, url, "", True, 800, 1000] for url in urls]}
    return snapshot(media, scopes=["product_main", "product_detail"], unresolved_videos=0,
        completion={"schema": "taobao_product_dom_scope_v1", "scroll_context": "document",
                    "scroll_top": 4200 if at_bottom else 1000, "client_height": 800, "scroll_height": 5000,
                    "at_bottom": at_bottom, "detail_end_reached": at_bottom,
                    "scopes": {"product_main": scope([media[0]["url"]], 500, 600),
                               "product_detail": scope(detail_urls, 4000, 5000)}})


class ProductMediaCollectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_stable_bottom_completes_only_current_exposed_taobao_dom_scope(self):
        page = Page(snapshots=[ready_snapshot()])
        result = await collect_product_media(page, TB_URL, "taobao", max_scrolls=8)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["complete"])
        diagnostics = result["diagnostics"]
        self.assertEqual(diagnostics["completion_scope"], "current_product_page_exposed_dom")
        self.assertEqual(diagnostics["stable_bottom_observations"], 3)
        self.assertEqual(len(diagnostics["completion_evidence"]["stable_observations"]), 3)
        self.assertIn("其他 SKU", result["reason"])
        self.assertEqual(page.snapshot_count, 3)

    async def test_long_detail_without_new_urls_does_not_stop_before_product_end(self):
        initial = ready_snapshot(at_bottom=False)
        bottom = ready_snapshot(detail_urls=["https://img.alicdn.com/detail.jpg?sig=a%2Bb", "https://img.alicdn.com/late.jpg"])
        page = Page(snapshots=[initial] * 5 + [bottom] * 3)
        result = await collect_product_media(page, TB_URL, "taobao", max_scrolls=20)
        self.assertTrue(result["complete"])
        self.assertEqual(page.snapshot_count, 8)
        self.assertTrue(any(row["url"].endswith("late.jpg") for row in result["media"]))

    async def test_growing_recommendation_tail_does_not_reset_product_stability(self):
        snapshots = []
        for height in (64381, 70000, 80000):
            value = ready_snapshot()
            value["completion"].update(scroll_height=height, scope_bottom=5000)
            snapshots.append(value)
        page = Page(snapshots=snapshots)
        result = await collect_product_media(page, TB_URL, "taobao", max_scrolls=200)
        self.assertTrue(result["complete"])
        self.assertEqual(page.snapshot_count, 3)
        self.assertEqual(result["diagnostics"]["last_dom_observation"]["scroll_height"], 80000)

    async def test_missing_detail_root_is_bounded_partial_not_two_hundred_scrolls(self):
        value = ready_snapshot()
        del value["completion"]["scopes"]["product_detail"]
        value["completion"].update(at_bottom=False, detail_end_reached=False, scope_bottom=None)
        page = Page(snapshots=[value])
        result = await collect_product_media(page, TB_URL, "taobao", max_scrolls=200)
        self.assertEqual(page.snapshot_count, 12)
        self.assertEqual(result["status"], "partial")
        self.assertFalse(result["complete"])
        self.assertEqual(result["diagnostics"]["exit_reason"], "detail_root_missing")
        self.assertIsNone(result["diagnostics"]["completion_evidence"])
        unavailable = result["diagnostics"]["detail_container_unavailable"]
        self.assertEqual(unavailable["reason"], "platform_detail_root_not_exposed")
        self.assertEqual(unavailable["observations"], 12)
        self.assertNotIn("product_detail", unavailable["exposed_scopes"])
        self.assertFalse(result["diagnostics"]["detail_tab_retry"]["control_visible"])
        self.assertIn("平台未展露，非读取失败", result["reason"])

    async def test_detail_root_materializing_before_wait_limit_can_complete(self):
        missing, loaded = ready_snapshot(), ready_snapshot()
        del missing["completion"]["scopes"]["product_detail"]
        page = Page(snapshots=[missing] * 11 + [loaded] * 3)
        result = await collect_product_media(page, TB_URL, "taobao", max_scrolls=200)
        self.assertTrue(result["complete"])
        self.assertEqual(page.snapshot_count, 14)
        self.assertEqual(result["diagnostics"]["missing_detail_observations"], 0)

    async def test_page_height_node_or_url_change_resets_stable_bottom_observations(self):
        for change in ("height", "nodes", "url"):
            first, changed = ready_snapshot(), ready_snapshot()
            if change == "height":
                changed["completion"].update(scroll_height=5200, scroll_top=4400)
                changed["completion"]["scopes"]["product_detail"]["root_bounds"][0].update(height=4200, bottom=5200)
            elif change == "nodes":
                changed = ready_snapshot(detail_urls=["https://img.alicdn.com/detail.jpg", "https://img.alicdn.com/new.jpg"])
            else:
                changed = ready_snapshot(detail_urls=["https://img.alicdn.com/replaced.jpg"])
            with self.subTest(change=change):
                page = Page(snapshots=[first, first, changed, changed, changed])
                result = await collect_product_media(page, TB_URL, "taobao", max_scrolls=8)
                self.assertTrue(result["complete"])
                self.assertEqual(page.snapshot_count, 5)
                evidence = result["diagnostics"]["completion_evidence"]["stable_observations"]
                self.assertEqual(len({r["signature"] for r in evidence}), 1)

    async def test_bottom_with_missing_roots_load_failures_iframe_or_expand_remains_partial(self):
        cases = []
        for field in ("loading_images", "broken_images", "lazy_pending", "unresolved_images",
                      "iframe_count", "expand_controls", "busy_markers"):
            value = ready_snapshot()
            value["completion"]["scopes"]["product_detail"][field] = 1
            cases.append((field, value))
        for scope_name in ("product_main", "product_detail"):
            value = ready_snapshot()
            del value["completion"]["scopes"][scope_name]
            cases.append(("missing_" + scope_name, value))
        value = ready_snapshot()
        value["completion"]["detail_end_reached"] = False
        cases.append(("detail_end", value))
        for name, value in cases:
            with self.subTest(case=name):
                result = await collect_product_media(Page(snapshots=[value]), TB_URL, "taobao", max_scrolls=5)
                self.assertFalse(result["complete"])
                self.assertIsNone(result["diagnostics"]["completion_evidence"])
                self.assertEqual(result["diagnostics"]["stable_bottom_observations"], 0)

    async def test_resolved_video_slot_clears_terminal_debt_but_keeps_peak(self):
        early, loaded = ready_snapshot(), ready_snapshot()
        early["completion"]["scopes"]["product_main"].update(video_slots=1, unresolved_video_slots=1)
        early["unresolved_videos"] = 1
        loaded["completion"]["scopes"]["product_main"]["video_slots"] = 1
        loaded["media"].append({"url": "https://cloud.video.taobao.com/main.mp4", "media_type": "video", "source_type": "product_video"})
        result = await collect_product_media(Page(snapshots=[early, loaded, loaded, loaded]), TB_URL, "taobao", max_scrolls=8)
        self.assertTrue(result["complete"])
        self.assertEqual(result["diagnostics"]["unresolved_videos"], 0)
        self.assertEqual(result["diagnostics"]["unresolved_videos_peak"], 1)

    async def test_unresolved_entry_at_stable_end_returns_partial_before_scroll_limit(self):
        value = ready_snapshot()
        value["completion"]["scopes"]["product_main"].update(video_slots=2, unresolved_video_slots=1)
        value["unresolved_videos"] = 1
        value["media"].append({"url": "https://cloud.video.taobao.com/main.mp4", "media_type": "video", "source_type": "product_video"})
        page = Page(snapshots=[value])
        result = await collect_product_media(page, TB_URL, "taobao", max_scrolls=200)
        self.assertEqual(page.snapshot_count, 12)
        self.assertEqual(result["status"], "partial")
        self.assertFalse(result["complete"])
        diagnostics = result["diagnostics"]
        self.assertEqual(diagnostics["exit_reason"], "unresolved_at_stable_end")
        self.assertEqual(diagnostics["unresolved_videos"], 1)
        self.assertEqual(diagnostics["completion_blocker"], "product_main_unresolved_video_slots")
        self.assertIsNone(diagnostics["completion_evidence"])
        self.assertFalse(diagnostics["completeness_verified"])
        evidence = diagnostics["unresolved_end_evidence"]["stable_observations"]
        self.assertEqual(len(evidence), 12)
        self.assertEqual(len({row["signature"] for row in evidence}), 1)

    async def test_unresolved_entry_never_stops_early_before_both_ends_or_without_metadata(self):
        value = ready_snapshot()
        value["completion"]["scopes"]["product_main"].update(video_slots=1, unresolved_video_slots=1)
        value["unresolved_videos"] = 1
        for change in ("page_flag", "page_position", "detail_flag", "detail_position", "missing_nodes"):
            changed = copy.deepcopy(value)
            observation = changed["completion"]
            if change == "page_flag":
                observation["at_bottom"] = False
            elif change == "page_position":
                observation["scroll_top"] = 1000
            elif change == "detail_flag":
                observation["detail_end_reached"] = False
            elif change == "detail_position":
                observation["scopes"]["product_detail"]["root_bounds"][0]["bottom"] = 5200
            elif change == "missing_nodes":
                del observation["scopes"]["product_detail"]["node_states"]
            with self.subTest(change=change):
                page = Page(snapshots=[changed] * 15 + [value] * 12)
                result = await collect_product_media(page, TB_URL, "taobao", max_scrolls=60)
                self.assertEqual(page.snapshot_count, 27)
                self.assertEqual(result["diagnostics"]["exit_reason"], "unresolved_at_stable_end")
                self.assertFalse(result["complete"])

    async def test_loading_changes_reset_unresolved_end_stability(self):
        first = ready_snapshot()
        first["completion"]["scopes"]["product_main"].update(video_slots=1, unresolved_video_slots=1)
        first["unresolved_videos"] = 1
        for change in ("height", "nodes", "url", "blocker", "node_state"):
            changed = copy.deepcopy(first)
            scope = changed["completion"]["scopes"]["product_detail"]
            if change == "height":
                changed["completion"].update(scroll_height=5200, scroll_top=4400)
                scope["root_bounds"][0].update(height=4200, bottom=5200)
            elif change == "nodes":
                scope["image_nodes"] += 1
                scope["loaded_images"] += 1
                scope["node_states"].append(list(scope["node_states"][0]))
            elif change == "url":
                changed["media"][1]["url"] = "https://img.alicdn.com/replaced.jpg"
            elif change == "blocker":
                changed["completion"]["scopes"]["product_main"]["busy_markers"] = 1
            else:
                scope["node_states"][0][4] = 900
            with self.subTest(change=change):
                page = Page(snapshots=[first] * 11 + [changed] * 12)
                result = await collect_product_media(page, TB_URL, "taobao", max_scrolls=60)
                self.assertEqual(page.snapshot_count, 23)
                self.assertEqual(result["diagnostics"]["exit_reason"], "unresolved_at_stable_end")
                evidence = result["diagnostics"]["unresolved_end_evidence"]["stable_observations"]
                self.assertEqual([row["turn"] for row in evidence], list(range(11, 23)))
                self.assertEqual(len({row["signature"] for row in evidence}), 1)
                self.assertFalse(result["complete"])

    async def test_pending_entry_resolving_before_stable_stop_can_complete_scope(self):
        pending, loaded = ready_snapshot(), ready_snapshot()
        pending["completion"]["scopes"]["product_main"].update(video_slots=1, unresolved_video_slots=1)
        pending["unresolved_videos"] = 1
        loaded["completion"]["scopes"]["product_main"]["video_slots"] = 1
        loaded["media"].append({"url": "https://cloud.video.taobao.com/main.mp4", "media_type": "video", "source_type": "product_video"})
        page = Page(snapshots=[pending] * 11 + [loaded] * 3)
        result = await collect_product_media(page, TB_URL, "taobao", max_scrolls=60)
        self.assertEqual(page.snapshot_count, 14)
        self.assertTrue(result["complete"])
        self.assertEqual(result["diagnostics"]["stable_unresolved_end_observations"], 0)
        self.assertIsNone(result["diagnostics"]["unresolved_end_evidence"])

    def test_blocked_gallery_still_retains_detail_scope_diagnostics(self):
        value = ready_snapshot()
        value["completion"]["scopes"]["product_main"].update(video_slots=2, unresolved_video_slots=1,
            video_slot_details=[{"kind": "main_player_container", "resolved_player_count": 1},
                                {"kind": "video_entry_unbound", "resolved_player_count": 0}])
        ready, code, evidence, _ = _taobao_completion_observation(value)
        self.assertFalse(ready)
        self.assertEqual(code, "product_main_unresolved_video_slots")
        self.assertIn("product_detail", evidence["scopes"])
        self.assertEqual(len(evidence["scopes"]["product_main"]["video_slot_details"]), 2)

    async def test_loading_limit_cannot_substitute_for_three_stable_observations(self):
        result = await collect_product_media(Page(snapshots=[ready_snapshot()]), TB_URL, "taobao", max_scrolls=2)
        self.assertFalse(result["complete"])
        self.assertEqual(result["diagnostics"]["exit_reason"], "loading_limit")

    async def test_other_platform_cannot_use_taobao_completion_evidence(self):
        result = await collect_product_media(Page(url=DY_URL, snapshots=[ready_snapshot()]), DY_URL, "douyin", max_scrolls=5)
        self.assertFalse(result["complete"])

    def test_missing_or_inconsistent_completion_metadata_never_defaults_to_ready(self):
        cases = [snapshot(), {**ready_snapshot(), "completion": None}]
        missing = ready_snapshot()
        del missing["completion"]["scopes"]["product_main"]["broken_images"]
        cases.append(missing)
        missing = ready_snapshot()
        missing["completion"]["scroll_top"] = 10  # Boolean bottom flag alone is not proof.
        cases.append(missing)
        missing = ready_snapshot()
        missing["completion"]["scopes"]["product_main"]["video_slots"] = 1
        cases.append(missing)  # A resolved slot must have an observed video URL.
        missing = ready_snapshot()
        missing["completion"]["scope_bottom"] = 800  # Must agree with actual root bounds.
        cases.append(missing)
        for value in cases:
            self.assertFalse(_taobao_completion_observation(value)[0])

    async def test_xhs_visible_verification_stops_without_extra_request(self):
        page = Page(url=XHS_URL, snapshots=[snapshot([], ui_text="请完成安全验证")])
        result = await collect_product_media(page, XHS_URL, "xiaohongshu")
        self.assertEqual(result["status"], "verification_required")
        self.assertEqual(page.requests, [])
        self.assertTrue(page.removed)

    async def test_stall_or_loading_limit_never_claims_complete(self):
        page = Page()
        result = await collect_product_media(page, TB_URL, "taobao", max_scrolls=1)
        self.assertEqual(result["status"], "partial")
        self.assertFalse(result["complete"])
        self.assertEqual(len(result["media"]), 1)
        self.assertEqual(page.navigations, 1)

    async def test_xhs_observed_response_collects_only_product_scopes(self):
        page = Page(url=XHS_URL, response=Response())
        result = await collect_product_media(page, XHS_URL, "xiaohongshu")
        self.assertEqual(len(result["media"]), 2)
        self.assertEqual(result["status"], "partial")
        self.assertFalse(result["complete"])
        self.assertTrue(page.removed)
        self.assertEqual(page.requests, [])

    async def test_xhs_single_read_only_fallback_and_restricted_response(self):
        page = Page(url=XHS_URL)
        result = await collect_product_media(page, XHS_URL, "xiaohongshu")
        self.assertEqual(page.requests, [XHS_RESPONSE])
        self.assertEqual(len(result["media"]), 2)
        page = Page(url=XHS_URL, response=Response(status=429))
        result = await collect_product_media(page, XHS_URL, "xiaohongshu")
        self.assertEqual(result["status"], "access_restricted")
        self.assertEqual(page.requests, [])
        self.assertEqual(result["media"], [])

    async def test_login_navigation_and_identity_change_do_not_collect(self):
        cases = [(Page(url="https://login.taobao.com/member/login.jhtml"), "login_required"),
                 (Page(url=TB_URL.replace("950216355977", "111")), "product_mismatch"),
                 (Page(change_url=TB_URL.replace("950216355977", "111")), "product_mismatch"),
                 (Page(navigate_error=True), "error")]
        for page, expected in cases:
            result = await collect_product_media(page, TB_URL, "taobao", max_scrolls=1)
            self.assertEqual(result["status"], expected)
            self.assertEqual(result["media"], [])
            self.assertNotIn("raw exception", str(result))

    async def test_douyin_video_cover_remains_unresolved_and_empty_not_complete(self):
        page = Page(url=DY_URL, snapshots=[snapshot([], unresolved_videos=1)])
        result = await collect_product_media(page, DY_URL, "douyin", max_scrolls=1)
        self.assertEqual(result["status"], "unsupported_page")
        self.assertFalse(result["complete"])
        self.assertEqual(result["diagnostics"]["unresolved_videos"], 1)

    async def test_cancellation_keeps_media_observed_before_stop(self):
        page = Page(pause=True)
        task = asyncio.create_task(collect_product_media(page, TB_URL, "taobao"))
        await asyncio.wait_for(page.ready.wait(), 1)
        task.cancel()
        result = await asyncio.wait_for(task, 1)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(len(result["media"]), 1)
        self.assertFalse(result["complete"])


if __name__ == "__main__":
    unittest.main()
