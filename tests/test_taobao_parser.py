import asyncio
import json
from types import SimpleNamespace
import unittest

from buyer_show.adapters.taobao import _ALL_REVIEWS_CONTAINER, _DOM, _DOM_SCHEMA, _network_match, collect, parse_reviews


URL = "https://detail.tmall.com/item.htm?id=674605142427"
A = "https://img.alicdn.com/imgextra/A_original_!!123.jpg?token=keep"
B = "https://img.alicdn.com/imgextra/B.jpg"
C = "https://img.alicdn.com/imgextra/C.jpg"


class TaobaoParserTests(unittest.TestCase):
    def test_video_alternates_stay_one_asset_and_keep_same_native_parent(self):
        primary = "https://cloud.video.taobao.com/play/u/null/p/1/v.mp4?sig=a%2Bb"
        backup = "https://pingjia.alicdn.com/actual-source.mp4?sig=c%2Bd"
        follow = "https://video-sh.cloudvideocdn.taobao.com/append.mp4"
        row = {"id": "parent", "feedback": "first", "video": {
            "cloudVideoUrl": primary, "sourceVideoUrl": backup, "coverUrl": A},
            "appendedFeed": {"appendedFeedback": "later", "video": {
                "cloudVideoUrl": follow, "sourceVideoUrl": backup}}}
        parsed = parse_reviews({"data": {"rateList": [row]}}, URL)
        self.assertEqual(parsed[0]["videos"], [primary])
        self.assertEqual(parsed[0]["video_sources"], {primary: [primary, backup]})
        self.assertEqual(parsed[1]["videos"], [follow])
        self.assertEqual(parsed[1]["video_sources"], {follow: [follow, backup]})
        self.assertEqual([r["review_id"] for r in parsed], ["parent", "parent"])
        self.assertEqual([r["review_type"] for r in parsed], ["initial", "append"])
        self.assertEqual(parsed[0]["images"], [])

    def test_video_sources_omitted_without_two_distinct_legal_urls_and_for_dom(self):
        primary = "https://cloud.video.taobao.com/one.mp4"
        for other in (primary, A, "https://evil.test/other.mp4", "https://cloud.video.taobao.com.evil.test/v.mp4",
                      "https://u:p@pingjia.alicdn.com/v.mp4", "https://pingjia.alicdn.com:8080/v.mp4", "blob:https://detail.tmall.com/x", None):
            with self.subTest(candidate=other):
                row = parse_reviews({"comments": [{"id": "r", "video": {"cloudVideoUrl": primary, "sourceVideoUrl": other}}]}, URL)[0]
                self.assertEqual(row["videos"], [primary])
                self.assertNotIn("video_sources", row)
        row = parse_reviews({"schema": _DOM_SCHEMA, "comments": [{"id": "dom", "videos": [primary],
                            "video_sources": {primary: [primary, "https://pingjia.alicdn.com/other.mp4"]}}]}, URL)[0]
        self.assertNotIn("video_sources", row)

    def test_duplicate_native_rows_merge_alternates_without_adding_video_count(self):
        primary = "https://cloud.video.taobao.com/one.mp4"
        one, two = "https://pingjia.alicdn.com/one.mp4", "https://pingjia.alicdn.com/two.mp4"
        rows = [{"id": "r", "video": {"cloudVideoUrl": primary, "sourceVideoUrl": u}} for u in (one, two, one)]
        parsed = parse_reviews({"data": {"rateList": rows}}, URL)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["videos"], [primary])
        self.assertEqual(parsed[0]["video_sources"], {primary: [primary, one, two]})

    def test_dom_thumbnail_matches_original_without_rewriting_or_double_counting(self):
        original = {"review_id": "native", "review_type": "initial", "content": "same", "images": [A], "videos": []}
        thumbnail = dict(original, review_id="synthetic:dom", images=[A.split("?")[0] + "_240x240q90.jpg_.webp"])
        self.assertIs(_network_match(thumbnail, [original]), original)
        self.assertEqual(original["images"], [A])
        # Identical text/media is not enough to choose between two real buyers.
        self.assertIsNone(_network_match(thumbnail, [original, dict(original, review_id="other")]))
        self.assertIsNone(_network_match(dict(thumbnail, review_type="append"), [original]))

    def test_observed_v6_initial_append_pictures_and_real_video_not_cover(self):
        payload = {"data": {"rateList": [{"id": "r", "auctionNumId": "674605142427", "feedback": "first",
            "feedPicPathList": [A, B], "headPicUrl": C,
            "video": {"cloudVideoUrl": "https://cloud.video.taobao.com/a.mp4", "sourceVideoUrl": "https://pingjia.alicdn.com/a.mp4", "coverUrl": C},
            "appendedFeed": {"appendedFeedback": "later", "appendFeedPicPathList": [C]}}]}}
        rows = parse_reviews(payload, URL)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["images"], [A, B])
        self.assertEqual(rows[0]["videos"], ["https://cloud.video.taobao.com/a.mp4"])
        self.assertEqual(rows[1]["images"], [C])
        self.assertEqual(rows[1]["review_type"], "append")
        self.assertEqual(rows[1]["review_id"], "r")
        self.assertEqual(rows[1]["content"], "later")
        payload["data"]["rateList"][0]["auctionNumId"] = "111"
        self.assertEqual(parse_reviews(payload, URL), [])

    def test_v6_video_source_fallback_rejects_untrusted_or_cover_url(self):
        row = {"id": "v", "feedback": "", "video": {"cloudVideoUrl": A, "sourceVideoUrl": "https://pingjia.alicdn.com/a.mp4", "coverUrl": B}}
        self.assertEqual(parse_reviews({"data": {"rateList": [row]}}, URL)[0]["videos"], ["https://pingjia.alicdn.com/a.mp4"])
        row["video"]["sourceVideoUrl"] = "https://evil.test/a.mp4"
        self.assertEqual(parse_reviews({"data": {"rateList": [row]}}, URL)[0]["videos"], [])

    def test_dom_video_only_initial_and_append_keep_parent_and_signed_urls(self):
        video = "https://cloud.video.taobao.com/play/u/1/p/1/e/6/t/1/77.mp4?auth=abc%2B1&x=2"
        follow = "https://cloud.video.taobao.com/88.mp4"
        rows = parse_reviews({"schema": _DOM_SCHEMA, "comments": [
            {"id": "parent", "videos": [video, video],
             "appendComment": {"videos": [follow]}}
        ]}, URL)
        self.assertEqual([(r["review_id"], r["review_type"], r["images"], r["videos"]) for r in rows],
                         [("parent", "initial", [], [video]), ("parent", "append", [], [follow])])

    def test_unverified_network_video_fields_and_cover_are_not_videos(self):
        payload = {"rateDetail": {"rateList": [{"id": "r", "content": "x", "videos": ["https://cloud.video.taobao.com/1.mp4"],
                      "video": {"cover": A, "url": "https://cloud.video.taobao.com/2.mp4"}}]}}
        self.assertEqual(parse_reviews(payload, URL)[0]["videos"], [])
        payload = {"schema": _DOM_SCHEMA, "comments": [{"content": "", "videos": [A, "blob:https://detail.tmall.com/1",
                         "https://u:p@cloud.video.taobao.com/1.mp4", "javascript:alert(1)"]}]}
        self.assertEqual(parse_reviews(payload, URL)[0]["videos"], [])

    def test_video_only_synthetic_rows_remain_distinct_and_duplicates_merge(self):
        a, b = "https://cloud.video.taobao.com/a.mp4", "https://cloud.video.taobao.com/b.mp4"
        payload = {"schema": _DOM_SCHEMA, "comments": [{"videos": [a]}, {"videos": [b]}]}
        self.assertEqual(len(parse_reviews(payload, URL)), 2)
        payload["comments"] = [{"id": "r", "videos": [a]}, {"id": "r", "videos": [b]}]
        self.assertEqual(parse_reviews(payload, URL)[0]["videos"], [a, b])

    def test_initial_append_and_multi_image_association(self):
        payload = {"rateDetail": {"rateList": [{"id": 42, "rateContent": "好&amp;用", "pics": [A, B],
                     "appendComment": {"commentId": 999, "content": "追评", "pics": [C]}}]}}
        rows = parse_reviews(payload, URL)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["images"], [A, B])
        self.assertEqual(rows[1]["images"], [C])
        self.assertEqual(rows[0]["review_id"], rows[1]["review_id"])
        self.assertEqual(rows[1]["review_type"], "append")
        self.assertEqual(rows[0]["content"], "好&用")

    def test_excludes_product_avatar_recommendations_and_nested_fake_reviews(self):
        payload = {"images": [A], "data": {"item": {"images": [B], "comments": [{"content": "fake", "pics": [C]}]},
                   "comments": [{"id": "real", "content": "text", "avatar": A,
                                 "product": {"pics": [B]}, "recommendations": [{"pics": [C]}]}]}}
        self.assertEqual(parse_reviews(payload, URL)[0]["images"], [])
        self.assertEqual(parse_reviews({"data": {"item": payload}}, URL), [])
        self.assertEqual(parse_reviews([{"content": "ambiguous", "pics": [A]}], URL), [])

    def test_jsonp_observed_urls_and_duplicate_rows(self):
        review = {"rateId": "r", "pics": ["//img.alicdn.com/imgextra/B.jpg", {"full": A, "thumb": B}, A]}
        payload = "jsonp123(" + json.dumps({"data": {"rateList": [review, review]}}) + ");"
        rows = parse_reviews(payload, URL)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["images"], [B, A])

    def test_wrong_product_and_untrusted_source_are_rejected(self):
        row = {"id": 1, "content": "x", "pics": [A]}
        self.assertEqual(parse_reviews({"itemId": "7", "comments": [row]}, URL), [])
        self.assertEqual(parse_reviews({"comments": [dict(row, itemId="7")]}, URL), [])
        self.assertEqual(parse_reviews({"rateDetail": {"itemId": "7", "rateList": [row]}}, URL), [])
        self.assertEqual(parse_reviews({"comments": [row]}, "https://evil.test/item.htm?id=674605142427"), [])
        self.assertEqual(parse_reviews({"comments": [row]}, "https://login.taobao.com/?id=674605142427"), [])

    def test_text_only_and_photo_only_reviews_and_stable_synthetic_id(self):
        payload = {"comments": [{"id": "text", "content": "无图"}, {"pics": [A]}]}
        rows = parse_reviews(payload, URL)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["images"], [])
        self.assertEqual(rows[1]["images"], [A])
        self.assertTrue(rows[1]["review_id"].startswith("synthetic:"))
        self.assertEqual(rows, parse_reviews(payload, URL))

    def test_bad_images_and_malformed_payload(self):
        row = {"id": 1, "pics": [None, 23, "data:image/png;base64,x", "http://localhost/a.jpg",
                                  "https://img.alicdn.com/avatar/a.jpg", "https://img.alicdn.com/blank.gif", B]}
        self.assertEqual(parse_reviews({"comments": [row]}, URL)[0]["images"], [B])
        for payload in (None, "<html>login</html>", "f({broken})", {"comments": [None, "x", {}]}):
            self.assertEqual(parse_reviews(payload, URL), [])

    def test_multiple_appends_merge_without_losing_images(self):
        rows = parse_reviews({"comments": [{"id": 1, "content": "first", "appendComments": [
            {"content": "second", "pics": [A]}, {"content": "third", "pics": [B]}]}]}, URL)
        self.assertEqual(rows[1]["images"], [A, B])
        self.assertEqual(rows[1]["content"], "second\nthird")


class _EmptyLocator:
    async def count(self):
        return 0

    def filter(self, **kwargs):
        return self

    def get_by_text(self, *args, **kwargs):
        return self


class _ReviewResponse:
    url = "https://rate.tmall.com/list_detail_rate.htm?itemId=674605142427"
    status = 200
    request = SimpleNamespace(url=url, post_data=None)

    async def text(self):
        return json.dumps({"rateDetail": {"rateList": [
            {"id": "network", "rateContent": "接口评价", "pics": [A]}
        ]}})


class _CancellablePage:
    """A page that pauses only after one network and DOM collection round."""
    url = URL

    def __init__(self, cleanup_fails=False, closed=False):
        self.listener = None
        self.cleanup_fails = cleanup_fails
        self.closed = closed
        self.remove_attempted = False
        self.waits = 0
        self.evaluations = 0
        self.collection_ready = asyncio.Event()
        self.resume = asyncio.Event()

    def on(self, event, listener):
        assert event == "response"
        self.listener = listener

    def remove_listener(self, event, listener):
        self.remove_attempted = True
        assert event == "response" and listener is self.listener
        self.listener = None
        if self.cleanup_fails:
            raise RuntimeError("page closed")

    async def goto(self, *args, **kwargs):
        self.listener(_ReviewResponse())

    async def wait_for_timeout(self, milliseconds):
        self.waits += 1
        if self.waits > 1:
            self.collection_ready.set()
            if self.closed:
                raise RuntimeError("page closed")
            await self.resume.wait()

    def locator(self, *args, **kwargs):
        return _EmptyLocator()

    def get_by_text(self, *args, **kwargs):
        return _EmptyLocator()

    async def evaluate(self, script):
        self.evaluations += 1
        if self.evaluations == 1:
            return [{"id": "dom", "content": "页面评价", "pics": [B]}]
        return None


class TaobaoCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_v6_prevents_unidentified_dom_snapshot_double_count(self):
        class Response:
            status = 200
            url = "https://h5api.m.taobao.com/h5/mtop.taobao.rate.detaillist.get/6.0/"
            request = SimpleNamespace(url=url + '?data=' + json.dumps({"auctionNumId": "674605142427", "pageNo": 1, "foldFlag": "0"}), post_data=None)
            async def text(self):
                return json.dumps({"data": {"rateList": [{"id": "native", "auctionNumId": "674605142427", "feedback": "full", "feedPicPathList": [A]}], "hasNext": "false", "foldCount": "3"}})
        class Page(_CancellablePage):
            async def goto(self, *args, **kwargs):
                self.listener(Response())
            async def wait_for_timeout(self, milliseconds):
                pass
            async def evaluate(self, script):
                return [{"content": "rendered differently", "pics": [A + "_240x240.jpg"]}] if script == _DOM else {"moved": False}
        result = await collect(Page(), URL, max_scrolls=1)
        self.assertEqual([row["review_id"] for row in result["records"]], ["native"])
        self.assertEqual(result["records"][0]["images"], [A])
        self.assertEqual(result["diagnostics"]["unmatched_dom_records"], 1)
        self.assertEqual(result["diagnostics"]["synthetic_ids"], 0)
        self.assertFalse(result["complete"])

    async def test_normal_and_folded_end_states_are_separate_and_native_ids_merge(self):
        class Response:
            status = 200
            url = "https://h5api.m.taobao.com/h5/mtop.taobao.rate.detaillist.get/6.0/"
            def __init__(self, folded, has_next, images):
                self.request = SimpleNamespace(url=self.url + '?data=' + json.dumps({"auctionNumId": "674605142427", "pageNo": 1, "foldFlag": folded}), post_data=None)
                self.payload = {"data": {"rateList": [{"id": "same", "auctionNumId": "674605142427", "feedback": "review", "feedPicPathList": images}], "hasNext": has_next, "foldCount": "3"}}
                self.payload["data"]["rateList"][0]["video"] = {"cloudVideoUrl": "https://cloud.video.taobao.com/same.mp4", "sourceVideoUrl": "https://pingjia.alicdn.com/source" + folded + ".mp4"}
            async def text(self):
                return json.dumps(self.payload)
        class Page(_CancellablePage):
            async def goto(self, *args, **kwargs):
                self.listener(Response("0", "false", [A]))
                self.listener(Response("1", "true", [B]))
            async def wait_for_timeout(self, milliseconds):
                pass
            async def evaluate(self, script):
                return [] if script == _DOM else {"moved": False}
        result = await collect(Page(), URL, max_scrolls=1)
        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(result["records"][0]["images"], [A, B])
        self.assertEqual(result["records"][0]["videos"], ["https://cloud.video.taobao.com/same.mp4"])
        self.assertEqual(result["records"][0]["video_sources"], {"https://cloud.video.taobao.com/same.mp4": [
            "https://cloud.video.taobao.com/same.mp4", "https://pingjia.alicdn.com/source0.mp4", "https://pingjia.alicdn.com/source1.mp4"]})
        scopes = result["diagnostics"]["review_scopes"]
        self.assertTrue(scopes["normal"]["end_observed"])
        self.assertFalse(scopes["folded"]["end_observed"])
        self.assertTrue(result["diagnostics"]["folded_reviews_opened"])
        self.assertEqual(result["diagnostics"]["folded_review_count"], 3)
        self.assertFalse(result["complete"])

    async def test_waits_for_real_review_entry_before_checking_transitional_identity(self):
        class Entry:
            ready = False
            first = property(lambda self: self)
            async def wait_for(self, **kwargs):
                self.ready = True
                page.url = URL
            async def count(self):
                return int(self.ready)
            def nth(self, _index):
                return self
            async def is_visible(self):
                return True
            async def click(self, **kwargs):
                pass
        class ReadyPage(_CancellablePage):
            async def goto(self, *args, **kwargs):
                self.url = "about:blank"  # Intermediate navigation, not an accepted product.
                self.listener(_ReviewResponse())
            async def wait_for_timeout(self, milliseconds):
                pass
            def get_by_text(self, label, **kwargs):
                return entry if label == "查看全部评价" else _EmptyLocator()
            async def evaluate(self, script):
                return [] if script == _DOM else {"moved": False}
        entry, page = Entry(), ReadyPage()
        result = await collect(page, URL, max_scrolls=1)
        self.assertTrue(entry.ready)
        self.assertTrue(result["diagnostics"]["all_reviews_clicked"])
        self.assertFalse(result["diagnostics"]["all_reviews_opened"])
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["diagnostics"]["page_location"]["product_id"], "674605142427")
        self.assertEqual([r["review_id"] for r in result["records"]], ["network"])

    async def test_identity_changes_during_dom_read_discard_foreign_records(self):
        class ChangedPage(_CancellablePage):
            async def evaluate(self, script):
                cards = await super().evaluate(script)
                self.url = URL.replace("674605142427", "111")
                return cards
        page = ChangedPage()
        result = await collect(page, URL)
        self.assertEqual(result["status"], "product_mismatch")
        self.assertEqual([r["review_id"] for r in result["records"]], ["network"])
        self.assertFalse(result["complete"])
        self.assertTrue(page.remove_attempted)

    async def cancelled_snapshot(self, page):
        task = asyncio.create_task(collect(page, URL))
        try:
            await asyncio.wait_for(page.collection_ready.wait(), timeout=1)
            task.cancel()
            return await asyncio.wait_for(task, timeout=1)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    def assert_saved_snapshot(self, page, result):
        self.assertFalse(result["complete"])
        self.assertTrue(page.remove_attempted)
        self.assertIsNone(page.listener)
        self.assertEqual({row["review_id"] for row in result["records"]}, {"network", "dom"})
        self.assertEqual({url for row in result["records"] for url in row["images"]}, {A, B})

    async def test_cancellation_retains_network_and_dom_records(self):
        page = _CancellablePage()
        result = await self.cancelled_snapshot(page)
        self.assertEqual(result["status"], "cancelled")
        self.assert_saved_snapshot(page, result)

    async def test_listener_cleanup_error_does_not_replace_cancelled_snapshot(self):
        page = _CancellablePage(cleanup_fails=True)
        result = await self.cancelled_snapshot(page)
        self.assertEqual(result["status"], "cancelled")
        self.assertTrue(result["diagnostics"]["listener_cleanup_failed"])
        self.assert_saved_snapshot(page, result)

    async def test_closed_page_and_cleanup_error_keep_partial_snapshot(self):
        page = _CancellablePage(cleanup_fails=True, closed=True)
        result = await collect(page, URL)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["diagnostics"]["error_type"], "RuntimeError")
        self.assert_saved_snapshot(page, result)


class TaobaoReviewEntryTests(unittest.IsolatedAsyncioTestCase):
    class Locator(_EmptyLocator):
        def __init__(self, page, kind):
            self.page, self.kind = page, kind
        first = property(lambda self: self)
        async def count(self):
            return 1
        def nth(self, index):
            return self
        async def is_visible(self):
            return self.kind == "entry" or self.page.drawer_visible
        async def wait_for(self, **kwargs):
            if self.kind == "entry":
                return
            self.page.drawer_waits += 1
            self.page.wait_options = kwargs
            if self.page.mode == "delayed_drawer":
                self.page.drawer_visible = True
                return
            if self.page.mode == "login_during_wait":
                self.page.url = "https://login.taobao.com/member/login.jhtml"
            raise TimeoutError("deliberately not echoed")
        async def click(self, **kwargs):
            self.page.clicks += 1
            if self.page.mode == "login_after_click":
                self.page.url = "https://login.taobao.com/member/login.jhtml"
            elif self.page.mode == "click_error":
                raise RuntimeError("signed URL must not appear in diagnostics")

    class Page(_CancellablePage):
        def __init__(self, mode):
            super().__init__()
            self.mode = mode
            self.drawer_visible = False
            self.drawer_waits = self.clicks = self.scrolls = 0
        async def goto(self, *args, **kwargs):
            pass
        async def wait_for_timeout(self, milliseconds):
            pass
        def get_by_text(self, label, **kwargs):
            return TaobaoReviewEntryTests.Locator(self, "entry") if label == "查看全部评价" else _EmptyLocator()
        def locator(self, selector):
            return TaobaoReviewEntryTests.Locator(self, "drawer") if selector == _ALL_REVIEWS_CONTAINER else _EmptyLocator()
        async def evaluate(self, script):
            if script == _DOM:
                if self.drawer_visible:
                    return [{"id": "full", "content": "全部评价", "pics": [C]}]
                return [{"id": "summary", "content": "摘要", "pics": [A, B]}]
            self.scrolls += 1
            return {"scope": "review_container", "moved": False}

    async def test_click_without_real_drawer_keeps_summary_and_returns_partial(self):
        page = self.Page("no_drawer")
        result = await collect(page, URL)
        self.assertEqual(result["status"], "partial")
        self.assertFalse(result["complete"])
        self.assertTrue(result["diagnostics"]["all_reviews_clicked"])
        self.assertFalse(result["diagnostics"]["all_reviews_opened"])
        self.assertIsNone(result["diagnostics"]["all_reviews_view_evidence"])
        self.assertEqual(result["records"][0]["images"], [A, B])
        self.assertEqual((page.clicks, page.drawer_waits, page.scrolls), (1, 1, 0))
        self.assertIn("未确认完整评价列表打开", result["reason"])

    async def test_delayed_real_drawer_is_confirmed_before_loading_reviews(self):
        page = self.Page("delayed_drawer")
        result = await collect(page, URL, max_scrolls=1)
        self.assertTrue(result["diagnostics"]["all_reviews_opened"])
        self.assertEqual(result["diagnostics"]["all_reviews_view_evidence"], "visible_drawer_review_container")
        self.assertEqual({row["review_id"] for row in result["records"]}, {"summary", "full"})
        self.assertEqual(page.wait_options, {"state": "visible", "timeout": 8000})
        self.assertEqual((page.clicks, page.drawer_waits, page.scrolls), (1, 1, 1))

    async def test_login_after_click_or_during_wait_keeps_summary_and_stops(self):
        for mode in ("login_after_click", "login_during_wait"):
            with self.subTest(mode=mode):
                page = self.Page(mode)
                result = await collect(page, URL)
                self.assertEqual(result["status"], "login_required")
                self.assertFalse(result["complete"])
                self.assertFalse(result["diagnostics"]["all_reviews_opened"])
                self.assertEqual(result["records"][0]["images"], [A, B])
                self.assertEqual((page.clicks, page.scrolls), (1, 0))
                self.assertEqual(page.drawer_waits, int(mode == "login_during_wait"))

    async def test_failed_click_keeps_summary_and_does_not_expose_exception_message(self):
        page = self.Page("click_error")
        result = await collect(page, URL)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["records"][0]["images"], [A, B])
        self.assertFalse(result["diagnostics"]["all_reviews_clicked"])
        self.assertFalse(result["diagnostics"]["all_reviews_opened"])
        self.assertNotIn("signed URL", json.dumps(result))
        self.assertEqual((page.clicks, page.drawer_waits, page.scrolls), (1, 0, 0))


if __name__ == "__main__":
    unittest.main()
