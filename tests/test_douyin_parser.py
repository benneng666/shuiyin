"""Synthetic normalization fixtures; these do not establish live-site success."""

import asyncio
import json
import copy
import shutil
import subprocess
import unittest

from buyer_show.adapters.douyin import _SNAPSHOT_JS, collect, parse_reviews, product_id_from_url


SOURCE = "https://haohuo.jinritemai.com/ecommerce/trade/detail/index.html?id=3845439021845184916&origin_type=604"
PRODUCT = "3845439021845184916"


class DouyinReviewParserTests(unittest.TestCase):
    def test_video_only_review_and_append_preserve_more_than_five_urls(self):
        videos = [f"https://v3.douyinvod.com/buyer-{i}.mp4?sig=a%2Bb" for i in range(8)]
        rows = parse_reviews({"reviews": [{"review_id": "r", "video_urls": json.dumps(videos),
              "append_review": {"videos": ["https://v3.douyinvod.com/append.mp4"]}}]}, SOURCE)
        self.assertEqual(rows[0]["images"], [])
        self.assertEqual(rows[0]["videos"], videos)
        self.assertEqual(rows[1]["videos"], ["https://v3.douyinvod.com/append.mp4"])
        self.assertEqual(rows[1]["review_id"], "r")
        self.assertEqual(rows[1]["review_type"], "append")

    def test_video_covers_nested_products_and_blob_not_payloads(self):
        row = {"content": "", "videos": ["https://p3-item.ecombdimg.com/cover.jpg", "blob:https://www.douyin.com/x",
                "https://u:p@v3.douyinvod.com/a.mp4", "//v3.douyinvod.com/real.mp4"],
               "cover_url": "https://p3-item.ecombdimg.com/cover.jpg",
               "product": {"videos": ["https://v3.douyinvod.com/product.mp4"]},
               "reply": {"videos": ["https://v3.douyinvod.com/seller.mp4"]}}
        self.assertEqual(parse_reviews([row], SOURCE)[0]["videos"], ["https://v3.douyinvod.com/real.mp4"])

    def test_video_only_rows_with_same_text_are_not_collapsed(self):
        rows = [{"content": "", "videos": [f"https://v3.douyinvod.com/{i}.mp4"]} for i in range(2)]
        self.assertEqual(len(parse_reviews(rows, SOURCE)), 2)
        rows = [{"review_id": "r", "videos": row["videos"]} for row in rows]
        self.assertEqual(len(parse_reviews(rows, SOURCE)[0]["videos"]), 2)

    def test_product_url_not_video_or_lookalike_host(self):
        self.assertEqual(product_id_from_url(SOURCE), PRODUCT)
        self.assertEqual(product_id_from_url("https://haohuo.jinritemai.com/views/product/item2?id=123"), "123")
        self.assertEqual(product_id_from_url("https://www.douyin.com/video/123?id=456"), "")
        self.assertEqual(product_id_from_url("https://jinritemai.com.evil.test/views/product/item2?id=123"), "")

    def test_only_review_media_not_product_avatar_or_merchant_reply(self):
        payload = {"product_id": PRODUCT, "images": ["https://img.test/main.jpg"],
            "reviews": [{"review_id": "r1", "content": "晒单", "imgs": ["https://img.test/buyer.jpg"],
                "user": {"avatar": "https://img.test/user.jpg"},
                "product": {"images": ["https://img.test/inside-main.jpg"]},
                "reply": {"images": ["https://img.test/merchant.jpg"]}}]}
        records = parse_reviews(payload, SOURCE)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["images"], ["https://img.test/buyer.jpg"])
        self.assertEqual(records[0]["platform"], "douyin")
        self.assertEqual(records[0]["product_id"], PRODUCT)

    def test_upstream_json_array_and_signed_urls_preserved(self):
        url = "https://p3-item.ecombdimg.com/a.webp?sig=abc%2B123&size=200"
        records = parse_reviews([{"content": "", "imgs": json.dumps([url, url, "//img.test/b.jpg"])}], SOURCE)
        self.assertEqual(records[0]["images"], [url, "https://img.test/b.jpg"])
        self.assertTrue(records[0]["review_id"].startswith("synthetic:"))

    def test_append_review_kept_separate(self):
        data = {"reviews": [{"review_id": "r1", "content": "初评", "images": ["https://img.test/a.jpg"],
            "append_review": {"content": "追评", "images": ["https://img.test/b.jpg"]}}]}
        result = parse_reviews(data, SOURCE)
        self.assertEqual([r["review_type"] for r in result], ["initial", "append"])
        self.assertEqual(result[1]["review_id"], "r1")
        self.assertEqual(result[1]["images"], ["https://img.test/b.jpg"])

    def test_synthetic_parent_id_is_shared_with_append_and_stable_when_append_changes(self):
        row = {"content": "初评", "imgs": ["https://img.test/a.jpg"],
               "append_review": {"review_id": "separate-child-id", "content": "追评", "imgs": ["https://img.test/b.jpg"]}}
        first = parse_reviews({"reviews": [row]}, SOURCE)
        self.assertTrue(first[0]["review_id"].startswith("synthetic:"))
        self.assertEqual(first[0]["review_id"], first[1]["review_id"])
        row["append_review"]["content"] = "后来修改追评"
        second = parse_reviews({"reviews": [row]}, SOURCE)
        self.assertEqual(first[0]["review_id"], second[1]["review_id"])

    def test_separate_append_uses_explicit_parent_review_id(self):
        result = parse_reviews({"reviews": [{"review_id": "append-child", "parent_review_id": "initial-parent",
            "review_type": "append", "content": "追评", "images": ["https://img.test/b.jpg"]}]}, SOURCE)
        self.assertEqual(result[0]["review_id"], "initial-parent")

    def test_reject_video_comments_even_with_product_source(self):
        video = {"reviews": [{"cid": "1", "aweme_id": "88", "content": "视频配图", "images": ["https://img.test/a.jpg"]}]}
        self.assertEqual(parse_reviews(video, SOURCE), [])
        self.assertEqual(parse_reviews({"comments": [{"cid": "1", "image_list": []}]}, SOURCE), [])
        self.assertEqual(parse_reviews({"reviews": [{"content": "x", "imgs": []}]}, "https://www.douyin.com/video/123"), [])

    def test_does_not_accept_other_product_or_unassociated_short_link(self):
        self.assertEqual(parse_reviews({"product_id": "999", "reviews": [{"content": "x", "imgs": []}]}, SOURCE), [])
        self.assertEqual(parse_reviews({"reviews": [{"content": "x", "imgs": []}]}, "https://v.douyin.com/example/"), [])
        linked = {"product_id": PRODUCT, "reviews": [{"review_id": "r", "content": "x", "imgs": []}]}
        self.assertEqual(parse_reviews(linked, "https://v.douyin.com/example/")[0]["product_id"], PRODUCT)

    def test_malformed_data_does_not_create_images(self):
        for value in (None, "not json", {"images": ["https://img.test/main.jpg"]}, {"reviews": [None, 1, "x"]}):
            self.assertEqual(parse_reviews(value, SOURCE), [])
        row = {"content": "x", "imgs": ["data:image/png;base64,xxx", "javascript:alert(1)",
            "https://user:pass@img.test/a.jpg", "https://img.test/avatar.jpg", {"avatar": "https://img.test/u.jpg"},
            {"url": "https://img.test/buyer.jpg"}]}
        self.assertEqual(parse_reviews([row], SOURCE)[0]["images"], ["https://img.test/buyer.jpg"])

    def test_duplicates_merge_images_without_losing_distinct_reviews(self):
        rows = [{"review_id": "r1", "content": "很好", "images": ["https://img.test/a.jpg"]},
                {"review_id": "r1", "content": "很好", "images": ["https://img.test/b.jpg"]},
                {"review_id": "r2", "content": "很好", "images": ["https://img.test/c.jpg"]}]
        result = parse_reviews({"data": {"review_list": rows}}, SOURCE)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["images"], ["https://img.test/a.jpg", "https://img.test/b.jpg"])

    def test_same_text_without_ids_does_not_drop_distinct_buyer_photos(self):
        rows = [{"content": "很好", "imgs": ["https://img.test/a.jpg"]},
                {"content": "很好", "imgs": ["https://img.test/b.jpg"]}]
        self.assertEqual(len(parse_reviews(rows, SOURCE)), 2)


class _EmptyLocator:
    @property
    def first(self):
        return self

    def filter(self, **_kwargs):
        return self

    async def count(self):
        return 0


class _FakePage:
    """No browser: navigation and snapshots are explicit regression fixtures."""
    def __init__(self, snapshots, resolved_url=SOURCE, scroll=True, change_after_snapshot=None):
        self.url = "about:blank"
        self.resolved_url = resolved_url
        self.snapshots = snapshots
        self.snapshot_calls = 0
        self.scroll = scroll
        self.change_after_snapshot = change_after_snapshot or {}

    async def goto(self, _url, **_kwargs):
        self.url = self.resolved_url

    async def wait_for_timeout(self, _milliseconds):
        pass

    def locator(self, _selector):
        return _EmptyLocator()

    def get_by_text(self, _label):
        return _EmptyLocator()

    async def evaluate(self, script):
        if script != _SNAPSHOT_JS:
            return self.scroll
        index = min(self.snapshot_calls, len(self.snapshots) - 1)
        self.snapshot_calls += 1
        data = copy.deepcopy(self.snapshots[index])
        if self.snapshot_calls in self.change_after_snapshot:
            self.url = self.change_after_snapshot[self.snapshot_calls]
        return data


def _snapshot(**changes):
    data = {"reviews": [{"review_id": "r1", "content": "很好", "imgs": ["https://img.test/a.jpg"]}],
            "ui_text": "", "end": False, "empty": False, "at_bottom": True, "review_scope_found": True}
    data.update(changes)
    return data


class DouyinCollectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_unresolved_video_or_append_prevents_false_completion(self):
        for field in ("unresolved_videos", "ambiguous_appends"):
            outcome = await collect(_FakePage([_snapshot(end=True, **{field: 1})]), SOURCE)
            self.assertEqual(outcome["status"], "partial")
            self.assertFalse(outcome["complete"])

    async def test_collector_retains_video_only_review(self):
        snapshot = _snapshot(end=True, reviews=[{"review_id": "v", "content": "", "videos": ["https://v3.douyinvod.com/a.mp4"]}])
        outcome = await collect(_FakePage([snapshot]), SOURCE)
        self.assertEqual(outcome["records"][0]["videos"], ["https://v3.douyinvod.com/a.mp4"])
        self.assertEqual(outcome["diagnostics"]["video_count"], 1)

    async def test_explicit_end_uses_completed_status(self):
        outcome = await collect(_FakePage([_snapshot(end=True)]), SOURCE)
        self.assertEqual(outcome["status"], "completed")
        self.assertTrue(outcome["complete"])

    async def test_explicit_empty_uses_completed_with_reason(self):
        outcome = await collect(_FakePage([_snapshot(reviews=[], empty=True)]), SOURCE)
        self.assertEqual(outcome["status"], "completed")
        self.assertTrue(outcome["complete"])
        self.assertEqual(outcome["records"], [])
        self.assertIn("暂无评价", outcome["reason"])

    async def test_redirect_to_different_product_is_rejected_before_read(self):
        page = _FakePage([_snapshot(end=True)], resolved_url=SOURCE.replace(PRODUCT, "999"))
        outcome = await collect(page, SOURCE)
        self.assertEqual(outcome["status"], "product_mismatch")
        self.assertEqual(outcome["records"], [])
        self.assertFalse(outcome["complete"])
        self.assertEqual(page.snapshot_calls, 0)

    async def test_missing_resolved_id_never_falls_back_to_input(self):
        outcome = await collect(_FakePage([_snapshot(end=True)], resolved_url="https://www.douyin.com/"), SOURCE)
        self.assertEqual(outcome["status"], "unsupported_page")
        self.assertEqual(outcome["records"], [])
        self.assertFalse(outcome["complete"])

    async def test_navigation_while_reading_discards_foreign_snapshot(self):
        page = _FakePage([_snapshot(end=True)], change_after_snapshot={2: SOURCE.replace(PRODUCT, "999")})
        outcome = await collect(page, SOURCE)
        self.assertEqual(outcome["status"], "product_mismatch")
        self.assertEqual(outcome["records"], [])
        self.assertFalse(outcome["complete"])

    async def test_short_link_locks_identity_after_resolution(self):
        outcome = await collect(_FakePage([_snapshot(end=True)]), "https://v.douyin.com/example/")
        self.assertEqual(outcome["status"], "completed")
        self.assertEqual(outcome["records"][0]["product_id"], PRODUCT)

    async def test_buyer_login_words_do_not_block_collection(self):
        data = _snapshot(body_text="商品评价 请先登录", ui_text="商品评价", end=True,
                         reviews=[{"review_id": "r1", "content": "商家说请先登录", "imgs": ["https://img.test/a.jpg"]}])
        outcome = await collect(_FakePage([data]), SOURCE)
        self.assertEqual(outcome["status"], "completed")

    async def test_actual_login_ui_overrides_visible_end_marker(self):
        outcome = await collect(_FakePage([_snapshot(ui_text="请先登录", end=True)]), SOURCE)
        self.assertEqual(outcome["status"], "login_required")
        self.assertFalse(outcome["complete"])

    async def test_stall_never_means_completed(self):
        outcome = await collect(_FakePage([_snapshot()]), SOURCE, max_scrolls=8)
        self.assertEqual(outcome["status"], "partial")
        self.assertFalse(outcome["complete"])
        self.assertIn("连续滚动", outcome["reason"])

    async def test_cancellation_preserves_review_already_read(self):
        class PausingPage(_FakePage):
            def __init__(self):
                super().__init__([_snapshot()])
                self.waiting_after_first_review = asyncio.Event()

            async def wait_for_timeout(self, milliseconds):
                if milliseconds == 900:
                    self.waiting_after_first_review.set()
                    await asyncio.Event().wait()

        page = PausingPage()
        task = asyncio.create_task(collect(page, SOURCE))
        await asyncio.wait_for(page.waiting_after_first_review.wait(), timeout=1)
        task.cancel()
        outcome = await asyncio.wait_for(task, timeout=1)
        self.assertEqual(outcome["status"], "cancelled")
        self.assertFalse(outcome["complete"])
        self.assertEqual(len(outcome["records"]), 1)
        self.assertEqual(outcome["records"][0]["review_id"], "r1")
        self.assertEqual(outcome["records"][0]["images"], ["https://img.test/a.jpg"])
        self.assertEqual(outcome["diagnostics"]["image_count"], 1)

    async def test_sold_out_h5_is_a_negative_sample(self):
        # Parent browser observation for product 3845439021845184916 on
        # 2026-10-29: H5 shows "购物车 已抢光", no usable review entry.
        outcome = await collect(_FakePage([_snapshot(reviews=[], ui_text="购物车\n已抢光", review_scope_found=False)]), SOURCE)
        self.assertEqual(outcome["status"], "product_unavailable")
        self.assertEqual(outcome["records"], [])
        self.assertFalse(outcome["complete"])


class DouyinSnapshotDomTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node is optional; required only for the in-memory DOM regression")
    def test_buyer_text_is_not_a_platform_end_empty_or_login_marker(self):
        # Evaluate the actual extraction script against a tiny in-memory DOM.
        # No browser, network, external package, or platform execution is used.
        harness = r"""
const extract = EXTRACT_FUNCTION;
class Element {
  constructor(tag, cls = '', text = '') {
    this.tagName = tag; this.className = cls; this.ownText = text;
    this.children = []; this.parentElement = null;
    this.clientHeight = 200; this.scrollHeight = 200; this.scrollTop = 0;
  }
  add(child) { child.parentElement = this; this.children.push(child); return child; }
  matches(s) { return s.startsWith('.') ? this.className.split(' ').includes(s.slice(1)) : this.tagName === s; }
  closest(s) { for (let p = this; p; p = p.parentElement) if (s.split(',').some(v => p.matches(v.trim()))) return p; return null; }
  descendants() { return this.children.flatMap(c => [c, ...c.descendants()]); }
  querySelectorAll(s) {
    if (s === '*') return this.descendants();
    if (s === '.qZmIVPuL img') return this.descendants().filter(e => e.tagName === 'img' && e.closest('.qZmIVPuL'));
    return this.descendants().filter(e => e.matches(s));
  }
  querySelector(s) { return this.querySelectorAll(s)[0] || null; }
  get innerText() { return [this.ownText, ...this.children.map(c => c.innerText)].filter(Boolean).join('\n'); }
  getClientRects() { return [{}]; }
  getAttribute() { return ''; }
}
const body = new Element('body');
const root = body.add(new Element('div', 'eRfaxg7R'));
for (const words of ['没有更多了', '暂无评价', '请先登录']) {
  const item = root.add(new Element('div', 'Ez3MC6d7'));
  item.add(new Element('span', 'mgNuPdjB', words));
}
global.NodeFilter = {SHOW_TEXT: 4};
global.getComputedStyle = e => ({visibility: 'visible', overflowY: e === root ? 'auto' : 'visible'});
global.document = {
  body, documentElement: new Element('html'),
  querySelectorAll: s => body.querySelectorAll(s),
  createTreeWalker(scope) {
    const texts = [scope, ...scope.descendants()].filter(e => e.ownText).map(e => ({parentElement: e, textContent: e.ownText}));
    let i = -1;
    return {currentNode: null, nextNode() { i++; this.currentNode = texts[i] || null; return !!this.currentNode; }};
  }
};
const buyerOnly = extract();
const marker = root.add(new Element('span', '', '没有更多了'));
body.add(new Element('div', '', '请先登录'));
const platformEnd = extract();
marker.ownText = '暂无评价';
const platformEmpty = extract();
process.stdout.write(JSON.stringify({buyerOnly, platformEnd, platformEmpty}));
""".replace("EXTRACT_FUNCTION", _SNAPSHOT_JS)
        process = subprocess.run([shutil.which("node"), "-e", harness], capture_output=True,
                                 text=True, encoding="utf-8", timeout=10)
        self.assertEqual(process.returncode, 0, process.stderr)
        data = json.loads(process.stdout)
        self.assertFalse(data["buyerOnly"]["end"])
        self.assertFalse(data["buyerOnly"]["empty"])
        self.assertNotIn("请先登录", data["buyerOnly"]["ui_text"])
        self.assertEqual(len(data["buyerOnly"]["reviews"]), 3)
        self.assertTrue(data["platformEnd"]["end"])
        self.assertIn("请先登录", data["platformEnd"]["ui_text"])
        self.assertTrue(data["platformEmpty"]["empty"])


if __name__ == "__main__":
    unittest.main()
