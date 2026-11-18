"""Protect provenance and completeness at the unsupported XHS boundary."""

import unittest

from buyer_show.adapters.xhs import collect, parse_reviews


PRODUCT = "https://www.xiaohongshu.com/goods-detail/6a83ca5ffa5d5c0001a98ca5"


class FakePage:
    def __init__(self, url=PRODUCT, text=""):
        self.url = url
        self.text = text
        self.reads = 0
        self.navigations = []

    async def goto(self, url, wait_until, timeout):
        self.navigations.append(url)
        # Short-link targets and redirects are represented by the fixture URL.
        if "xhslink." not in url:
            self.url = url
        return None

    def locator(self, selector):
        if selector != "body":
            raise AssertionError(selector)
        return self

    async def inner_text(self, timeout):
        self.reads += 1
        return self.text


class XiaohongshuProvenanceTests(unittest.IsolatedAsyncioTestCase):
    def test_detail_images_and_template_placeholder_are_not_reviews(self):
        payload = {
            "success": True,
            "data": {
                "template": '[{"key":"unboxing","parent":"app.slot.comments"}]',
                "template_data": [{
                    "carouselH5": {"images": [{"url": "https://example.com/hero.jpg"}]},
                    "graphicDetailsV4": {"images": [{"url": "https://example.com/detail.jpg"}]},
                }],
            },
        }
        self.assertEqual(parse_reviews(payload, PRODUCT), [])

    def test_note_comment_pictures_are_rejected_even_with_product_source(self):
        payload = {"data": {"comments": [{
            "id": "comment-id", "note_id": "6a63a800000000000f029c91",
            "content": "图片", "pictures": [{"url_default": "https://example.com/comment.jpg"}],
        }]}}
        self.assertEqual(parse_reviews(payload, PRODUCT), [])

    async def test_visible_product_and_review_label_do_not_imply_success(self):
        result = await collect(FakePage(text="商品评价 查看全部 商品主图"), PRODUCT)
        self.assertEqual(result["status"], "unsupported_review_entry")
        self.assertEqual(result["records"], [])
        self.assertFalse(result["complete"])
        self.assertTrue(result["diagnostics"]["visible_review_entry_marker"])

    async def test_app_prompt_is_incomplete(self):
        result = await collect(FakePage(text="打开小红书App查看更多"), PRODUCT)
        self.assertEqual(result["status"], "app_required")
        self.assertFalse(result["complete"])

    async def test_note_and_lookalike_host_are_not_products(self):
        for source in (
            "https://www.xiaohongshu.com/explore/6a63a800000000000f029c91?item_id=6a83ca5ffa5d5c0001a98ca5",
            "https://www.xiaohongshu.com.evil.example/goods-detail/6a83ca5ffa5d5c0001a98ca5",
        ):
            with self.subTest(source=source):
                page = FakePage()
                result = await collect(page, source)
                self.assertEqual(result["status"], "unsupported_review_entry")
                self.assertEqual(page.reads, 0)
                self.assertIsNone(result["diagnostics"]["product_id"])

    async def test_redirected_short_link_requires_matching_product_page(self):
        result = await collect(FakePage(), "https://xhslink.cn/o/public-share")
        self.assertEqual(result["diagnostics"]["product_id"], "6a83ca5ffa5d5c0001a98ca5")
        self.assertFalse(result["complete"])
        note_page = FakePage("https://www.xiaohongshu.com/explore/6a63a800000000000f029c91")
        result = await collect(note_page, "https://xhslink.cn/o/public-share")
        self.assertIsNone(result["diagnostics"]["product_id"])
        self.assertEqual(note_page.reads, 0)

    async def test_navigation_to_requested_product_from_blank_page(self):
        page = FakePage("about:blank")
        result = await collect(page, PRODUCT)
        self.assertEqual(page.navigations, [PRODUCT])
        self.assertEqual(page.url, PRODUCT)
        self.assertFalse(result["complete"])

    async def test_product_mismatch_after_navigation_does_not_read_other_product(self):
        class RedirectedPage(FakePage):
            async def goto(self, url, wait_until, timeout):
                self.navigations.append(url)

        page = RedirectedPage("https://www.xiaohongshu.com/goods-detail/64f076e669262b000145d2c0")
        result = await collect(page, PRODUCT)
        self.assertTrue(result["diagnostics"]["product_identity_mismatch"])
        self.assertEqual(page.reads, 0)
        self.assertFalse(result["complete"])

    async def test_navigation_error_is_not_empty_success(self):
        class FailingPage(FakePage):
            async def goto(self, url, wait_until, timeout):
                raise TimeoutError("not persisted")

        result = await collect(FailingPage("about:blank"), PRODUCT)
        self.assertEqual(result["status"], "unsupported_review_entry")
        self.assertEqual(result["diagnostics"]["navigation_error"], "TimeoutError")
        self.assertFalse(result["complete"])

    async def test_diagnostics_drop_url_query_secrets(self):
        source = PRODUCT + "?xsec_token=do-not-store&web_session=secret"
        result = await collect(FakePage(url=source), source)
        self.assertNotIn("do-not-store", str(result))
        self.assertNotIn("secret", str(result))


if __name__ == "__main__":
    unittest.main()
