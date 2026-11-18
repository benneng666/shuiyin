"""The folded-review notice is not itself authorization to click a control."""
import json
import unittest
from types import SimpleNamespace

from buyer_show.adapters.taobao import _DOM, _FOLDED_NOTICE, collect, _folded_entry
from tests.test_taobao_parser import URL, A, _EmptyLocator, _CancellablePage


class _FoldLocator(_EmptyLocator):
    def __init__(self, page, control=False):
        self.page, self.control = page, control

    async def count(self):
        return int(not self.control or self.page.mode in {"button", "disabled", "hidden", "login"})

    def nth(self, index):
        return self

    async def is_visible(self):
        return not (self.control and self.page.mode == "hidden")

    async def evaluate(self, script):
        if self.page.mode == "login":
            self.page.url = "https://login.taobao.com/member/login.jhtml"
        return {"semantic": self.page.mode == "self_button", "text": "展开折叠评价" if self.page.mode == "self_button" else "已折叠72条对你帮助不大的评价"}

    def locator(self, selector):
        return _FoldLocator(self.page, control=True)

    async def is_enabled(self):
        return self.page.mode != "disabled"

    async def get_attribute(self, name):
        return None

    async def click(self, **kwargs):
        self.page.fold_clicks += 1


class _Page(_CancellablePage):
    def __init__(self, mode, native_end=True):
        super().__init__()
        self.mode, self.native_end, self.fold_clicks = mode, native_end, 0

    async def goto(self, *args, **kwargs):
        response = SimpleNamespace(
            url="https://h5api.m.taobao.com/h5/mtop.taobao.rate.detaillist.get/6.0/", status=200)
        response.request = SimpleNamespace(url=response.url + '?data=' + json.dumps({
            "auctionNumId": "674605142427", "pageNo": 1, "foldFlag": "0"}), post_data=None)
        async def text():
            return json.dumps({"data": {"rateList": [{"id": "native", "feedback": "review", "feedPicPathList": [A]}],
                "hasNext": "false" if self.native_end else "true", "foldCount": "72"}})
        response.text = text
        self.listener(response)

    async def wait_for_timeout(self, milliseconds):
        pass

    def locator(self, selector):
        return _FoldLocator(self) if selector == _FOLDED_NOTICE else _EmptyLocator()

    async def evaluate(self, script):
        return [{"id": "native", "content": "review", "pics": [A]}] if script == _DOM else {"moved": False}


class FoldedReviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_information_only_notice_is_not_clicked_and_retains_partial_records(self):
        page = _Page("notice")
        result = await collect(page, URL, max_scrolls=10)
        self.assertEqual(page.fold_clicks, 0)
        self.assertEqual(result["diagnostics"]["folded_ui_state"], "notice_only")
        self.assertTrue(result["diagnostics"]["folded_notice_seen"])
        self.assertFalse(result["diagnostics"]["folded_expand_attempted"])
        self.assertTrue(result["diagnostics"]["review_scopes"]["normal"]["end_observed"])
        self.assertFalse(result["diagnostics"]["review_scopes"]["folded"]["end_observed"])
        self.assertEqual(result["records"][0]["images"], [A])
        self.assertEqual(result["status"], "partial")
        self.assertFalse(result["complete"])
        self.assertIn("未发现可确认的展开控件", result["reason"])
        unreachable = result["diagnostics"]["folded_unreachable"]
        self.assertEqual(unreachable["count"], 72)
        self.assertEqual(unreachable["reason"], "platform_notice_only_no_expand_control")
        self.assertFalse(unreachable["platform_expand_control_seen"])
        self.assertIn("平台不可达 72 条折叠评价", result["reason"])
        self.assertIn("已保存评价记录", result["reason"])

    async def test_explicit_expand_button_only_clicked_once_and_not_assumed_loaded(self):
        for mode in ("button", "self_button"):
            with self.subTest(mode=mode):
                page = _Page(mode)
                result = await collect(page, URL, max_scrolls=3)
                self.assertEqual(page.fold_clicks, 1)
                self.assertEqual(result["diagnostics"]["folded_ui_state"], "expand_control")
                self.assertTrue(result["diagnostics"]["folded_expand_attempted"])
                self.assertFalse(result["diagnostics"]["folded_reviews_opened"])
                self.assertFalse(result["complete"])
                self.assertNotIn("folded_unreachable", result["diagnostics"])

    async def test_hidden_and_disabled_buttons_are_not_expand_actions(self):
        for mode in ("hidden", "disabled"):
            with self.subTest(mode=mode):
                notice, control = await _folded_entry(_Page(mode))
                self.assertIsNotNone(notice)
                self.assertIsNone(control)

    async def test_login_appearing_before_expand_stops_without_click(self):
        page = _Page("login")
        result = await collect(page, URL, max_scrolls=3)
        self.assertEqual(result["status"], "login_required")
        self.assertEqual(page.fold_clicks, 0)
        self.assertFalse(result["diagnostics"]["folded_expand_attempted"])
        self.assertEqual(result["records"][0]["images"], [A])


if __name__ == "__main__":
    unittest.main()
