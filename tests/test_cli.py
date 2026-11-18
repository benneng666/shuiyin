import asyncio
import contextlib
import io
import signal
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

from buyer_show.__main__ import main


@contextlib.contextmanager
def mock_batch(callback):
    engine = ModuleType("buyer_show.engine")
    engine.run_batch = callback
    with patch.dict(sys.modules, {"buyer_show.engine": engine}):
        yield


class CliTests(unittest.TestCase):
    def test_help_describes_expanded_media_scope_and_platform_limitations(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as raised:
            main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("主图、详情图、商品视频及评价/追评图片视频", output.getvalue())
        self.assertIn("小红书商品评价尚未支持", output.getvalue())

    def test_final_summary_handles_video_and_legacy_image_reports(self):
        async def fake_batch(*args, **kwargs):
            return [
                {"complete": False, "stats": {"unique_images": 2, "unique_videos": 3,
                                               "downloads_failed": 1, "videos_failed": 2}},
                {"complete": True, "stats": {"unique_images": 4}},
            ]

        output = io.StringIO()
        with mock_batch(fake_batch), contextlib.redirect_stdout(output):
            code = main(["https://item.taobao.com/item.htm?id=123"])
        self.assertEqual(code, 1)
        self.assertIn("图片 6，视频 3；下载失败 3", output.getvalue())
        self.assertIn("素材来源见商品清单", output.getvalue())

    def test_login_prompt_explains_independent_browser_session(self):
        continued = []

        async def fake_batch(*args, **kwargs):
            continued.append(await args[4]("请登录"))
            return [{"complete": False}]

        output = io.StringIO()
        with mock_batch(fake_batch), patch("builtins.input", return_value="y"), contextlib.redirect_stdout(output):
            main(["https://item.taobao.com/item.htm?id=123"])
        self.assertEqual(continued, [True])
        self.assertIn("独立 Edge / Chrome", output.getvalue())
        self.assertIn("日常浏览器登录不共享", output.getvalue())

    def test_ctrl_c_uses_graceful_stop_and_restores_signal_handler(self):
        old = signal.getsignal(signal.SIGINT)
        observed = []

        async def fake_batch(*args, **kwargs):
            signal.raise_signal(signal.SIGINT)
            observed.append(kwargs["should_stop"]())
            await asyncio.sleep(0)
            return []

        with mock_batch(fake_batch), contextlib.redirect_stdout(io.StringIO()):
            code = main(["https://item.taobao.com/item.htm?id=123"])
        self.assertEqual(code, 130)
        self.assertEqual(observed, [True])
        self.assertIs(signal.getsignal(signal.SIGINT), old)
