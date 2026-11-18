import unittest

from buyer_show.links import ProductLink, classify_url, extract_links


TAOBAO = "https://item.taobao.com/item.htm?id=123456789012"
XHS = "https://www.xiaohongshu.com/goods-detail/6a83ca5ffa5d5c0001a98ca5"
DOUYIN = "https://haohuo.jinritemai.com/ecommerce/trade/detail/index.html?id=3738302600051818881"


class ProductLinkTests(unittest.TestCase):
    def test_known_product_routes(self):
        cases = [
            (TAOBAO, "taobao", "123456789012"),
            ("https://detail.tmall.com/item.htm?id=123456789012", "taobao", "123456789012"),
            ("https://h5.m.taobao.com/awp/core/detail.htm?id=123456789012", "taobao", "123456789012"),
            (XHS, "xiaohongshu", "6a83ca5ffa5d5c0001a98ca5"),
            (DOUYIN, "douyin", "3738302600051818881"),
        ]
        for url, platform, product_id in cases:
            with self.subTest(url=url):
                self.assertEqual(classify_url(url), ProductLink(platform, product_id, url))

    def test_query_signatures_are_byte_for_byte_preserved(self):
        url = TAOBAO + "&sign=a%2Fb%3D+Z&x=2&x=1#details"
        self.assertEqual(classify_url(url).url, url)
        self.assertEqual(extract_links("看看这个 " + url + " 复制后打开")[0].url, url)

    def test_short_links_remain_unconfirmed(self):
        for host, platform in (("m.tb.cn", "taobao"), ("v.douyin.com", "douyin"),
                               ("xhslink.com", "xiaohongshu"), ("xhslink.cn", "xiaohongshu")):
            with self.subTest(host=host):
                url = f"https://{host}/o/shared?id=123&sign=abc%3D"
                result = classify_url(url)
                self.assertEqual(result.platform, platform)
                self.assertTrue(result.is_short)
                self.assertIsNone(result.product_id)
                self.assertEqual(result.url, url)

    def test_multiline_share_prose_and_wrappers(self):
        text = f"推荐：{TAOBAO}，复制打开\n小红书（{XHS}）\n[抖音商品]({DOUYIN})"
        self.assertEqual([item.platform for item in extract_links(text)], ["taobao", "xiaohongshu", "douyin"])

    def test_advertising_redirect_preserves_signature_and_requires_resolution(self):
        url = "https://click.simba.taobao.com/cc_im?s=123&e=a%2Fb+z%3D&a=skuId%3D456"
        result = classify_url(url)
        self.assertEqual(result, ProductLink("taobao", None, url, is_short=True))
        self.assertEqual(extract_links("商品：" + url)[0].url, url)
        for unsupported in (
            "https://click.simba.taobao.com/other?s=123",
            "https://click.simba.taobao.com/cc_im/extra?s=123",
            "https://click.simba.taobao.com.evil.test/cc_im?s=123",
            "https://user:pass@click.simba.taobao.com/cc_im?s=123",
            "https://click.simba.taobao.com:444/cc_im?s=123",
        ):
            with self.subTest(url=unsupported), self.assertRaises(ValueError):
                classify_url(unsupported)

    def test_deduplicate_product_identity_but_keep_first_signed_link(self):
        first = TAOBAO + "&sign=first%3D"
        second = "https://detail.tmall.com/item.htm?id=123456789012&sign=other"
        short = "https://m.tb.cn/h.someShare?tk=abc"
        links = extract_links("\n".join([first, first, second, short, short]))
        self.assertEqual(len(links), 2)
        self.assertEqual(links[0].url, first)
        self.assertTrue(links[1].is_short)

    def test_social_links_are_explicit_errors_even_in_mixed_input(self):
        for wrong, label in (
            ("https://www.xiaohongshu.com/explore/6a63a800000000000f029c91", "笔记"),
            ("https://www.xiaohongshu.com/discovery/item/6a63a800000000000f029c91", "笔记"),
            ("https://www.douyin.com/video/7689454534989609202", "视频"),
            ("https://www.douyin.com/note/7689454534989609202", "图文"),
            ("https://www.douyin.com/?modal_id=7689454534989609202", "视频"),
        ):
            with self.subTest(wrong=wrong):
                with self.assertRaisesRegex(ValueError, label):
                    extract_links(TAOBAO + "\n" + wrong)

    def test_exact_domain_security_boundary(self):
        for url in (
            "https://item.taobao.com.evil.com/item.htm?id=123",
            "https://evil.com/?url=" + TAOBAO,
            "https://user:pass@item.taobao.com/item.htm?id=123",
            "https://evil.com@item.taobao.com/item.htm?id=123",
            "https://item.taobao.com@evil.com/item.htm?id=123",
            "http://localhost/item.htm?id=123",
            "http://127.0.0.1/item.htm?id=123",
            "https://item.taobao.com:444/item.htm?id=123",
            "https://item.taobao.com:/item.htm?id=123",
            "https://item.taobao.com./item.htm?id=123",
            "https://item.taobao.com\\@evil.com/item.htm?id=123",
        ):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    classify_url(url)

    def test_only_standard_explicit_ports_allowed(self):
        for url in ("https://item.taobao.com:443/item.htm?id=123", "http://item.taobao.com:80/item.htm?id=123"):
            self.assertEqual(classify_url(url).product_id, "123")

    def test_control_characters_cannot_be_hidden_by_urlsplit(self):
        for char in ("\n", "\r", "\t", "\x00"):
            with self.subTest(char=repr(char)):
                with self.assertRaises(ValueError):
                    classify_url("https://item.tao" + char + "bao.com/item.htm?id=123")

    def test_ids_and_paths_are_not_guessed(self):
        invalid = [
            "https://item.taobao.com/item.htm",
            "https://item.taobao.com/item.htm?id=abc",
            "https://item.taobao.com/item.htm?id=0",
            "https://item.taobao.com/item.htm?id=123&id=456",
            "https://item.taobao.com/item.htm?id=123&id=",
            "https://item.taobao.com/not-item?id=123",
            "https://www.douyin.com/product/3738302600051818881",
            "https://haohuo.jinritemai.com/video/123?id=3738302600051818881",
            "https://haohuo.jinritemai.com/ecommerce/trade/detail/index.html?product_id=3738302600051818881",
            "https://www.xiaohongshu.com/goods-detail/not-a-product-id",
            "https://www.xiaohongshu.com/shop/6a8276edf4861a0015578413",
            "https://m.tb.cn/",
        ]
        for url in invalid:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    classify_url(url)

    def test_no_urls_gives_helpful_error(self):
        with self.assertRaisesRegex(ValueError, "没有找到"):
            extract_links("商品口令但没有链接")

    def test_error_does_not_echo_signature(self):
        with self.assertRaises(ValueError) as raised:
            extract_links("https://evil.com/path?sign=private-signature")
        self.assertNotIn("private-signature", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
