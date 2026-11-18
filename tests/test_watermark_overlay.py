"""Whole overlay-block evidence: complete masks without erasing detail copy."""
import unittest

from buyer_show.watermark_overlay import plan_overlay_blocks, merge_overlay_decision


def scan(index, rows, *, product='taobao:1', sources=('product_main',), size=(750, 1000), sha=None):
    width, height = size
    return {'item_id': str(index), 'product_key': product,
            'source_types': list(sources), 'sha256': sha or ('sha-%s' % index),
            'detection': {'width': width, 'height': height, 'scan_complete': True,
                          'source_sha256': sha or ('sha-%s' % index),
                          'regions': [{'x': x, 'y': y, 'w': w, 'h': h,
                                       'label': label, 'confidence': .99}
                                      for x, y, w, h, label in rows]}}


LOGO = [
    (28, 22, 179, 46, '边角文字候选：TEENIE'),
    (29, 61, 180, 40, '边角文字候选：WEENIE'),
    (51, 101, 132, 24, '边角文字候选：ALLTHAT BEAR'),
    (49, 122, 135, 40, '边角文字候选：官方正品'),
]


class OverlayBlockTests(unittest.TestCase):
    def test_repeated_corner_logo_becomes_one_edge_to_edge_block(self):
        plans = plan_overlay_blocks([scan(i, LOGO) for i in range(3)])
        for index in range(3):
            plan = plans[str(index)]
            self.assertEqual(len(plan['approved']), 1)
            box = plan['approved'][0]
            self.assertEqual((box['x'], box['y']), (0, 0))
            self.assertGreater(box['w'], 200)  # covers graphic/badge, not just the lines
            self.assertGreater(box['h'], 160)
            self.assertEqual(box['evidence']['basis'], 'repeated_corner_overlay')
            self.assertGreaterEqual(box['evidence']['independent_images'], 3)
            self.assertFalse(plan['review'])

    def test_single_image_logo_is_visible_but_not_auto_erased(self):
        plan = plan_overlay_blocks([scan(0, LOGO)])['0']
        self.assertFalse(plan['approved'])
        self.assertTrue(plan['review'])

    def test_detail_page_service_headings_are_never_auto_erased(self):
        rows = [(56, 121, 165, 51, '边角文字候选：官方保证'),
                (55, 163, 238, 51, '边角文字候选：乐享细致服务'),
                (366, 127, 362, 24, '店铺标记候选：我们郑重地承诺正品')]
        plans = plan_overlay_blocks([scan(i, rows, sources=('product_detail',)) for i in range(3)])
        for index in range(3):
            self.assertFalse(plans[str(index)]['approved'])
            self.assertTrue(plans[str(index)]['review'])

    def test_platform_promo_banner_is_approved_even_on_one_image(self):
        rows = [(100, 940, 550, 40, '平台标记候选：天猫双11全球狂欢季，限时立减15%')]
        plan = plan_overlay_blocks([scan(0, rows)])['0']
        self.assertEqual(len(plan['approved']), 1)
        box = plan['approved'][0]
        # Only the promo artwork is masked; the designed colour field around
        # it stays real, so the model does not have to invent a large flat area.
        self.assertGreater(box['x'], 0)
        self.assertLess(box['w'], 750)
        self.assertLess(box['y'] + box['h'], 1000)
        self.assertEqual(box['evidence']['basis'], 'platform_banner')

    def test_oversized_block_is_not_promoted_to_automatic(self):
        rows = [(100, 700, 500, 250, '平台标记候选：天猫双11全球狂欢季，限时立减15%')]
        plans = plan_overlay_blocks([scan(i, rows) for i in range(3)])
        self.assertFalse(plans['0']['approved'])
        self.assertTrue(plans['0']['review'])

    def test_ordinary_product_copy_is_not_an_overlay_candidate(self):
        rows = [(50, 111, 573, 69, '边角文字候选：大空间装载更多可能'),
                (45, 188, 292, 55, '边角文字候选：通勤出游逛街'),
                (122, 836, 91, 33, '边角文字候选：14寸电脑')]
        plan = plan_overlay_blocks([scan(i, rows) for i in range(3)])
        for index in range(3):
            self.assertFalse(plan[str(index)]['approved'])
            self.assertFalse(plan[str(index)]['review'])

    def test_merge_removes_uncertain_lines_already_inside_an_approved_block(self):
        block = {'x': 0, 'y': 0, 'w': 220, 'h': 173, 'detection_kind': 'overlay_block',
                 'evidence': {'zone': 'left_top', 'strengths': ['brand'], 'texts': ['TEENIE'],
                              'basis': 'repeated_corner_overlay', 'independent_images': 3, 'box': {}}}
        line = {'x': 28, 'y': 22, 'w': 179, 'h': 46}
        decision = {'item_id': '0', 'status': 'needs_review', 'approved_regions': [],
                    'review_regions': [line], 'proposed_regions': [line],
                    'review_reasons': ['检测到商家或边角品牌文字，但缺少同商品至少3张独立图片的稳定位置及局部背景变化证据。'],
                    'evidence': {'policy_version': 'old'}, 'warnings': []}
        plan = {'approved': [block], 'review': [], 'evidence': [], 'policy_version': 'x'}
        merged = merge_overlay_decision(decision, plan, 750, 1000)
        self.assertEqual(merged['approved_regions'], [block])
        self.assertFalse(merged['review_regions'])
        self.assertEqual(merged['status'], 'auto')
        self.assertFalse(merged['review_reasons'])
        self.assertIn('overlay_blocks', merged['evidence'])
        self.assertEqual(decision['review_regions'], [line])  # caller's data untouched

    def test_low_confidence_badge_text_joins_the_same_corner_block(self):
        # A 0.84-confidence badge word used to be dropped, leaving half the
        # graphic badge in the image as model "context".
        scans = [scan(i, LOGO, size=(1440, 1920)) for i in range(3)]
        scans[0]['detection']['regions'].append(
            {'x': 1283, 'y': 1572, 'w': 157, 'h': 180,
             'label': '边角文字候选：COLLEG', 'confidence': .84})
        plans = plan_overlay_blocks(scans)
        zones = {b['evidence']['zone'] for b in plans['0']['approved']}
        self.assertIn('left_top', zones)
        self.assertIn('right_bottom', zones)


if __name__ == '__main__':
    unittest.main()
