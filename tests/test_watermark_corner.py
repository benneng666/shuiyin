import unittest
from buyer_show.watermark import _candidate_regions


class CornerCandidateTests(unittest.TestCase):
    def test_brand_text_requires_explicit_extra_candidate_mode(self):
        row = [[[20,20],[220,20],[220,60],[20,60]], 'TEENIE WEENIE', .98]
        self.assertEqual(_candidate_regions([row], 750, 1000), [])
        found = _candidate_regions([row], 750, 1000, include_corner_text=True)
        self.assertEqual(len(found), 1)
        self.assertTrue(found[0]['label'].startswith('边角文字候选'))

    def test_more_candidates_still_exclude_central_and_product_spec_text(self):
        rows = [
            [[[220,400],[420,400],[420,450],[220,450]], 'PRINT ON BAG', .99],
            [[[20,20],[220,20],[220,60],[20,60]], '商品规格 纯棉', .99],
        ]
        self.assertEqual(_candidate_regions(rows, 750, 1000, include_corner_text=True), [])


if __name__ == '__main__':
    unittest.main()
