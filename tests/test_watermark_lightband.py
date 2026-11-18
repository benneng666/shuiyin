"""Evidence-bound detection of light overlay headlines that OCR misses."""
import unittest
from unittest import mock

import numpy as np

from buyer_show.watermark_lightband import (
    detect_bands, plan_light_bands, merge_light_band_decision)


def canvas(width=750, height=1000, lines=((80, 300), (80, 380)), white_block=False):
    image = np.full((height, width, 3), 185, np.uint8)
    rng = np.random.default_rng(7)
    image[:, :] = np.clip(image.astype(np.int16) + rng.integers(-6, 7, image.shape), 0, 255).astype(np.uint8)
    import cv2
    for x, y in lines:
        cv2.putText(image, 'PROMO HEADLINE', (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    1.15, (252, 252, 252), 5, cv2.LINE_AA)
    if white_block:
        image[600:860, 100:500] = 255
    return image


def scan(index, *, product='taobao:1', sources=('product_main',), regions=(), size=(750, 1000)):
    width, height = size
    return {'item_id': str(index), 'product_key': product, 'path': 'image-%s.webp' % index,
            'source_types': list(sources), 'sha256': 'sha-%s' % index,
            'detection': {'width': width, 'height': height, 'scan_complete': True,
                          'source_sha256': 'sha-%s' % index, 'regions': list(regions)}}


class DetectBandsTests(unittest.TestCase):
    def test_finds_light_headline_lines_not_large_white_areas(self):
        bands = detect_bands(canvas(white_block=True))
        self.assertGreaterEqual(len(bands), 1)
        self.assertTrue(any(250 <= band['y'] <= 400 and band['w'] > 300 for band in bands))
        self.assertFalse(any(band['y'] > 600 and band['h'] > 100 for band in bands))

    def test_band_is_text_height_not_a_whole_corner(self):
        for band in detect_bands(canvas()):
            self.assertLess(band['h'], 100)
            self.assertGreater(band['w'] / band['h'], 2)


class PlanLightBandsTests(unittest.TestCase):
    def test_repeated_headline_across_main_photos_is_approved(self):
        image = canvas()
        plans = plan_light_bands([scan(0), scan(1)], loader=lambda path: image)
        self.assertTrue(plans['0']['approved'])
        self.assertTrue(plans['1']['approved'])
        self.assertEqual(plans['0']['approved'][0]['evidence']['basis'], 'repeated_light_headline')
        self.assertEqual(plans['0']['approved'][0]['detection_kind'], 'light_band')

    def test_single_photo_without_ocr_or_peer_stays_unapproved(self):
        plans = plan_light_bands([scan(0)], loader=lambda path: canvas())
        self.assertFalse(plans['0']['approved'])
        self.assertTrue(plans['0']['evidence'])

    def test_ocr_line_corroborates_a_band_on_one_photo(self):
        image = canvas()
        bands = detect_bands(image)
        box = bands[0]
        regions = [{'x': box['x'], 'y': box['y'], 'w': box['w'], 'h': box['h'],
                    'label': '边角文字候选：促销', 'confidence': .99}]
        plans = plan_light_bands([scan(0, regions=regions)], loader=lambda path: image)
        self.assertTrue(any(region['evidence']['basis'] == 'ocr_corroborated_headline'
                            for region in plans['0']['approved']))

    def test_detail_images_are_never_auto_approved(self):
        image = canvas()
        plans = plan_light_bands([scan(i, sources=('product_detail',)) for i in range(3)],
                                 loader=lambda path: image)
        self.assertEqual(plans, {})

    def test_band_already_covered_by_whole_block_is_not_duplicated(self):
        image = canvas()
        bands = detect_bands(image)
        box = bands[0]
        overlay = {'0': {'approved': [dict(box)], 'review': [], 'evidence': []},
                   '1': {'approved': [], 'review': [], 'evidence': []}}
        plans = plan_light_bands([scan(0), scan(1)], overlay, loader=lambda path: image)
        self.assertFalse(any(region['x'] == box['x'] and region['y'] == box['y']
                             for region in plans['0']['approved']))

    def test_merge_adds_regions_and_keeps_evidence_trail(self):
        image = canvas()
        plans = plan_light_bands([scan(0), scan(1)], loader=lambda path: image)
        decision = {'item_id': '0', 'status': 'needs_review', 'approved_regions': [],
                    'review_regions': [], 'proposed_regions': [], 'review_reasons': [],
                    'evidence': {}, 'warnings': []}
        merged = merge_light_band_decision(decision, plans['0'], 750, 1000)
        self.assertEqual(len(merged['approved_regions']), len(plans['0']['approved']))
        self.assertEqual(merged['status'], 'auto')
        self.assertIn('light_bands', merged['evidence'])
        self.assertEqual(decision['approved_regions'], [])  # input is not mutated


class DesignBandTests(unittest.TestCase):
    def test_detects_uniform_bottom_strip_but_not_a_photo(self):
        from buyer_show.watermark_lightband import detect_design_band
        image = np.full((400, 300, 3), 200, np.uint8)
        rng = np.random.default_rng(3)
        image[:300] = np.clip(image[:300].astype(np.int16)
                              + rng.integers(-110, 111, image[:300].shape), 0, 255).astype(np.uint8)
        image[300:350] = (10, 10, 10)
        image[350:] = (190, 70, 40)
        image[320, 20:60] = 255  # white lettering in the middle of a row
        # A row of white lettering breaks the run; the returned top is still
        # inside the designed strip and covers that lettering row.
        top = detect_design_band(image)
        self.assertIsNotNone(top)
        self.assertTrue(300 <= top < 350)
        self.assertIsNone(detect_design_band(
            np.clip(np.full((400, 300, 3), 128, np.int16)
                    + rng.integers(-110, 111, (400, 300, 3)), 0, 255).astype(np.uint8)))

    def test_repair_uses_row_colours_and_preserves_exterior_and_alpha(self):
        import tempfile
        from pathlib import Path
        from PIL import Image
        from buyer_show.watermark import repair_image
        image = np.full((120, 100, 4), 255, np.uint8)
        image[:, :, :3] = (175, 175, 175)
        image[80:, :, :3] = (190, 60, 40)
        image[90, 10:80, :3] = 255  # white promo lettering
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'strip.png'
            Image.fromarray(image).save(source)
            result = repair_image(str(source),
                                  [{'x': 0, 'y': 80, 'w': 100, 'h': 40, 'fill': 'row_band'}],
                                  output_root=Path(folder) / 'out', method='auto')
            with Image.open(result['output']) as im:
                repaired = np.asarray(im.convert('RGBA'))
            with Image.open(result['mask']) as im:
                mask = np.asarray(im.convert('L'))
        self.assertTrue(np.array_equal(repaired[:80], image[:80]))
        self.assertTrue(np.array_equal(repaired[:, :, 3], image[:, :, 3]))
        self.assertTrue(np.all(mask[80:] > 0) and np.all(mask[:80] == 0))
        # Every rebuilt row is a single colour taken from that real row.
        self.assertLessEqual(float(np.abs(repaired[90, :, :3].astype(int)
                                          - np.median(repaired[90, :, :3], axis=0)).mean()), 1)
        self.assertLess(int(repaired[90, 0, 0]), 220)  # lettering is gone

    def test_lama_receives_a_mask_without_the_rebuilt_strip(self):
        import tempfile
        from pathlib import Path
        from PIL import Image
        from buyer_show.watermark import repair_image
        image = np.full((80, 60, 4), 255, np.uint8)
        image[:, :, :3] = (180, 180, 180)
        regions = [{'x': 0, 'y': 60, 'w': 60, 'h': 20, 'fill': 'row_band'},
                   {'x': 5, 'y': 5, 'w': 15, 'h': 10}]
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'sample.png'
            Image.fromarray(image).save(source)
            with mock.patch('buyer_show.watermark_ai.inpaint_rgb') as backend:
                backend.return_value = (np.zeros((80, 60, 3), np.uint8), {'roi_count': 0})
                repair_image(str(source), regions, output_root=Path(folder) / 'out', method='lama')
            ai_mask = backend.call_args[0][1]
        self.assertTrue(backend.called)
        self.assertTrue(np.all(ai_mask[60:] == 0))
        self.assertTrue(np.all(ai_mask[5:15, 5:20] != 0))

    def test_merge_drops_blocks_inside_a_rebuilt_strip(self):
        decision = {'item_id': '0', 'status': 'auto',
                    'approved_regions': [{'x': 20, 'y': 90, 'w': 40, 'h': 15}],
                    'review_regions': [{'x': 30, 'y': 95, 'w': 12, 'h': 6}],
                    'proposed_regions': [], 'review_reasons': [], 'evidence': {}, 'warnings': []}
        plan = {'approved': [{'x': 0, 'y': 80, 'w': 100, 'h': 40, 'fill': 'row_band',
                              'label': 'strip'}], 'evidence': [], 'policy_version': 'x'}
        merged = merge_light_band_decision(decision, plan, 100, 120)
        self.assertEqual(len(merged['approved_regions']), 1)
        self.assertEqual(merged['approved_regions'][0]['fill'], 'row_band')
        self.assertFalse(merged['review_regions'])


class OverlayHeadlineTests(unittest.TestCase):
    def test_proven_template_approves_dark_headline_on_another_main_photo(self):
        from tests.test_watermark_overlay import LOGO, scan as overlay_scan
        headline = [(50, 111, 573, 69, '边角文字候选：大空间装载更多可能'),
                    (45, 188, 292, 55, '边角文字候选：通勤/出游/逛街,')]
        scans = [overlay_scan(i, LOGO) for i in range(2)] + [overlay_scan(2, headline)]
        plans = __import__('buyer_show.watermark_overlay', fromlist=['plan_overlay_blocks']).plan_overlay_blocks(scans)
        approved = plans['2']['approved']
        headlines = [region for region in approved
                     if region['detection_kind'] == 'pasted_headline']
        self.assertEqual(len(headlines), 1)  # all lines form one masked area
        self.assertNotIn('fill', headlines[0])  # straps run through it: keep LaMa
        self.assertGreater(headlines[0]['h'], 180)  # includes the OCR-missed line
        for region in approved:
            self.assertLessEqual(region['x'] + region['w'], 750)
            self.assertLessEqual(region['y'] + region['h'], 1000)

    def test_headline_is_not_approved_without_a_proven_product_template(self):
        from tests.test_watermark_overlay import scan as overlay_scan
        headline = [(50, 111, 573, 69, '边角文字候选：大空间装载更多可能')]
        plans = __import__('buyer_show.watermark_overlay', fromlist=['plan_overlay_blocks']).plan_overlay_blocks(
            [overlay_scan(0, headline)])
        self.assertFalse(any(region['detection_kind'] == 'pasted_headline'
                             for region in plans['0']['approved']))
