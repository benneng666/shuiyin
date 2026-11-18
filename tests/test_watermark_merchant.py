"""Merchant candidates, bounded long-image OCR, and conservative repair."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
from PIL import Image

from buyer_show import watermark as wm
from tests.test_watermark import box


class MerchantDetectionTests(unittest.TestCase):
    def test_product_profile_detects_store_marks_and_requested_brand_at_center(self):
        rows = [box(340, 350, 180, 30, 'MEXICAN'), box(380, 490, 170, 24, '稻草人官方旗舰店'),
                box(30, 20, 150, 24, '货号 MEXICAN 328'), box(50, 60, 160, 24, '规格 750 x 1000'),
                box(10, 700, 180, 24, '柔软舒适透气'), box(300, 600, 140, 24, 'other brand')]
        self.assertEqual(wm._candidate_regions(rows, 1000, 800), [])
        found = wm._candidate_regions(rows, 1000, 800, merchant_terms='MEXICAN,稻草人', detection_profile='product')
        self.assertEqual(len(found), 2)
        self.assertTrue(all('商家关键词' in r['label'] for r in found))
        self.assertTrue(any(r['x'] > 200 and r['y'] > 200 for r in found))

    def test_merchant_profile_without_keywords_does_not_classify_all_center_text(self):
        rows = [box(350, 350, 180, 24, '云朵箱包专营店'), box(350, 410, 180, 24, 'MEXICAN'),
                box(350, 450, 180, 24, '材质：全棉'), box(350, 500, 180, 24, '设计灵感源于自然')]
        found = wm._candidate_regions(rows, 1000, 800, detection_profile='product')
        self.assertEqual(len(found), 1)
        self.assertIn('专营店', found[0]['label'])

    def test_keyword_matching_normalizes_ocr_spacing_but_not_latin_substrings(self):
        rows = [box(300, 300, 150, 30, 'ＭＥ XICAN'), box(300, 400, 150, 30, 'mexicanized'),
                box(300, 500, 300, 30, 'Designed and manufactured by Mexican')]
        found = wm._candidate_regions(rows, 1000, 800, merchant_terms='mexican；稻草人\nMexican', detection_profile='product')
        self.assertEqual(len(found), 2)

    def test_candidate_accumulation_is_bounded(self):
        candidates = []
        incoming = [dict(x=5,y=5,w=10,h=10,label='shop'+str(i),confidence=.95) for i in range(300)]
        self.assertEqual(wm._merge_candidates(candidates, incoming), 44)
        self.assertEqual(len(candidates), wm.MAX_REGIONS)

    def test_parameter_text_excluded_even_with_extra_corner_and_keyword(self):
        rows = [box(20, 20, 250, 20, 'MEXICAN 材质：牛皮'), box(20, 50, 250, 20, 'MEXICAN 规格表')]
        self.assertEqual(wm._candidate_regions(rows, 1000, 800, include_corner_text=True,
                          merchant_terms='MEXICAN', detection_profile='product'), [])

    def test_options_are_bounded_and_validated_before_loading(self):
        for args in ({'detection_profile':'unknown'}, {'merchant_terms':'X'}, {'merchant_terms':['brand']},
                     {'merchant_terms':','.join('brand'+str(i) for i in range(33))}):
            with self.subTest(args=args), mock.patch.object(wm, '_load_image') as load, self.assertRaises(ValueError):
                wm.detect_image('unused', **args)
            load.assert_not_called()

    def test_long_image_tile_coordinates_and_overlap_deduplicate(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'long.png'
            Image.new('RGB', (750, 4000), 'white').save(path)
            inputs = []
            outputs = [[box(40, 1510, 160, 30, 'MEXICAN')], [box(40, 70, 160, 30, 'MEXICAN')],
                       [box(40, 120, 160, 30, 'MEXICAN')]]
            def engine(pixels):
                inputs.append(pixels.shape)
                return outputs[len(inputs)-1], None
            with mock.patch.object(wm, '_get_ocr', return_value=engine):
                result = wm.detect_image(path, merchant_terms='MEXICAN', detection_profile='product')
            self.assertEqual(inputs, [(1600, 750, 3), (1600, 750, 3), (1120, 750, 3)])
            self.assertEqual(len(result['regions']), 2)
            self.assertEqual([r['y'] for r in result['regions']], [1507, 2997])
            self.assertEqual(result['ocr_tiles'], 3)
            self.assertTrue(result['scan_complete'])
            self.assertTrue(any('实物品牌标牌' in s for s in result['warnings']))

    def test_tile_limit_reports_unscanned_area_and_cancellation_stops_between_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'long.png'; Image.new('RGB', (300, 5000), 'white').save(path)
            engine = mock.Mock(return_value=([], None))
            with mock.patch.object(wm, '_get_ocr', return_value=engine), mock.patch.object(wm, 'MAX_OCR_TILES', 2):
                result = wm.detect_image(path, detection_profile='product')
            self.assertEqual(engine.call_count, 2)
            self.assertFalse(result['scan_complete'])
            self.assertTrue(any('尚未扫描' in s for s in result['warnings']))
            cancelled = False
            def stop_after_one(pixels):
                nonlocal cancelled
                cancelled = True
                return [], None
            with mock.patch.object(wm, '_get_ocr', return_value=stop_after_one), self.assertRaises(asyncio.CancelledError):
                wm.detect_image(path, should_stop=lambda:cancelled)


class AdaptiveRepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.source = self.root/'original.png'
        self.pixels = np.full((100, 150, 4), 255, dtype=np.uint8)
        self.pixels[:, :, :3] = [235, 228, 215]
        self.regions = [{'x':40,'y':40,'w':14,'h':12}]
        self.pixels[40:52,40:54,:3] = [12, 35, 73]

    def tearDown(self):
        self.temp.cleanup()

    def run_repair(self, **options):
        Image.fromarray(self.pixels).save(self.source)
        before = self.source.read_bytes()
        result = wm.repair_image(self.source, self.regions, **options)
        data = np.asarray(Image.open(result['output']).convert('RGBA'))
        mask = np.asarray(Image.open(result['mask']))
        report = json.loads(Path(result['manifest']).read_text(encoding='utf-8'))
        self.assertTrue(np.array_equal(data[mask == 0], self.pixels[mask == 0]))
        self.assertTrue(np.array_equal(data[:,:,3], self.pixels[:,:,3]))
        self.assertEqual(self.source.read_bytes(), before)
        return data, mask, report

    def test_auto_fills_flat_color_and_records_per_region_method(self):
        with mock.patch.object(cv2, 'inpaint', side_effect=AssertionError('flat region does not need inpaint')):
            data, mask, report = self.run_repair()
        self.assertTrue(np.all(data[mask != 0,:3] == [235,228,215]))
        self.assertEqual(report['requested_method'], 'auto')
        self.assertEqual(report['region_methods'][0]['method'], 'flat_color')
        self.assertEqual(report['region_methods'][0]['color_rgb'], [235,228,215])

    def test_auto_excludes_other_approved_watermark_from_boundary_samples(self):
        self.regions.append({'x':54,'y':40,'w':8,'h':12})
        self.pixels[40:52,54:62,:3] = [255,0,0]
        data, mask, report = self.run_repair()
        self.assertEqual([r['method'] for r in report['region_methods']], ['flat_color','flat_color'])
        self.assertTrue(np.all(data[mask != 0,:3] == [235,228,215]))

    def test_complex_boundary_falls_back_to_telea(self):
        rng = np.random.default_rng(7)
        self.pixels[:,:,:3] = rng.integers(0,256,(100,150,3),dtype=np.uint8)
        _, _, report = self.run_repair()
        self.assertEqual(report['region_methods'][0]['method'], 'telea')
        self.assertEqual(report['region_methods'][0]['reason'], 'textured_or_gradient_boundary')

    def test_vertical_column_background_continues_without_diamond_artifacts(self):
        self.pixels = np.full((300, 300, 4), 255, dtype=np.uint8)
        column = np.rint(170 + 22*np.sin(np.arange(300)*.10)).astype(np.uint8)
        self.pixels[:,:,:3] = column[None,:,None]
        self.regions = [{'x':90,'y':90,'w':70,'h':70}]
        expected = self.pixels.copy()
        self.pixels[90:160,90:160,:3] = [10,20,30]
        with mock.patch.object(cv2, 'inpaint', side_effect=AssertionError('consistent columns need no inpaint')):
            data, mask, report = self.run_repair()
        self.assertEqual(report['region_methods'][0]['method'], 'vertical_interpolation')
        self.assertTrue(np.array_equal(data[mask != 0,:3], expected[mask != 0,:3]))
        self.assertGreater(report['region_methods'][0]['profile_correlation'], .99)

    def test_horizontal_bands_use_symmetric_directional_validation(self):
        self.pixels = np.full((300, 300, 4), 255, dtype=np.uint8)
        stripe = np.rint(170 + 22*np.sin(np.arange(300)*.10)).astype(np.uint8)
        self.pixels[:,:,:3] = stripe[:,None,None]
        self.regions = [{'x':90,'y':90,'w':70,'h':70}]
        expected = self.pixels.copy()
        self.pixels[90:160,90:160,:3] = 0
        data, mask, report = self.run_repair()
        self.assertEqual(report['region_methods'][0]['method'], 'horizontal_interpolation')
        self.assertTrue(np.array_equal(data[mask != 0,:3], expected[mask != 0,:3]))

    def test_gentle_horizontal_color_gradient_is_interpolated_not_flattened(self):
        self.pixels = np.full((300,300,4),255,dtype=np.uint8)
        self.pixels[:,:,0] = np.rint(100 + np.arange(300)*.20).astype(np.uint8)[None,:]
        self.pixels[:,:,1] = np.rint(50 + np.arange(300)*.60).astype(np.uint8)[:,None]
        self.pixels[:,:,2] = 180
        self.regions = [{'x':70,'y':90,'w':40,'h':30}]
        expected = self.pixels.copy()
        self.pixels[90:120,70:110,:3] = 0
        data, mask, report = self.run_repair()
        self.assertEqual(report['region_methods'][0]['method'], 'horizontal_interpolation')
        self.assertLessEqual(np.abs(data[mask != 0,:3].astype(int)-expected[mask != 0,:3].astype(int)).max(), 1)

    def test_disagreeing_content_on_top_bottom_rejects_directional_fill(self):
        self.regions = [{'x':40,'y':40,'w':30,'h':25}]
        self.pixels[:,:,:3] = 200
        self.pixels[32:40,40:70,:3] = 150
        self.pixels[65:73,40:70,:3] = 240
        self.pixels[40:65,40:70,:3] = 0
        _, _, report = self.run_repair()
        self.assertEqual(report['region_methods'][0]['method'], 'telea')
        self.assertEqual(report['region_methods'][0]['directional_reason'], 'no_consistent_clean_direction')

    def test_other_mask_and_transparency_cannot_supply_directional_samples(self):
        original = np.full((300,300,4),255,dtype=np.uint8)
        original[:,:,:3] = np.rint(170 + 22*np.sin(np.arange(300)*.10)).astype(np.uint8)[None,:,None]
        region = {'x':90,'y':90,'w':70,'h':70}
        mask = np.zeros((300,300),dtype=np.uint8);mask[90:160,90:160]=255
        self.assertIsNotNone(wm._directional_background(original,mask,region)[0])
        mask[82:90,90:160] = 255
        self.assertIsNone(wm._directional_background(original,mask,region)[0])
        mask[82:90,90:160] = 0
        original[82:90,90:160,3] = 0
        self.assertIsNone(wm._directional_background(original,mask,region)[0])

    def test_large_region_never_uses_directional_interpolation(self):
        original = np.full((100,100,4),255,dtype=np.uint8)
        mask = np.zeros((100,100),dtype=np.uint8)
        plan, stats = wm._directional_background(original,mask,{'x':30,'y':30,'w':30,'h':30})
        self.assertIsNone(plan)
        self.assertEqual(stats['directional_reason'], 'large_or_narrow_region')

    def test_transparent_boundary_and_overlapping_boxes_are_conservative(self):
        self.pixels[:,:,3] = 180
        _, _, report = self.run_repair()
        self.assertEqual(report['region_methods'][0]['method'], 'telea')
        self.assertEqual(report['region_methods'][0]['reason'], 'insufficient_opaque_boundary')
        self.pixels[:,:,3] = 255
        self.regions.append({'x':45,'y':42,'w':14,'h':12})
        _, _, report = self.run_repair()
        self.assertEqual([r['reason'] for r in report['region_methods']], ['overlapping_regions']*2)

    def test_explicit_ns_and_telea_never_switch_to_flat_fill(self):
        for method, flag in [('telea',cv2.INPAINT_TELEA),('ns',cv2.INPAINT_NS)]:
            with self.subTest(method=method), mock.patch.object(cv2, 'inpaint', wraps=cv2.inpaint) as inpaint:
                _, _, report = self.run_repair(method=method)
                self.assertEqual(inpaint.call_args.args[3], flag)
                self.assertEqual(report['region_methods'][0]['method'], method)
        with self.assertRaisesRegex(ValueError, 'auto'):
            wm.repair_image(self.source, self.regions, method='unknown')

    def test_auto_cancellation_before_publication_keeps_source_and_no_success(self):
        Image.fromarray(self.pixels).save(self.source)
        before = self.source.read_bytes(); cancelled = False
        real = wm._flat_boundary
        def boundary(*args):
            nonlocal cancelled
            result = real(*args); cancelled = True; return result
        with mock.patch.object(wm, '_flat_boundary', side_effect=boundary), self.assertRaises(asyncio.CancelledError):
            wm.repair_image(self.source, self.regions, should_stop=lambda:cancelled)
        self.assertEqual(self.source.read_bytes(), before)
        self.assertFalse((self.root/'processed').exists())


if __name__ == '__main__': unittest.main()
