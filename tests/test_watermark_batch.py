import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from buyer_show.watermark_batch import run_batch, select_items, _smart_method, _avoid_uncertain_overlap, merge_graphics_decision


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def item(self, name='image.png', color='white', sources=None, product='taobao:1'):
        path = self.root / name
        Image.new('RGB', (120, 160), color).save(path)
        return {'path': str(path), 'media_type': 'image', 'product_key': product,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'source_types': sources or ['product_main']}

    @staticmethod
    def empty_scan(path, **kwargs):
        return {'width': 120, 'height': 160, 'scan_complete': True,
                'source_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                'text_regions': [], 'regions': []}

    def run_empty(self, items, **kwargs):
        with patch('buyer_show.watermark_batch.scan_image', side_effect=self.empty_scan):
            return run_batch(items, output_root=self.root / 'output', **kwargs)

    def test_default_scope_product_only_and_opt_ins(self):
        main = self.item()
        review = self.item('review.png', color='blue', sources=['review_initial'])
        local = self.item('local.png', color='red', sources=['unclassified'])
        video = dict(main, path=str(self.root / 'product.mp4'), media_type='video', sha256='v', source_types=['product_video'])
        self.assertEqual(select_items([main, review, local, video]), [main | {'categories': []}, video | {'categories': []}])
        self.assertEqual(len(select_items([main, review, local], include_reviews=True, include_local=True)), 3)

    def test_duplicate_hash_is_not_merged_across_product_boundaries(self):
        main = self.item()
        details = dict(main, source_types=['product_detail'])
        other = dict(main, product_key='taobao:2')
        values = select_items([main, details, other])
        self.assertEqual(len(values), 2)
        self.assertEqual(values[0]['source_types'], ['product_detail', 'product_main'])

    def test_review_only_scope_does_not_export_mixed_item_into_product_category(self):
        item = self.item(sources=['product_main', 'review_initial'])
        result = self.run_empty([item], include_products=False, include_reviews=True)
        self.assertFalse(result['scope']['product'])
        self.assertEqual([value['category'] for value in result['items'][0]['exports']], ['买家秀/初评/图片'])

    def test_no_detection_preserves_bytes_and_classification_and_separate_runs(self):
        item = self.item(sources=['product_main', 'product_detail'])
        original = Path(item['path']).read_bytes()
        result = self.run_empty([item])
        row = result['items'][0]
        self.assertEqual(row['status'], 'unchanged')
        self.assertEqual(result['counts']['exported_files'], 2)
        self.assertEqual({v['category'] for v in row['exports']}, {'商品主图/3x4', '详情图'})
        for value in row['exports']:
            self.assertEqual(Path(value['path']).read_bytes(), original)
        self.assertEqual(Path(item['path']).read_bytes(), original)
        self.assertTrue(Path(result['html']).is_file())
        second = self.run_empty([item])
        self.assertNotEqual(second['batch_dir'], result['batch_dir'])

    def test_result_page_shows_video_method_and_fallback_warning_escaped(self):
        from buyer_show.watermark_batch import _html_report
        item = {'path': str(self.root/'sample.mp4'), 'media_type': 'video', 'product_key': 'test:video',
                'status': 'processed', 'reason': '完成', 'repair_method': 'delogo',
                'warnings': ['精修预算已用尽 <check>']}
        _html_report({'batch_dir': str(self.root), 'items': [item], 'counts': {'processed': 1}})
        page = (self.root/'打开结果.html').read_text(encoding='utf-8')
        self.assertIn('实际方式：快速视频修补', page)
        self.assertIn('精修预算已用尽 &lt;check&gt;', page)
        self.assertNotIn('<check>', page)

    def test_untrusted_category_and_product_names_cannot_write_outside_batch(self):
        item = self.item(product='../../CON:bad') | {'categories': ['../../escaped/foo']}
        result = self.run_empty([item])
        for exported in result['items'][0]['exports']:
            self.assertTrue(Path(exported['path']).resolve().is_relative_to(Path(result['batch_dir'])))

    def test_uncertain_is_original_not_accidentally_repaired(self):
        item = self.item()
        decision = {'item_id': '0', 'status': 'needs_review', 'approved_regions': [],
                    'proposed_regions': [{'x': 2, 'y': 2, 'w': 10, 'h': 10}],
                    'review_reasons': ['uncertain']}
        with patch('buyer_show.watermark_batch.analyze_images', return_value=[decision]), \
                patch('buyer_show.watermark_batch.repair_image') as repair:
            result = self.run_empty([item])
        repair.assert_not_called()
        self.assertEqual(result['items'][0]['status'], 'needs_review')
        self.assertEqual(Path(result['items'][0]['output']).read_bytes(), Path(item['path']).read_bytes())

    def test_partial_approval_uses_only_approved_mask_and_preserves_exterior(self):
        item = self.item(color='gray')
        from PIL import ImageDraw
        with Image.open(item['path']) as image:
            ImageDraw.Draw(image).rectangle((12, 12, 18, 18), fill='black')
            image.save(item['path'])
        item['sha256'] = hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()
        approved = {'x': 10, 'y': 10, 'w': 12, 'h': 12}
        unsure = {'x': 70, 'y': 80, 'w': 15, 'h': 15}
        decision = {'item_id': '0', 'status': 'auto', 'approved_regions': [approved],
                    'proposed_regions': [approved, unsure], 'review_regions': [unsure]}
        with patch('buyer_show.watermark_batch.analyze_images', return_value=[decision]):
            result = self.run_empty([item], image_method='auto')
        row = result['items'][0]
        self.assertTrue(row['partially_processed'])
        self.assertEqual(row['status'], 'needs_review')
        processing = json.loads(Path(row['manifest']).read_text(encoding='utf-8'))
        self.assertEqual(processing['regions'], [approved])
        self.assertEqual(processing['source_sha256'], item['sha256'])
        import numpy as np
        before = np.asarray(Image.open(item['path']).convert('RGBA'))
        after = np.asarray(Image.open(row['output']).convert('RGBA'))
        mask = np.ones(before.shape[:2], bool)
        mask[10:22, 10:22] = False
        self.assertTrue(np.array_equal(before[mask], after[mask]))
        self.assertFalse(np.array_equal(before, after))

    def test_one_failed_detection_does_not_abort_other_files(self):
        first, second = self.item(), self.item('second.png', color='blue')
        def scan(path, **kwargs):
            if path == first['path']:
                raise RuntimeError('ocr unavailable')
            return self.empty_scan(path)
        with patch('buyer_show.watermark_batch.scan_image', side_effect=scan):
            result = run_batch([first, second], output_root=self.root / 'output')
        self.assertEqual(result['counts']['failed'], 1)
        self.assertEqual(result['counts']['unchanged'], 1)
        self.assertEqual(result['counts']['exported_files'], 2)
        self.assertEqual(result['status'], 'completed_with_issues')

    def test_cancel_returns_completed_outputs_and_report(self):
        items = [self.item(), self.item('second.png', color='blue')]
        stopped = False
        def event(value):
            nonlocal stopped
            if value['status'] == 'unchanged':
                stopped = True
        result = self.run_empty(items, should_stop=lambda: stopped, on_event=event)
        self.assertEqual(result['status'], 'cancelled')
        self.assertEqual(result['counts']['unchanged'], 1)
        self.assertEqual(result['counts']['cancelled'], 1)
        self.assertTrue(Path(result['items'][0]['output']).is_file())
        self.assertEqual(json.loads(Path(result['manifest']).read_text(encoding='utf-8'))['status'], 'cancelled')

    def test_cancel_during_scan_is_not_failure(self):
        with patch('buyer_show.watermark_batch.scan_image', side_effect=asyncio.CancelledError):
            result = run_batch([self.item()], output_root=self.root / 'output')
        self.assertEqual(result['status'], 'cancelled')
        self.assertEqual(result['counts']['cancelled'], 1)

    def test_modified_original_is_not_processed_or_published(self):
        item = self.item()
        Path(item['path']).write_bytes(b'changed by user')
        with patch('buyer_show.watermark_batch.scan_image') as scan:
            result = run_batch([item], output_root=self.root / 'output')
        scan.assert_not_called()
        self.assertEqual(result['counts']['failed'], 1)
        self.assertEqual(result['counts']['exported_files'], 0)

    def test_video_empty_plan_uses_video_route_preserves_exact_original(self):
        path = self.root / 'movie.mp4'
        path.write_bytes(b'video-fixture')
        item = {'path': str(path), 'media_type': 'video', 'source_types': ['product_video'], 'product_key': 'taobao:1'}
        with patch('buyer_show.watermark_batch.plan_video', return_value={'status': 'not_detected'}), \
                patch('buyer_show.watermark_batch.repair_video') as repair:
            result = run_batch([item], output_root=self.root / 'output')
        repair.assert_not_called()
        self.assertEqual(result['counts']['unchanged'], 1)
        self.assertEqual(result['items'][0]['exports'][0]['category'], '商品视频')
        self.assertEqual(Path(result['items'][0]['output']).read_bytes(), path.read_bytes())

    def test_smart_selects_flat_fast_backend_and_texture_ai(self):
        item = self.item()
        regions = [{'x': 20, 'y': 20, 'w': 25, 'h': 20}]
        self.assertEqual(_smart_method(item['path'], regions, lambda: False), 'auto')
        import numpy as np
        image = np.random.default_rng(1).integers(0, 256, (160, 120, 3), dtype=np.uint8)
        Image.fromarray(image).save(item['path'])
        self.assertEqual(_smart_method(item['path'], regions, lambda: False), 'lama')

    def test_smart_sends_gradient_background_to_ai_not_interpolation(self):
        """A smooth gradient used to keep the fast path via directional fill."""
        import numpy as np
        from buyer_show.watermark import _load_image, _flat_boundary, _directional_background
        item = self.item()
        gradient = np.zeros((160, 120, 4), dtype=np.uint8)
        gradient[:, :, 3] = 255
        gradient[:, :, :3] = (np.arange(120, dtype=np.uint8) * 2)[None, :, None]
        Image.fromarray(gradient).save(item['path'])
        regions = [{'x': 20, 'y': 20, 'w': 25, 'h': 20}]
        image, _, _ = _load_image(item['path'])
        try:
            original = np.asarray(image)
            mask = np.zeros((160, 120), dtype=np.uint8)
            mask[20:40, 20:45] = 255
            self.assertIsNone(_flat_boundary(original, mask, regions[0])[0])
            self.assertIsNotNone(_directional_background(original, mask, regions[0])[0])
        finally:
            image.close()
        self.assertEqual(_smart_method(item['path'], regions, lambda: False), 'lama')

    def test_smart_sends_large_flat_box_to_ai(self):
        item = self.item()
        # 1200 px = 6.25% of the image: flat fill would succeed, but the box
        # is too large to trust an exact-colour extrapolation.
        regions = [{'x': 20, 'y': 20, 'w': 40, 'h': 30}]
        self.assertEqual(_smart_method(item['path'], regions, lambda: False), 'lama')

    def test_partial_approval_never_erases_overlapping_uncertain_region(self):
        approved = {'x': 1, 'y': 1, 'w': 20, 'h': 10, 'start': 0, 'end': 2}
        uncertain = {'x': 2, 'y': 2, 'w': 20, 'h': 10, 'start': 1, 'end': 3}
        self.assertEqual(_avoid_uncertain_overlap([approved], [uncertain], 'image'), ([], [approved]))
        self.assertEqual(_avoid_uncertain_overlap([approved], [uncertain], 'video'), ([], [approved]))
        uncertain['start'] = 2
        self.assertEqual(_avoid_uncertain_overlap([approved], [uncertain], 'video'), ([approved], []))

    def test_policy_failure_is_not_written_as_completed_or_cancelled(self):
        with patch('buyer_show.watermark_batch.analyze_images', side_effect=RuntimeError('policy failure')):
            result = self.run_empty([self.item()])
        self.assertEqual(result['status'], 'completed_with_issues')
        self.assertEqual(result['counts']['failed'], 1)
        self.assertEqual(result['counts']['cancelled'], 0)

    def test_cancellation_after_repair_commit_keeps_result_path(self):
        item = self.item()
        decision = {'item_id': '0', 'status': 'auto', 'approved_regions': [{'x': 10, 'y': 10, 'w': 12, 'h': 12}]}
        ticks, stopped = 0, False
        def event(value):
            nonlocal ticks, stopped
            if value['stage'] == 'repair' and value['status'] == 'processing':
                ticks += 1
                if ticks == 2:
                    stopped = True
        with patch('buyer_show.watermark_batch.analyze_images', return_value=[decision]):
            result = self.run_empty([item], image_method='auto', on_event=event, should_stop=lambda: stopped)
        self.assertEqual(result['status'], 'cancelled')
        self.assertTrue(result['items'][0]['repair_committed'])
        self.assertTrue(Path(result['items'][0]['output']).is_file())
        self.assertTrue(Path(result['items'][0]['manifest']).is_file())

    def test_graphic_supplement_preserves_uncertainty_and_caps_union_area(self):
        anchor = {'x': 10, 'y': 10, 'w': 30, 'h': 20}
        small = {'x': 50, 'y': 10, 'w': 10, 'h': 10}
        large = {'x': 0, 'y': 40, 'w': 90, 'h': 40}
        uncertain = {'x': 5, 'y': 85, 'w': 10, 'h': 10}
        decision = {'status': 'auto', 'approved_regions': [anchor], 'review_regions': [uncertain],
                    'review_reasons': ['unknown text']}
        extra = {'approved_regions': [anchor, small, large], 'evidence': {'kind': 'repeated_shape'}}
        result = merge_graphics_decision(decision, extra, 100, 100)
        self.assertEqual(result['approved_regions'], [anchor, small])
        self.assertEqual(result['review_regions'], [uncertain, large])
        self.assertEqual(decision['approved_regions'], [anchor])
        self.assertIn('unknown text', result['review_reasons'])
        self.assertEqual(result['evidence']['graphics']['kind'], 'repeated_shape')


if __name__ == '__main__':
    unittest.main()
