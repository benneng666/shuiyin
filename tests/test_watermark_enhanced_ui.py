"""Manual re-detection reuses only known merchant context, never old masks."""
from types import SimpleNamespace
import hashlib
import json
from pathlib import Path
import queue
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
from PIL import Image

from buyer_show.watermark_ui import (WatermarkWindow, worker_main, _read_preview,
    _merge_extra_candidates, _accept_source_identity, SOURCE_CHANGED)


class EnhancedDetectionTests(unittest.TestCase):
    def test_manual_redetection_uses_previous_product_merchants_but_honors_edit(self):
        item = {'meta': {'width': 100, 'height': 100}, 'merchant_tokens': ['acmebrand'],
                'regions': [{'x': 1, 'y': 1, 'w': 10, 'h': 10}]}
        start = Mock()
        options = {'detection_profile': 'product', 'merchant_terms': '', 'include_corner_text': True}
        window = SimpleNamespace(busy=False, current='source.png', selected_region=None,
                                 _item=lambda: item, _start=start, _detection_options=lambda: dict(options))
        WatermarkWindow._detect(window)
        self.assertEqual(start.call_args.args[0]['merchant_terms'], 'acmebrand')
        self.assertNotIn('regions', start.call_args.args[0])
        options['merchant_terms'] = 'user-store'
        WatermarkWindow._detect(window)
        self.assertEqual(start.call_args.args[0]['merchant_terms'], 'user-store')

    def test_worker_adds_enhanced_review_candidates_without_repair(self):
        base = {'width': 100, 'height': 100, 'source_sha256': 'known', 'scan_complete': True,
                'regions': [], 'warnings': [], 'text_regions': []}
        extra = {'regions': [{'x': 10, 'y': 20, 'w': 30, 'h': 10, 'label': '店名', 'confidence': .97}],
                 'warnings': [], 'scan_complete': True, 'evidence': {'enhanced': True}}
        events = queue.Queue()
        with patch('buyer_show.watermark.detect_image', return_value=base) as detect, \
                patch('buyer_show.watermark_text_extra.detect_extra_text', return_value=extra) as enhanced, \
                patch('buyer_show.watermark.repair_image') as repair:
            worker_main({'action': 'detect', 'source': 'source.png', 'detection_profile': 'product',
                         'merchant_terms': 'acmebrand'}, events, threading.Event())
        self.assertTrue(detect.call_args.kwargs['return_text_regions'])
        self.assertEqual(enhanced.call_args.args[1], 'acmebrand')
        repair.assert_not_called()
        kind, result = events.get_nowait()
        self.assertEqual(kind, 'result')
        self.assertEqual(result['regions'][0]['x'], 10)
        self.assertEqual(result['enhanced_detection'], {'enhanced': True})


class EnhancedSourceBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.png'
        Image.new('RGB', (100, 100), 'blue').save(self.source)
        self.sha = hashlib.sha256(self.source.read_bytes()).hexdigest()

    def test_different_prefix_same_text_overlap_deduplicates_without_expanding_mask(self):
        old = {'x': 10, 'y': 10, 'w': 70, 'h': 15, 'label': '商家关键词候选：ACME BRAND', 'confidence': .98}
        extra = {**old, 'x': 8, 'w': 74, 'label': '已知店名候选：ＡＣＭＥＢＲＡＮＤ', 'confidence': .99}
        regions = [dict(old)]
        self.assertEqual(_merge_extra_candidates(regions, [extra]), 0)
        self.assertEqual(regions, [old])
        # Different words and graphics keep their independent metadata.
        _merge_extra_candidates(regions, [{**extra, 'text': 'OTHER'},
            {**extra, 'detection_kind': 'graphic'}])
        self.assertEqual(len(regions), 3)
        self.assertEqual(regions[-1]['detection_kind'], 'graphic')

    def test_preview_hash_binds_exact_image_bytes_without_changing_meta_contract(self):
        preview = _read_preview(self.source)
        self.assertEqual(preview['source_sha256'], self.sha)
        self.assertEqual(preview['meta'], {'width': 100, 'height': 100})
        Image.new('RGB', (100, 100), 'red').save(self.source)
        self.assertNotEqual(_read_preview(self.source)['source_sha256'], preview['source_sha256'])

    def test_video_preview_also_returns_file_hash_and_rejects_change_during_extract(self):
        source = self.root / 'video.mp4'
        source.write_bytes(b'old video fixture')
        info = {'width': 100, 'height': 100, 'duration': 2, 'has_audio': True}
        with patch('buyer_show.watermark_video.probe_video', return_value=info), \
             patch('buyer_show.watermark_video.extract_frame', return_value=Image.new('RGB',(100,100),'blue')):
            result = _read_preview(source)
        self.assertEqual(result['source_sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertTrue(result['meta']['has_audio'])
        def replace(*_args):
            source.write_bytes(b'replaced video fixture')
            return Image.new('RGB',(100,100),'red')
        with patch('buyer_show.watermark_video.probe_video', return_value=info), \
             patch('buyer_show.watermark_video.extract_frame', side_effect=replace):
            with self.assertRaisesRegex(ValueError, '源文件内容已变化'):
                _read_preview(source)

    def test_same_size_replacement_never_calls_image_or_video_repair(self):
        for source, target in [(self.source, 'buyer_show.watermark.repair_image'),
            (self.root/'video.mp4', 'buyer_show.watermark_video.repair_video')]:
            if source.suffix == '.mp4':
                source.write_bytes(b'old video')
            expected = hashlib.sha256(source.read_bytes()).hexdigest()
            if source.suffix == '.png':
                Image.new('RGB', (100,100), 'red').save(source)
            else:
                source.write_bytes(b'new video')
            events = queue.Queue()
            with patch(target) as repair:
                worker_main({'action':'repair','source':str(source),'regions':[],
                    'expected_source_sha256':expected},events,threading.Event())
            repair.assert_not_called()
            kind, message = events.get_nowait()
            self.assertEqual(kind, 'error')
            self.assertIn('源文件内容已变化',message)

    def test_detection_rechecks_source_after_backend_before_returning_regions(self):
        def detect(*args, **kwargs):
            Image.new('RGB',(100,100),'red').save(self.source)
            return {'width':100,'height':100,'source_sha256':self.sha,'regions':[],'warnings':[]}
        events = queue.Queue()
        with patch('buyer_show.watermark.detect_image',side_effect=detect):
            worker_main({'action':'detect','source':str(self.source),
                'expected_source_sha256':self.sha},events,threading.Event())
        self.assertEqual(events.get_nowait(),('error',SOURCE_CHANGED))

    def test_backend_manifest_must_match_candidate_source_even_after_precheck(self):
        manifest = self.root/'processing.json'
        manifest.write_text(json.dumps({'source_sha256':'different source'}),encoding='utf-8')
        events = queue.Queue()
        with patch('buyer_show.watermark.repair_image',return_value={
                'output':str(self.root/'result.png'),'manifest':str(manifest)}) as repair, \
                patch('buyer_show.watermark_ui._read_preview') as preview:
            worker_main({'action':'repair','source':str(self.source),'regions':[],
                'expected_source_sha256':self.sha},events,threading.Event())
        repair.assert_called_once()
        preview.assert_not_called()
        self.assertEqual(events.get_nowait()[0],'error')

    def test_real_repair_with_bound_hash_returns_valid_manifest_and_preserves_source(self):
        events = queue.Queue()
        worker_main({'action':'repair','source':str(self.source),'regions':[{'x':20,'y':20,'w':10,'h':10}],
            'expected_source_sha256':self.sha,'method':'auto'},events,threading.Event())
        results = [value for kind,value in list(events.queue) if kind=='result']
        self.assertEqual(len(results),1,list(events.queue))
        processing = json.loads(Path(results[0]['manifest']).read_text(encoding='utf-8'))
        self.assertEqual(processing['source_sha256'],self.sha)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(),self.sha)

    def test_preview_replacement_keeps_old_manual_binding_until_clear_and_reload(self):
        old = {'x':5,'y':5,'w':10,'h':10,'_ui_origin':'manual','_ui_source_sha256':self.sha}
        item = {'regions':[old],'regions_source_sha256':self.sha,'preview_source_sha256':self.sha,
                'meta':{'width':100,'height':100},'preview':b'old','result':None}
        replacement = 'b'*64
        self.assertFalse(_accept_source_identity(item,replacement))
        self.assertEqual(item['regions'][0]['_ui_source_sha256'],self.sha)
        self.assertEqual(item['preview_source_sha256'],self.sha)
        window = SimpleNamespace(busy=False,_item=lambda:item,_mark_edited=Mock(),
            _sync_regions=Mock(),_refresh_preview=Mock(),selected_region=0)
        WatermarkWindow._clear_regions(window)
        window._refresh_preview.assert_called_once()
        self.assertEqual(item['regions'],[])
        self.assertNotIn('regions_source_sha256',item)
        self.assertTrue(_accept_source_identity(item,replacement))

    def test_detect_on_replaced_preview_cannot_rebind_old_frames(self):
        item = {'regions':[], 'preview_source_sha256':self.sha}
        self.assertFalse(_accept_source_identity(item,'c'*64,detection=True))
        self.assertTrue(item['source_changed'])
        self.assertEqual(item['preview_source_sha256'],self.sha)

    def test_repair_job_keeps_manual_source_identity_out_of_backend_regions(self):
        item = {'meta':{'width':100,'height':100},'regions':[{'x':5,'y':5,'w':10,'h':10,
            '_ui_origin':'manual','_ui_source_sha256':self.sha}], 'regions_source_sha256':self.sha,
            'preview_source_sha256':self.sha}
        start=Mock()
        window=SimpleNamespace(busy=False,_item=lambda:item,current=str(self.source),selected_region=None,
            _preview_seconds=lambda:0,zoom=Mock(),repair_method=Mock(),_start=start,_warn=Mock())
        window.zoom.get.return_value='适应窗口';window.repair_method.get.return_value='自动（按背景选择）'
        WatermarkWindow._repair(window)
        job=start.call_args.args[0]
        self.assertEqual(job['expected_source_sha256'],self.sha)
        self.assertNotIn('_ui_source_sha256',job['regions'][0])
        item['source_changed']=True;start.reset_mock()
        WatermarkWindow._repair(window)
        start.assert_not_called()


if __name__ == '__main__':
    unittest.main()
