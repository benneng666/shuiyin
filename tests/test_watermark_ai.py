"""CPU model transport and mask-preservation contracts; no network/model needed."""
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from buyer_show import watermark as wm
from buyer_show import watermark_ai as ai


class ModelDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'models' / 'model.onnx'
        self.contents = b'test pinned ONNX bytes'
        for patcher in (mock.patch.object(ai, 'model_path', return_value=self.path),
                        mock.patch.object(ai, 'MODEL_SIZE', len(self.contents)),
                        mock.patch.object(ai, 'MODEL_SHA256', hashlib.sha256(self.contents).hexdigest()),
                        mock.patch.object(ai, '_VERIFIED_KEY', None)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_status_is_readonly_and_verification_precedes_reuse(self):
        with mock.patch.object(ai, '_download_stream') as download:
            self.assertFalse(ai.model_status()['installed'])
            self.assertFalse(self.path.parent.exists())
            self.path.parent.mkdir()
            self.path.write_bytes(self.contents)
            self.assertTrue(ai.model_status()['installed'])
            self.assertFalse(ai.model_status()['integrity_verified_in_this_process'])
            self.assertEqual(ai.ensure_model(), self.path)
            self.assertTrue(ai.model_status()['integrity_verified_in_this_process'])
            download.assert_not_called()

    def test_partial_transfer_resumes_and_publishes_only_verified_bytes(self):
        calls, progress = [], []
        def reader(url, offset, messages, aborted):
            calls.append((url, offset))
            if len(calls) == 1:
                messages.put(('start', False))
                messages.put(('data', self.contents[:7]))
                messages.put(('error', 'timeout'))
            else:
                messages.put(('start', True))
                messages.put(('data', self.contents[offset:]))
                messages.put(('done', None))
        with mock.patch.object(ai, '_download_stream', side_effect=reader):
            self.assertEqual(ai.ensure_model(emit=progress.append), self.path)
        self.assertEqual([offset for _, offset in calls], [0, 7])
        self.assertEqual(self.path.read_bytes(), self.contents)
        self.assertEqual(list(self.path.parent.glob('*.part')), [])
        self.assertTrue(any('MB' in message for message in progress))

    def test_server_ignoring_range_truncates_partial(self):
        calls = []
        def reader(url, offset, messages, aborted):
            calls.append(offset)
            messages.put(('start', False))
            messages.put(('data', self.contents[:7] if not offset else self.contents))
            messages.put(('error' if not offset else 'done', None))
        with mock.patch.object(ai, '_download_stream', side_effect=reader):
            ai.ensure_model()
        self.assertEqual(calls, [0, 7])
        self.assertEqual(self.path.read_bytes(), self.contents)

    def test_hash_mismatch_never_replaces_existing_file(self):
        self.path.parent.mkdir()
        self.path.write_bytes(b'existing corrupt cache')
        def reader(url, offset, messages, aborted):
            messages.put(('start', False))
            messages.put(('data', b'x' * len(self.contents)))
            messages.put(('done', None))
        with mock.patch.object(ai, '_download_stream', side_effect=reader), self.assertRaisesRegex(ai.LamaError, '校验失败'):
            ai.ensure_model()
        self.assertEqual(self.path.read_bytes(), b'existing corrupt cache')
        self.assertEqual(list(self.path.parent.glob('*.part')), [])

    def test_stalled_download_is_cancellable_and_cleans_only_its_stage(self):
        stop = threading.Event()
        self.path.parent.mkdir()
        unrelated = self.path.parent / 'unrelated.part'
        unrelated.write_bytes(b'keep')
        def reader(url, offset, messages, aborted):
            messages.put(('start', False))
            stop.set()
            aborted.wait(1)
        started = time.monotonic()
        with mock.patch.object(ai, '_download_stream', side_effect=reader), self.assertRaises(asyncio.CancelledError):
            ai.ensure_model(should_stop=stop.is_set)
        self.assertLess(time.monotonic() - started, 1)
        self.assertFalse(self.path.exists())
        self.assertEqual(list(self.path.parent.iterdir()), [unrelated])


class InferenceTests(unittest.TestCase):
    def test_tiled_full_resolution_keeps_a_one_pixel_mask_without_resizing(self):
        calls = []
        class Session:
            def run(_self, names, feed):
                calls.append(feed)
                self.assertEqual(feed['image'].shape, (1, 3, 512, 512))
                self.assertEqual(feed['mask'].shape, (1, 1, 512, 512))
                self.assertEqual(feed['image'].dtype, np.float32)
                self.assertEqual(feed['mask'].dtype, np.float32)
                self.assertTrue(np.allclose(feed['image'], 128 / 255))
                self.assertEqual(set(np.unique(feed['mask'])), {0, 1})
                self.assertGreater(feed['mask'].sum(), 0)
                return [np.full((1, 3, 512, 512), 172, np.float32)]
        rgb = np.full((1001, 1003, 3), 128, np.uint8)
        mask = np.zeros(rgb.shape[:2], bool)
        mask[501, 501] = True
        result = ai._predict_roi(Session(), rgb, mask, lambda: False)
        self.assertEqual(result.shape, rgb.shape)
        self.assertTrue(np.allclose(result[501, 501], 172, atol=.001))
        # No whole-ROI downscale: unpublished pixels keep their original value.
        self.assertTrue(np.all(result[0, 0] == 128))
        self.assertTrue(calls)

    def test_large_roi_uses_multiple_overlapping_512_tiles_at_native_size(self):
        calls = []
        class Session:
            def run(_self, names, feed):
                calls.append(feed['image'].shape)
                return [np.full((1, 3, 512, 512), 91, np.float32)]
        rgb = np.full((900, 900, 3), 40, np.uint8)
        mask = np.zeros((900, 900), bool)
        mask[20:60, 20:60] = True
        mask[800:840, 800:840] = True
        result = ai._predict_roi(Session(), rgb, mask, lambda: False)
        self.assertEqual(result.shape, rgb.shape)
        self.assertGreater(len(calls), 1)  # more than one tile, no global rescale
        self.assertTrue(all(shape == (1, 3, 512, 512) for shape in calls))
        self.assertTrue(np.all(result[30, 30] == 91))
        self.assertTrue(np.all(result[400, 400] == 40))

    def test_cancel_after_native_call_discards_result_and_nonfinite_is_rejected(self):
        stop = threading.Event()
        session = mock.Mock()
        def run(*_):
            stop.set()
            return [np.zeros((1, 3, 512, 512), np.float32)]
        session.run.side_effect = run
        rgb, mask = np.zeros((30, 40, 3), np.uint8), np.ones((30, 40), bool)
        with self.assertRaises(asyncio.CancelledError):
            ai._predict_roi(session, rgb, mask, stop.is_set)
        session.run.side_effect = None
        session.run.return_value = [np.full((1, 3, 512, 512), np.nan, np.float32)]
        with self.assertRaisesRegex(ai.LamaError, '数值异常'):
            ai._predict_roi(session, rgb, mask, lambda: False)

    def test_adjacent_letters_group_and_far_regions_keep_all_owned_pixels(self):
        mask = np.zeros((800, 1300), bool)
        mask[10:25, 10:20] = True
        mask[12:30, 30:45] = True
        mask[700:710, 1200:1210] = True
        groups = ai._roi_groups(mask)
        self.assertEqual(len(groups), 2)
        coverage = np.zeros(mask.shape, np.uint8)
        for (x, y, w, h), owned in groups:
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x+w, 1300)
            self.assertLessEqual(y+h, 800)
            coverage[y:y+h, x:x+w] += owned
        self.assertTrue(np.array_equal(coverage, mask.astype(np.uint8)))
        with mock.patch.object(ai, 'MAX_ROIS', 1), self.assertRaisesRegex(ai.LamaError, '分散'):
            ai._roi_groups(mask)

    def test_roi_composition_preserves_exterior_even_when_model_changes_everything(self):
        rgb = np.random.default_rng(7).integers(0, 256, (800, 1000, 3), dtype=np.uint8)
        mask = np.zeros((800, 1000), np.uint8)
        mask[0:25, 0:30] = 255
        mask[700:710, 900:920] = 255
        before = rgb.copy()
        def prediction(session, crop, local_mask, should_stop):
            return np.full(crop.shape, 200, np.float32)
        with mock.patch.object(ai, '_get_session', return_value=object()), mock.patch.object(ai, '_predict_roi', side_effect=prediction):
            result, info = ai.inpaint_rgb(rgb, mask)
        self.assertTrue(np.array_equal(rgb, before))
        self.assertTrue(np.array_equal(result[mask == 0], rgb[mask == 0]))
        self.assertTrue(np.all(result[5:20, 5:20] == 200))
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(info['roi_count'], 2)
        self.assertEqual(info['sha256'], ai.MODEL_SHA256)

    def test_empty_mask_does_not_load_and_invalid_mask_rejected(self):
        rgb = np.zeros((10, 20, 3), np.uint8)
        with mock.patch.object(ai, '_get_session') as session:
            result, _ = ai.inpaint_rgb(rgb, np.zeros((10, 20), bool))
            session.assert_not_called()
        self.assertTrue(np.array_equal(result, rgb))
        for mask in (np.zeros((9, 20)), np.full((10, 20), np.nan), np.full((10, 20), 'x')):
            with self.assertRaises(ValueError):
                ai.inpaint_rgb(rgb, mask)


class ImageApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'original.png'
        self.original = np.random.default_rng(5).integers(0, 256, (120, 140, 4), dtype=np.uint8)
        self.original[:20, :, 3] = 0
        Image.fromarray(self.original).save(self.source)
        self.before = self.source.read_bytes()
        self.regions = [{'x': 5, 'y': 10, 'w': 30, 'h': 40}]

    def test_lama_preserves_alpha_exterior_source_and_writes_provenance(self):
        def infer(rgb, mask, **kwargs):
            self.assertTrue(np.all(rgb[:20] == 255))
            return np.full(rgb.shape, 200, np.uint8), {'sha256': ai.MODEL_SHA256, 'roi_count': 1}
        with mock.patch.object(ai, 'inpaint_rgb', side_effect=infer):
            value = wm.repair_image(self.source, self.regions, method='lama')
        result = np.asarray(Image.open(value['output']).convert('RGBA'))
        mask = np.asarray(Image.open(value['mask']))
        self.assertTrue(np.array_equal(result[mask == 0], self.original[mask == 0]))
        self.assertTrue(np.array_equal(result[:, :, 3], self.original[:, :, 3]))
        self.assertTrue(np.all(result[mask != 0, :3] == 200))
        self.assertEqual(self.source.read_bytes(), self.before)
        manifest = json.loads(Path(value['manifest']).read_text(encoding='utf-8'))
        self.assertEqual(manifest['requested_method'], 'lama')
        self.assertEqual(manifest['region_methods'][0]['method'], 'lama_onnx_cpu')
        self.assertEqual(manifest['ai_model']['sha256'], ai.MODEL_SHA256)
        self.assertTrue(any('真实像素' in warning for warning in manifest['warnings']))

    def test_lama_cancel_and_download_failure_publish_no_success(self):
        for error in (asyncio.CancelledError(), ai.LamaError('AI 模型下载未完成')):
            with mock.patch.object(ai, 'inpaint_rgb', side_effect=error):
                with self.assertRaises(type(error) if isinstance(error, asyncio.CancelledError) else RuntimeError):
                    wm.repair_image(self.source, self.regions, method='lama')
            self.assertFalse((self.root / 'processed').exists())
            self.assertEqual(self.source.read_bytes(), self.before)

    def test_lama_bounded_alpha_composition_exactly_matches_original_float_formula(self):
        # Cross a block boundary, including alpha 0/1/127/128/254/255 and RGB
        # rounding boundaries. The opaque case must not allocate float buffers.
        pixels = np.random.default_rng(19).integers(0, 256, (600, 513, 4), dtype=np.uint8)
        pixels[:, :, 3] = np.arange(600 * 513, dtype=np.uint32).reshape(600, 513) % 256
        for opaque in (False, True):
            with self.subTest(opaque=opaque):
                original = pixels.copy()
                if opaque:
                    original[:, :, 3] = 255
                Image.fromarray(original).save(self.source)
                before = self.source.read_bytes()
                alpha = original[:, :, 3:4].astype(np.float32) / 255
                expected = np.rint(original[:, :, :3] * alpha + 255 * (1-alpha)).astype(np.uint8)
                def infer(rgb, mask, **kwargs):
                    self.assertTrue(np.array_equal(rgb, expected))
                    self.assertEqual(rgb.dtype, np.uint8)
                    return np.zeros_like(rgb), {'roi_count': 1}
                with mock.patch.object(ai, 'inpaint_rgb', side_effect=infer), \
                     mock.patch.object(np, 'rint', wraps=np.rint) as rounded:
                    value = wm.repair_image(self.source, self.regions, method='lama')
                if opaque:
                    rounded.assert_not_called()
                else:
                    self.assertGreater(len(rounded.call_args_list), 1)
                    self.assertTrue(all(call.args[0].size < expected.size for call in rounded.call_args_list))
                result = np.asarray(Image.open(value['output']).convert('RGBA'))
                mask = np.asarray(Image.open(value['mask']))
                self.assertTrue(np.array_equal(result[:, :, 3], original[:, :, 3]))
                self.assertTrue(np.array_equal(result[mask == 0], original[mask == 0]))
                self.assertEqual(self.source.read_bytes(), before)


class ImageEdgeFeatherTests(unittest.TestCase):
    def test_mask_edges_never_mix_watermarked_original_back_in(self):
        rgb = np.full((80, 100, 3), 200, np.uint8)
        mask = np.zeros((80, 100), np.uint8)
        mask[20:60, 90:100] = 1
        mask[0:12, 0:20] = 1
        with mock.patch.object(ai, '_get_session', return_value=object()), \
                mock.patch.object(ai, '_predict_roi', side_effect=lambda session, pixels, local, stop: np.zeros_like(pixels)):
            result, info = ai.inpaint_rgb(rgb, mask)
        # Interior, right edge and top-left corner all use 100% prediction;
        # the old 2px feather mixed the original watermark strokes back in.
        self.assertTrue(np.all(result[35, 90] == 0))
        self.assertTrue(np.all(result[30:50, 99] == 0))
        self.assertTrue(np.all(result[0, 0] == 0))
        self.assertTrue(np.array_equal(result[mask == 0], rgb[mask == 0]))
        self.assertEqual(info['inside_mask_feather_pixels'], 0)
        self.assertEqual(info['mask_edge_composition'], 'prediction_only_no_watermark_mixed_back')
        self.assertTrue(info['tiled_full_resolution'])


if __name__ == '__main__':
    unittest.main()
