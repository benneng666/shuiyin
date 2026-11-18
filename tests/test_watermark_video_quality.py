import asyncio
import unittest
from unittest.mock import patch

import numpy as np

from buyer_show.watermark_video_quality import QualityFrameProcessor, PrecisionMaskUnavailable, _crossing_structure


class QualityFrameTests(unittest.TestCase):
    def frame(self):
        a = np.zeros((128, 160, 3), np.uint8)
        a[:, 70:] = (170, 90, 40)
        a[::6, :] = 220
        return a

    def box(self, **kw):
        return dict(x=60, y=50, w=24, h=18, start=0, end=1, **kw)

    def test_simple_flat_frame_preserved_without_loading_ai(self):
        frame = np.full((128,160,3), 175, np.uint8)
        p = QualityFrameProcessor(160,128,[self.box()])
        with patch.object(p, '_session', side_effect=AssertionError('unused')):
            self.assertTrue(np.array_equal(p(frame, .2), frame))
        self.assertEqual(p.diagnostics['keyframes'], 0)

    def test_edge_watermark_is_not_reused_as_its_own_background(self):
        for x, y, w, h in [(0, 0, 24, 18), (136, 110, 24, 18), (0, 60, 24, 18),
                            (60, 0, 24, 18), (136, 60, 24, 18), (60, 110, 24, 18),
                            (0, 0, 160, 8)]:
            with self.subTest(box=(x, y, w, h)):
                frame = np.full((128, 160, 3), 175, np.uint8)
                frame[y:y+h, x:x+w] = 255
                p = QualityFrameProcessor(160, 128, [dict(x=x, y=y, w=w, h=h, start=0, end=1)])
                with patch.object(p, '_session', side_effect=AssertionError('flat background')):
                    repaired = p(frame, .2)
                self.assertTrue(np.all(repaired == 175))
                self.assertTrue(np.all(frame[y:y+h, x:x+w] == 255))

    def test_precision_holes_and_exterior_are_byte_exact(self):
        frame = self.frame(); original = frame.copy()
        mask = np.zeros((18,24),np.uint8);mask[3:15,5:10]=1;mask[8:10,7:9]=0
        p = QualityFrameProcessor(160,128,[self.box()],pixel_masks=[mask],use_ai=False)
        with patch.object(p,'_session',side_effect=AssertionError('unused')):
            result=p(frame,.5)
        full=np.zeros(frame.shape[:2],bool);full[50:68,60:84]=mask!=0
        self.assertTrue(np.array_equal(result[~full],frame[~full]))
        self.assertTrue(np.array_equal(frame,original))
        self.assertFalse(np.array_equal(result[full],frame[full]))

    def test_half_open_intervals_and_many_disjoint_time_segments(self):
        boxes=[dict(x=60,y=50,w=24,h=18,start=i,end=i+1) for i in range(128)]
        p=QualityFrameProcessor(160,128,boxes,use_ai=False)
        self.assertTrue(np.array_equal(p(self.frame(),128),self.frame()))
        self.assertEqual(len(p.states),0)

    def test_three_simultaneous_regions_rejected_before_model(self):
        with self.assertRaises(ValueError):
            QualityFrameProcessor(160,128,[self.box()]*3)

    def test_invalid_coordinates_mask_and_time_fail_closed(self):
        bad=dict(self.box(),x=60.5)
        with self.assertRaises(ValueError):QualityFrameProcessor(160,128,[bad])
        with self.assertRaises(ValueError):QualityFrameProcessor(160,128,[self.box()],pixel_masks=[np.ones((17,24))])
        p=QualityFrameProcessor(160,128,[self.box()],use_ai=False)
        p(self.frame(),.5)
        with self.assertRaises(ValueError):p(self.frame(),.4)
        with self.assertRaises(ValueError):p(self.frame(),float('nan'))

    def test_cancel_before_model_loading(self):
        p=QualityFrameProcessor(160,128,[self.box()],should_stop=lambda:True)
        with self.assertRaises(asyncio.CancelledError):p(self.frame(),0)

    def test_neural_budget_and_exterior(self):
        p=QualityFrameProcessor(160,128,[dict(self.box(),end=3)],max_keyframes=1)
        def predict(rgb,mask):
            p.diagnostics['keyframes']+=1
            return np.full_like(rgb,123)
        full=np.zeros((128,160),bool);full[50:68,60:84]=1
        with patch.object(p,'_predict',side_effect=predict):
            first=p(self.frame(),0)
            last=p(self.frame(),2)
        self.assertTrue(np.array_equal(first[~full],self.frame()[~full]))
        self.assertTrue(np.array_equal(last[~full],self.frame()[~full]))
        self.assertEqual(p.diagnostics['keyframes'],1)
        self.assertEqual(p.diagnostics['budget_fallback_frames'],1)

    def test_automatic_prepare_refuses_unreliable_fine_mask(self):
        p=QualityFrameProcessor(160,128,[self.box()],use_ai=False)
        report=dict(masks=[None],evidence=[{'accepted':False}],source_sha256='test-hash')
        with patch('buyer_show.watermark_video_precision.detect_precision_masks',return_value=report):
            with self.assertRaises(PrecisionMaskUnavailable):p.prepare('unused',{'duration':1},[0,.1])
        self.assertEqual(p.diagnostics['precision_masks'],0)

    def test_prepare_preserves_source_change_and_cancellation_errors(self):
        for error in (ValueError('source changed'),asyncio.CancelledError()):
            p=QualityFrameProcessor(160,128,[self.box()],use_ai=False)
            with patch('buyer_show.watermark_video_precision.detect_precision_masks',side_effect=error):
                with self.assertRaises(type(error)):p.prepare('unused',{'duration':1},[0,.1])

    def test_prepare_checks_mask_geometry_before_processing(self):
        p=QualityFrameProcessor(160,128,[self.box()],use_ai=False)
        report=dict(masks=[np.ones((40,40),np.uint8)],evidence=[],source_sha256='test-hash')
        with patch('buyer_show.watermark_video_precision.detect_precision_masks',return_value=report):
            with self.assertRaises(ValueError):p.prepare('unused',{'duration':1},[0,.1])

    def test_neural_prepare_uses_full_approved_box_not_refined_glyphs(self):
        mask=np.ones((18,24),np.uint8);mask[8:10,8:10]=0
        p=QualityFrameProcessor(160,128,[self.box()],pixel_masks=[mask])
        with patch('buyer_show.watermark_video_precision.detect_precision_masks',side_effect=AssertionError('unneeded')):
            p.prepare('unused',{'duration':1},[0,.1])
        self.assertIsNone(p.pixel_masks[0])
        self.assertEqual(p.diagnostics['precision_masks'],0)

    def test_nearby_object_does_not_trigger_inpainting_of_flat_selection(self):
        source=np.full((128,160,3),160,np.uint8)
        source[25:43,40:100]=20
        source[54:64,64:80]=255
        p=QualityFrameProcessor(160,128,[self.box()])
        with patch.object(p,'_session',side_effect=AssertionError('nearby object is outside')):
            result=p(source,0)
        self.assertEqual(int(result[50:68,60:84].min()),160)
        self.assertEqual(int(result[50:68,60:84].max()),160)

    def test_disappeared_structure_clears_cached_neural_object(self):
        p=QualityFrameProcessor(160,128,[self.box()])
        def predict(rgb,mask):
            p.diagnostics['keyframes']+=1
            return np.full_like(rgb,37)
        with patch.object(p,'_predict',side_effect=predict):
            p(self.frame(),0)
            self.assertTrue(p.states)
            flat=np.full((128,160,3),160,np.uint8);flat[54:64,64:80]=255
            output=p(flat,.033)
        self.assertFalse(p.states)
        self.assertTrue(np.all(output[50:68,60:84]==160))

    def test_crossing_edges_require_two_real_sides(self):
        flat=np.full((128,160,3),180,np.uint8)
        self.assertFalse(_crossing_structure(flat,(60,50,24,18))[0])
        flat[:,70:]=30
        self.assertTrue(_crossing_structure(flat,(60,50,24,18))[0])


if __name__=='__main__':unittest.main()
