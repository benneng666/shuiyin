import asyncio
import unittest

import cv2
import numpy as np

from buyer_show.watermark_video_precision import fit_precision_mask


class PrecisionMaskTests(unittest.TestCase):
    def crops(self, *, alpha=1., moving=False, marker=True):
        values=[]
        for i, background in enumerate(np.linspace(70,220,24)):
            im=np.full((54,148,3),round(background),np.uint8)
            if marker:
                layer=np.zeros((54,148),np.uint8)
                x=13+(i%5)*3 if moving else 13
                cv2.putText(layer,'SHOP',(x,36),cv2.FONT_HERSHEY_SIMPLEX,.8,255,2,cv2.LINE_AA)
                a=layer.astype(np.float32)/255*alpha
                im=np.rint(im*(1-a[:,:,None])+255*a[:,:,None]).astype(np.uint8)
            values.append(im)
        return values

    def test_accepts_independent_fixed_text_inside_region(self):
        mask,info=fit_precision_mask(self.crops(),(5,5,138,44))
        self.assertTrue(info['accepted'],info)
        self.assertEqual(mask.shape,(44,138))
        self.assertEqual(mask.dtype,np.uint8)
        self.assertGreater((mask==0).sum(),0)
        self.assertGreater(info['fit_samples'],8)

    def test_translucent_contrast_is_fitted(self):
        mask,info=fit_precision_mask(self.crops(alpha=.35),(5,5,138,44))
        self.assertIsNotNone(mask,info)
        self.assertLess(info['mean_detected_alpha'],.5)

    def test_very_faint_overlay_keeps_full_rectangle(self):
        mask,info=fit_precision_mask(self.crops(alpha=.08),(5,5,138,44))
        self.assertIsNone(mask)
        self.assertFalse(info['accepted'])

    def test_changing_gray_without_text_is_not_a_mask(self):
        mask,info=fit_precision_mask(self.crops(marker=False),(5,5,138,44))
        self.assertIsNone(mask)
        self.assertFalse(info['accepted'])

    def test_duplicate_frames_do_not_fake_independent_evidence(self):
        crop=self.crops()[0]
        mask,info=fit_precision_mask([crop]*24,(5,5,138,44))
        self.assertIsNone(mask)
        self.assertEqual(info['independent_samples'],1)

    def test_moving_text_is_rejected(self):
        mask,info=fit_precision_mask(self.crops(moving=True),(5,5,138,44))
        self.assertIsNone(mask,info)

    def test_single_fixed_physical_edge_is_rejected(self):
        crops=self.crops(marker=False)
        for im in crops:im[:,65:82]=20
        mask,info=fit_precision_mask(crops,(5,5,138,44))
        self.assertIsNone(mask,info)

    def test_uniform_changed_rectangle_is_not_sparse_text(self):
        crops=self.crops(marker=False)
        for im in crops:im[5:49,5:143]=30
        mask,info=fit_precision_mask(crops,(5,5,138,44))
        self.assertIsNone(mask,info)

    def test_visible_context_needed_on_all_sides(self):
        mask,info=fit_precision_mask(self.crops(),(0,0,138,44))
        self.assertIsNone(mask)
        self.assertEqual(info['reason'],'insufficient_visible_border')

    def test_cancel_is_not_silently_downgraded_to_fallback(self):
        with self.assertRaises(asyncio.CancelledError):
            fit_precision_mask(self.crops(),(5,5,138,44),should_stop=lambda:True)

    def test_inputs_are_not_modified(self):
        crops=self.crops(); copies=[im.copy() for im in crops]
        fit_precision_mask(crops,(5,5,138,44))
        self.assertTrue(all(np.array_equal(a,b) for a,b in zip(crops,copies)))


if __name__=='__main__':unittest.main()
