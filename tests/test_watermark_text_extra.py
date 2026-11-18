"""Known-store-only second pass: body text and physical labels stay intact."""
import asyncio
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from buyer_show.watermark_text_extra import detect_extra_text,augment_text,_pad


def text_row(text,x=20,y=20,w=200,h=25,confidence=.99):
    return dict(x=x,y=y,w=w,h=h,label='OCR文字：'+text,confidence=confidence)


def decision(index,tokens=()):
    return {'item_id':str(index),'status':'not_detected','approved_regions':[],
            'proposed_regions':[],'review_regions':[],'review_reasons':[],
            'evidence':{'merchant_tokens':list(tokens),'regions':[]}}


class ExtraTextDetectorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'original.png'
        Image.new('RGB',(600,500),(230,230,230)).save(self.path)
        self.base={'width':600,'height':500,'scan_complete':True,'text_regions':[]}

    def tearDown(self): self.temp.cleanup()

    def test_no_known_store_never_starts_ocr_or_invents_a_merchant(self):
        with patch('buyer_show.watermark_text_extra._ocr') as ocr:
            result=detect_extra_text('unused',merchant_terms='')
        self.assertEqual(result['regions'],[]);ocr.assert_not_called()

    def test_text_height_padding_reaches_edge_and_keeps_large_heading_budget(self):
        edge=_pad({'x':8,'y':821,'w':264,'h':49},800,1067)
        self.assertEqual(edge,{'x':0,'y':813,'w':280,'h':65})
        small=_pad({'x':20,'y':20,'w':100,'h':10},600,500)
        self.assertEqual(small['x'],17)
        title=_pad({'x':49,'y':114,'w':643,'h':91},750,750)
        self.assertLessEqual(title['w']*title['h'],750*750*.12)

    def test_plain_known_store_line_is_returned_without_expensive_extra_ocr(self):
        self.base['text_regions']=[text_row('大橘猫BANGGIRL',x=170,y=400,w=260,h=30),
                                  text_row('规格 20cm'),text_row('材质 牛皮',y=70),text_row('重量 0.3kg',y=120)]
        before=self.path.read_bytes()
        with patch('buyer_show.watermark_text_extra._ocr') as ocr:
            result=detect_extra_text(self.path,'大橘猫BANGGIRL',base_detection=self.base)
        ocr.assert_not_called();self.assertEqual(len(result['regions']),1)
        self.assertEqual(result['regions'][0]['kind'],'known_brand_text')
        self.assertEqual(result['evidence']['product_copy_rows'],3)
        self.assertEqual(self.path.read_bytes(),before)

    def test_embedded_store_does_not_make_entire_parameter_line_erasable(self):
        self.base['text_regions']=[text_row('材质 牛皮 大橘猫BANGGIRL',w=400)]
        result=detect_extra_text(self.path,'大橘猫BANGGIRL',base_detection=self.base)
        self.assertEqual(result['regions'][0]['kind'],'embedded_brand')
        self.assertFalse(result['regions'][0]['brand_only_line'])

    def test_plus_phone_copy_and_physical_other_brand_do_not_trigger_enhancement(self):
        self.base['text_regions']=[text_row('小包在左下会有和PLUS手机对比图',confidence=.7),
                                  text_row('BOLUN',x=200,y=220,w=70,h=20,confidence=.7)]
        with patch('buyer_show.watermark_text_extra._ocr') as ocr:
            result=detect_extra_text(self.path,'大橘猫BANGGIRL',base_detection=self.base)
        self.assertEqual(result['regions'],[]);ocr.assert_not_called()

    def test_faint_weak_full_line_can_be_confirmed_by_known_chinese_store_alias(self):
        self.base['text_regions']=[text_row('大橘猫B#GCIRL',x=10,y=300,w=250,h=40,confidence=.60)]
        # Local OCR identifies the Chinese store name, while the original weak
        # full-line box retains the full transparent signature's footprint.
        ocr_row=[[[20,30],[230,30],[230,100],[20,100]],'大橘猫',.99]
        with patch('buyer_show.watermark_text_extra._ocr',return_value=[ocr_row]):
            result=detect_extra_text(self.path,'大橘猫BANGGIRL',base_detection=self.base)
        self.assertEqual(len(result['regions']),1)
        region=result['regions'][0]
        self.assertEqual(region['kind'],'enhanced_overlay')
        self.assertEqual(region['matched_term'],'大橘猫')
        self.assertTrue(region['full_line_supported'])
        self.assertGreaterEqual(region['w'],250)
        self.assertLessEqual(result['evidence']['ocr_calls'],4)

    def test_unconfirmed_faint_line_is_reviewable_not_a_no_watermark_claim(self):
        self.base['text_regions']=[text_row('大橘猫B#GCIRL',x=10,y=300,w=250,h=40,confidence=.60)]
        with patch('buyer_show.watermark_text_extra._ocr',return_value=[]):
            result=detect_extra_text(self.path,'大橘猫BANGGIRL',base_detection=self.base)
        self.assertTrue(any(r['kind']=='uncertain_brand' for r in result['regions']))

    def test_cancellation_is_honored_before_loading(self):
        with self.assertRaises(asyncio.CancelledError):
            detect_extra_text('unused','BANGGIRL',should_stop=lambda:True)

    def test_exhausted_ocr_budget_reports_unchecked_regions(self):
        self.base['text_regions']=[text_row('大橘猫B#GCIRL',x=10,y=y,w=250,h=40,confidence=.60)
                                  for y in (60,200,340)]
        with patch('buyer_show.watermark_text_extra._ocr',return_value=[]) as ocr:
            result=detect_extra_text(self.path,'大橘猫BANGGIRL',base_detection=self.base)
        self.assertEqual(ocr.call_count,4)
        self.assertFalse(result['scan_complete'])
        self.assertTrue(any('上限' in warning for warning in result['warnings']))
        self.assertEqual(result['evidence']['scanned_rois'],2)
        self.assertEqual(result['evidence']['skipped_rois'],1)

    def test_partial_chinese_ocr_uses_product_learned_full_line_footprint(self):
        row=[[[20,60],[230,60],[230,130],[20,130]],'大橘猫',.99]
        hint={'x':10,'y':300,'w':250,'h':40}
        with patch('buyer_show.watermark_text_extra._ocr',return_value=[row]):
            result=detect_extra_text(self.path,'大橘猫BANGGIRL',base_detection=self.base,roi_hints=[hint])
        region=result['regions'][0]
        self.assertEqual(region['kind'],'enhanced_overlay')
        self.assertEqual(region['footprint_basis'],'same_product_repeated_full_line_layout')
        self.assertGreaterEqual(region['w'],250)
        self.assertLess(region['h'],80)

    def test_failed_layout_probe_with_two_known_stamps_stays_reviewable(self):
        self.base['text_regions']=[text_row('大橘猫BANGGIRL',x=10,y=20,w=250,h=30),
                                  text_row('大橘猫BANGGIRL',x=300,y=420,w=250,h=30)]
        with patch('buyer_show.watermark_text_extra._ocr',return_value=[]):
            result=detect_extra_text(self.path,'大橘猫BANGGIRL',base_detection=self.base,
                                     roi_hints=[{'x':10,'y':300,'w':250,'h':40}])
        proposals=[r for r in result['regions'] if r['kind']=='uncertain_brand']
        self.assertEqual(len(proposals),1)
        self.assertEqual(proposals[0]['confidence'],0)
        self.assertIn('未确认',proposals[0]['label'])


class ExtraTextBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.scans=[]
        for i in range(3):
            path=self.root/f'{i}.png';Image.new('RGB',(600,500),(80+i*35,)*3).save(path)
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            self.scans.append({'item_id':str(i),'product_key':'product-a','source_types':['product_detail'],
                'path':str(path),'sha256':sha,'detection':{'width':600,'height':500,'scan_complete':True,
                'source_sha256':sha,'text_regions':[]}})
        self.decisions=[decision(0,['大橘猫banggirl','banggirl']),decision(1),decision(2)]

    def tearDown(self): self.temp.cleanup()

    @staticmethod
    def overlay(path,*args,**kwargs):
        i=int(Path(path).stem)
        return {'regions':[dict(x=10,y=300,w=250,h=40,label='浅色店名：大橘猫',text='大橘猫',confidence=.99,
            matched_term='大橘猫',kind='enhanced_overlay',enhancement='clahe_local',brand_only_line=True,
            surrounding_rgb=[80+i*35]*12)],'warnings':[],'scan_complete':True,'evidence':{}}

    def test_three_independent_faint_overlays_need_preexisting_product_brand(self):
        before=copy.deepcopy(self.decisions)
        with patch('buyer_show.watermark_text_extra.detect_extra_text',side_effect=self.overlay):
            result=augment_text(self.scans,self.decisions)
        self.assertTrue(all(len(d['approved_regions'])==1 for d in result))
        self.assertEqual(self.decisions,before)
        self.assertEqual([d['evidence']['merchant_tokens'] for d in result],[d['evidence']['merchant_tokens'] for d in before])
        self.assertTrue(all('extra_text' in s['detection'] for s in self.scans))

    def test_chinese_latin_and_full_store_name_vote_for_one_existing_merchant(self):
        names=['大橘猫','banggirl','大橘猫banggirl']
        def detect(path,*args,**kwargs):
            value=self.overlay(path);name=names[int(Path(path).stem)]
            value['regions'][0].update(matched_term=name,text=name)
            return value
        with patch('buyer_show.watermark_text_extra.detect_extra_text',side_effect=detect):
            result=augment_text(self.scans,self.decisions)
        self.assertTrue(all(len(d['approved_regions'])==1 for d in result))
        self.assertTrue(all(d['evidence']['regions'][-1]['merchant_family']=='banggirl' for d in result))

    def test_already_read_faint_full_store_lines_do_not_require_corner_or_enhanced_ocr(self):
        for scan in self.scans:
            scan['detection']['text_regions']=[text_row('大橘猫BANGGIRL',x=10,y=300,w=250,h=40)]
        with patch('buyer_show.watermark_text_extra._ocr') as ocr:
            result=augment_text(self.scans,self.decisions)
        self.assertTrue(all(len(d['approved_regions'])==1 for d in result));ocr.assert_not_called()

    def test_platform_and_small_english_signature_do_not_establish_a_new_store(self):
        self.decisions[0]['evidence']['merchant_tokens']=['淘宝','frenchhomegoods']
        self.decisions[0]['approved_regions']=[text_row('淘宝',x=500,y=400,w=60,h=20),
                                             text_row('French Home Goods',x=480,y=430,w=100,h=20)]
        with patch('buyer_show.watermark_text_extra.detect_extra_text') as detector:
            augment_text(self.scans,self.decisions)
        detector.assert_not_called()

    def test_unrelated_product_and_no_brand_are_not_queried(self):
        self.scans[1]['product_key']='other-product';self.scans[2]['product_key']=''
        with patch('buyer_show.watermark_text_extra.detect_extra_text',side_effect=self.overlay) as detector:
            result=augment_text(self.scans,self.decisions)
        self.assertEqual(detector.call_count,1)
        self.assertTrue(all(not d['approved_regions'] for d in result))

    def test_duplicate_bytes_static_badge_and_two_images_are_not_auto(self):
        for condition in ('duplicate','static','two'):
            with self.subTest(condition=condition):
                scans=copy.deepcopy(self.scans);decisions=copy.deepcopy(self.decisions)
                if condition=='duplicate':
                    for s in scans:s['detection']['source_sha256']='same'
                if condition=='two':scans=scans[:2];decisions=decisions[:2]
                def detect(path,*args,**kwargs):
                    value=self.overlay(path)
                    if condition=='static':value['regions'][0]['surrounding_rgb']=[170]*12
                    return value
                with patch('buyer_show.watermark_text_extra.detect_extra_text',side_effect=detect):
                    result=augment_text(scans,decisions)
                self.assertTrue(all(not d['approved_regions'] for d in result))

    def test_known_store_information_caption_can_process_one_image_without_erasing_parameters(self):
        self.scans=self.scans[:1];self.decisions=self.decisions[:1]
        self.scans[0]['detection']['text_regions']=[text_row('大橘猫BANGGIRL',x=170,y=400,w=260,h=30),
            text_row('规格 20cm'),text_row('材质 牛皮',y=70),text_row('重量 0.3kg',y=120),
            text_row('材质 牛皮 大橘猫BANGGIRL',y=180,w=400),text_row('BOLUN',x=300,y=250,w=70,h=20)]
        before=Path(self.scans[0]['path']).read_bytes()
        result=augment_text(self.scans,self.decisions)[0]
        self.assertEqual(len(result['approved_regions']),1)
        self.assertGreater(result['approved_regions'][0]['y'],390)
        self.assertTrue(any('材质' in r['label'] for r in result['review_regions']))
        self.assertEqual(Path(self.scans[0]['path']).read_bytes(),before)

    def test_known_small_physical_label_is_not_promoted_as_information_caption(self):
        for scan in self.scans:
            scan['detection']['text_regions']=[text_row('BANGGIRL',x=300,y=250,w=70,h=20),
                text_row('规格 20cm'),text_row('材质 牛皮',y=70),text_row('重量 0.3kg',y=120)]
        result=augment_text(self.scans,self.decisions)
        self.assertTrue(all(not d['approved_regions'] for d in result))

    def test_incomplete_extra_scan_preserves_reason_even_if_verified_region_is_approved(self):
        def detect(path,*args,**kwargs):
            value=self.overlay(path);value['scan_complete']=False;return value
        with patch('buyer_show.watermark_text_extra.detect_extra_text',side_effect=detect):
            result=augment_text(self.scans,self.decisions)
        self.assertTrue(all(d['approved_regions'] for d in result))
        self.assertTrue(all(d['evidence']['extra_text_scan_complete'] is False for d in result))
        self.assertTrue(all(any('未覆盖' in reason for reason in d['review_reasons']) for d in result))

    def test_one_file_failure_does_not_abandon_later_files_and_cancel_still_escapes(self):
        def detect(path,*args,**kwargs):
            if Path(path).stem=='1':raise RuntimeError('synthetic local model failure')
            return self.overlay(path)
        with patch('buyer_show.watermark_text_extra.detect_extra_text',side_effect=detect) as detector:
            result=augment_text(self.scans,self.decisions)
        self.assertEqual(detector.call_count,3)
        self.assertEqual(result[1]['status'],'needs_review')
        self.assertEqual(self.scans[1]['detection']['extra_text']['evidence']['error_type'],'RuntimeError')
        self.assertIn('extra_text',self.scans[2]['detection'])
        with patch('buyer_show.watermark_text_extra.detect_extra_text',side_effect=asyncio.CancelledError),self.assertRaises(asyncio.CancelledError):
            augment_text(self.scans,self.decisions)

    def test_legacy_decision_without_optional_evidence_remains_compatible(self):
        value=decision(0);del value['evidence']
        value['approved_regions']=[text_row('already approved')]
        result=augment_text(self.scans[:1],[value])
        self.assertEqual(result[0]['approved_regions'],value['approved_regions'])
        self.assertIn('evidence',result[0])


if __name__=='__main__':unittest.main()
