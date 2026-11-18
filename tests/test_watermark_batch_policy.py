"""Image batch decisions: evidence, isolation, and preservation regressions."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from buyer_show import watermark as wm
from buyer_show.watermark_batch_policy import analyze_images, scan_image, text_is_product_copy
from tests.test_watermark import box


def row(text='BANGGIRL', **updates):
    return {'x':20,'y':30,'w':240,'h':35,'label':'OCR文字：'+text,'confidence':.98,**updates}


def scan(index, rows=None, **updates):
    rows=copy.deepcopy(rows if rows is not None else [row()])
    for region in rows:
        region.setdefault('surrounding_rgb',[80+int(index)*25]*12)
    return {'item_id':str(index),'product_key':'taobao:item-a','source_types':['product_main'],
            'detection':{'width':1000,'height':1000,'scan_complete':True,'source_sha256':f'unique-{index}',
                         'text_regions':rows},**updates}


class BatchImageEvidenceTests(unittest.TestCase):
    def test_three_independent_product_images_allow_one_click_without_keywords(self):
        decisions=analyze_images([scan(i) for i in range(3)])
        self.assertEqual([d['status'] for d in decisions],['auto']*3)
        for decision in decisions:
            self.assertEqual(len(decision['approved_regions']),1)
            self.assertIn('banggirl',decision['evidence']['merchant_tokens'])
            self.assertEqual(decision['evidence']['regions'][0]['independent_images'],3)
            self.assertIn('not watermark probability',decision['evidence']['confidence_meaning'])

    def test_unmarked_and_normal_product_copy_are_never_erased_or_queued_for_review(self):
        values=[scan(0,[])]+[scan(i,[row('细节展示'),row('规格 28 x 32 cm',y=80),
                                    row('DETAIL DISPLAY',y=130),row('产品信息/information',y=160),
                                    row('信息',y=190)]) for i in range(1,4)]
        for decision in analyze_images(values):
            self.assertEqual(decision['status'],'not_detected')
            self.assertEqual(decision['approved_regions'],[])
            self.assertEqual(decision['proposed_regions'],[])
            self.assertIn('不等于确认无水印',decision['note'])

    def test_brand_is_not_a_measurement_merely_because_its_name_contains_cm(self):
        self.assertFalse(text_is_product_copy('ACME'))
        self.assertTrue(text_is_product_copy('宽度 28cm'))
        decisions=analyze_images([scan(i,[row('ACME')]) for i in range(3)])
        self.assertTrue(all(d['status']=='auto' for d in decisions))

    def test_central_and_small_repeated_physical_labels_are_not_approved(self):
        for region in (row('BOLUN',x=460,y=500,w=90,h=25),
                       row('BOLUN',x=20,y=20,w=90,h=25),
                       row('ACME 实物吊牌'),row('ACME 印花')):
            with self.subTest(region=region):
                decisions=analyze_images([scan(i,[region]) for i in range(3)],merchant_terms='BOLUN,ACME')
                self.assertTrue(all(not d['approved_regions'] for d in decisions))

    def test_same_physical_badge_patch_across_otherwise_different_images_is_not_auto(self):
        decisions=analyze_images([scan(i,[row('BANGGIRL',surrounding_rgb=[170]*12)]) for i in range(3)])
        self.assertTrue(all(d['status']=='needs_review' and not d['approved_regions'] for d in decisions))
        values=[scan(i) for i in range(3)]
        for value in values:
            del value['detection']['text_regions'][0]['surrounding_rgb']
        self.assertTrue(all(not d['approved_regions'] for d in analyze_images(values)))

    def test_duplicate_bytes_missing_product_and_cross_product_never_supply_three_votes(self):
        copies=[scan(i) for i in range(3)]
        for value in copies:
            value['detection']['source_sha256']='same-image'
        missing=[scan(i,product_key='') for i in range(3)]
        mixed=[scan(i,product_key=f'product-{i}') for i in range(3)]
        no_hash=[scan(i) for i in range(3)]
        for value in no_hash:
            del value['detection']['source_sha256']
        for values in (copies,missing,mixed,no_hash):
            with self.subTest(values=values):
                decisions=analyze_images(values)
                self.assertTrue(all(d['status']=='needs_review' and not d['approved_regions'] for d in decisions))

    def test_explicit_keyword_or_high_ocr_score_alone_is_not_auto_permission(self):
        for terms in ('','BANGGIRL'):
            decision=analyze_images([scan(0,[row(confidence=1)])],merchant_terms=terms)[0]
            self.assertEqual(decision['status'],'needs_review')
            self.assertEqual(decision['approved_regions'],[])

    def test_chinese_ocr_decoration_variants_keep_the_exact_latin_brand_evidence(self):
        values=[scan(0,[row('大橘猫Banggirl')]),scan(1,[row('大杨猫BANGGIRL')]),
                scan(2,[row('三大橘猫Banggirl')])]
        self.assertTrue(all(d['status']=='auto' for d in analyze_images(values)))
        # Similar but different Latin brands cannot vote for each other.
        values=[scan(i,[row(name)]) for i,name in enumerate(('BRANDA','BRANDB','BRANDC'))]
        self.assertTrue(all(d['status']=='needs_review' for d in analyze_images(values)))

    def test_known_chinese_shop_text_requires_geometry_and_independent_evidence(self):
        values=[scan(i,[row('云朵官方旗舰店')]) for i in range(3)]
        self.assertTrue(all(d['status']=='auto' for d in analyze_images(values)))
        for value in values:
            value['detection']['text_regions'][0]['y']=500
        self.assertTrue(all(d['status']=='needs_review' for d in analyze_images(values)))

    def test_motion_size_low_score_and_incomplete_scan_block_automatic_approval(self):
        changes={
            'different-position':lambda v:v['detection']['text_regions'][0].update(x=150,y=180),
            'oversize':lambda v:v['detection']['text_regions'][0].update(w=600,h=100),
            'low-score':lambda v:v['detection']['text_regions'][0].update(confidence=.60),
            'truncated':lambda v:v['detection'].update(text_regions_truncated=True),
            'incomplete':lambda v:v['detection'].update(scan_complete=False),
        }
        for name,change in changes.items():
            with self.subTest(name=name):
                values=[scan(i) for i in range(3)]; change(values[2])
                self.assertTrue(all(not d['approved_regions'] for d in analyze_images(values)))

    def test_approved_and_ambiguous_regions_can_coexist_without_erasing_ambiguity(self):
        values=[scan(i,[row()]) for i in range(3)]
        values[0]['detection']['text_regions'].append(row('OTHERBRAND',x=700,y=900,w=240))
        decision=analyze_images(values)[0]
        self.assertEqual(decision['status'],'auto')
        self.assertEqual(len(decision['approved_regions']),1)
        self.assertEqual(len(decision['review_regions']),1)
        self.assertEqual(len(decision['proposed_regions']),2)
        self.assertEqual(decision['evidence']['merchant_tokens'],['banggirl'])

    def test_analysis_is_deterministic_and_does_not_mutate_detection_or_originals(self):
        values=[scan(i) for i in range(3)]; before=copy.deepcopy(values)
        self.assertEqual(analyze_images(values),analyze_images(values))
        self.assertEqual(values,before)


class RawOcrBatchContractTests(unittest.TestCase):
    def test_raw_text_opt_in_carries_copy_for_filtering_without_changing_legacy_regions(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'input.png'; Image.new('RGB',(1000,1000),'white').save(path)
            before=path.read_bytes()
            rows=[box(20,30,240,35,'BANGGIRL'),box(20,80,240,35,'材质：牛皮'),
                  box(460,500,90,25,'BOLUN')]
            with mock.patch.object(wm,'_get_ocr',return_value=lambda pixels:(rows,None)):
                legacy=wm.detect_image(path)
                result=scan_image(path)
            self.assertNotIn('text_regions',legacy)
            self.assertEqual(legacy['regions'],[])
            self.assertEqual(len(result['text_regions']),3)
            self.assertTrue(result['scan_complete'])
            self.assertEqual(len(result['source_sha256']),64)
            self.assertEqual(path.read_bytes(),before)

    def test_raw_text_is_bounded_and_marks_truncation(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'input.png'; Image.new('RGB',(1000,1000),'white').save(path)
            rows=[box(20,20+i,50,10,f'text{i}') for i in range(wm.MAX_REGIONS+3)]
            with mock.patch.object(wm,'_get_ocr',return_value=lambda pixels:(rows,None)):
                result=scan_image(path)
            self.assertEqual(len(result['text_regions']),wm.MAX_REGIONS)
            self.assertTrue(result['text_regions_truncated'])
            self.assertEqual(analyze_images([{'detection':result}])[0]['status'],'needs_review')


def logo_rows():
    return [row('BANGGIRL',x=570,y=800,w=400,h=60),
            row('淘宝',x=870,y=865,w=70,h=25,confidence=.84),
            row('French Home Goods',x=800,y=897,w=170,h=20,confidence=.82),
            row('Recommended',x=830,y=920,w=140,h=18,confidence=.85)]


class MerchantSignatureBlockTests(unittest.TestCase):
    def test_repeated_compact_signature_sublines_are_approved_as_separate_masks(self):
        values=[scan(i,logo_rows()) for i in range(3)]
        decisions=analyze_images(values)
        for decision in decisions:
            self.assertEqual(decision['status'],'auto')
            self.assertEqual(len(decision['approved_regions']),4)
            self.assertEqual(decision['review_regions'],[])
            self.assertEqual(decision['review_reasons'],[])
            self.assertEqual(len([e for e in decision['evidence']['regions'] if e.get('role')=='logo_companion']),3)
            self.assertEqual(decision['evidence']['merchant_tokens'],['banggirl'])
            self.assertTrue(all(r['h']<80 for r in decision['approved_regions']))

    def test_copyright_year_belongs_only_to_a_proven_adjacent_logo(self):
        values=[scan(i,[row(),row('(C)2023',x=20,y=68,w=80,h=20)]) for i in range(3)]
        for decision in analyze_images(values):
            self.assertEqual(len(decision['approved_regions']),2)
        unrelated=[scan(i,[row('(C)2023',x=20,y=68,w=80,h=20)]) for i in range(3)]
        self.assertTrue(all(not d['approved_regions'] for d in analyze_images(unrelated)))

    def test_sublines_can_use_ocr_spelling_variants_without_fuzzy_global_brand_matching(self):
        variants=['French Hlome Goods','Fretch Jlome','Freneh Hlome Goods']
        values=[]
        for i,text in enumerate(variants):
            rows=logo_rows(); rows[2]['label']='OCR文字：'+text
            values.append(scan(i,rows))
        self.assertTrue(all(len(d['approved_regions'])==4 and not d['review_regions'] for d in analyze_images(values)))

    def test_one_missing_trailing_letter_of_long_proven_brand_can_join(self):
        values=[scan(i,[row()]) for i in range(3)]+[scan(3,[row('BANGGIR')])]
        decisions=analyze_images(values)
        self.assertTrue(all(d['status']=='auto' for d in decisions))
        self.assertIn('banggirl',decisions[3]['evidence']['merchant_tokens'])
        for value in ('BANGGIRT','BANGGI','ACM'):
            with self.subTest(value=value):
                values[-1]=scan(3,[row(value)])
                self.assertEqual(analyze_images(values)[3]['approved_regions'],[])

    def test_ambiguous_longer_brand_does_not_repair_unknown_prefix(self):
        values=[scan(i,[row('BRANDKEYA')]) for i in range(3)]
        values.extend(scan(i+3,[row('BRANDKEYB')]) for i in range(3))
        values.append(scan(6,[row('BRANDKEY')]))
        self.assertEqual(analyze_images(values)[-1]['approved_regions'],[])

    def test_no_anchor_or_only_two_independent_layouts_never_authorizes_signature(self):
        no_anchor=[scan(i,logo_rows()[1:]) for i in range(3)]
        self.assertTrue(all(not d['approved_regions'] for d in analyze_images(no_anchor)))
        values=[scan(i,logo_rows() if i<2 else logo_rows()[:1]) for i in range(3)]
        self.assertTrue(all(len(d['approved_regions'])==1 for d in analyze_images(values)))

    def test_inconsistent_neighbor_layout_is_not_joined_even_with_three_brand_anchors(self):
        values=[scan(i,logo_rows()) for i in range(3)]
        # The third layout puts every small line above the anchor. Both blocks
        # are close, but the lower layout now has only two independent votes.
        for region in values[2]['detection']['text_regions'][1:]:
            region['y']=755
            region['h']=15
        decisions=analyze_images(values)
        self.assertTrue(all(len(d['approved_regions'])==1 for d in decisions))

    def test_parameters_prices_body_text_and_physical_badge_are_never_added(self):
        for text in ('材质：牛皮','售价 99 元','宽度28cm','Product details','Soft leather bag',
                     'This bag is made of leather','BOLUN 实物标牌'):
            with self.subTest(text=text):
                values=[scan(i,[logo_rows()[0],row(text,x=800,y=890,w=170,h=20)]) for i in range(3)]
                self.assertTrue(all(len(d['approved_regions'])==1 for d in analyze_images(values)))
        rows=logo_rows()+[row('BOLUN',x=440,y=500,w=70,h=20),row('BANGGIRL',x=35,y=710,w=200,h=25)]
        self.assertTrue(all(len(d['approved_regions'])==4 for d in analyze_images([scan(i,rows) for i in range(3)])))

    def test_expansion_stays_bounded_even_when_nearby_text_repeats(self):
        rows=[row('BANGGIRL',x=500,y=750,w=500,h=78),
              row('French Home Goods',x=750,y=890,w=200,h=35,confidence=.82)]
        self.assertTrue(all(len(d['approved_regions'])==1 for d in analyze_images([scan(i,rows) for i in range(3)])))
        # A chain of small English rows cannot recursively extend the block.
        rows=logo_rows()+[row('Veryfar',x=830,y=965,w=140,h=18)]
        for decision in analyze_images([scan(i,rows) for i in range(3)]):
            self.assertEqual(len(decision['approved_regions']),4)
            self.assertEqual(len(decision['review_regions']),1)

    def test_static_physical_logo_patch_cannot_become_an_anchor_for_signature_expansion(self):
        values=[scan(i,logo_rows()) for i in range(3)]
        for value in values:
            value['detection']['text_regions'][0]['surrounding_rgb']=[170]*12
        self.assertTrue(all(not d['approved_regions'] for d in analyze_images(values)))


if __name__=='__main__':
    unittest.main()
