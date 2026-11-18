"""Conservative, explainable image decisions for an unattended local batch.

OCR confidence is never interpreted as a watermark probability. Repetition is
scoped to a product and independent image bytes; ordinary product copy is not
eligible. Ambiguous candidates remain visible for review and are not erased.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict

POLICY_VERSION = 'product-overlay-evidence-v2'
_COPY = re.compile(r'手机|对比图|测量|晾干|机洗|误差|信息|information|细节|展示|尺码|尺寸|规格|材质|面料|颜色|色彩|重量|容量|型号|货号|参数|长度|宽度|高度|厚度|价格|售价|提示|说明|保障|保证|退换|退货|包邮|运费|质检|定制|赠品|优惠|折扣|购物|售后|客服|搭配|实拍|上身|多色|可选|新款|经典|时尚|潮流|风格|百搭|通勤|轻盈|柔软|耐磨|优质|舒适|简约|款式|设计|场景|结构|拉链|肩带|口袋|单肩|斜挎|双肩|背包|女包|男包|手提|内里|正面|背面|侧面|效果|长宽|推荐|温馨|名称|产地|适用|开合|可调|加厚|吊牌|标牌|铭牌|印花|标识牌|水洗标|实物|size|detail|display|colour|color|weight|material|capacity|length|height|width|product|quality|fashion|design|style|comfort|shoulder|pocket|zipper|collection|photograph|model|copyright|allrights|recommended|becommended|becoumended', re.I)
_STORE = re.compile(r'旗舰店|专卖店|专营店|直营店|官方店铺|官方商城|商家水印|店铺水印|officialstore|flagshipstore',re.I)
_PLATFORM = re.compile(r'^(?:淘宝|天猫|抖音|小红书|douyin|taobao|tmall|xiaohongshu|rednote)$',re.I)


def normalize_text(text):
    return re.sub(r'[^0-9a-z\u3400-\u9fff]', '', unicodedata.normalize('NFKC',str(text)).casefold())


def text_is_product_copy(text):
    token=normalize_text(text)
    return bool(_COPY.search(token) or re.search(r'\d',token))


def scan_image(path, *, merchant_terms='', should_stop=lambda:False, emit=lambda text:None):
    from .watermark import detect_image, _load_image, _check_stop
    result=detect_image(path,should_stop=should_stop,emit=emit,include_corner_text=True,
                        merchant_terms=merchant_terms,detection_profile='product',return_text_regions=True)
    # A physical product badge can recur in several otherwise different photos.
    # Record the immediate background outside each text box, not the whole image:
    # a changed room behind an identical copied badge cannot supply evidence.
    from PIL import Image, ImageStat
    source, info, _=_load_image(path)
    if info['sha256']!=result['source_sha256']:
        source.close()
        raise RuntimeError('图片在检测期间发生变化，请重新开始。')
    matte=Image.new('RGBA',source.size,'white'); matte.alpha_composite(source)
    try:
        for region in result['text_regions']:
            _check_stop(should_stop)
            x,y,w,h=(region[k] for k in ('x','y','w','h'))
            radius=max(8,min(40,round(h*.5)))
            samples=[]
            for bounds in ((max(0,x-radius),y,x,y+h),(x+w,y,min(source.width,x+w+radius),y+h),
                           (x,max(0,y-radius),x+w,y),(x,y+h,x+w,min(source.height,y+h+radius))):
                if bounds[2]<=bounds[0] or bounds[3]<=bounds[1]:
                    samples.extend([None]*3)
                else:
                    patch=matte.crop(bounds)
                    samples.extend(round(v,2) for v in ImageStat.Stat(patch).mean[:3])
                    patch.close()
            region['surrounding_rgb']=samples
    finally:
        source.close(); matte.close()
    return result


def _text(region):
    value=region.get('text')
    if not isinstance(value,str):
        value=str(region.get('label','')).split('：',1)[-1]
    return value.strip()


def _box(row,width,height):
    try:
        values=[float(row[k]) for k in ('x','y','w','h','confidence')]
        x,y,w,h,score=values
        if (not all(math.isfinite(v) for v in values) or min(w,h)<=0 or min(x,y)<0
                or x+w>width or y+h>height or not 0<=score<=1): return None
        result={'x':round(x),'y':round(y),'w':round(w),'h':round(h),'confidence':score,
                'label':str(row.get('label',_text(row)))[:180], 'text':_text(row)}
        context=row.get('surrounding_rgb')
        if (isinstance(context,list) and len(context)==12
                and all(v is None or type(v) in (int,float) and math.isfinite(v) and 0<=v<=255 for v in context)):
            result['surrounding_rgb']=context
        return result
    except (KeyError,TypeError,ValueError): return None


def _corner(box,width,height):
    x,y,w,h=(box[k] for k in ('x','y','w','h'))
    vertical='top' if y<=height*.20 else 'bottom' if y+h>=height*.80 else None
    horizontal='left' if x<=width*.20 else 'right' if x+w>=width*.80 else None
    return (horizontal+'_'+vertical) if horizontal and vertical else None


def _signature(box,width,height):
    return (box['x']/width,box['y']/height,box['w']/width,box['h']/height)


def _stable(a,b):
    # Large typographic overlays may move slightly between a product's crops.
    tolerance=.075 if min(a[2],b[2])>=.25 else .035
    return (abs(a[0]-b[0])<=tolerance and abs(a[1]-b[1])<=tolerance
            and .60<=a[2]/max(.000001,b[2])<=1.67
            and .60<=a[3]/max(.000001,b[3])<=1.67)


def _brandlike(text):
    token=normalize_text(text)
    if not 2<=len(token)<=28 or text_is_product_copy(token): return False
    # Unknown short Chinese generic headings need stronger semantic evidence.
    return bool(_STORE.search(token) or _PLATFORM.fullmatch(token)
                or re.search(r'[a-z]{4,}',token))


def _brand_key(text):
    """A long Latin brand run survives OCR errors in its Chinese decoration.

    Do not fuzzily equate arbitrary words: only the exact Latin run is shared.
    E.g. 大杨猫BANGGIRL and 大橘猫Banggirl can support each other.
    """
    token=normalize_text(text)
    runs=re.findall(r'[a-z]{5,}',token)
    return max(runs,key=len) if len(runs)==1 else token


def _different_background(a,b):
    a=a.get('surrounding_rgb'); b=b.get('surrounding_rgb')
    if a is None or b is None: return False
    differences=[abs(x-y) for x,y in zip(a,b) if x is not None and y is not None]
    return (len(differences)>=6 and max(differences)>=18
            and math.sqrt(sum(v*v for v in differences)/len(differences))>=12)


def _terms(value):
    return list(dict.fromkeys(normalize_text(t) for t in re.split(r'[,，;；、\r\n]+',value)
                              if len(normalize_text(t))>=2))


def _pad(box,width,height):
    padding=max(3,min(8,math.ceil(box['h']*.10)))
    left,top=max(0,box['x']-padding),max(0,box['y']-padding)
    right,bottom=min(width,box['x']+box['w']+padding),min(height,box['y']+box['h']+padding)
    return {'x':left,'y':top,'w':right-left,'h':bottom-top,'label':'批量商家水印：'+box['text'],
            'confidence':round(box['confidence'],6)}


def _brand_voters(row, product, groups):
    """Allow one *missing trailing letter* of a proven long Latin brand.

    This is deliberately not edit distance/fuzzy matching: short tokens,
    substitutions, arbitrary prefixes and ambiguous longer brands cannot join.
    """
    key=row['brand_key']
    def stable_voters(brand):
        return [p for p in groups.get((product,brand,row['corner']),[])
                if _stable(row['signature'],p['signature'])]
    voters=stable_voters(key)
    if len({p['identity'] for p in voters})>=3 or not re.fullmatch(r'[a-z]{7,}',key):
        return key,voters
    alternatives=[]
    for group_product,brand,corner in groups:
        if group_product!=product or corner!=row['corner'] or len(brand)!=len(key)+1 or not brand.startswith(key):
            continue
        candidates=stable_voters(brand)
        if len({p['identity'] for p in candidates})>=3:
            alternatives.append((brand,candidates))
    return alternatives[0] if len(alternatives)==1 else (key,voters)


def _signature_text(text):
    """Short signature sublines; product copy and body prose are never expanded."""
    token=normalize_text(text)
    if re.fullmatch(r'(?:c|copyright)20\d{2}',token) or ('©' in text and re.fullmatch(r'20\d{2}',token)):
        return True
    # These are ignored as standalone headings. They can belong to a small,
    # repeated logo signature only when a proven brand anchors the whole block.
    if token in ('recommended','becommended','becoumended','copyright','allrightsreserved'):
        return True
    if text_is_product_copy(text): return False
    if _PLATFORM.fullmatch(token) or _STORE.search(token): return True
    plain=unicodedata.normalize('NFKC',text).casefold()
    words=re.findall(r'[a-z]+',plain)
    return bool(re.fullmatch(r'[a-z]{2,36}',token) and 1<=len(words)<=5
                and not set(words).intersection({'the','this','that','with','for','from','our','your','made','price','sale',
                                                 'is','are','can','has','have','bag','leather','cotton','cm','mm','kg'}))


def _companion_plan(item,anchor):
    """A bounded footprint is evidence only; masks remain individual OCR rows."""
    a=anchor['box']; width,height=item['width'],item['height']; children=[]
    for row in item['rows']:
        b=row['box']
        if row is anchor or b['confidence']<.65 or not _signature_text(b['text']): continue
        if b['w']>a['w']*.80 or b['h']>a['h']*.80 or b['w']*b['h']>min(a['w']*a['h']*.35,width*height*.012):
            continue
        # Only tight upper/lower signature rows; same-height adjacent product
        # text is not swept into the logo, and no recursive growth is performed.
        above=b['y']+b['h']<=a['y']+a['h']*.15 and b['y']>=a['y']-a['h']*.65
        below=b['y']>=a['y']+a['h']*.75 and b['y']+b['h']<=a['y']+a['h']*2.30
        if not (above or below): continue
        if b['x']<a['x']-width*.025 or b['x']+b['w']>a['x']+a['w']+width*.025: continue
        if anchor['corner'].startswith('left'):
            aligned=b['x']<=a['x']+a['w']*.15 and b['x']+b['w']<=a['x']+a['w']*.80
        else:
            aligned=b['x']>=a['x']+a['w']*.40
        if aligned: children.append(row)
    if not children: return None
    left=min(row['box']['x'] for row in children); right=max(row['box']['x']+row['box']['w'] for row in children)
    top=min(row['box']['y'] for row in children); bottom=max(row['box']['y']+row['box']['h'] for row in children)
    block=(min(left,a['x']),min(top,a['y']),max(right,a['x']+a['w']),max(bottom,a['y']+a['h']))
    if (block[2]-block[0])*(block[3]-block[1])>width*height*.085 or block[3]-block[1]>height*.18:
        return None
    return {'anchor':anchor,'children':children,'identity':item['identity'],'product':item['product'],
            'signature':((left-a['x'])/a['w'],(top-a['y']-a['h'])/a['h'],
                         (right-a['x'])/a['w'],(bottom-a['y']-a['h'])/a['h']),
            'bounds':list(block)}


def _expand_signature_blocks(prepared,decisions):
    groups=defaultdict(list); plans=[]
    for index,(item,decision) in enumerate(zip(prepared,decisions)):
        if decision['status']!='auto': continue
        for anchor in item.get('anchors',[]):
            plan=_companion_plan(item,anchor)
            if plan is None: continue
            plan['index']=index
            groups[(plan['product'],anchor['canonical_brand'],anchor['corner'])].append(plan)
            plans.append(plan)
    for plan in plans:
        anchor=plan['anchor']; signature=plan['signature']
        siblings=groups[(plan['product'],anchor['canonical_brand'],anchor['corner'])]
        voters=[p for p in siblings if all(abs(a-b)<=limit for a,b,limit in
                zip(signature,p['signature'],(.18,.28,.18,.28)))]
        if len({p['identity'] for p in voters})<3: continue
        item=prepared[plan['index']]; decision=decisions[plan['index']]
        for child in plan['children']:
            region=_pad(child['box'],item['width'],item['height'])
            if region in decision['approved_regions']: continue
            # Remove only the exact promoted row, never an unrelated proposal.
            decision['review_regions']=[r for r in decision['review_regions'] if r!=region]
            decision['approved_regions'].append(region)
            decision['evidence']['regions'].append({'text':child['box']['text'],'token':child['token'],
                'basis':'repeated_adjacent_signature_layout','role':'logo_companion',
                'anchor_brand':anchor['canonical_brand'],'independent_images':len({p['identity'] for p in voters}),
                'product_key':item['product'],'corner':anchor['corner'],'block_bounds':plan['bounds'],
                'ocr_text_confidence':child['box']['confidence']})
    from .watermark import _union_area
    for item,decision in zip(prepared,decisions):
        if decision['status']!='auto': continue
        if _union_area(decision['approved_regions'])>item['width']*item['height']*.12:
            decision['review_regions']=decision['approved_regions']+decision['review_regions']
            decision['approved_regions']=[]; decision['evidence']['merchant_tokens']=[]
            decision['status']='needs_review'
            decision['review_reasons'].append('署名块扩展后总面积较大，保留原图并转待确认。')
        elif not decision['review_regions']:
            decision['review_reasons']=[]
        decision['proposed_regions']=decision['approved_regions']+decision['review_regions']


def analyze_images(scans, *, merchant_terms=''):
    """Return ordered decisions without file I/O or side effects.

    scans: item_id, product_key, source_types, detection; source_sha256 from
    detector (or source_sha256/sha256 on scan) ensures copies cannot vote twice.
    """
    explicit=_terms(merchant_terms)
    prepared=[]; groups=defaultdict(list)
    for index,scan in enumerate(scans):
        detection=scan.get('detection') or {}
        width,height=detection.get('width',0),detection.get('height',0)
        valid_size=(type(width) in (int,float) and type(height) in (int,float)
                    and math.isfinite(width) and math.isfinite(height) and min(width,height)>0)
        complete=(detection.get('scan_complete') is True and not detection.get('candidates_truncated')
                  and not detection.get('text_regions_truncated'))
        identity=str(detection.get('source_sha256') or scan.get('source_sha256') or scan.get('sha256') or '')
        product=str(scan.get('product_key') or '')
        rows=[]; seen=set()
        for raw in detection.get('text_regions',detection.get('regions',[])) if valid_size else []:
            box=_box(raw,width,height)
            if box is None: continue
            token=normalize_text(box['text'])
            key=(token,box['x'],box['y'],box['w'],box['h'])
            if key in seen or len(token)<2: continue
            seen.add(key)
            corner=_corner(box,width,height)
            matched=next((t for t in explicit if t in token),None)
            # Small labels printed on a photographed item may repeat across
            # photos. Repetition alone must not authorize erasing those labels.
            typography=(box['w']>=width*.18 or bool(_STORE.search(token))
                        or bool(_PLATFORM.fullmatch(token)))
            eligible=bool(corner and typography and box['confidence']>=.90 and box['w']*box['h']<=width*height*.045
                          and box['h']<=height*.12 and not text_is_product_copy(token)
                          and (_brandlike(token) or matched))
            row={'box':box,'token':token,'brand_key':_brand_key(token),'corner':corner,'signature':_signature(box,width,height),
                 'eligible':eligible,'matched_term':matched,'identity':identity,'index':index}
            rows.append(row)
            if eligible and complete and product and identity:
                groups[(product,row['brand_key'],corner)].append(row)
        prepared.append({'scan':scan,'detection':detection,'width':width,'height':height,
                         'complete':complete,'rows':rows,'product':product,'identity':identity,'valid_size':valid_size})
    decisions=[]
    for index,item in enumerate(prepared):
        approved=[]; review=[]; reasons=[]; evidence=[]; tokens=[]
        item['anchors']=[]
        scan=item['scan']; det=item['detection']; width,height=item['width'],item['height']
        for row in item['rows']:
            token,box=row['token'],row['box']
            if text_is_product_copy(token): continue
            canonical_brand,voters=_brand_voters(row,item['product'],groups)
            identities={p['identity'] for p in voters}
            repeated=len(identities)>=3
            changing_background=any(p['identity']!=row['identity'] and _different_background(box,p['box'])
                                    for p in voters)
            # Only stable multi-image overlays are automatic. A keyword by
            # itself can also be a physical sign or product label.
            accepted=item['complete'] and row['eligible'] and repeated and changing_background
            suspicious=(row['matched_term'] or _STORE.search(token) or _PLATFORM.fullmatch(token)
                        or (row['corner'] and _brandlike(token)))
            if accepted:
                row['canonical_brand']=canonical_brand; item['anchors'].append(row)
                region=_pad(box,width,height); approved.append(region); tokens.extend((token,canonical_brand))
                evidence.append({'text':box['text'],'token':token,'independent_images':len(identities),
                                 'product_key':item['product'],'corner':row['corner'],
                                 'basis':'repeated_short_text_at_stable_corner_with_changing_background',
                                 'changing_local_background':changing_background,'ocr_text_confidence':box['confidence']})
            elif suspicious:
                review.append(_pad(box,width,height))
                reasons.append('检测到商家或边角品牌文字，但缺少同商品至少3张独立图片的稳定位置及局部背景变化证据。')
        if not item['valid_size'] or not item['complete']:
            reasons.append('图片文字扫描不完整，未自动擦除；请检查扫描上限或检测结果。')
            review=approved+review; approved=[]; tokens=[]
        # A bounded total area keeps automatic changes local.
        from .watermark import _union_area
        if item['valid_size'] and _union_area(approved)>width*height*.12:
            review=approved+review; approved=[]; tokens=[]
            reasons.append('自动候选总面积较大，保留原图并转待确认。')
        status='auto' if approved else 'needs_review' if review or reasons else 'not_detected'
        decisions.append({'item_id':scan.get('item_id',str(index)),'status':status,
                          'approved_regions':approved,'proposed_regions':approved+review,'review_regions':review,
                          'review_reasons':list(dict.fromkeys(reasons)),
                          'evidence':{'policy_version':POLICY_VERSION,'merchant_tokens':list(dict.fromkeys(tokens)),
                                      'regions':evidence,'scan_complete':item['complete'],
                                      'confidence_meaning':'OCR text confidence only, not watermark probability'},
                          'note':'未检出符合自动规则的水印不等于确认无水印；原图保留。' if status=='not_detected' else ''})
    _expand_signature_blocks(prepared,decisions)
    return decisions


def augment_text(scans,decisions,*,should_stop=lambda:False,emit=lambda message:None):
    """Optional second pass after product-level merchant evidence is known."""
    from .watermark_text_extra import augment_text as augment
    return augment(scans,decisions,should_stop=should_stop,emit=emit)
