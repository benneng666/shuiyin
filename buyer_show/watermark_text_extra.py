"""Second-pass, known-merchant-only text evidence for local image batches.

Enhanced OCR proposes regions; it cannot discover a new trusted merchant. Plain
store headings are distinguished from product copy, and low-opacity text still
needs repeated layout on independent product images before unattended repair.
"""
from __future__ import annotations

import copy
import math
import re
from collections import defaultdict
from difflib import SequenceMatcher
from statistics import median

from .watermark_batch_policy import normalize_text, text_is_product_copy

MAX_EXTRA_ROIS=3
MAX_EXTRA_OCR_CALLS=4


def _terms(value):
    parts=re.split(r'[,，;；、\r\n]+',value) if isinstance(value,str) else list(value or [])
    full={normalize_text(t) for t in parts if isinstance(t,str) and 2<=len(normalize_text(t))<=80
          and not text_is_product_copy(t)}
    aliases=set(full)
    for term in full:
        aliases.update(s for s in re.findall(r'[\u3400-\u9fff]{3,}|[a-z]{6,}',term)
                       if s not in {'旗舰店','专卖店','专营店','官方旗舰店','officialstore','flagshipstore'})
    return sorted(full,key=len,reverse=True),sorted(aliases,key=len,reverse=True)


def _text(row):
    return str(row.get('text') or str(row.get('label','')).split('：',1)[-1]).strip()


def _known(text,terms,aliases):
    token=normalize_text(text)
    exact=next((t for t in aliases if token==t),None)
    embedded=next((t for t in aliases if t in token),None)
    return exact,embedded


def _family(term,terms):
    """Connect exact Chinese/Latin aliases of the already established merchant.

    Shared aliases must come from an existing whole merchant name. This never
    connects a newly recognized word to another product or creates a new brand.
    """
    connected={term}; changed=True
    while changed:
        changed=False
        for full in terms:
            parts={full,*re.findall(r'[\u3400-\u9fff]{3,}|[a-z]{6,}',full)}
            if connected.intersection(parts) and not parts.issubset(connected):
                connected.update(parts);changed=True
    latin=sorted(t for t in connected if re.fullmatch(r'[a-z]{6,}',t))
    return latin[0] if latin else sorted(connected,key=lambda t:(-len(t),t))[0]


def _trusted_tokens(scan,decision):
    tokens=decision.get('evidence',{}).get('merchant_tokens',[])
    tokens=[t for t in tokens if not re.fullmatch(r'淘宝|天猫|抖音|小红书|taobao|tmall|douyin|xiaohongshu',normalize_text(t))]
    approved=decision.get('approved_regions',[]);width=scan.get('detection',{}).get('width',0)
    if approved and width:
        # Signature footers sometimes gained independent repetition evidence in
        # an old batch. Only the main typographic anchors establish store identity
        # for this more permissive second pass, not "French Home Goods" etc.
        main={normalize_text(_text(r)) for r in approved if r['w']>=width*.25
              and normalize_text(_text(r)) in tokens}
        return [t for t in tokens if any(normalize_text(t) in m for m in main)]
    return tokens


def _possible(text,terms,aliases):
    if text_is_product_copy(text): return False
    token=normalize_text(text)
    if not 2<=len(token)<=40: return False
    if any(t in token for t in aliases): return True
    return any(len(t)>=6 and SequenceMatcher(None,token,t).ratio()>=.55 for t in aliases)


def _iou(a,b):
    ix=max(0,min(a['x']+a['w'],b['x']+b['w'])-max(a['x'],b['x']))
    iy=max(0,min(a['y']+a['h'],b['y']+b['h'])-max(a['y'],b['y']))
    intersection=ix*iy
    return intersection/max(1,a['w']*a['h']+b['w']*b['h']-intersection)


def _pad(row,width,height,padding=None):
    x,y,w,h=(int(row[k]) for k in ('x','y','w','h'))
    if padding is None:
        padding=max(3,min(8,math.ceil(h*.15)))
        # Keep a large standalone heading within the existing 12% automatic
        # area budget instead of rejecting it solely for added padding.
        while padding>3 and (w+padding*2)*(h+padding*2)>width*height*.12:
            padding-=1
    left,top=max(0,x-padding),max(0,y-padding)
    right,bottom=min(width,x+w+padding),min(height,y+h+padding)
    return {'x':left,'y':top,'w':right-left,'h':bottom-top}


def _context(image,box):
    from PIL import ImageStat
    x,y,w,h=(box[k] for k in ('x','y','w','h')); radius=max(8,min(40,round(h*.5)))
    values=[]; dispersions=[]
    for bounds in ((max(0,x-radius),y,x,y+h),(x+w,y,min(image.width,x+w+radius),y+h),
                   (x,max(0,y-radius),x+w,y),(x,y+h,x+w,min(image.height,y+h+radius))):
        if bounds[2]<=bounds[0] or bounds[3]<=bounds[1]:
            values.extend([None]*3); continue
        patch=image.crop(bounds); stat=ImageStat.Stat(patch)
        values.extend(round(v,2) for v in stat.mean[:3]); dispersions.extend(stat.stddev[:3]);patch.close()
    return values,round(max(dispersions,default=255),2)


def _valid_rows(detection,width,height):
    rows=[]
    for source in detection.get('text_regions',detection.get('regions',[]))[:256]:
        try:
            row={k:int(source[k]) for k in ('x','y','w','h')};score=float(source.get('confidence',0))
            if (min(row['x'],row['y'])<0 or min(row['w'],row['h'])<=0 or
                    row['x']+row['w']>width or row['y']+row['h']>height or not .5<=score<=1): continue
            row.update(text=_text(source),confidence=score);rows.append(row)
        except (ValueError,TypeError,KeyError): continue
    return rows


def _ocr(pixels,should_stop):
    from .watermark import _get_ocr,_OCR_LOCK,_check_stop
    while not _OCR_LOCK.acquire(timeout=.1): _check_stop(should_stop)
    try:
        _check_stop(should_stop); result,_=_get_ocr()(pixels)
    finally: _OCR_LOCK.release()
    _check_stop(should_stop)
    return result or []


def detect_extra_text(path,merchant_terms='',*,base_detection=None,roi_hints=None,
                      should_stop=lambda:False,emit=lambda message:None):
    """Return reviewable known-store candidates, never an approval to repair.

    Contract: regions / warnings / scan_complete / evidence. OCR runs only on
    bounded weak-line ROIs or product-derived hints, at most four extra calls.
    """
    from .watermark import _load_image,_check_stop,detect_image
    from PIL import Image
    _check_stop(should_stop)
    terms,aliases=_terms(merchant_terms)
    if not terms:
        return {'regions':[],'warnings':['增强识别需要已知店名；未使用新识别文字建立商家身份。'],
                'scan_complete':True,'evidence':{'ocr_calls':0,'known_terms':[]}}
    source,info,_=_load_image(path); matte=Image.new('RGBA',source.size,'white');matte.alpha_composite(source)
    width,height=source.size
    try:
        base=base_detection if base_detection is not None else detect_image(path,merchant_terms=','.join(terms),
            detection_profile='product',return_text_regions=True,should_stop=should_stop,emit=emit)
        if base.get('source_sha256') and base['source_sha256']!=info['sha256']:
            raise ValueError('图片在检测期间发生变化，请重新开始。')
        rows=_valid_rows(base,width,height); copy_rows=sum(text_is_product_copy(r['text']) for r in rows)
        candidates=[]; rois=[]; calls=0; checked_rois=0; warnings=[]
        for row in rows:
            exact,embedded=_known(row['text'],terms,aliases)
            if exact or embedded:
                candidate={**_pad(row,width,height),'label':'已知店名候选：'+row['text'],
                    'text':row['text'],'confidence':row['confidence'],'matched_term':exact or embedded,
                    'kind':'known_brand_text' if exact else 'embedded_brand','enhancement':'base_ocr',
                    'brand_only_line':bool(exact),'raw_box':{k:row[k] for k in ('x','y','w','h')}}
                context,dispersion=_context(matte,row);candidate['surrounding_rgb']=context
                candidate['product_copy_rows']=copy_rows;candidate['background_dispersion']=dispersion
                candidates.append(candidate)
            if (row['confidence']<.92 and _possible(row['text'],terms,aliases)
                    and not (exact and row['confidence']>=.90)
                    and row['h']<=height*.14 and row['w']*row['h']<=width*height*.06):
                rois.append({'box':row,'hint':False})
        for hint in (roi_hints or [])[:MAX_EXTRA_ROIS]:
            try:
                box={k:int(hint[k]) for k in ('x','y','w','h')}
                if min(box['x'],box['y'])<0 or min(box['w'],box['h'])<=0: continue
                if box['x']+box['w']>width or box['y']+box['h']>height: continue
                if (not any(_iou(box,r['box'])>.3 for r in rois)
                        and not any(c['kind']=='known_brand_text' and _iou(box,c)>.3 for c in candidates)):
                    rois.append({'box':box,'hint':True})
            except (ValueError,TypeError,KeyError): continue
        if len(rois)>MAX_EXTRA_ROIS: warnings.append('已限制额外文字识别范围，仍有疑似区域需要确认。')
        import cv2
        import numpy as np
        for entry in rois[:MAX_EXTRA_ROIS]:
            _check_stop(should_stop)
            if calls>=MAX_EXTRA_OCR_CALLS:
                warnings.append('已达到局部增强次数上限，剩余疑似区域需要确认。')
                break
            checked_rois+=1
            row=entry['box']; mx=max(12,round(row['h']*.4));my=max(16,round(height*.075)) if entry['hint'] else max(16,round(row['h']*.4))
            left,top=max(0,row['x']-mx),max(0,row['y']-my)
            right,bottom=min(width,row['x']+row['w']+mx),min(height,row['y']+row['h']+my)
            patch=matte.crop((left,top,right,bottom)).convert('RGB');pixels=np.asarray(patch);patch.close()
            gray=cv2.cvtColor(pixels,cv2.COLOR_RGB2GRAY);hits=[]; method_used=''
            for method in ('clahe_local','highpass_local'):
                if calls>=MAX_EXTRA_OCR_CALLS: break
                _check_stop(should_stop)
                adjusted=(cv2.createCLAHE(clipLimit=3,tileGridSize=(8,4)).apply(gray) if method=='clahe_local'
                          else np.clip((gray.astype(float)-cv2.GaussianBlur(gray,(0,0),5))*3+127,0,255).astype('uint8'))
                scale=min(2,1200/max(adjusted.shape)); enlarged=cv2.resize(adjusted,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
                calls+=1;emit(f'正在增强识别浅色店名（局部 {calls}/{MAX_EXTRA_OCR_CALLS}）。')
                for found in _ocr(enlarged,should_stop):
                    try:
                        points,text,score=found[:3];exact,_=_known(str(text),terms,aliases)
                        if not exact or float(score)<.80: continue
                        xs=[float(p[0])/scale+left for p in points];ys=[float(p[1])/scale+top for p in points]
                        hit={'x':max(0,int(min(xs))),'y':max(0,int(min(ys))),
                             'w':int(math.ceil(max(xs))-int(min(xs))),'h':int(math.ceil(max(ys))-int(min(ys))),
                             'text':str(text),'confidence':float(score),'matched_term':exact}
                        if hit['w']>0 and hit['h']>0 and hit['x']+hit['w']<=width and hit['y']+hit['h']<=height:
                            hits.append(hit)
                    except (TypeError,ValueError,IndexError): continue
                if hits: method_used=method;break
            if hits:
                best=max(hits,key=lambda h:(len(h['matched_term']),h['confidence']))
                # A complete weak OCR line is more reliable for the full word's
                # footprint than a newly split Chinese prefix alone.
                full_line=(not entry['hint'] and row['w']/max(1,row['h'])>=4.3
                           and _possible(row.get('text',''),terms,aliases))
                target=row if full_line else best
                footprint='original_weak_full_line' if full_line else 'recognized_text_only'
                layout=row if entry['hint'] else next((h for h in (roi_hints or []) if _iou(h,row)>.15),None)
                family=_family(best['matched_term'],terms)
                mixed=next((t for t in terms if family in t and re.search(r'[\u3400-\u9fff]{3,}',t)),None)
                if not full_line and layout and mixed:
                    # Chinese prefix and Latin suffix are often separate OCR
                    # detections. Reuse only a full-line footprint observed on
                    # >=3 other product images; move it to this image's observed
                    # glyph baseline rather than erasing at the predicted y.
                    total_h=max(layout['h'],best['h']);total_w=max(layout['w'],best['w'])
                    if re.fullmatch(r'[\u3400-\u9fff]{3,}',best['matched_term']):
                        target={'x':min(layout['x'],best['x']),'y':best['y'],'w':total_w,'h':total_h}
                    elif re.fullmatch(r'[a-z]{6,}',best['matched_term']):
                        target={'x':layout['x'],'y':max(0,best['y']+best['h']-total_h),'w':total_w,'h':total_h}
                    if target is not best:
                        target['w']=max(target['w'],layout['x']+layout['w']-target['x'])
                        target['w']=min(target['w'],width-target['x']);target['h']=min(target['h'],height-target['y'])
                        full_line=True;footprint='same_product_repeated_full_line_layout'
                region={**_pad(target,width,height),'label':'浅色店名增强候选：'+best['text'],
                    'text':best['text'],'confidence':round(best['confidence'],6),'matched_term':best['matched_term'],
                    'kind':'enhanced_overlay','enhancement':method_used,'brand_only_line':True,
                    'raw_box':{k:target[k] for k in ('x','y','w','h')},'full_line_supported':full_line,'footprint_basis':footprint,
                    'recognition_evidence':[{'text':h['text'],'confidence':round(h['confidence'],6)} for h in hits]}
                region['surrounding_rgb'],region['background_dispersion']=_context(matte,target)
                candidates=[c for c in candidates if _iou(c,region)<.65];candidates.append(region)
            elif not entry['hint']:
                candidates.append({**_pad(row,width,height),'label':'浅色疑似店名（需确认）：'+row.get('text',''),
                    'text':row.get('text',''),'confidence':float(row.get('confidence',.5)),
                    'kind':'uncertain_brand','enhancement':'unconfirmed_local','brand_only_line':False})
            elif sum(c['kind']=='known_brand_text' for c in candidates)>=2:
                # Two visible store stamps plus a repeated third-stamp layout
                # make a failed local OCR check reviewable. It is not evidence
                # to erase the predicted rectangle or to call the image clean.
                candidates.append({**_pad(row,width,height),'label':'同商品浅色店名位置（未确认）',
                    'text':'','confidence':0.0,'kind':'uncertain_brand',
                    'enhancement':'product_layout_unconfirmed','brand_only_line':False})
        return {'regions':candidates,'warnings':warnings,'scan_complete':len(rois)<=MAX_EXTRA_ROIS and checked_rois==len(rois),
                'evidence':{'ocr_calls':calls,'known_terms':terms,'product_copy_rows':copy_rows,
                            'planned_rois':len(rois),'scanned_rois':checked_rois,'skipped_rois':len(rois)-checked_rois,
                            'source_sha256':info['sha256'],'confidence_meaning':'OCR text confidence only'},
                'width':width,'height':height}
    finally:
        source.close();matte.close()


def _group_hints(scans,decisions,known):
    pools=defaultdict(list); hints={}
    for scan,decision in zip(scans,decisions):
        product=scan.get('product_key','');det=scan.get('detection',{});width,height=det.get('width',0),det.get('height',0)
        if not width or not height or product not in known: continue
        terms,aliases=_terms(known[product]);identity=det.get('source_sha256') or scan.get('sha256')
        for row in _valid_rows(det,width,height):
            if (row['confidence']<.92 and _possible(row['text'],terms,aliases) and row['w']>=width*.18
                    and row['w']/row['h']>=4.3 and row['h']<=height*.1
                    and not any(_iou(row,r)>.35 for r in decision.get('approved_regions',[]))):
                pools[product].append((identity,[row[k]/div for k,div in zip(('x','y','w','h'),(width,height,width,height))]))
    for product,pool in pools.items():
        result=[]
        for _,shape in pool:
            group=[(identity,s) for identity,s in pool if abs(s[0]-shape[0])<=.08 and abs(s[1]-shape[1])<=.08 and .6<=s[2]/shape[2]<=1.7]
            if len({identity for identity,_ in group if identity})<3: continue
            candidate=[median(s[j] for _,s in group) for j in range(4)]
            if not any(abs(old[0]-candidate[0])<.1 and abs(old[1]-candidate[1])<.1 for old in result): result.append(candidate)
        hints[product]=result[:MAX_EXTRA_ROIS]
    return hints


def augment_text(scans,decisions,*,should_stop=lambda:False,emit=lambda message:None):
    """Return decisions with extra evidence; retain the original trusted tokens."""
    from .watermark import _check_stop,_union_area
    from .watermark_batch_policy import _stable,_different_background
    result=copy.deepcopy(decisions);known=defaultdict(set)
    for decision in result: decision.setdefault('evidence',{})
    for scan,decision in zip(scans,decisions):
        if scan.get('product_key'):
            known[scan['product_key']].update(_trusted_tokens(scan,decision))
    hints=_group_hints(scans,decisions,known); extras=[];groups=defaultdict(list)
    for index,(scan,decision) in enumerate(zip(scans,result)):
        _check_stop(should_stop);product=scan.get('product_key','');det=scan.get('detection') or {};path=scan.get('path')
        # Do not reclassify product prose merely because it contains PLUS etc.
        decision['review_regions']=[r for r in decision.get('review_regions',[]) if not text_is_product_copy(_text(r))]
        if not path or not known.get(product) or not det.get('scan_complete'): continue
        emit(f'正在补充识别已知店名 {index+1}/{len(scans)}。')
        width,height=det.get('width',0),det.get('height',0)
        roi_hints=[{k:max(0,round(v*div)) for k,v,div in zip(('x','y','w','h'),shape,(width,height,width,height))} for shape in hints.get(product,[])]
        roi_hints=[box for box in roi_hints if not any(_iou(box,r)>.30 for r in decision.get('approved_regions',[]))]
        try:
            extra=detect_extra_text(path,list(known[product]),base_detection=det,roi_hints=roi_hints,should_stop=should_stop,emit=emit)
        except Exception as exc:
            # Cancellation derives from BaseException and deliberately escapes.
            # One OCR/model/file error must not discard the remaining batch.
            extra={'regions':[],'warnings':[f'本图增强识别失败（{type(exc).__name__}），原件保留。'],
                   'scan_complete':False,'evidence':{'error_type':type(exc).__name__,'ocr_calls':0}}
            if not decision.get('approved_regions'): decision['status']='needs_review'
        det['extra_text']=extra;identity=det.get('source_sha256') or scan.get('sha256') or ''
        decision.setdefault('evidence',{})['extra_text_scan_complete']=bool(extra['scan_complete'])
        decision['evidence']['extra_text_ocr_calls']=extra.get('evidence',{}).get('ocr_calls',0)
        if not extra['scan_complete']:
            decision.setdefault('review_reasons',[]).append('额外店名补查未覆盖全部疑似区域，已核验区域可处理，其余需要确认。')
        for candidate in extra['regions']:
            candidate=copy.deepcopy(candidate);candidate.update(index=index,product=product,identity=identity)
            candidate['merchant_family']=_family(candidate.get('matched_term',''),_terms(known[product])[0])
            candidate['signature']=(candidate['x']/width,candidate['y']/height,candidate['w']/width,candidate['h']/height)
            extras.append(candidate)
            if candidate['kind'] in ('enhanced_overlay','known_brand_text') and identity:
                groups[(product,candidate['merchant_family'])].append(candidate)
    for candidate in extras:
        index=candidate['index'];scan=scans[index];decision=result[index];det=scan['detection'];width,height=det['width'],det['height']
        if any(_iou(candidate,r)>.55 for r in decision.get('approved_regions',[])): continue
        region={k:candidate[k] for k in ('x','y','w','h','label','confidence')}
        body=candidate['kind']=='embedded_brand';accepted=False;basis='';voters=[]
        if candidate['kind']=='known_brand_text' and candidate['confidence']>=.92 and 'product_detail' in scan.get('source_types',[]):
            raw=candidate['raw_box'];area=raw['w']*raw['h']/(width*height)
            heading=raw['y']<=height*.30 and raw['w']>=width*.60
            flat_caption=raw['w']>=width*.25 and candidate.get('background_dispersion',255)<=14
            accepted=bool(candidate.get('product_copy_rows',0)>=3 and raw['w']/raw['h']>=4 and raw['h']<=height*.14
                          and area<=.12 and (heading or flat_caption))
            basis='known_merchant_standalone_line_in_product_information'
        if not accepted and candidate['kind'] in ('enhanced_overlay','known_brand_text'):
            voters=[c for c in groups[(candidate['product'],candidate['merchant_family'])] if _stable(candidate['signature'],c['signature'])]
            identities={v['identity'] for v in voters}
            raw=candidate.get('raw_box',candidate)
            accepted=(len(identities)>=3 and raw['w']>=width*.18 and raw['w']/raw['h']>=4
                and raw['w']*raw['h']<=width*height*.045 and raw['h']<=height*.12
                and any(c['identity']!=candidate['identity'] and _different_background(candidate,c) for c in voters))
            basis='enhanced_known_brand_at_repeated_position_with_changing_background'
        if accepted and _union_area(decision.get('approved_regions',[])+[region])<=width*height*.12:
            decision.setdefault('approved_regions',[]).append(region)
            decision['review_regions']=[r for r in decision['review_regions'] if _iou(candidate,r)<.60]
            decision['evidence'].setdefault('regions',[]).append({'text':candidate['text'],'basis':basis,
                'role':'extra_known_brand_text','matched_term':candidate.get('matched_term'),
                'merchant_family':candidate.get('merchant_family'),'footprint_basis':candidate.get('footprint_basis','original_complete_brand_line'),
                'enhancement':candidate['enhancement'],'independent_images':len({v['identity'] for v in voters}) if voters else None,
                'product_key':candidate['product'],'ocr_text_confidence':candidate['confidence']})
        elif not any(_iou(candidate,r)>.60 for r in decision['review_regions']):
            decision['review_regions'].append(region)
            decision.setdefault('review_reasons',[]).append('店名与商品正文同行，未自动擦除整行。' if body else '发现已知店名候选，但额外叠加证据不足，保留待确认。')
    for decision in result:
        approved=decision.get('approved_regions',[]);review=decision.get('review_regions',[])
        decision['proposed_regions']=approved+review
        if approved:
            decision['status']='auto'
            decision['note']=''
            if not review and decision['evidence'].get('extra_text_scan_complete') is not False:
                decision['review_reasons']=[]
        elif review: decision['status']='needs_review'
    return result
