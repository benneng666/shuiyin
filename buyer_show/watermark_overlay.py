"""Complete pasted-overlay blocks, not just their OCR text lines.

A merchant overlay is usually a *block*: several text lines plus a logo, a
badge, a coloured corner or a full-width promo strip. Approving only the OCR
lines leaves the graphic/background behind and, worse, feeds those leftover
strokes to the inpaint model as if they were real background (dark blocks and
streaks). This module groups lines into bounded corner/edge blocks and expands
them to the image edge so the whole pasted layer is masked.

Evidence stays scoped to one product. Detail-page typography is excluded by
position, by line width, by copy keywords, and by requiring the same corner
layout on independent images of the same product.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

POLICY_VERSION = 'overlay-block-evidence-v1'

# Words that only make sense on a pasted shop/platform overlay.
_PLATFORM = re.compile(r'天猫|淘宝|双11|双十一|狂欢|限时|立减|到手价|RMB|全球狂欢|活动价', re.I)
_STORE = re.compile(r'官方正品|官方保证|旗舰店|专营店|直营店|专卖店|正品保证|品牌授权', re.I)
_PROMO = re.compile(r'多色选择|优选|穿搭需求|满足|满印|高级面料|质感丰盈|时尚大容量|通勤一包|狂欢|到手价|限时|立减', re.I)
_LATIN = re.compile(r'[A-Za-z]{4,}')
# Long explanatory copy / table text printed on a detail image is not a watermark.
_COPY = re.compile(
    r'尺寸|材质|保养|清洁|误差|说明|洗涤|护理|信息|参数|承诺|郑重|法律|污渍|供应商|一口价|'
    r'不可机洗|测量|产地|渠道|建议|避免|尽量|规格|货号|重量|容量|单位|人工|如实|请以|'
    r'旗舰店购买|提供的指导价|后台设置|适用范围|商品均为正品|放心选购|'
    r'cm|mm|kg|ml|\d{3,}', re.I)

# Wide headlines pasted on a main photo are also overlay artwork, but they
# are dark text on a light design field and OCR often finds only one line.
# Kept separate from _PROMO so corner-badge classification stays unchanged.
_HEADLINE = re.compile(
    r'大空间|装载|更多可能|通勤|出游|逛街|大容量|一包搞定|多色选择|穿搭需求|'
    r'优选|舒适面料|质感丰盈|满印|满足穿搭', re.I)

_MIN_CONFIDENCE = .80
_MAX_CORNER_LINE_RATIO = .45
_MAX_BLOCK_AREA_RATIO = .12
_MAX_TOTAL_AREA_RATIO = .22


def _text(region):
    label = str(region.get('label') or '')
    return label.split('：', 1)[-1].strip() if '：' in label else (region.get('text') or label).strip()


def _kind(region):
    label = str(region.get('label') or '')
    if label.startswith('平台标记'):
        return 'platform'
    if label.startswith('店铺标记'):
        return 'store'
    return 'corner'


def _strength(text, kind):
    """None for ordinary product copy; otherwise the overlay evidence class."""
    if _COPY.search(text):
        return None
    if _PLATFORM.search(text) or kind == 'platform':
        return 'platform'
    if _STORE.search(text) or kind == 'store':
        return 'store'
    if _LATIN.search(text):
        return 'brand'
    if _PROMO.search(text):
        return 'promo'
    return None


def _zone(box, width, height):
    x, y, w, h = (box[k] for k in ('x', 'y', 'w', 'h'))
    left, top, right, bottom = x / width, y / height, (x + w) / width, (y + h) / height
    # A wide line close to the top/bottom edge belongs to a full-width strip.
    if w >= width * .35 and (top < .10 or bottom > .90):
        return 'bottom_banner' if bottom > .90 else 'top_banner'
    if top < .28 and left < .35:
        return 'left_top'
    if top < .28 and right > .65:
        return 'right_top'
    if bottom > .78 and left < .35:
        return 'left_bottom'
    if bottom > .78 and right > .65:
        return 'right_bottom'
    return None


def _union(boxes):
    return (min(b['x'] for b in boxes), min(b['y'] for b in boxes),
            max(b['x'] + b['w'] for b in boxes), max(b['y'] + b['h'] for b in boxes))


def _block(zone, bounds, width, height, strengths=()):
    left, top, right, bottom = bounds
    pad = max(6, round(min(width, height) * .015))
    if zone == 'left_top':
        left, top, right, bottom = 0, 0, right + pad, bottom + pad
    elif zone == 'right_top':
        # A platform badge is a solid colour block sitting on plain background:
        # OCR often misses its rounded left tip, so a wide bounded margin is
        # safe. Light promo lettering printed across hair/clothing is different:
        # widening the mask there makes the model repaint the person, so keep
        # the mask tight around the detected lines.
        if 'platform' in strengths:
            left = min(left - pad, round(width * .45))
        else:
            left = left - pad * 2
        left, top, right, bottom = left, 0, width, bottom + pad
    else:
        # Bottom corners and promo strips sit on large designed colour fields.
        # Erasing the whole field forces the model to invent a big flat area;
        # mask only the overlay artwork plus a small margin so the surrounding
        # real colour/texture is preserved and continued.
        margin = max(10, round(pad * 1.6))
        left, top, right, bottom = left - margin, top - margin, right + margin, bottom + margin
    x, y = max(0, left), max(0, top)
    x1, y1 = min(width, right), min(height, bottom)
    return {'x': x, 'y': y, 'w': max(1, x1 - x), 'h': max(1, y1 - y)}


def _contains(outer, inner):
    return (outer['x'] <= inner['x'] and outer['y'] <= inner['y']
            and outer['x'] + outer['w'] >= inner['x'] + inner['w']
            and outer['y'] + outer['h'] >= inner['y'] + inner['h'])


def _intersection(a, b):
    return (max(0, min(a['x']+a['w'], b['x']+b['w']) - max(a['x'], b['x']))
            * max(0, min(a['y']+a['h'], b['y']+b['h']) - max(a['y'], b['y'])))


def _signature(box, width, height):
    return (box['x'] / width, box['y'] / height, box['w'] / width, box['h'] / height)


def _stable(a, b):
    return all(abs(u - v) <= .06 for u, v in zip(a, b))


def _latin_tokens(texts):
    tokens = set()
    for text in texts:
        tokens.update(match.group(0).casefold() for match in re.finditer(r'[A-Za-z]{5,}', text))
    return tokens


def _rows(scan):
    detection = scan.get('detection') or {}
    width, height = detection.get('width'), detection.get('height')
    if not (type(width) is int and type(height) is int and width > 0 and height > 0):
        return []
    values = []
    for raw in detection.get('regions', []) or []:
        try:
            box = {k: int(raw[k]) for k in ('x', 'y', 'w', 'h')}
            confidence = float(raw.get('confidence', 0))
        except (KeyError, TypeError, ValueError):
            continue
        if (min(box['w'], box['h']) <= 0 or box['x'] < 0 or box['y'] < 0
                or box['x'] + box['w'] > width or box['y'] + box['h'] > height
                or confidence < _MIN_CONFIDENCE):
            continue
        text = _text(raw)
        strength = _strength(text, _kind(raw))
        zone = _zone(box, width, height)
        if not strength or not zone:
            continue
        if zone.endswith('_top') or zone.endswith('_bottom'):
            if box['w'] > width * _MAX_CORNER_LINE_RATIO:
                continue
        values.append({'box': box, 'text': text, 'strength': strength,
                       'zone': zone, 'confidence': confidence})
    return values


def _groups(scan):
    """Group nearby overlay lines in one corner/edge into a single block."""
    detection = scan['detection']
    width, height = detection['width'], detection['height']
    by_zone = defaultdict(list)
    for row in _rows(scan):
        by_zone[row['zone']].append(row)
    result = []
    for zone, rows in by_zone.items():
        # A full-width strip absorbs all other lines in the same edge band.
        if zone.endswith('banner'):
            result.append((zone, rows))
            continue
        rows.sort(key=lambda r: (r['box']['y'], r['box']['x']))
        current = []
        for row in rows:
            if not current:
                current = [row]
                continue
            previous = current[-1]['box']
            gap = row['box']['y'] - (previous['y'] + previous['h'])
            limit = max(18, round(max(previous['h'], row['box']['h']) * 1.6))
            if gap <= limit:
                current.append(row)
            else:
                result.append((zone, current))
                current = [row]
        if current:
            result.append((zone, current))
    blocks = []
    for zone, rows in result:
        bounds = _union([r['box'] for r in rows])
        box = _block(zone, bounds, width, height,
                     strengths=sorted({r['strength'] for r in rows}))
        if box['w'] * box['h'] > width * height * _MAX_BLOCK_AREA_RATIO:
            # Oversized: keep the tight text-only proposal visible for review.
            box = {'x': bounds[0], 'y': bounds[1],
                   'w': bounds[2] - bounds[0], 'h': bounds[3] - bounds[1]}
            evidence_ok = False
        else:
            evidence_ok = True
        blocks.append({
            'zone': zone, 'box': box, 'rows': rows,
            'strengths': sorted({r['strength'] for r in rows}),
            'texts': [r['text'] for r in rows],
            'signature': _signature(box, width, height),
            'size_ok': evidence_ok,
        })
    return blocks


def plan_overlay_blocks(scans, *, should_stop=lambda: False):
    """Return per-item complete blocks, with cross-image approval evidence.

    scans: the exact records produced by ``scan_image`` (item_id, product_key,
    source_types, detection). No file I/O and no side effects.
    """
    prepared, layouts = [], defaultdict(list)
    for index, scan in enumerate(scans):
        should_stop()
        detection = scan.get('detection') or {}
        if not detection or not (scan.get('source_types') or []):
            prepared.append(None)
            continue
        blocks = _groups(scan) if detection.get('scan_complete') is not False else []
        identity = str(detection.get('source_sha256') or scan.get('sha256') or index)
        product = str(scan.get('product_key') or '')
        # Detail images legitimately contain printed headings, service notices
        # and size tables. Only main product photos may vote for, or receive,
        # an automatic whole-block erase; detail candidates stay reviewable.
        is_main = 'product_main' in (scan.get('source_types') or [])
        entry = {'item_id': str(scan.get('item_id', index)), 'product': product,
                 'identity': identity, 'width': detection['width'], 'height': detection['height'],
                 'blocks': blocks, 'is_main': is_main,
                 'regions': list(detection.get('regions') or [])}
        prepared.append(entry)
        for block in blocks:
            if product and is_main:
                # One corner may carry a slightly different mix of text lines
                # (brand + promo + shop) across photos. Group by position and
                # compare evidence classes when voting, not by exact class tuple.
                layouts[(product, block['zone'])].append((entry, block))
    # A proven repeated shop logo also corroborates a matching brand heading
    # elsewhere in the same product, without promoting unrelated detail copy.
    product_brands = defaultdict(set)
    for entry in prepared:
        if entry is None or not entry['is_main']:
            continue
        for block in entry['blocks']:
            peers = layouts.get((entry['product'], block['zone']), [])
            voters = {p['identity'] for p, b in peers
                      if _stable(block['signature'], b['signature'])
                      and set(block['strengths']) & set(b['strengths'])}
            if len(voters) >= 2:
                product_brands[entry['product']].update(_latin_tokens(block['texts']))
    plans = {}
    for entry in prepared:
        if entry is None:
            continue
        approved, review, evidence = [], [], []
        repeated_corner = False
        for block in entry['blocks']:
            box, zone, strengths = block['box'], block['zone'], block['strengths']
            peers = layouts.get((entry['product'], zone), [])
            independent = {p['identity'] for p, b in peers
                           if _stable(block['signature'], b['signature'])
                           and set(strengths) & set(b['strengths'])}
            repeated = len(independent) >= 2
            platform_banner = bool(zone.endswith('banner') and 'platform' in strengths)
            basis = ('repeated_corner_overlay' if repeated else
                     'platform_banner' if platform_banner else None)
            if not entry['is_main']:
                basis = None
            if basis == 'repeated_corner_overlay':
                repeated_corner = True
            record = {'texts': block['texts'][:8], 'zone': zone, 'strengths': strengths,
                      'independent_images': len(independent), 'basis': basis,
                      'box': {k: box[k] for k in ('x', 'y', 'w', 'h')},
                      'policy_version': POLICY_VERSION}
            if basis and block['size_ok']:
                approved.append({**box, 'label': '整块商家叠加水印：' + '/'.join(block['texts'][:3]),
                                 'detection_kind': 'overlay_block', 'evidence': record})
            else:
                review.append({**box, 'label': '整块水印候选（待确认）：' + '/'.join(block['texts'][:3]),
                               'detection_kind': 'overlay_block', 'evidence': record})
                evidence.append(record)
        # A proven corner overlay in this exact image corroborates the other
        # corners of the same pasted template (e.g. a promo badge opposite a
        # repeated shop logo), while still bounding every mask.
        if repeated_corner:
            kept = []
            for region in review:
                record = region['evidence']
                if (region['detection_kind'] == 'overlay_block' and record['basis'] is None
                        and {'brand', 'promo', 'store', 'platform'} & set(record['strengths'])):
                    record = {**record, 'basis': 'companion_of_proven_corner_overlay'}
                    approved.append({**region, 'evidence': record})
                else:
                    kept.append(region)
            review = kept
        # A brand heading on its own (no corner watermark in this particular
        # photo) is only approved when the same brand was proven as a repeated
        # overlay in other images of this exact product.
        if entry['is_main'] and product_brands.get(entry['product']):
            kept = []
            for region in review:
                record = region['evidence']
                if (region['detection_kind'] == 'overlay_block' and record['basis'] is None
                        and 'brand' in record['strengths']
                        and product_brands[entry['product']] & _latin_tokens(record['texts'])):
                    record = {**record, 'basis': 'same_brand_heading_as_proven_overlay'}
                    approved.append({**region, 'evidence': record})
                else:
                    kept.append(region)
            review = kept
        # Bounded union; excess blocks stay visible but are never auto-erased.
        from .watermark import _union_area
        width, height = entry['width'], entry['height']
        while approved and _union_area(approved) > width * height * _MAX_TOTAL_AREA_RATIO:
            review.append(approved.pop())
        plans[entry['item_id']] = {'approved': approved, 'review': review, 'evidence': evidence,
                                   'policy_version': POLICY_VERSION}
    # Wide promo headlines are part of the same pasted template. They may be
    # dark text on a light field, so the light-band detector cannot see them;
    # use the OCR lines directly, but only for a product whose main photos have
    # already proven a pasted overlay. Tight text-height boxes avoid repainting
    # the whole upper half of a photo.
    proven_products = {entry['product'] for entry in prepared
                       if entry and entry['is_main'] and entry['product']
                       and plans[entry['item_id']]['approved']}
    for entry in prepared:
        if (entry is None or not entry['is_main'] or entry['product'] not in proven_products):
            continue
        width, height = entry['width'], entry['height']
        item_plan = plans[entry['item_id']]
        existing = list(item_plan['approved'])
        headline_boxes, texts = [], []
        for raw in entry['regions']:
            try:
                box = {key: int(raw[key]) for key in ('x', 'y', 'w', 'h')}
                confidence = float(raw.get('confidence', 0))
            except (KeyError, TypeError, ValueError):
                continue
            text = _text(raw)
            if (confidence < .90 or min(box['w'], box['h']) <= 0 or box['y'] > height * .42
                    or box['x'] < 0 or box['y'] < 0 or box['x'] + box['w'] > width
                    or box['y'] + box['h'] > height or _COPY.search(text)
                    or not _HEADLINE.search(text)):
                continue
            headline_boxes.append(box)
            texts.append(text[:60])
        if not headline_boxes:
            continue
        # OCR often finds only part of a multi-line headline. Mask the whole
        # title area, extending one more line height, so a missed line cannot
        # be repainted into garbled glyphs. Keep the generative backend here:
        # product handles and straps run through this area and must be rebuilt,
        # not flattened to a constant colour.
        left = min(box['x'] for box in headline_boxes)
        top = min(box['y'] for box in headline_boxes)
        right = max(box['x'] + box['w'] for box in headline_boxes)
        bottom = max(box['y'] + box['h'] for box in headline_boxes)
        tail = max(40, round(1.2 * max(box['h'] for box in headline_boxes)))
        margin = 6
        x, y = max(0, left-margin), max(0, top-margin)
        region = {'x': x, 'y': y,
                  'w': min(width-x, right-left+2*margin),
                  'h': min(height-y, bottom-top+tail+margin)}
        if any(_intersection(region, other) / max(1, region['w']*region['h']) > .20
               for other in existing):
            continue
        record = {'texts': texts[:6], 'boxes': len(headline_boxes),
                  'basis': 'same_product_pasted_headline_complete_area',
                  'policy_version': POLICY_VERSION}
        region.update(label='主图促销标题区：合并漏检行',
                      detection_kind='pasted_headline', evidence=record)
        item_plan['approved'].append(region)
        existing.append(region)
    return plans


def merge_overlay_decision(decision, plan, width, height):
    """Fold complete blocks into a policy decision, keeping its evidence trail."""
    merged = dict(decision)
    approved = list(decision.get('approved_regions') or [])
    review = list(decision.get('review_regions') or [])
    reasons = list(decision.get('review_reasons') or [])
    known = {tuple(r[k] for k in ('x', 'y', 'w', 'h')) for r in approved}
    for region in plan.get('approved') or []:
        values = [region.get(k) for k in ('x', 'y', 'w', 'h')]
        if (any(type(v) is not int for v in values) or min(values[:2]) < 0 or min(values[2:]) <= 0
                or values[0] + values[2] > width or values[1] + values[3] > height):
            continue
        key = tuple(values)
        if key not in known:
            approved.append(region)
            known.add(key)
    # An OCR line inside an approved whole block is no longer "uncertain":
    # leaving it in review would veto the block in _avoid_uncertain_overlap.
    review = [r for r in review if not any(_contains(a, r) for a in approved)]
    review.extend(plan.get('review') or [])
    if approved:
        reasons = [r for r in reasons if '缺少同商品至少3张' not in r and '候选依据不足' not in r]
    merged['evidence'] = {**decision.get('evidence', {}), 'overlay_blocks': plan.get('evidence', []),
                          'overlay_policy_version': POLICY_VERSION}
    warnings = list(decision.get('warnings') or [])
    if plan.get('approved'):
        warnings.append('已按整块覆盖商家叠加水印（含图形与底色）；遮罩外像素与透明度保持不变。')
    merged.update(approved_regions=approved, review_regions=review,
                  proposed_regions=approved + review, review_reasons=list(dict.fromkeys(reasons)),
                  warnings=list(dict.fromkeys(warnings)),
                  status='auto' if approved else 'needs_review' if review or reasons else decision.get('status', 'not_detected'))
    return merged
