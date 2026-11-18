"""Detect light overlay headlines that OCR misses, and mask their whole line.

Large white promo headlines printed over a photo are a blind spot for OCR on
light backgrounds. A bright top-hat transform isolates thin bright strokes;
horizontal closing groups them into text *line bands*. Only bands corroborated
by another main photo of the same product, by an OCR line, or by an adjacent
already-proven headline line are approved. A band is a tight rectangle the
height of the text: every anti-aliased stroke is inside the mask, while the
surrounding real texture stays available as inpaint context.

This module never edits files. Image loading is injectable for tests.
"""
from __future__ import annotations

from collections import defaultdict

POLICY_VERSION = 'light-band-evidence-v1'
_THRESHOLDS = (30, 42)
_MAX_BANDS_PER_IMAGE = 8
_MAX_TOTAL_AREA_RATIO = .38
_DESIGN_BAND_DEVIATION = 32
_DESIGN_BAND_MIN_FRACTION = .10
_DESIGN_BAND_START_FRACTION = .70


def _load_rgb(path):
    import numpy as np
    from PIL import Image, ImageOps
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert('RGB'))


def _bands_at(rgb, threshold, hclose=21):
    import cv2
    import numpy as np
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    ksize = max(13, min(35, round(min(width, height) * .035) // 2 * 2 + 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    score = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
    spread = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    score[spread > 45] *= .4
    binary = (score > threshold).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    band = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (hclose, 3)))
    band = cv2.dilate(band, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    count, _, stats, _ = cv2.connectedComponentsWithStats(band, 8)
    found = []
    for label in range(1, count):
        x, y, w, h = (int(stats[label, i]) for i in range(4))
        if w < width * .15 or h < height * .018 or h > height * .13 or w / h < 2.2:
            continue
        coverage = float((binary[y:y+h, x:x+w] > 0).mean())
        if coverage < .18:
            continue
        found.append({'x': x, 'y': y, 'w': w, 'h': h, 'coverage': round(coverage, 3)})
    return found


def _overlap(a, b):
    ix = max(0, min(a['x']+a['w'], b['x']+b['w']) - max(a['x'], b['x']))
    iy = max(0, min(a['y']+a['h'], b['y']+b['h']) - max(a['y'], b['y']))
    return ix * iy


def _merge(boxes):
    boxes = sorted(boxes, key=lambda b: (b['y'], b['x']))
    merged = []
    for box in boxes:
        if merged:
            last = merged[-1]
            x0, y0 = min(box['x'], last['x']), min(box['y'], last['y'])
            x1, y1 = max(box['x']+box['w'], last['x']+last['w']), max(box['y']+box['h'], last['y']+last['h'])
            inter = _overlap(box, last)
            union = box['w']*box['h'] + last['w']*last['h'] - inter
            if inter / max(1, union) > .30 and (y1-y0) < max(box['h'], last['h']) * 2.6:
                merged[-1] = {'x': x0, 'y': y0, 'w': x1-x0, 'h': y1-y0,
                              'coverage': max(box['coverage'], last['coverage'])}
                continue
        merged.append(dict(box))
    return merged


def detect_bands(rgb):
    """Return merged light text-line bands in absolute pixel coordinates."""
    return _merge([box for threshold in _THRESHOLDS for box in _bands_at(rgb, threshold)])


def detect_design_band(rgb):
    """Return the first row of a bottom, horizontally uniform designed strip.

    Promo strips are flat colour fields: per-row median absolute deviation is
    low even where white lettering or a badge covers part of the row. A photo
    row is highly varied, so this does not silently select product imagery.
    """
    import numpy as np
    pixels = rgb.astype(np.int16)
    height = pixels.shape[0]
    median = np.median(pixels, axis=1)
    deviation = np.abs(pixels - median[:, None, :]).mean(axis=(1, 2))
    start = int(height * _DESIGN_BAND_START_FRACTION)
    run_top = None
    result = None
    for y in range(height - 1, start - 1, -1):
        if deviation[y] < _DESIGN_BAND_DEVIATION:
            run_top = y
        else:
            if run_top is not None and height - run_top >= height * _DESIGN_BAND_MIN_FRACTION:
                result = run_top
                break
            run_top = None
    if result is None and run_top is not None and height - run_top >= height * _DESIGN_BAND_MIN_FRACTION:
        result = run_top
    if result is None:
        return None
    # A date line or badge can break the uniform run, leaving the strip's real
    # top edge a few rows above the detected run. Extend upward while rows are
    # still close to the strip colour, bounded to 2% of the image height so a
    # photo can never be swallowed.
    floor = max(start, result - int(height * .02))
    noisy = _DESIGN_BAND_DEVIATION * 1.6
    while result > floor and deviation[result-1] < noisy:
        result -= 1
    return result


def _ocr_boxes(scan):
    detection = scan.get('detection') or {}
    boxes = []
    for raw in detection.get('regions', []) or []:
        try:
            box = {key: int(raw[key]) for key in ('x', 'y', 'w', 'h')}
        except (KeyError, TypeError, ValueError):
            continue
        if min(box['w'], box['h']) > 0:
            boxes.append(box)
    return boxes


def plan_light_bands(scans, overlay_plans=None, *, loader=None,
                     should_stop=lambda: False):
    """Return per-item approved light bands plus a full evidence trail.

    ``overlay_plans`` is the output of ``plan_overlay_blocks``; bands almost
    fully contained in an already approved whole block are dropped, because a
    single merged inpaint mask must not be generated twice.
    """
    read = loader or _load_rgb
    overlay_plans = overlay_plans or {}
    prepared = []
    for index, scan in enumerate(scans):
        should_stop()
        detection = scan.get('detection') or {}
        width, height = detection.get('width'), detection.get('height')
        is_main = 'product_main' in (scan.get('source_types') or [])
        if (not is_main or detection.get('scan_complete') is False
                or not (type(width) is int and type(height) is int and width > 0 and height > 0)
                or not scan.get('path')):
            prepared.append(None)
            continue
        try:
            rgb = read(scan['path'])
        except Exception:
            prepared.append(None)
            continue
        if getattr(rgb, 'shape', (0, 0))[:2] != (height, width):
            prepared.append(None)
            continue
        bands = detect_bands(rgb)
        prepared.append({'item_id': str(scan.get('item_id', index)), 'product': str(scan.get('product_key') or ''),
                         'identity': str(scan.get('sha256') or detection.get('source_sha256') or index),
                         'width': width, 'height': height, 'bands': bands, 'rgb': rgb,
                         'ocr': _ocr_boxes(scan)})

    plans = {}
    for entry in prepared:
        if entry is None:
            continue
        width, height = entry['width'], entry['height']
        approved_boxes = [region for region in
                          (overlay_plans.get(entry['item_id'], {}).get('approved') or [])]
        approved, evidence, proven_lines = [], [], []
        prior_blocks = overlay_plans.get(entry['item_id'], {}).get('approved') or []
        has_banner = any((region.get('evidence') or {}).get('zone') == 'bottom_banner'
                         or (region.get('evidence') or {}).get('basis') == 'platform_banner'
                         for region in prior_blocks)
        design_top = None
        if has_banner:
            top = detect_design_band(entry['rgb'])
            if top is not None and top >= height * _DESIGN_BAND_START_FRACTION:
                design_top = top
        for band in entry['bands']:
            if design_top is not None and band['y'] + band['h'] > design_top:
                clipped = design_top - band['y']
                if clipped < band['h'] * .35:
                    evidence.append({'box': {k: band[k] for k in ('x', 'y', 'w', 'h')},
                                    'basis': None, 'covered_by_block': False,
                                    'clipped_by_design_band': True,
                                    'policy_version': POLICY_VERSION})
                    continue
                band = {**band, 'h': clipped}
            norm = (band['x']/width, band['y']/height, band['w']/width, band['h']/height)
            voters = set()
            for other in prepared:
                if other is None or other is entry or other['product'] != entry['product']:
                    continue
                for peer in other['bands']:
                    pn = (peer['x']/other['width'], peer['y']/other['height'],
                          peer['w']/other['width'], peer['h']/other['height'])
                    # Width must also agree: a full-width promo strip must not
                    # corroborate a small printed label at the same height.
                    if (abs(norm[0]-pn[0]) <= .08 and abs(norm[1]-pn[1]) <= .06
                            and abs(norm[2]-pn[2]) <= .30 and abs(norm[3]-pn[3]) <= .08):
                        voters.add(other['identity'])
                        break
            ocr_hit = any(_overlap(band, box) / max(1, min(band['w']*band['h'], box['w']*box['h'])) > .45
                          for box in entry['ocr'])
            # A whole block already covering this line makes a second mask redundant.
            covered = any(_overlap(band, box) / max(1, band['w']*band['h']) > .70
                          for box in approved_boxes)
            adjacent = any(
                (0 <= band['y'] - (prior['y']+prior['h']) <= 1.4*band['h']
                 or 0 <= prior['y'] - (band['y']+band['h']) <= 1.4*prior['h'])
                and min(band['x']+band['w'], prior['x']+prior['w']) - max(band['x'], prior['x'])
                >= .5 * min(band['w'], prior['w'])
                for prior in proven_lines)
            basis = ('repeated_light_headline' if len(voters) >= 1 else
                     'ocr_corroborated_headline' if ocr_hit else
                     'adjacent_headline_line' if adjacent else None)
            record = {'box': {k: band[k] for k in ('x', 'y', 'w', 'h')},
                      'normalized': [round(value, 4) for value in norm],
                      'coverage': band['coverage'], 'independent_images': len(voters),
                      'ocr_corroborated': ocr_hit, 'covered_by_block': covered,
                      'basis': basis, 'policy_version': POLICY_VERSION}
            if basis and not covered:
                margin = 5
                x, y = max(0, band['x']-margin), max(0, band['y']-margin)
                region = {'x': x, 'y': y,
                          'w': min(width-x, band['w']+2*margin),
                          'h': min(height-y, band['h']+2*margin),
                          'label': '漏检浅色促销文字行：%s' % record['basis'],
                          'detection_kind': 'light_band', 'evidence': record}
                approved.append(region)
                proven_lines.append(band)
            else:
                # A line already erased by a whole block still proves that the
                # line directly below belongs to the same pasted headline.
                if covered and (len(voters) >= 1 or ocr_hit):
                    proven_lines.append(band)
                evidence.append(record)
        if design_top is not None:
            approved.append({'x': 0, 'y': design_top, 'w': width, 'h': height-design_top,
                             'fill': 'row_band', 'label': '底部设计色带：按每行真实底色重建',
                             'detection_kind': 'design_band',
                             'evidence': {'basis': 'proven_platform_banner_uniform_rows',
                                          'top': design_top, 'row_deviation_limit': _DESIGN_BAND_DEVIATION,
                                          'policy_version': POLICY_VERSION}})
        # Bounded union; excess bands stay visible in the evidence trail.
        from .watermark import _union_area
        while approved and _union_area(approved + approved_boxes) > width*height*_MAX_TOTAL_AREA_RATIO:
            evidence.append(approved.pop()['evidence'])
        if len(approved) > _MAX_BANDS_PER_IMAGE:
            evidence.extend(region['evidence'] for region in approved[_MAX_BANDS_PER_IMAGE:])
            approved = approved[:_MAX_BANDS_PER_IMAGE]
        plans[entry['item_id']] = {'approved': approved, 'evidence': evidence,
                                   'policy_version': POLICY_VERSION}
    return plans


def merge_light_band_decision(decision, plan, width, height):
    """Fold proven light bands into a policy decision, keeping the trail."""
    merged = dict(decision)
    approved = list(decision.get('approved_regions') or [])
    known = {tuple(region[k] for k in ('x', 'y', 'w', 'h')) for region in approved}
    for region in plan.get('approved') or []:
        values = [region.get(k) for k in ('x', 'y', 'w', 'h')]
        if (any(type(value) is not int for value in values) or min(values[:2]) < 0
                or min(values[2:]) <= 0 or values[0]+values[2] > width
                or values[1]+values[3] > height):
            continue
        key = tuple(values)
        if key not in known:
            approved.append(region)
            known.add(key)
    # A rebuilt designed strip subsumes whole blocks that sit entirely inside
    # it; keeping both would publish two different repairs for the same pixels.
    design = [region for region in approved if region.get('fill') == 'row_band']
    if design:
        def contained(region):
            return any(strip['x'] <= region['x'] and strip['y'] <= region['y']
                       and strip['x']+strip['w'] >= region['x']+region['w']
                       and strip['y']+strip['h'] >= region['y']+region['h']
                       for strip in design)
        approved = [region for region in approved
                    if region.get('fill') == 'row_band' or not contained(region)]
    review = [region for region in (decision.get('review_regions') or [])
              if not any(approved_region['x'] <= region['x']
                         and approved_region['y'] <= region['y']
                         and approved_region['x']+approved_region['w'] >= region['x']+region['w']
                         and approved_region['y']+approved_region['h'] >= region['y']+region['h']
                         for approved_region in approved)]
    merged['evidence'] = {**decision.get('evidence', {}), 'light_bands': plan.get('evidence', []),
                          'light_band_policy_version': POLICY_VERSION}
    warnings = list(decision.get('warnings') or [])
    if plan.get('approved'):
        warnings.append('已补充 OCR 漏检的浅色促销文字行（贴合文字高度的窄矩形），其余像素保持不变。')
    merged.update(approved_regions=approved, review_regions=review,
                  proposed_regions=approved + review,
                  warnings=list(dict.fromkeys(warnings)),
                  status='auto' if approved else decision.get('status', 'not_detected'))
    return merged
