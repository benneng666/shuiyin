"""One local worker, product-scoped detection, and a complete derived media set.

Only policy-approved regions reach a repair backend. Uncertain/no-hit files
are copied byte-for-byte; originals and the evidence trail are never replaced.
"""
from __future__ import annotations

import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import tempfile
import time
from uuid import uuid4

from PIL import Image, ImageOps

from .watermark_batch_policy import scan_image, analyze_images
from .watermark_overlay import plan_overlay_blocks, merge_overlay_decision
from .watermark_batch_video import plan_video
from .watermark import repair_image
from .watermark_video import repair_video

PRODUCT_SOURCES = frozenset({'product_main', 'product_detail', 'product_video'})
REVIEW_SOURCES = frozenset({'review_initial', 'review_append'})
TERMINAL = ('processed', 'unchanged', 'needs_review', 'failed', 'cancelled')
LABELS = {'processed': '已处理', 'unchanged': '未检出，原样', 'needs_review': '待确认',
          'failed': '失败', 'cancelled': '已取消', 'pending': '等待', 'processing': '处理中'}


def _stop(should_stop):
    if should_stop():
        raise asyncio.CancelledError()


def _sha(path, should_stop=lambda: False):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(1024 * 1024):
            _stop(should_stop)
            value.update(chunk)
    return value.hexdigest()


def _notify(callback, value):
    try:
        callback(value)
    except Exception:
        pass  # Presentation callbacks must not invalidate published results.


def _slug(value):
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', str(value)).strip(' .')[:64]
    if not value or re.fullmatch(r'CON|PRN|AUX|NUL|COM\d|LPT\d', value, re.I):
        value = 'local'
    return value


def select_items(items, *, include_products=True, include_reviews=False, include_local=False):
    """Merge duplicates only inside one product, keeping every association."""
    selected = {}
    for item in items:
        kind = item.get('media_type', 'image')
        if kind not in {'image', 'video'} or not item.get('path'):
            continue
        sources = set(item.get('source_types') or [])
        eligible = bool(include_products and sources & PRODUCT_SOURCES or include_reviews and sources & REVIEW_SOURCES
                        or include_local and not sources & (PRODUCT_SOURCES | REVIEW_SOURCES))
        if not eligible:
            continue
        path = str(Path(item['path']).expanduser().absolute())
        product = str(item.get('product_key') or '')
        identity = item.get('sha256') or os.path.normcase(path)
        key = (product, kind, identity)
        if key not in selected:
            selected[key] = {**item, 'path': path, 'media_type': kind, 'product_key': product,
                             'source_types': sorted(sources), 'categories': list(item.get('categories') or [])}
        else:
            entry = selected[key]
            entry['source_types'] = sorted(set(entry['source_types']) | sources)
            entry['categories'] = sorted(set(entry['categories']) | set(item.get('categories') or []))
    return list(selected.values())


def _categories(item, include_products, include_reviews, include_local):
    """Generate known folders, never interpret an imported category as a path."""
    sources = set(item['source_types'])
    kind = item['media_type']
    result = []
    if include_products and 'product_main' in sources and kind == 'image':
        from .media_catalog import _ratio
        with Image.open(item['path']) as image:
            normalized = ImageOps.exif_transpose(image)
            try:
                ratio = _ratio(*normalized.size)
            finally:
                normalized.close()
        result.append('商品主图/' + ratio)
    if include_products and 'product_detail' in sources and kind == 'image':
        result.append('详情图')
    if include_products and ('product_video' in sources or kind == 'video' and sources & PRODUCT_SOURCES):
        result.append('商品视频')
    if include_reviews:
        for source, label in (('review_initial', '初评'), ('review_append', '追评')):
            if source in sources:
                result.append('买家秀/' + label + ('/视频' if kind == 'video' else '/图片'))
    if not result and include_local:
        result.append('本地素材/' + ('视频' if kind == 'video' else '图片'))
    return list(dict.fromkeys(result or ['其他商品素材']))


def _smart_method(path, regions, should_stop):
    """Keep exact flat fills; use the CPU AI backend for everything else.

    Telea/NS replace the whole rectangle with interpolated pixels. On a
    textured or gradient background that is the visible smear reported after
    watermark removal. Only a nearly uniform, small box can be filled
    losslessly with an exact colour; rule-based gradients still blur, so they
    go through LaMa too. Pure-colour boxes keep the fast path unchanged.
    """
    import numpy as np
    from .watermark import _load_image, validate_regions, _flat_boundary
    image, _, _ = _load_image(path)
    try:
        width, height = image.size
        boxes = validate_regions(regions, width, height)
        original = np.asarray(image)
        mask = np.zeros((height, width), dtype=np.uint8)
        for box in boxes:
            x, y, w, h = (box[key] for key in ('x', 'y', 'w', 'h'))
            mask[y:y+h, x:x+w] = 255
        for index, box in enumerate(boxes):
            _stop(should_stop)
            if any(i != index and max(box['x'], other['x']) < min(box['x']+box['w'], other['x']+other['w'])
                   and max(box['y'], other['y']) < min(box['y']+box['h'], other['y']+other['h'])
                   for i, other in enumerate(boxes)):
                return 'lama'
            color, _ = _flat_boundary(original, mask, box)
            if color is None or box['w'] * box['h'] > width * height * .05:
                return 'lama'
        return 'auto'
    finally:
        image.close()


def _avoid_uncertain_overlap(approved, uncertain, media_type):
    def overlap(a, b):
        spatial = (max(a['x'], b['x']) < min(a['x']+a['w'], b['x']+b['w'])
                   and max(a['y'], b['y']) < min(a['y']+a['h'], b['y']+b['h']))
        temporal = (media_type != 'video' or max(a.get('start', 0), b.get('start', 0))
                    < min(a.get('end', float('inf')), b.get('end', float('inf'))))
        return spatial and temporal
    kept, rejected = [], []
    for region in approved:
        (rejected if any(overlap(region, other) for other in uncertain) else kept).append(region)
    return kept, rejected


def merge_graphics_decision(decision, extra, width, height):
    """Merge bounded extra masks without promoting unrelated uncertain text."""
    from .watermark import _union_area
    merged = dict(decision)
    approved = list(decision.get('approved_regions') or [])
    review = list(decision.get('review_regions') or [])
    reasons = list(decision.get('review_reasons') or [])
    warnings = list(decision.get('warnings') or []) + list(extra.get('warnings') or [])
    key = lambda box: tuple(box[k] for k in ('x', 'y', 'w', 'h'))
    known = {key(box) for box in approved}
    for box in extra.get('approved_regions') or []:
        values = [box.get(k) for k in ('x', 'y', 'w', 'h')]
        if (any(type(v) is not int for v in values) or min(values[:2]) < 0 or min(values[2:]) <= 0
                or values[0]+values[2] > width or values[1]+values[3] > height):
            warnings.append('图形检测返回无效区域，未自动使用。')
            reasons.append('补充图形范围无效，需检查。')
            continue
        if key(box) in known:
            continue
        if len(approved) >= 256 or _union_area(approved + [box]) > width * height * .12:
            review.append(box)
            reasons.append('图形补充后自动范围较大；超出部分保留待确认。')
        else:
            approved.append(box)
            known.add(key(box))
    review.extend(extra.get('review_regions') or [])
    reasons.extend(extra.get('review_reasons') or [])
    # Preserve separate evidence, so a geometric match is never called OCR.
    merged['evidence'] = {**decision.get('evidence', {}), 'graphics': extra.get('evidence', {})}
    merged.update(approved_regions=approved, review_regions=review,
                  proposed_regions=approved + review, review_reasons=list(dict.fromkeys(reasons)),
                  warnings=list(dict.fromkeys(warnings)),
                  status='auto' if approved else 'needs_review' if review or reasons else 'not_detected')
    return merged


def _copy(source, target, should_stop):
    """Exclusive destination in this run; bounded copy and atomic publication."""
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.with_name('.' + target.name + '.' + uuid4().hex + '.part')
    try:
        digest = hashlib.sha256()
        with Path(source).open('rb') as reader, stage.open('xb') as writer:
            while chunk := reader.read(1024 * 1024):
                _stop(should_stop)
                writer.write(chunk)
                digest.update(chunk)
        _stop(should_stop)
        # Run directories are unique; avoid replacing even an unexpected file.
        if target.exists():
            raise FileExistsError(target)
        os.rename(stage, target)
        return digest.hexdigest()
    finally:
        stage.unlink(missing_ok=True)


def _atomic_json(path, data):
    stage = path.with_name('.' + path.name + '.tmp')
    stage.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    # Windows indexers/antivirus can briefly hold the previous manifest open.
    for attempt in range(5):
        try:
            os.replace(stage, path)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(.05 * (attempt + 1))


def _html_report(summary):
    esc = lambda value: html.escape(str(value), quote=True)
    rows = []
    root = Path(summary['batch_dir'])
    for item in summary['items']:
        links = []
        for label, path in [('原件', item['path']), ('结果', item.get('output'))]:
            if path:
                links.append('<a href="' + esc(Path(path).as_uri()) + '">' + label + '</a>')
        name = Path(item['path']).name
        details = [item.get('reason', '')]
        if item.get('media_type') == 'video' and item.get('repair_method'):
            methods = {'smart': '智能视频精修', 'lama_temporal': 'AI视频精修', 'delogo': '快速视频修补'}
            details.append('实际方式：' + methods.get(item['repair_method'], item['repair_method']))
        details.extend(str(w) for w in item.get('warnings', []))
        rows.append('<tr><td>' + esc(item.get('product_key') or '本地') + '</td><td>' + esc(name)
                    + '</td><td>' + esc(LABELS[item['status']]) + '</td><td>' + '<br>'.join(esc(d) for d in dict.fromkeys(details) if d)
                    + '</td><td>' + ' · '.join(links) + '</td></tr>')
    page = '<!doctype html><meta charset="utf-8"><title>一键去水印结果</title><style>body{font:15px system-ui;margin:36px;background:#f4f6f9;color:#172334}table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left;overflow-wrap:anywhere}a{color:#1568b3}h1{font-size:25px}</style><h1>一键去水印结果</h1><p>'
    page += esc(' · '.join(LABELS[k] + ' ' + str(summary['counts'].get(k, 0)) for k in TERMINAL))
    page += '</p><p>完整素材按商品与类型分文件夹。未检出素材原样保留；待确认素材未擦除不确定区域。原件均保留。视频为有限抽样识别，可能漏检。</p><table><tr><th>商品</th><th>素材</th><th>结果</th><th>说明</th><th>查看</th></tr>' + ''.join(rows) + '</table>'
    (root / '打开结果.html').write_text(page, encoding='utf-8')


def run_batch(items, *, output_root=None, include_products=True, include_reviews=False, include_local=False,
              image_method='smart', video_method='smart', merchant_terms='', detection_profile='product',
              include_corner_text=True, overlay_blocks=True, should_stop=lambda: False,
              emit=lambda text: None, on_event=lambda value: None):
    if image_method not in {'smart', 'auto', 'lama'}:
        raise ValueError('批量修复方法必须为 smart、auto 或 lama。')
    if video_method not in {'smart', 'delogo', 'lama_temporal'}:
        raise ValueError('视频修复方法必须为 smart、delogo 或 lama_temporal。')
    selected = select_items(items, include_products=include_products, include_reviews=include_reviews, include_local=include_local)
    if not selected:
        raise ValueError('当前范围没有素材；请先载入采集任务，或勾选本地素材/买家秀。')
    base = Path(output_root).expanduser().resolve() if output_root else Path(selected[0]['path']).parent.parent / 'processed'
    base.mkdir(parents=True, exist_ok=True)
    batch_dir = Path(tempfile.mkdtemp(prefix='batch-' + datetime.now().strftime('%Y%m%d-%H%M%S-'), dir=base)).resolve()
    manifest = batch_dir / 'batch.json'
    records = []
    for index, item in enumerate(selected):
        records.append({**item, 'item_id': str(index), 'status': 'pending', 'output': None,
                        'manifest': None, 'regions': [], 'approved_regions': [], 'warnings': [], 'exports': []})
    summary = {'schema_version': 1, 'batch_dir': str(batch_dir), 'manifest': str(manifest),
               'created_at': datetime.now(timezone.utc).isoformat(), 'status': 'processing',
               'scope': {'product': include_products, 'reviews': include_reviews, 'local': include_local},
               'image_method': image_method, 'video_method': video_method, 'overlay_blocks': overlay_blocks,
               'items': records, 'counts': {}, 'warnings': [], 'html': str(batch_dir / '打开结果.html')}

    def checkpoint(record=None, stage=None):
        counts = Counter(row['status'] for row in records)
        summary['counts'] = {key: counts[key] for key in (*TERMINAL, 'pending', 'processing')}
        summary['counts'].update(total=len(records),
                                 partially_processed=sum(bool(row.get('partially_processed')) for row in records),
                                 exported_files=sum(len(row['exports']) for row in records))
        fallback_count = sum(bool(row.get('video_fallback')) for row in records)
        summary['video_fallback_count'] = fallback_count
        prefix = '视频修复提示：'
        summary['warnings'] = [w for w in summary['warnings'] if not str(w).startswith(prefix)]
        if fallback_count:
            summary['warnings'].append(f'{prefix}{fallback_count} 个视频因范围或推理预算限制含快速修补，请在本批结果中核对说明。')
        _atomic_json(manifest, summary)
        if record is not None:
            _notify(on_event, {'index': int(record['item_id']) + 1, 'total': len(records),
                              'path': record['path'], 'product_key': record['product_key'],
                              'status': record['status'], 'stage': stage,
                              'counts': dict(summary['counts'])})

    def verify(record):
        actual = _sha(record['path'], should_stop)
        expected = record.get('source_sha256') or record.get('sha256')
        if expected and actual != expected:
            raise ValueError('原始文件已变化，停止处理这份素材。')
        record['source_sha256'] = actual
        return actual

    def export(record, source, expected_sha=None):
        product = record['product_key']
        group = _slug(product or '本地素材') + '-' + hashlib.sha256(product.encode()).hexdigest()[:8]
        source = Path(source)
        extension = source.suffix.lower()
        if not re.fullmatch(r'\.[a-z0-9]{1,8}', extension):
            raise ValueError('素材扩展名无效。')
        filename = str(int(record['item_id']) + 1).zfill(4) + '_' + record['source_sha256'][:12] + extension
        folders = _categories(record, include_products, include_reviews, include_local)
        for folder in folders:
            target = batch_dir / '完整素材' / group / folder / filename
            digest = _copy(source, target, should_stop)
            expected = record['source_sha256'] if source == Path(record['path']) else expected_sha
            if expected and digest != expected:
                target.unlink()
                raise ValueError('导出内容与校验记录不一致；未发布该副本。')
            record['exports'].append({'path': str(target), 'sha256': digest, 'category': folder})
            if not record.get('output'):
                record['output'] = str(target)
        record['output_sha256'] = record['exports'][0]['sha256']

    checkpoint()
    scans, decisions, group_terms = [], {}, {}
    cancelled = False
    fatal_error = None
    try:
        # First pass provides cross-image evidence scoped to a single product.
        for record in records:
            _stop(should_stop)
            record['status'] = 'processing'
            checkpoint(record, 'detect')
            _notify(emit, f"识别 {int(record['item_id'])+1}/{len(records)}：{Path(record['path']).name}")
            try:
                verify(record)
                if record['media_type'] == 'image':
                    detection = scan_image(record['path'], merchant_terms=merchant_terms, should_stop=should_stop, emit=emit)
                    if detection.get('source_sha256') and detection['source_sha256'] != record['source_sha256']:
                        raise ValueError('检测期间原始文件发生变化。')
                    scans.append({'item_id': record['item_id'], 'product_key': record['product_key'], 'path': record['path'],
                                  'source_types': record['source_types'], 'sha256': record['source_sha256'],
                                  'detection': detection})
                record['status'] = 'pending'
            except Exception as exc:
                record.update(status='failed', reason=str(exc)[:500])
            checkpoint(record, 'detect')
        base_decisions = analyze_images(scans, merchant_terms=merchant_terms)
        from .watermark_text_extra import augment_text
        from .watermark_graphics import analyze_graphics
        _notify(emit, '正在补查浅色店名、品牌图形与装饰线条…')
        enriched = augment_text(scans, base_decisions, should_stop=should_stop, emit=emit)
        graphic_results = {row['item_id']: row for row in analyze_graphics(scans, enriched,
                          should_stop=should_stop, emit=emit)}
        if overlay_blocks:
            _notify(emit, '正在合并整块商家水印（图形、角标与横幅）…')
        overlay_plans = plan_overlay_blocks(scans, should_stop=should_stop) if overlay_blocks else {}
        if overlay_blocks:
            from .watermark_lightband import plan_light_bands, merge_light_band_decision
            _notify(emit, '正在补查 OCR 漏检的浅色促销文字行…')
            light_plans = plan_light_bands(scans, overlay_plans, should_stop=should_stop)
        else:
            light_plans = {}
        scan_by_id = {scan['item_id']: scan for scan in scans}
        for decision in enriched:
            scan = scan_by_id[decision['item_id']]
            extra = graphic_results.get(decision['item_id'])
            if extra:
                decision = merge_graphics_decision(decision, extra, scan['detection']['width'], scan['detection']['height'])
            plan = overlay_plans.get(decision['item_id'])
            if plan and (plan['approved'] or plan['review'] or plan['evidence']):
                decision = merge_overlay_decision(decision, plan,
                                                  scan['detection']['width'], scan['detection']['height'])
            light = light_plans.get(decision['item_id'])
            if light and (light['approved'] or light['evidence']):
                decision = merge_light_band_decision(decision, light,
                                                     scan['detection']['width'], scan['detection']['height'])
            trace = batch_dir / '检测记录' / ('image-' + decision['item_id'] + '.json')
            trace.parent.mkdir(exist_ok=True)
            _atomic_json(trace, {'scan': scan, 'decision': decision})
            decisions[decision['item_id']] = decision
            record = records[int(decision['item_id'])]
            record['detection_manifest'] = str(trace)
            group_terms.setdefault(record['product_key'], set()).update(decision.get('evidence', {}).get('merchant_tokens', []))
        for record in records:
            _stop(should_stop)
            if record['status'] == 'failed':
                # Detection failure does not remove a readable original from the deliverable.
                try:
                    verify(record)
                    export(record, record['path'])
                    record['reason'] += ' 原件已保留到完整素材。'
                except Exception as exc:
                    record['warnings'].append('原样导出失败：' + str(exc)[:300])
                checkpoint(record, 'repair')
                continue
            record['status'] = 'processing'
            checkpoint(record, 'repair')
            try:
                verify(record)
                if record['media_type'] == 'video':
                    # Unidentified local media cannot inherit another product's brand.
                    inferred = group_terms.get(record['product_key'], set()) if record['product_key'] else set()
                    terms = merchant_terms or ','.join(sorted(inferred))
                    decision = plan_video(record['path'], merchant_terms=terms, should_stop=should_stop, emit=emit)
                    if decision.get('source_sha256') and decision['source_sha256'] != record['source_sha256']:
                        raise ValueError('视频检测与原始文件不一致。')
                else:
                    decision = decisions[record['item_id']]
                record['evidence'] = decision.get('evidence', {})
                record['warnings'] = list(decision.get('warnings', []))
                approved = decision.get('approved_regions') or []
                proposed = decision.get('proposed_regions', decision.get('regions', []))
                review_reasons = decision.get('review_reasons') or []
                review_regions = decision.get('review_regions') or []
                if record['media_type'] == 'video':
                    review_regions = [track['region'] for track in decision.get('evidence', {}).get('tracks', [])
                                      if not track.get('approved') and track.get('region')]
                approved, overlaps = _avoid_uncertain_overlap(approved, review_regions, record['media_type'])
                if overlaps:
                    review_regions = review_regions + overlaps
                    review_reasons = review_reasons + ['可靠框与未确认框重叠，已保留该区域供人工检查。']
                record.update(regions=proposed, approved_regions=approved, review_regions=review_regions,
                              review_reasons=review_reasons)
                review = bool(review_reasons or review_regions or decision['status'] == 'needs_review')
                verify(record)
                if approved:
                    _notify(emit, f"自动修复 {int(record['item_id'])+1}/{len(records)}：{len(approved)} 个可靠区域")
                    if record['media_type'] == 'image':
                        method = _smart_method(record['path'], approved, should_stop) if image_method == 'smart' else image_method
                        record['repair_method'] = method
                        repaired = repair_image(record['path'], approved, output_root=batch_dir / '处理记录',
                                                should_stop=should_stop, emit=emit, method=method)
                    else:
                        record['repair_method'] = video_method
                        repaired = repair_video(record['path'], approved, output_root=batch_dir / '处理记录',
                                                should_stop=should_stop, emit=emit, method=video_method)
                        record['repair_method'] = repaired.get('method', video_method)
                    record.update(output=repaired['output'], manifest=repaired['manifest'],
                                  mask=repaired.get('mask'), repair_committed=True)
                    # Immediately journal a committed backend output, including on cancel.
                    checkpoint(record, 'repair')
                    verify(record)
                    processing = json.loads(Path(repaired['manifest']).read_text(encoding='utf-8'))
                    if processing.get('source_sha256') != record['source_sha256']:
                        raise ValueError('处理记录与原始文件不一致。')
                    record['warnings'] += processing.get('warnings', [])
                    if record['media_type'] == 'video':
                        algorithm = processing.get('quality', {}).get('algorithm', {})
                        record['video_fallback'] = bool(processing.get('quality_fallback_reason')
                            or algorithm.get('budget_fallback_frames'))
                        record['video_quality'] = {key: algorithm.get(key, 0) for key in
                            ('keyframes', 'flow_frames', 'spatial_frames', 'budget_fallback_frames')}
                    expected = (processing.get('output', {}).get('sha256') if record['media_type'] == 'image'
                                else processing.get('validation', {}).get('sha256'))
                    if not expected:
                        raise ValueError('处理记录缺少输出哈希，未导出未校验结果。')
                    export(record, repaired['output'], expected)
                    record['status'] = 'needs_review' if review else 'processed'
                    record['partially_processed'] = review
                    record['reason'] = '已处理可靠区域；其余候选集中待确认。' if review else '已自动处理可靠商家水印区域。'
                else:
                    export(record, record['path'])
                    record['status'] = 'needs_review' if review else 'unchanged'
                    record['reason'] = '候选依据不足，原样保留，未自动擦除。' if review else '未检出水印候选，原样保留。'
                verify(record)
            except Exception as exc:
                record.update(status='failed', reason=str(exc)[:500])
                if not record['exports'] and not record.get('repair_committed'):
                    try:
                        verify(record)
                        export(record, record['path'])
                    except Exception as copy_exc:
                        record['warnings'].append('原样导出失败：' + str(copy_exc)[:300])
            checkpoint(record, 'repair')
    except asyncio.CancelledError:
        cancelled = True
    except Exception as exc:
        fatal_error = str(exc)[:500]
        summary['warnings'].append('批量判定未完成：' + fatal_error)
    finally:
        for record in records:
            if record['status'] in {'pending', 'processing'}:
                record.update(status='failed' if fatal_error else 'cancelled',
                              reason=fatal_error or '批量任务已取消；已生成的结果保留。')
        summary['status'] = ('cancelled' if cancelled else 'completed_with_issues'
                             if any(row['status'] == 'failed' for row in records) else 'completed')
        summary['finished_at'] = datetime.now(timezone.utc).isoformat()
        checkpoint()
        try:
            _html_report(summary)
        except OSError as exc:
            summary['warnings'].append('HTML 汇总写入失败：' + str(exc)[:200])
            checkpoint()
    return summary
