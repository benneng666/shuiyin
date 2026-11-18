"""Conservative fixed-text masks inside regions already approved for repair.

This is a mask refinement stage, never a watermark detector. Pixel transmission is
estimated from independent frames with approximately planar visible surroundings.
Unreliable evidence returns None so callers retain their original repair region.
"""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import cv2
import numpy as np

MAX_BOXES = 6
MAX_SAMPLES = 24
MAX_MASK_PIXELS = 65536
MIN_FIT_FRAMES = 8


def _stop(should_stop):
    if should_stop():
        raise asyncio.CancelledError()


def _hash(path, should_stop):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(1024 * 1024):
            _stop(should_stop)
            digest.update(chunk)
    return digest.hexdigest()


def fit_precision_mask(crops, box, *, should_stop=lambda: False):
    """Return (local uint8 mask or None, evidence) from RGB/BGR context crops.

    ``box`` is (x, y, width, height) inside every same-sized crop. No source clean
    reference or generated repair enters this function. The returned mask cannot
    select a pixel outside the caller's approved rectangle.
    """
    evidence = {'method': 'fixed-text-transmission-v1', 'accepted': False,
                'input_samples': len(crops), 'fallback': 'approved_rectangle'}

    def reject(reason):
        evidence['reason'] = reason
        return None, evidence

    _stop(should_stop)
    if len(crops) < MIN_FIT_FRAMES:
        return reject('too_few_samples')
    x, y, width, height = map(int, box)
    if width < 8 or height < 8 or width * height > MAX_MASK_PIXELS:
        return reject('unsupported_mask_size')
    shape = np.asarray(crops[0]).shape
    if len(shape) != 3 or shape[2] != 3 or any(np.asarray(c).shape != shape for c in crops):
        return reject('inconsistent_samples')
    h, w, _ = shape
    # All four visible sides are needed to estimate a meaningful background plane.
    if min(x, y, w-x-width, h-y-height) < 4:
        return reject('insufficient_visible_border')
    yy, xx = np.mgrid[:h, :w]
    selected = (xx >= x) & (xx < x+width) & (yy >= y) & (yy < y+height)
    ring = ~selected
    coords = np.stack((np.ones_like(xx), xx/max(w, 1), yy/max(h, 1)), axis=-1)
    design = coords[ring]
    inverse = np.linalg.pinv(design)
    values, backgrounds, residuals, seen = [], [], [], set()
    for crop in crops[:MAX_SAMPLES]:
        _stop(should_stop)
        array = np.asarray(crop, dtype=np.uint8)
        signature = hashlib.sha256(array.tobytes()).digest()
        if signature in seen:
            continue
        seen.add(signature)
        array = array.astype(np.float64)
        plane = coords @ (inverse @ array[ring])
        residual = float(np.percentile(np.abs(plane[ring]-array[ring]), 85))
        if residual > 8:
            continue
        values.append(array[selected])
        backgrounds.append(plane[selected])
        residuals.append(residual)
    evidence['independent_samples'] = len(seen)
    evidence['fit_samples'] = len(values)
    if len(values) < MIN_FIT_FRAMES:
        return reject('too_few_planar_independent_frames')
    a = np.asarray(backgrounds, dtype=np.float32)
    b = np.asarray(values, dtype=np.float32)
    ac, bc = a-a.mean(axis=0), b-b.mean(axis=0)
    variance = (ac**2).sum(axis=(0, 2))
    transmission = np.clip((ac*bc).sum(axis=(0, 2))/np.maximum(variance, 1), 0, 1)
    bias = b.mean(axis=0)-transmission[:, None]*a.mean(axis=0)
    predicted = transmission[None, :, None]*a+bias
    error = np.sqrt(((predicted-b)**2).mean(axis=(0, 2)))
    effect = np.abs(predicted-a).mean(axis=(0, 2))
    confidence = (error < 4) & (variance/len(a) > 100) & (effect > 3)
    alpha = np.where(confidence, 1-transmission, 0).reshape(height, width)
    raw = (alpha > .035).astype(np.uint8)
    coverage = float(raw.mean())
    mean_alpha = float(alpha[raw != 0].mean()) if raw.any() else 0.
    evidence.update(raw_fraction=round(coverage, 6),
                    background_std=float(np.sqrt(np.mean(variance)/(len(a)*3))),
                    planar_residual_mean=float(np.mean(residuals)))
    if not .025 <= coverage <= .78:
        return reject('no_sparse_fixed_text_support')
    if mean_alpha < .2:
        return reject('overlay_too_faint_for_precise_mask')
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    valid = np.flatnonzero(areas >= 4)+1
    evidence['components'] = len(valid)
    _, hierarchy = cv2.findContours(raw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0 if hierarchy is None else int((hierarchy[0, :, 3] >= 0).sum())
    evidence['enclosed_glyph_holes'] = holes
    # Outlined characters may touch through their black outlines. Several enclosed
    # character holes are acceptable evidence in that case; one solid logo is not.
    if len(valid) < 2 and holes < 3:
        return reject('not_multiple_text_components')
    if int(areas.max()) > .90 * int(raw.sum()) and holes < 3:
        return reject('dominant_single_shape')
    # Any broad/fixed edge reaching both opposite borders is not a glyph template.
    for label in valid:
        left, top, cw, ch, area = map(int, stats[label])
        if cw >= width-1 or ch >= height-1:
            return reject('support_reaches_opposite_borders')
    # Leave a tiny contextual margin for anti-aliasing and compression ringing.
    # Dilation remains clamped to the approved rectangle by array construction.
    mask = cv2.dilate(raw*255, np.ones((3, 3), dtype=np.uint8))
    if float((mask != 0).mean()) > .86:
        return reject('refinement_too_dense')
    outside = (mask.ravel() == 0)
    # A fitted character mask must explain nearly all persistent local contrast.
    # Otherwise isolated fragments would remain after refinement.
    unexplained = (np.abs(b-a).mean(axis=(0, 2)) > 7) & outside
    unexplained_fraction = float(unexplained.sum()/max(1, outside.sum()))
    evidence['unexplained_outside_fraction'] = unexplained_fraction
    if unexplained_fraction > .025:
        return reject('unexplained_fragments_outside_mask')
    evidence.update(accepted=True, reason='independent_planar_frames_support_fixed_text',
                    selected_pixels=int((mask != 0).sum()),
                    selected_fraction=float((mask != 0).mean()),
                    mean_detected_alpha=mean_alpha,
                    meaning='mask_refinement_only_not_watermark_identity')
    return mask, evidence


def detect_precision_masks(source, boxes, *, source_info=None, sample_times=None,
                           should_stop=lambda: False, emit=lambda message: None):
    """Sample a local video, returning one local mask/None per approved box.

    This stage deliberately refuses moving, fading, large and uncertain templates.
    It does not approve regions or broaden the user's selection.
    """
    from .watermark_video import _frame, _local_source, _probe

    path = _local_source(source)
    original_hash = _hash(path, should_stop)
    info = source_info or _probe(path, should_stop)
    width, height, duration = int(info['width']), int(info['height']), float(info['duration'])
    evidence = [{'accepted': False, 'reason': 'not_analyzed', 'fallback': 'approved_rectangle'} for _ in boxes]
    masks = [None for _ in boxes]
    crops = [[] for _ in boxes]
    valid = {}
    for i, box in enumerate(boxes):
        _stop(should_stop)
        try:
            x, y, bw, bh = [int(box[k]) for k in ('x', 'y', 'w', 'h')]
            start, end = float(box.get('start', 0)), float(box.get('end', duration))
        except (ValueError, TypeError, KeyError):
            evidence[i]['reason'] = 'invalid_box'
            continue
        if i >= MAX_BOXES or bw*bh > MAX_MASK_PIXELS or min(bw, bh) < 8:
            evidence[i]['reason'] = 'unsupported_mask_size_or_count'
            continue
        if x < 5 or y < 5 or x+bw+5 > width or y+bh+5 > height:
            evidence[i]['reason'] = 'insufficient_visible_border'
            continue
        if not (0 <= start < end <= duration):
            evidence[i]['reason'] = 'invalid_interval'
            continue
        valid[i] = (x, y, bw, bh, start, end)
    if valid:
        emit('分析固定文字的轮廓，减少对水印周围画面的改动…')
        # Container duration can extend past the final video PTS because of audio.
        if sample_times is None:
            tail_margin = max(.1, 2/max(1, float(info.get('fps') or 25)))
            sample_times = np.linspace(0, max(0, duration-tail_margin), MAX_SAMPLES)
        else:
            sample_times = sorted(set(float(t) for t in sample_times))
            if any(not np.isfinite(t) or not 0 <= t < duration for t in sample_times):
                raise ValueError('文字轮廓采样时间必须位于视频范围内')
            if len(sample_times) > MAX_SAMPLES:
                sample_times = [sample_times[i] for i in np.linspace(0, len(sample_times)-1, MAX_SAMPLES).astype(int)]
        for t in sample_times:
            _stop(should_stop)
            active = [(i, b) for i, b in valid.items() if b[4] <= t < b[5]]
            if not active:
                continue
            frame = np.asarray(_frame(path, float(t), info, should_stop))
            for i, (x, y, bw, bh, _, _) in active:
                crops[i].append(frame[y-5:y+bh+5, x-5:x+bw+5].copy())
        for i, (_, _, bw, bh, _, _) in valid.items():
            masks[i], evidence[i] = fit_precision_mask(crops[i], (5, 5, bw, bh), should_stop=should_stop)
    if original_hash != _hash(path, should_stop):
        raise ValueError('视频源文件在文字轮廓分析期间发生变化，请重新加载')
    return {'masks': masks, 'evidence': evidence, 'source_sha256': original_hash,
            'sample_budget': MAX_SAMPLES, 'method': 'fixed-text-transmission-v1'}
