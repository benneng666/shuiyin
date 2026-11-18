"""Reviewable OCR candidates and conservative CPU image inpainting.

No model or native image library is loaded at import time. Detection identifies
possible platform/account overlays, not ownership, and never modifies a file.
The caller must present candidates for review. Telea reconstructs nearby colors;
it cannot recover the original pixels underneath an overlay.

Coordinates refer to the EXIF-transposed, full-resolution image. Outside the
approved rectangle union, decoded RGBA pixels are copied exactly. Alpha is kept
everywhere, including inside the mask. Animated/multiframe images are rejected.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import math
import numbers
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
import unicodedata
from datetime import datetime, timezone
from collections.abc import Mapping
from uuid import uuid4


ALGORITHM_VERSION = "adaptive-rgba-mask-v4"
INPAINT_RADIUS = 3.0
MAX_PIXELS = 40_000_000
MAX_SOURCE_BYTES = 150 * 1024 * 1024
MAX_REGIONS = 256
OCR_TILE_SIZE = 1600
OCR_TILE_OVERLAP = 160
MAX_OCR_TILES = 64
_OCR_ENGINE = None
_OCR_LOCK = threading.Lock()
_REVIEW_WARNING = "自动检测仅提供候选，请逐框审核；普通文字、商品图案也可能被误认，未识别到不代表没有水印。"
_CONFIDENCE_WARNING = "confidence 仅表示 OCR 对识别文字的置信度，不是水印概率。"
_PLATFORM = re.compile(r"抖音|小红书|淘宝|天猫|(?<![a-z])(?:rednote|red|douyin|xiaohongshu|taobao|tmall)(?![a-z])", re.I)
_ACCOUNT = re.compile(r"(?:^|\s)@[\w.\-]{3,}|(?:抖音号|小红书号|账号|帐号|用户\s*ID|\bID)\s*[:：]\s*[\w.\-]{3,}", re.I)
_PRODUCT_COPY = re.compile(r"同款|旗舰店|官方(?:店|授权)|天猫精灵|订单|包邮|折扣|商品|价格|规格|材质")
_MERCHANT = re.compile(r"旗舰店|专卖店|专营店|直营店|品牌授权店|官方店铺|官方商城|(?:商家|店铺)(?:水印|标识)|\b(?:official\s*store|flagship\s*store)\b", re.I)
_PARAMETERS = re.compile(r"规格|材质|成分|尺寸|重量|容量|型号|货号|参数|长度|宽度|高度|厚度|尺码|售价|价格|\d\s*(?:cm|mm|kg|ml|厘米|毫米|千克|毫升)\b", re.I)


def _detection_options(merchant_terms, detection_profile):
    if not isinstance(detection_profile, str) or detection_profile not in {"platform", "product"}:
        raise ValueError("检测模式必须为 platform 或 product。")
    if not isinstance(merchant_terms, str) or len(merchant_terms) > 1024:
        raise ValueError("商家关键词须为不超过 1024 字的文本。")
    terms = list(dict.fromkeys(re.sub(r"\s+", "", unicodedata.normalize("NFKC", s)).casefold()
                              for s in re.split(r"[,，;；、\r\n]+", merchant_terms) if s.strip()))
    if len(terms) > 32 or any(not 2 <= len(t) <= 64 for t in terms):
        raise ValueError("商家关键词最多 32 个，每个须为 2 至 64 字；请用逗号、分号或换行分隔。")
    return terms


def _keyword_match(text, terms):
    spaced = unicodedata.normalize("NFKC", text).casefold()
    compact = re.sub(r"\s+", "", spaced)
    return any((re.search(r"(?<![a-z0-9])" + r"\s*".join(map(re.escape, term)) + r"(?![a-z0-9])", spaced)
                if term.isascii() else term in compact) for term in terms)


def _check_stop(should_stop):
    if should_stop():
        raise asyncio.CancelledError()


def _emit(emit, message):
    # A closed log view must not turn a successfully written image into failure.
    try:
        emit(message)
    except Exception:
        pass


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限数值。")
    return float(value)


def _union_area(regions):
    """Exact rectangle union, independent of NumPy and image allocation."""
    xs = sorted({n for r in regions for n in (r["x"], r["x"] + r["w"])})
    area = 0
    for left, right in zip(xs, xs[1:]):
        spans = sorted((r["y"], r["y"] + r["h"]) for r in regions
                       if r["x"] < right and r["x"] + r["w"] > left)
        length, start, end = 0, None, None
        for y0, y1 in spans:
            if start is None:
                start, end = y0, y1
            elif y0 > end:
                length += end - start
                start, end = y0, y1
            else:
                end = max(end, y1)
        if start is not None:
            length += end - start
        area += (right - left) * length
    return area


def validate_regions(regions, width, height) -> list[dict]:
    """Normalize spatial boxes; reject out-of-bounds and >40% union masks.

    Fractional GUI coordinates round outward. Optional label/confidence survive;
    start/end are deliberately ignored for a still image, as are unknown keys.
    An empty list is valid for editing/detection; repair_image rejects it.
    """
    if (isinstance(width, bool) or isinstance(height, bool)
            or not isinstance(width, numbers.Integral) or not isinstance(height, numbers.Integral)
            or width <= 0 or height <= 0):
        raise ValueError("图片宽高必须为正整数。")
    if not isinstance(regions, (list, tuple)) or len(regions) > MAX_REGIONS:
        raise ValueError(f"遮罩必须是区域列表，最多 {MAX_REGIONS} 个区域。")
    normalized, seen = [], set()
    for index, region in enumerate(regions, 1):
        if not isinstance(region, Mapping) or not all(k in region for k in ("x", "y", "w", "h")):
            raise ValueError(f"第 {index} 个区域缺少 x、y、w、h。")
        x, y, w, h = (_number(region[k], f"第 {index} 个区域的 {k}") for k in ("x", "y", "w", "h"))
        if w <= 0 or h <= 0:
            raise ValueError(f"第 {index} 个区域的宽高必须大于零。")
        if x < 0 or y < 0 or x + w > width or y + h > height:
            raise ValueError(f"第 {index} 个区域超出图片边界，请重新框选。")
        x0, y0, x1, y1 = math.floor(x), math.floor(y), math.ceil(x + w), math.ceil(y + h)
        key = (x0, y0, x1 - x0, y1 - y0)
        if key in seen:
            continue
        seen.add(key)
        item = dict(zip(("x", "y", "w", "h"), key))
        if isinstance(region.get("label"), str):
            item["label"] = re.sub(r"[\x00-\x1f\x7f]", " ", region["label"])[:160]
        if "confidence" in region:
            confidence = _number(region["confidence"], f"第 {index} 个区域的置信度")
            if not 0 <= confidence <= 1:
                raise ValueError("OCR 文字置信度必须在 0 至 1 之间。")
            item["confidence"] = confidence
        # A designed colour strip is rebuilt from its own row colours instead
        # of being generated by an inpaint model. Allow-listed, never inferred.
        if region.get("fill") == "row_band":
            item["fill"] = "row_band"
        normalized.append(item)
    if _union_area(normalized) * 5 > int(width) * int(height) * 2:
        raise ValueError("遮罩覆盖面积不能超过图片的 40%；请缩小或减少区域。大面积 Telea 修复无法可靠还原内容。")
    return normalized


def _load_image(path):
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        source = Path(path).expanduser().resolve(strict=True)
        if not source.is_file() or source.stat().st_size > MAX_SOURCE_BYTES:
            raise ValueError("请选择不超过 150 MB 的本地图片文件。")
        # Hash exactly the byte snapshot being processed, even if another app
        # subsequently changes the source path. Never reopen it for writing.
        with source.open("rb") as stream:
            raw = stream.read(MAX_SOURCE_BYTES + 1)
        if len(raw) > MAX_SOURCE_BYTES:
            raise ValueError("图片文件超过 150 MB，暂不支持。")
        with Image.open(io.BytesIO(raw)) as opened:
            if getattr(opened, "n_frames", 1) != 1 or getattr(opened, "is_animated", False):
                raise ValueError("暂不支持动画或多帧图片；请先选择单帧静态图片。")
            if opened.width * opened.height > MAX_PIXELS:
                raise ValueError("图片超过 4000 万像素，暂不支持 CPU 修复。")
            if opened.mode not in {"RGB", "RGBA", "L", "LA", "P", "1"}:
                raise ValueError("暂不支持该图片色彩模式；请先另存为普通 RGB 或 RGBA 图片，保留原件。")
            opened.load()
            orientation = opened.getexif().get(274, 1)
            original_mode = opened.mode
            icc = opened.info.get("icc_profile") if original_mode in {"RGB", "RGBA", "P"} else None
            image = ImageOps.exif_transpose(opened).convert("RGBA")
        metadata = {"path": str(source), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                    "original_mode": original_mode, "exif_orientation": orientation,
                    "orientation_normalized": True}
        return image, metadata, icc
    except (FileNotFoundError, IsADirectoryError, PermissionError, OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"无法读取图片（{type(exc).__name__}），请检查文件是否有效及是否可访问。") from None
    except Image.DecompressionBombError:
        raise ValueError("图片像素数量过大，暂不支持处理。") from None


def _get_ocr():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _OCR_ENGINE = RapidOCR(
                det_use_cuda=False, cls_use_cuda=False, rec_use_cuda=False,
                det_use_dml=False, cls_use_dml=False, rec_use_dml=False,
                intra_op_num_threads=2, inter_op_num_threads=1, print_verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(f"自动检测模型初始化失败（{type(exc).__name__}）；请检查 OCR 组件，可继续手动框选。") from None
    return _OCR_ENGINE


def _candidate_regions(ocr_rows, width, height, include_corner_text=False, merchant_terms='', detection_profile='platform'):
    terms = _detection_options(merchant_terms, detection_profile)
    candidates, seen = [], set()
    for row in ocr_rows or []:
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue
        polygon, text, score = row[:3]
        if not isinstance(text, str):
            continue
        try:
            score = _number(score, "OCR 文字置信度")
            if not .65 <= score <= 1 or len(polygon) != 4:
                continue
            points = [(_number(p[0], "OCR x"), _number(p[1], "OCR y")) for p in polygon if len(p) == 2]
            if len(points) != 4:
                continue
        except (ValueError, TypeError, IndexError):
            continue
        x0, x1 = min(p[0] for p in points), max(p[0] for p in points)
        y0, y1 = min(p[1] for p in points), max(p[1] for p in points)
        if x1 <= x0 or y1 <= y0 or x1 < 0 or y1 < 0 or x0 >= width or y0 >= height:
            continue
        compact = re.sub(r"\s+", "", text)
        account = bool(_ACCOUNT.search(text))
        platform = bool(_PLATFORM.search(compact))
        merchant = detection_profile == "product" and bool(_MERCHANT.search(text)) and score >= .78
        keyword = detection_profile == "product" and _keyword_match(text, terms) and score >= .78
        corner = (x0 <= width * .20 or x1 >= width * .80) and (y0 <= height * .22 or y1 >= height * .78)
        # Colored RED product copy is not a platform marker by itself.
        if re.search(r"\bred\b", text, re.I) and not re.fullmatch(r"\s*red\s*", text, re.I) and not account:
            platform = False
        if detection_profile == "product" and _PARAMETERS.search(compact):
            continue
        if not (merchant or keyword or account) and _PRODUCT_COPY.search(compact):
            continue
        if len(compact) > 64 or (x1 - x0) * (y1 - y0) > width * height * .12:
            continue
        extra_corner = include_corner_text and corner and score >= .82 and len(compact) >= 2
        if not (merchant or keyword or platform and (corner or account) or account and corner and score >= .75 or extra_corner):
            continue
        padding = max(2, min(8, math.ceil((y1 - y0) * .08)))
        left, top = max(0, math.floor(x0) - padding), max(0, math.floor(y0) - padding)
        right, bottom = min(width, math.ceil(x1) + padding), min(height, math.ceil(y1) + padding)
        if right <= left or bottom <= top:
            continue
        key = (left, top, right - left, bottom - top)
        if key in seen:
            continue
        seen.add(key)
        label = ("商家关键词候选：" if keyword else "店铺标记候选：" if merchant else "平台标记候选：" if platform else "边角账号候选：" if account else "边角文字候选：") + re.sub(r"[\x00-\x1f\x7f]", " ", text)[:60]
        candidates.append(dict(zip(("x", "y", "w", "h"), key), label=label, confidence=round(score, 6)))
    return candidates


def _ocr_tiles(width, height):
    step = OCR_TILE_SIZE - OCR_TILE_OVERLAP
    def positions(length):
        count = max(1, math.ceil((length - OCR_TILE_SIZE) / step) + 1)
        return range(0, count * step, step)
    for y in positions(height):
        for x in positions(width):
            yield (x, y, min(width, x + OCR_TILE_SIZE), min(height, y + OCR_TILE_SIZE))


def _merge_candidates(regions, additions):
    dropped = 0
    for row in additions:
        duplicate = None
        for old in regions:
            if re.sub(r"\s", "", old["label"]).casefold() != re.sub(r"\s", "", row["label"]).casefold():
                continue
            ix = max(0, min(old['x'] + old['w'], row['x'] + row['w']) - max(old['x'], row['x']))
            iy = max(0, min(old['y'] + old['h'], row['y'] + row['h']) - max(old['y'], row['y']))
            if ix * iy >= .7 * min(old['w'] * old['h'], row['w'] * row['h']):
                duplicate = old; break
        if duplicate is None:
            if len(regions) < MAX_REGIONS:
                regions.append(row)
            else:
                dropped += 1
        else:
            x, y = min(duplicate['x'], row['x']), min(duplicate['y'], row['y'])
            right = max(duplicate['x'] + duplicate['w'], row['x'] + row['w'])
            bottom = max(duplicate['y'] + duplicate['h'], row['y'] + row['h'])
            duplicate.update(x=x, y=y, w=right-x, h=bottom-y, confidence=max(duplicate['confidence'], row['confidence']))
    return dropped


def _batch_text_regions(rows, width, height):
    """Bounded raw OCR evidence for batch analysis, never approval to erase."""
    result = []
    for row in rows or []:
        try:
            polygon, text, score = row[:3]
            if not isinstance(text, str) or not 2 <= len(text.strip()) <= 128 or len(polygon) != 4:
                continue
            score = _number(score, "OCR 文字置信度")
            points = [(_number(p[0], "x"), _number(p[1], "y")) for p in polygon]
            if not .5 <= score <= 1:
                continue
            left, top = max(0, math.floor(min(p[0] for p in points))), max(0, math.floor(min(p[1] for p in points)))
            right, bottom = min(width, math.ceil(max(p[0] for p in points))), min(height, math.ceil(max(p[1] for p in points)))
            if right <= left or bottom <= top:
                continue
            result.append({'x': left, 'y': top, 'w': right-left, 'h': bottom-top,
                           'label': 'OCR文字：' + text.strip(), 'confidence': round(score, 6)})
        except (TypeError, ValueError, IndexError):
            continue
    return result


def detect_image(path, *, should_stop=lambda: False, emit=lambda msg: None, include_corner_text=False,
                 merchant_terms='', detection_profile='platform', return_text_regions=False) -> dict:
    """Return reviewable pixel regions; confidence is OCR text confidence only.

    CPU inference is synchronous and serialized. Cancellation is checked before
    and after inference; it cannot interrupt an active ONNX native call.
    """
    _check_stop(should_stop)
    _detection_options(merchant_terms, detection_profile)
    image, source_info, _ = _load_image(path)
    width, height = image.size
    _emit(emit, "正在用 CPU 识别可能的水印文字；结果需要人工审核。")
    try:
        import numpy as np
        from PIL import Image
        # Composite onto white so invisible RGB under transparent pixels cannot
        # become false text candidates. Repair itself retains the original alpha.
        regions, tiles_done, limited, candidate_drops = [], 0, False, 0
        text_regions, text_drops = [], 0
        for index, bounds in enumerate(_ocr_tiles(width, height)):
            _check_stop(should_stop)
            if index >= MAX_OCR_TILES:
                limited = True; break
            tile = image.crop(bounds)
            matte = Image.new("RGBA", tile.size, (255, 255, 255, 255))
            matte.alpha_composite(tile)
            bgr = np.asarray(matte.convert("RGB"))[:, :, ::-1].copy()
            while not _OCR_LOCK.acquire(timeout=.1):
                _check_stop(should_stop)
            try:
                _check_stop(should_stop)
                engine = _get_ocr()
                _check_stop(should_stop)
                rows, _timings = engine(bgr)
            finally:
                _OCR_LOCK.release()
            _check_stop(should_stop)
            translated = []
            for row in rows or []:
                try:
                    polygon = [[p[0] + bounds[0], p[1] + bounds[1]] for p in row[0]]
                    translated.append([polygon, row[1], row[2]])
                except (TypeError, IndexError):
                    continue
            candidate_drops += _merge_candidates(regions, _candidate_regions(translated, width, height,
                include_corner_text=include_corner_text, merchant_terms=merchant_terms, detection_profile=detection_profile))
            if return_text_regions:
                text_drops += _merge_candidates(text_regions, _batch_text_regions(translated, width, height))
            tiles_done += 1
            if width > OCR_TILE_SIZE or height > OCR_TILE_SIZE:
                _emit(emit, f"已完成第 {tiles_done} 块文字检测。")
    except Exception as exc:
        raise RuntimeError(f"自动检测失败（{type(exc).__name__}），可继续手动框选。") from None
    warnings = [_REVIEW_WARNING, _CONFIDENCE_WARNING]
    if detection_profile == 'product':
        warnings.append("商家模式仅按店铺文字及指定关键词提出候选；实物品牌标牌、商品印花也可能命中，必须审核。非文字 Logo 仍需手动框选。")
    if limited:
        warnings.append(f"已达到 {MAX_OCR_TILES} 块 OCR 上限，剩余图片区域尚未扫描；可分段另存后继续检测。")
    if candidate_drops:
        warnings.append(f"候选超过 {MAX_REGIONS} 个，已限制返回数量；请分段审核，不能认定已列出全部候选。")
    if include_corner_text:
        warnings.append("已包含边角品牌/普通文字候选，这些不一定是水印；请删去需要保留的宣传文字或商品标识。")
    if not regions:
        warnings.append("未找到符合当前检测规则的候选；弧形文字、图形 Logo 或低清文字可能漏检，可手动添加区域。")
    if len(regions) > MAX_REGIONS or _union_area(regions) * 5 > width * height * 2:
        warnings.append("候选区域过多或覆盖超过 40%，修复前需要删除误检并缩小遮罩。")
    _emit(emit, f"识别到 {len(regions)} 个候选区域，请审核后修复。")
    result = {"width": width, "height": height, "regions": regions, "warnings": warnings,
            "detection_profile": detection_profile, "ocr_tiles": tiles_done, "scan_complete": not limited,
            "candidates_truncated": bool(candidate_drops), "source_sha256": source_info['sha256']}
    if return_text_regions:
        result.update(text_regions=text_regions, text_regions_truncated=bool(text_drops),
                      source_sha256=source_info['sha256'])
    return result


def _flat_boundary(original, mask, region):
    """Use only opaque original pixels outside *all* approved masks.

    At least one horizontal and one vertical side must provide enough context.
    A strict maximum channel deviation avoids extrapolating through texture or
    a gradient. This is a conservative heuristic, not hidden-pixel recovery.
    """
    import numpy as np
    x, y, w, h = (region[k] for k in ('x', 'y', 'w', 'h'))
    height, width = mask.shape
    if w * h > width * height * .08:
        return None, {'reason': 'large_region'}
    radius = max(3, min(8, math.ceil(min(w, h) * .15)))
    sides = [(max(0, x-radius), y, x, y+h), (x+w, y, min(width, x+w+radius), y+h),
             (x, max(0, y-radius), x+w, y), (x, y+h, x+w, min(height, y+h+radius))]
    samples, supported = [], []
    for left, top, right, bottom in sides:
        local = original[top:bottom, left:right]
        allowed = (mask[top:bottom, left:right] == 0) & (local[:, :, 3] == 255)
        pixels = local[:, :, :3][allowed]
        # Excluding a neighboring mask must leave meaningful context, not one
        # clean pixel that accidentally happens to have a convenient color.
        enough = len(pixels) >= max(8, (right-left) * (bottom-top) * .5)
        supported.append(enough)
        if enough: samples.append(pixels)
    if not any(supported[:2]) or not any(supported[2:]):
        return None, {'reason': 'insufficient_opaque_boundary'}
    pixels = np.concatenate(samples).astype(np.int16)
    center = np.median(pixels, axis=0)
    deviation = int(np.max(np.abs(pixels-center)))
    stats = {'boundary_pixels': len(pixels), 'max_channel_deviation': deviation}
    if deviation > 8:
        return None, dict(stats, reason='textured_or_gradient_boundary')
    return np.rint(center).astype(np.uint8), dict(stats, reason='nearly_uniform_boundary')


def _directional_background(original, mask, region):
    """Validate an axis-aligned low-variation continuation from independent edges.

    A pair of matching strips alone is insufficient: the predicted fill must
    also agree with at least one sufficiently clean perpendicular boundary.
    Every sample is from the original image, outside all approved masks.
    """
    import numpy as np
    x, y, w, h = (region[k] for k in ('x', 'y', 'w', 'h'))
    if w * h > mask.size * .08 or min(w, h) < 4:
        return None, {'directional_reason': 'large_or_narrow_region'}
    radius = 8
    accepted = []
    for direction in ('vertical', 'horizontal'):
        # Transposition gives the horizontal path identical validation rules.
        image, excluded = (original, mask) if direction == 'vertical' else (original.transpose(1, 0, 2), mask.T)
        xx, yy, ww, hh = (x, y, w, h) if direction == 'vertical' else (y, x, h, w)
        if yy < radius or yy + hh + radius > image.shape[0]:
            continue
        profiles, noise, sample_counts = [], [], []
        for start, end in ((yy-radius, yy), (yy+hh, yy+hh+radius)):
            strip = image[start:end, xx:xx+ww]
            clean = (excluded[start:end, xx:xx+ww] == 0) & (strip[:, :, 3] == 255)
            counts = clean.sum(axis=0)
            if np.any(counts < radius * .75):
                break
            samples = np.where(clean[:, :, None], strip[:, :, :3].astype(np.float32), np.nan)
            profile = np.nanmedian(samples, axis=0)
            deviations = np.abs(samples - profile[None, :, :])[np.broadcast_to(clean[:, :, None], samples.shape)]
            profiles.append(profile); noise.append(deviations); sample_counts.append(int(counts.sum()))
        if len(profiles) != 2:
            continue
        noise = np.concatenate(noise)
        noise_p95, noise_max = float(np.percentile(noise, 95)), float(noise.max())
        if noise_p95 > 4 or noise_max > 8:
            continue
        first, last = profiles
        delta = np.abs(last - first)
        delta_p95, delta_max = float(np.percentile(delta, 95)), float(delta.max())
        if delta_p95 > 12 or delta_max > 20:
            continue
        # Preserve real column/row structure only if both sampled edges agree
        # on it. For near-uniform profiles there is no useful correlation.
        profile_range = max(float(np.ptp(first, axis=0).max()), float(np.ptp(last, axis=0).max()))
        correlation = None
        if profile_range >= 12:
            centered_first, centered_last = first-first.mean(axis=0), last-last.mean(axis=0)
            denominator = float(np.sqrt(np.sum(centered_first**2) * np.sum(centered_last**2)))
            correlation = float(np.sum(centered_first * centered_last) / denominator) if denominator else 0.0
            if correlation < .96:
                continue
        weights = (np.arange(hh, dtype=np.float32) + (radius+1)/2) / (hh+radius)
        cross_errors, cross_samples = [], 0
        for column, profile_index in ((xx-1, 0), (xx+ww, -1)):
            if column < 0 or column >= image.shape[1]:
                continue
            boundary = image[yy:yy+hh, column]
            clean = (excluded[yy:yy+hh, column] == 0) & (boundary[:, 3] == 255)
            if int(clean.sum()) < max(8, hh * .8):
                continue
            prediction = first[profile_index][None, :] * (1-weights[:, None]) + last[profile_index][None, :] * weights[:, None]
            cross_errors.append(np.abs(boundary[:, :3].astype(np.float32) - prediction)[clean])
            cross_samples += int(clean.sum())
        if not cross_errors:
            continue
        cross_errors = np.concatenate(cross_errors)
        cross_p95, cross_max = float(np.percentile(cross_errors, 95)), float(cross_errors.max())
        if cross_p95 > 10 or cross_max > 16:
            continue
        stats = {'reason': 'consistent_low_variation_directional_boundaries',
                 'direction': direction, 'strip_radius': radius, 'strip_sample_counts': sample_counts,
                 'strip_deviation_p95': round(noise_p95, 4), 'strip_deviation_max': round(noise_max, 4),
                 'endpoint_difference_p95': round(delta_p95, 4), 'endpoint_difference_max': round(delta_max, 4),
                 'profile_range': round(profile_range, 4),
                 'profile_correlation': round(correlation, 6) if correlation is not None else None,
                 'cross_boundary_samples': cross_samples,
                 'cross_boundary_error_p95': round(cross_p95, 4), 'cross_boundary_error_max': round(cross_max, 4)}
        plan = {'direction': direction, 'first': first, 'last': last, 'radius': radius}
        accepted.append((cross_p95 + delta_p95 + 2*noise_p95, plan, stats))
    if not accepted:
        return None, {'directional_reason': 'no_consistent_clean_direction'}
    _, plan, stats = min(accepted, key=lambda value: value[0])
    return plan, stats


def _directional_pixels(plan, width, height):
    import numpy as np
    length = height if plan['direction'] == 'vertical' else width
    weights = (np.arange(length, dtype=np.float32) + (plan['radius']+1)/2) / (length+plan['radius'])
    values = plan['first'][None, :, :] * (1-weights[:, None, None]) + plan['last'][None, :, :] * weights[:, None, None]
    pixels = np.clip(np.rint(values), 0, 255).astype(np.uint8)
    return pixels if plan['direction'] == 'vertical' else pixels.transpose(1, 0, 2)


def repair_image(source, regions, output_root=None, should_stop=lambda: False, emit=lambda msg: None,
                 method='auto') -> dict:
    """Write a unique derived PNG/mask/manifest transaction; never overwrite source.

    The approved rectangle union is the exact mask (no hidden dilation). The
    entire alpha channel and all RGBA pixels outside it are kept unchanged.
    A failed/cancelled attempt removes its own staging directory only.
    """
    _check_stop(should_stop)
    if not isinstance(method, str) or method not in {'auto', 'telea', 'ns', 'lama'}:
        raise ValueError("修复方法必须为 auto、telea、ns 或 lama。")
    image, source_info, icc = _load_image(source)
    width, height = image.size
    approved = validate_regions(regions, width, height)
    if not approved:
        raise ValueError("请至少添加并确认一个修复区域。")
    _check_stop(should_stop)
    stage, base = None, None
    try:
        import cv2
        import numpy as np
        from PIL import Image
        original = np.asarray(image).copy()
        mask = np.zeros((height, width), dtype=np.uint8)
        for region in approved:
            x, y, w, h = (region[k] for k in ("x", "y", "w", "h"))
            mask[y:y + h, x:x + w] = 255
        _emit(emit, "正在修复已确认区域；原始图片会保留。")
        _check_stop(should_stop)
        result = original.copy()
        region_methods, flat, directional = [], [], []
        ai_info = None
        row_bands = [region for region in approved if region.get('fill') == 'row_band']
        ai_mask = mask.copy()
        for region in row_bands:
            x, y, w, h = (region[k] for k in ("x", "y", "w", "h"))
            ai_mask[y:y+h, x:x+w] = 0
        if method == 'lama':
            from .watermark_ai import inpaint_rgb
            # Do not expose invisible RGB as model context. Final composition
            # still retains the exact original alpha and exterior RGBA bytes.
            context_rgb = original[:, :, :3].copy()
            source_pixels = original.reshape(-1, 4)
            context_pixels = context_rgb.reshape(-1, 3)
            # Bound float temporaries even for a very wide/long detail image.
            # Opaque images use the RGB copy directly, without float conversion.
            for offset in range(0, len(source_pixels), 262_144):
                _check_stop(should_stop)
                pixels = source_pixels[offset:offset + 262_144]
                if np.all(pixels[:, 3] == 255):
                    continue
                alpha = pixels[:, 3:4].astype(np.float32) / 255
                context_pixels[offset:offset + len(pixels)] = np.rint(
                    pixels[:, :3] * alpha + 255 * (1-alpha)).astype(np.uint8)
            filled, ai_info = inpaint_rgb(context_rgb, ai_mask, should_stop=should_stop, emit=emit)
            _check_stop(should_stop)
            result[ai_mask != 0, :3] = filled[ai_mask != 0]
            region_methods = [{'region_index': index,
                               'method': 'row_band_fill' if region.get('fill') == 'row_band' else 'lama_onnx_cpu'}
                              for index, region in enumerate(approved)]
        for index, region in ([] if method == 'lama' else enumerate(approved)):
            _check_stop(should_stop)
            x, y, w, h = (region[k] for k in ('x', 'y', 'w', 'h'))
            color, plan, details = None, None, {}
            if method == 'auto':
                overlap = any(i != index and max(x, other['x']) < min(x+w, other['x']+other['w'])
                              and max(y, other['y']) < min(y+h, other['y']+other['h'])
                              for i, other in enumerate(approved))
                if overlap:
                    details = {'reason': 'overlapping_regions'}
                else:
                    color, details = _flat_boundary(original, mask, region)
                    if color is None:
                        _check_stop(should_stop)
                        plan, directional_details = _directional_background(original, mask, region)
                        details.update(directional_details)
            used = 'flat_color' if color is not None else (plan['direction'] + '_interpolation' if plan else ('ns' if method == 'ns' else 'telea'))
            record = {'region_index': index, 'method': used, **details}
            if color is not None:
                record['color_rgb'] = color.tolist()
                flat.append((region, color))
            if plan is not None:
                directional.append((region, plan))
            region_methods.append(record)
        if any(r['method'] in {'telea', 'ns'} for r in region_methods):
            bgr = original[:, :, :3][:, :, ::-1].copy()
            flag = cv2.INPAINT_NS if method == 'ns' else cv2.INPAINT_TELEA
            # Even automatically filled boxes remain masked in this operation;
            # their original watermarks must not contaminate another region.
            filled = cv2.inpaint(bgr, mask, INPAINT_RADIUS, flag)[:, :, ::-1]
            _check_stop(should_stop)
            result[mask != 0, :3] = filled[mask != 0]
        for region, color in flat:
            _check_stop(should_stop)
            x, y, w, h = (region[k] for k in ('x', 'y', 'w', 'h'))
            result[y:y+h, x:x+w, :3] = color
        for region, plan in directional:
            _check_stop(should_stop)
            x, y, w, h = (region[k] for k in ('x', 'y', 'w', 'h'))
            result[y:y+h, x:x+w, :3] = _directional_pixels(plan, w, h)
        # A designed colour strip (black band + promo bar) is nearly uniform
        # along each row. Generative inpainting of such a large flat area leaves
        # a visible rectangle seam; rebuild it from that row's own real colour.
        for region in row_bands:
            _check_stop(should_stop)
            x, y, w, h = (region[k] for k in ('x', 'y', 'w', 'h'))
            margin = max(1, min(24, width // 8))
            ramp = np.linspace(0, 1, w, dtype=np.float32)[:, None]
            # A whole row of lettering can dominate its own median. Compare
            # with the rows directly above/below and use their colour instead.
            context = original[max(0, y-1):min(height, y+h+1), :, :3].astype(np.int16)
            medians = np.median(context, axis=1)
            for offset, row_index in enumerate(range(y, y+h), start=1 if y > 0 else 0):
                _check_stop(should_stop)
                row = original[row_index, :, :3].astype(np.int16)
                base = medians[offset]
                neighbours = [medians[index] for index in (offset-1, offset+1)
                              if 0 <= index < len(medians)]
                if neighbours and np.abs(base - np.mean(neighbours, axis=0)).mean() > 35:
                    base = np.mean(neighbours, axis=0)
                left = row[max(0, x-margin):x]
                right = row[x+w:min(width, x+w+margin)]
                first = np.median(left, axis=0) if len(left) >= 4 else base
                last = np.median(right, axis=0) if len(right) >= 4 else base
                if np.abs(first-base).mean() > 45:
                    first = base
                if np.abs(last-base).mean() > 45:
                    last = base
                result[row_index, x:x+w, :3] = np.clip(
                    first[None, :] * (1-ramp) + last[None, :] * ramp, 0, 255).astype(np.uint8)
        _check_stop(should_stop)
        # Explicit composition, instead of trusting the algorithm to preserve
        # the mask exterior. The alpha plane is never passed into inpaint.
        base = (Path(output_root) if output_root is not None else Path(source_info["path"]).parent / "processed")
        base = base.expanduser().resolve()
        base.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".image-repair-", dir=base)).resolve()
        mask_path, result_path = stage / "mask.png", stage / "result.png"
        Image.fromarray(mask).save(mask_path, format="PNG")
        options = {"icc_profile": icc} if icc else {}
        Image.fromarray(result).save(result_path, format="PNG", **options)
        _check_stop(should_stop)
        # Ensure both derived files decode before publishing a success manifest.
        for path in (mask_path, result_path):
            with Image.open(path) as check:
                check.load()
                if check.size != (width, height):
                    raise RuntimeError("输出尺寸校验失败。")
        warnings = (["AI 根据附近画面生成选区内容，并非找回被遮挡的真实像素；可能生成不真实纹理，请检查完整图片。",
                     "LaMa 以 512×512 分块在全分辨率下推理，不再整体缩放；贴边选区会补镜像上下文；选区边缘完全使用修复结果，不混回原水印像素。"]
                    if method == 'lama' else
                    ["修复仅依据选区外的邻近颜色推测，无法恢复被遮挡的原始内容；请检查边缘和纹理。"])
        if row_bands:
            warnings.append("设计色带按每行真实底色与左右边界重建，未使用生成式修复；该色带内的叠加文字与图案会被移除。")
        if flat:
            warnings.append("部分选区使用近纯色边界填充；这是背景估计，请确认选区没有包含需要保留的商品细节。")
        if directional:
            warnings.append("部分选区按上下或左右边界作方向插值，适合延续一致的条纹或浅渐变；请检查是否误盖商品细节。")
        if np.any(original[:, :, 3] != 255):
            warnings.append("透明度已完整保留；透明层本身的水印或轮廓不会被修复。")
        if _union_area(approved) > width * height * .10:
            warnings.append("遮罩超过图片的 10%，大面积或复杂纹理修补可能出现明显模糊。")
        manifest = {
            "schema_version": 1, "status": "completed", "media_type": "image",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": source_info, "source_sha256": source_info["sha256"],
            "algorithm": {'auto': 'boundary_flat_color_or_directional_interpolation_or_opencv.INPAINT_TELEA', 'telea': 'opencv.INPAINT_TELEA', 'ns': 'opencv.INPAINT_NS',
                          'lama': 'Carve.LaMa.ONNX.CPU.ROI.mask_only'}[method],
            "algorithm_version": ALGORITHM_VERSION, "requested_method": method, "region_methods": region_methods,
            "ai_model": ai_info,
            "opencv_version": cv2.__version__, "inpaint_radius": INPAINT_RADIUS,
            "width": width, "height": height, "coordinate_space": "exif_normalized_pixels",
            "regions": approved, "mask_area_pixels": _union_area(approved),
            "mask_area_ratio": _union_area(approved) / (width * height),
            "outside_mask_pixels_preserved": True, "alpha_preserved": True,
            "output": {"file": "result.png", "sha256": hashlib.sha256(result_path.read_bytes()).hexdigest()},
            "mask": {"file": "mask.png", "sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest()},
            "confidence_meaning": "OCR text confidence, not watermark probability",
            "warnings": warnings,
        }
        (stage / "processing.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        _check_stop(should_stop)
        destination = base / ("image-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-") + uuid4().hex)
        # Publishing the whole unique directory is the commit point. No writes
        # or fallible work are performed after it; logging is best-effort.
        os.rename(stage, destination)
        stage = None
        _emit(emit, "图片修复完成，结果与遮罩已另存，原件未覆盖。")
        return {"output": str(destination / "result.png"), "manifest": str(destination / "processing.json"),
                "mask": str(destination / "mask.png"), "width": width, "height": height}
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        if method == 'lama':
            from .watermark_ai import LamaError
            if isinstance(exc, LamaError):
                raise RuntimeError(str(exc)) from None
        raise RuntimeError(f"图片修复失败（{type(exc).__name__}）；原件未改动，未发布成功结果。") from None
    finally:
        if stage is not None and base is not None:
            # Only delete the exact unique directory this call created, after
            # verifying its resolved parent. Never remove a user-selected root.
            if stage.parent == base and stage.name.startswith(".image-repair-"):
                shutil.rmtree(stage, ignore_errors=True)
