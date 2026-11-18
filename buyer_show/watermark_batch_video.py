"""Conservative, local CPU planning for unattended video watermark batches.

This module never repairs media. A caller may pass approved_regions to the
existing repair_video, which preserves audio and validates the full output.
OCR confidence is text confidence, not an overlay probability. Sparse visual
evidence cannot prove that a mark is not physically present in the scene.
"""
from __future__ import annotations

import math
import re
import tempfile
import unicodedata
from collections.abc import Mapping
from pathlib import Path

from PIL import Image

from . import watermark_video as video
from .watermark import _detection_options, _keyword_match, _MERCHANT, detect_image

POLICY_VERSION = "stationary-merchant-video-v2"
MAX_SAMPLES = 24
MAX_TRACKS = 8
MIN_TEXT_CONFIDENCE = .92
_COPY = re.compile(r"规格|材质|成分|尺寸|重量|容量|型号|参数|尺码|售价|价格|包邮|折扣|优惠|促销|"
                   r"限时|同款|新品|新款|点击|购买|欢迎|设计|功能|穿搭|教程|使用方法|"
                   r"实物|门头|吊牌|标牌|铭牌|标识牌|水洗标|皮标|印花|包装|"
                   r"\b(?:sale|discount|shop\s+now|buy\s+now|tutorial|recommended|home\s+goods)\b", re.I)
WARNINGS = ["未检出只表示本次抽样未发现候选，不代表视频没有水印。",
            "自动批准依据为保守的抽样启发式，不能保证识别所有实物标牌或未采样的跳变；原件必须保留。",
            "OCR confidence 仅表示文字识别置信度，不是水印概率。",
            "修复沿用 CPU delogo：可能模糊、闪烁或损伤背景，不能恢复被遮挡的真实像素。"]
REASONS = {
    "sampling_insufficient": "抽样数量、视频长度或时间间隔不足以自动判断。",
    "scan_incomplete": "候选检测未完整结束或候选数量超过自动判定上限。",
    "unsupported_video_layout": "视频方向、像素格式或轨道结构需要人工处理。",
    "not_explicit_merchant": "文字不属于明确商家关键词或店铺标记。",
    "product_copy_or_sign": "疑似商品说明、宣传字幕或实物标牌文字，保留人工审核。",
    "low_text_confidence": "部分抽样的文字识别置信度不足。",
    "not_small_corner": "候选不在小面积边角范围内。",
    "edge_context_missing": "候选紧贴画面边界，邻域插值缺少可靠背景。",
    "temporal_gap_or_motion": "候选在部分抽样中缺失、跳变、移动或位置不稳定。",
    "overlay_evidence_insufficient": "未确认固定字形与变化背景的分离证据，可能是实物标牌或静态画面。",
    "background_too_complex": "候选周围背景纹理复杂，不适合自动邻域插值。",
    "enhanced_text_unconfirmed": "增强识别仅发现含店名的正文或不确定文字，不能自动修复。",
}


def _text(label):
    value = str(label).split("：", 1)[-1]
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).casefold()


def _box(value, width, height):
    if not isinstance(value, Mapping):
        return None
    try:
        numbers = [float(value[key]) for key in ("x", "y", "w", "h")]
        if not all(math.isfinite(n) and n == int(n) for n in numbers):
            return None
        box = dict(zip(("x", "y", "w", "h"), map(int, numbers)))
        confidence = float(value.get("confidence", 0))
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            return None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if (min(box["x"], box["y"]) < 0 or min(box["w"], box["h"]) < 3
            or box["x"] + box["w"] > width or box["y"] + box["h"] > height):
        return None
    result = {**box, "label": str(value.get("label", "待审核候选"))[:120], "confidence": confidence}
    if "extra_text_kind" in value:
        result.update(extra_text_kind=value["extra_text_kind"],
                      extra_text_brand_only=value.get("extra_text_brand_only") is True)
    return result


def _tracks(samples, width, height):
    tracks = []
    for index, sample in enumerate(samples):
        used = set()
        for row in sample.get("regions", []):
            region = _box(row, width, height)
            if region is None:
                continue
            key = _text(region["label"])
            same = [track for track in tracks if track["id"] not in used
                    and track["text"] == key and video._iou(track["anchor"], region) >= .45]
            if same:
                track = max(same, key=lambda t: video._iou(t["anchor"], region))
            else:
                track = {"id": len(tracks), "text": key, "anchor": region, "observations": []}
                tracks.append(track)
            track["observations"].append({"sample": index, "region": region})
            used.add(track["id"])
    return tracks


def _reasons(codes):
    return [{"code": code, "message": REASONS[code]} for code in dict.fromkeys(codes)]


def classify_video_detection(detection, *, merchant_terms="", visual_evidence=None):
    """Pure policy: explicit per-sample observations and optional pixel evidence.

    No positive decision can be inferred from a merged detect_video rectangle.
    visual_evidence is produced by plan_video, not a user-facing override.
    """
    terms = _detection_options(merchant_terms, "product")
    width, height = detection.get("width", 0), detection.get("height", 0)
    duration = detection.get("duration", 0)
    if (type(width) is not int or type(height) is not int or min(width, height) <= 0
            or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0):
        raise ValueError("视频检测尺寸或时长无效")
    samples = detection.get("samples", [])
    if not isinstance(samples, list):
        samples = []
    raw_regions = detection.get("regions", [])
    tracks = _tracks(samples, width, height)
    global_codes = []
    times = [sample.get("seconds") for sample in samples]
    times_valid = all(isinstance(t, (int, float)) and math.isfinite(t) and 0 <= t < duration for t in times)
    dense = (8 <= len(times) <= MAX_SAMPLES and duration >= 2 and times_valid
             and times[0] <= .1 and times[-1] >= duration - .15
             and all(0 < b - a <= 2.01 for a, b in zip(times, times[1:])))
    if not dense:
        global_codes.append("sampling_insufficient")
    if (detection.get("scan_complete") is not True or len(tracks) > MAX_TRACKS
            or any(sample.get("scan_complete") is not True for sample in samples)):
        global_codes.append("scan_incomplete")
    if detection.get("repair_supported") is not True:
        global_codes.append("unsupported_video_layout")
    approved, track_evidence, rejected = [], [], []
    visuals = visual_evidence or {}
    for track in tracks:
        anchor, observations = track["anchor"], track["observations"]
        text = str(anchor["label"]).split("：", 1)[-1]
        codes = list(global_codes)
        if not (_keyword_match(text, terms) or _MERCHANT.search(text)):
            codes.append("not_explicit_merchant")
        if not 2 <= len(track["text"]) <= 32 or _COPY.search(text):
            codes.append("product_copy_or_sign")
        boxes = [value["region"] for value in observations]
        if any("extra_text_kind" in box and (box["extra_text_kind"] not in {"known_brand_text", "enhanced_overlay"}
                or box.get("extra_text_brand_only") is not True) for box in boxes):
            codes.append("enhanced_text_unconfirmed")
        if any(box["confidence"] < MIN_TEXT_CONFIDENCE for box in boxes):
            codes.append("low_text_confidence")
        union = {key: anchor[key] for key in ("x", "y", "w", "h")}
        for box in boxes[1:]:
            union = video._union_box(union, box)
        corner = ((union["x"] <= width * .15 or union["x"] + union["w"] >= width * .85)
                  and (union["y"] <= height * .15 or union["y"] + union["h"] >= height * .85))
        if (not corner or union["w"] * union["h"] > width * height * .025
                or union["w"] > width * .35 or union["h"] > height * .13):
            codes.append("not_small_corner")
        if min(union["x"], union["y"], width - union["x"] - union["w"], height - union["y"] - union["h"]) < 3:
            codes.append("edge_context_missing")
        if (len(observations) != len(samples) or len({o["sample"] for o in observations}) != len(samples)
                or any(video._iou(anchor, box) < .8
                       or abs(box["x"] - anchor["x"]) > 2 or abs(box["y"] - anchor["y"]) > 2
                       or abs(box["w"] - anchor["w"]) > 3 or abs(box["h"] - anchor["h"]) > 3 for box in boxes)):
            codes.append("temporal_gap_or_motion")
        visual = visuals.get(track["id"], {})
        if visual.get("screen_locked_ink_with_changing_background") is not True:
            codes.append("overlay_evidence_insufficient")
        if visual.get("simple_background") is not True:
            codes.append("background_too_complex")
        region = {**union, "start": 0.0, "end": duration,
                  "label": anchor["label"], "confidence": min(box["confidence"] for box in boxes)}
        if not codes:
            approved.append(region)
        else:
            rejected.extend(codes)
        track_evidence.append({"track": track["id"], "observed_samples": len(observations),
                               "total_samples": len(samples), "region": region,
                               "approved": not codes, "review_reasons": _reasons(codes), "visual": visual})
    if not tracks:
        # Unsupported repair geometry alone does not turn an empty, completed
        # detection into a positive finding. Missing/incomplete detection does.
        if detection.get("scan_complete") is not True or raw_regions or not samples:
            rejected.extend(global_codes or ["scan_incomplete"])
    status = "needs_review" if rejected else "approved" if approved else "not_detected"
    reason = ("固定商家文字候选通过保守抽样规则，可使用现有 CPU 视频修复。" if status == "approved" else
              "存在疑似标记但自动判定证据不足，原件保留，需人工审核。" if status == "needs_review" else
              "本次抽样未检出候选，原件保留；这不代表视频没有水印。")
    return {"status": status, "reason": reason, "width": width, "height": height, "duration": duration,
            "regions": list(raw_regions), "approved_regions": approved,
            "review_reasons": _reasons(rejected), "detection": detection,
            "warnings": list(dict.fromkeys(WARNINGS + list(detection.get("warnings", [])))),
            "evidence": {"policy": POLICY_VERSION, "tracking": "sampled_segments_only",
                         "tracks": track_evidence, "automatic_claim": "conservative_sample_evidence_only"}}


def _visual(patches, interior):
    import cv2
    import numpy as np
    stack = np.stack(patches).astype(np.float32)
    mean, deviation = stack.mean(axis=0), stack.std(axis=0)
    inside = np.zeros(mean.shape, dtype=bool)
    x, y, w, h = interior
    inside[y:y+h, x:x+w] = True
    ring = ~inside
    stable_ink = inside & (deviation <= 8) & ((mean <= 35) | (mean >= 220))
    ink_ratio = float(stable_ink.sum() / max(1, inside.sum()))
    moving_ratio = float((deviation[ring] >= 10).mean()) if ring.any() else 0
    ring_deviation = float(deviation[ring].mean()) if ring.any() else 0
    # Gray or translucent lettering need not be near absolute black/white.
    # Measure persistent strokes relative to each frame's local background;
    # do not treat an entire flat patch or one coincidental edge as lettering.
    relative_ratio, relative_iou, relative_components = 0.0, 0.0, 0
    relative_ink = False
    if ring.any() and len(patches) >= 8:
        levels = np.median(stack[:, ring], axis=1)
        residual = stack - levels[:, None, None]
        positive = inside & (residual.min(axis=0) >= 8)
        negative = inside & (residual.max(axis=0) <= -8)
        persistent = positive | negative
        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(persistent.astype(np.uint8), connectivity=8)
        valid_labels = [index for index in range(1, labels_count) if stats[index, cv2.CC_STAT_AREA] >= 3]
        persistent = np.isin(labels, valid_labels) & inside
        relative_components = len(valid_labels)
        relative_ratio = float(persistent.sum() / max(1, inside.sum()))
        if persistent.any():
            scores = []
            mean_residual = residual.mean(axis=0)
            for plane in residual:
                # The same contrast sign and shape must persist in every
                # sampled frame; motion is also rejected by box tracking.
                shape = inside & (((plane >= 8) & (mean_residual > 0)) |
                                  ((plane <= -8) & (mean_residual < 0)))
                scores.append(float((shape & persistent).sum() / max(1, (shape | persistent).sum())))
            relative_iou = min(scores)
        relative_ink = relative_components >= 2 and .035 <= relative_ratio <= .45 and relative_iou >= .70
    # A static sign on a static scene cannot pass the background-change gate.
    # This does not semantically prove that the mark is not a physical object.
    edge_ratios, spatial_stds = [], []
    for patch in patches:
        edges = cv2.Canny(patch, 70, 140) > 0
        edge_ratios.append(float(edges[ring].mean()) if ring.any() else 1)
        spatial_stds.append(float(patch[ring].std()) if ring.any() else 255)
    simple = max(edge_ratios) <= .13 and max(spatial_stds) <= 35
    return {"screen_locked_ink_with_changing_background": bool((ink_ratio >= .035 or relative_ink) and moving_ratio >= .25 and ring_deviation >= 10),
            "simple_background": bool(simple), "stable_ink_fraction": round(ink_ratio, 6),
            "relative_stroke_evidence": bool(relative_ink),
            "relative_stroke_fraction": round(relative_ratio, 6),
            "relative_stroke_shape_iou_min": round(relative_iou, 6),
            "relative_stroke_components": relative_components,
            "changing_background_fraction": round(moving_ratio, 6),
            "background_temporal_std": round(ring_deviation, 4),
            "background_edge_fraction_max": round(max(edge_ratios), 6),
            "background_spatial_std_max": round(max(spatial_stds), 4)}


def plan_video(path, *, merchant_terms="", should_stop=lambda: False, emit=lambda msg: None):
    """Sample at most 24 local frames, return a plan without modifying media."""
    _detection_options(merchant_terms, "product")
    video._stop(should_stop)
    source = video._local_source(path)
    info = video._probe(source, should_stop)
    source_sha = video._sha(source, should_stop)
    duration = info["duration"]
    count = min(MAX_SAMPLES, max(8, math.ceil(duration / 2) + 1))
    # Seeking later than the last frame PTS produces no frame even though the
    # container duration still includes that frame's display interval.
    fps = info.get("fps") or 0
    tail = min(duration / 2, max(.1, 1 / fps if fps > 0 else .25))
    times = [i * max(0, duration - tail) / (count - 1) for i in range(count)]
    samples, regions, warnings = [], [], []
    complete = True
    extra_enabled = video._known_text_enabled(merchant_terms, "product")
    import numpy as np
    # Only tiny anchor crops survive each iteration, never full video frames.
    anchors, crops = [], {}
    with tempfile.TemporaryDirectory(prefix="buyer_show_batch_video_") as directory:
        frame_path = Path(directory) / "frame.png"
        for index, seconds in enumerate(times):
            video._stop(should_stop)
            emit(f"批量视频判定：抽样 {index + 1}/{count}")
            frame = video._frame(source, seconds, info, should_stop)
            try:
                frame.save(frame_path)
                found = detect_image(frame_path, should_stop=should_stop, emit=emit,
                                     merchant_terms=merchant_terms, detection_profile="product", include_corner_text=False,
                                     **({"return_text_regions": True} if extra_enabled else {}))
                found = video._enhance_known_text(frame_path, found, merchant_terms=merchant_terms,
                    detection_profile="product", should_stop=should_stop, emit=emit)
                valid = [box for row in found.get("regions", [])[:32]
                         if (box := _box(row, info["width"], info["height"])) is not None]
                sample_complete = found.get("scan_complete") is True and len(found.get("regions", [])) == len(valid)
                complete = complete and sample_complete
                samples.append({"seconds": seconds, "regions": valid, "scan_complete": sample_complete,
                                **({"extra_text": found.get("extra_text", {})} if extra_enabled else {})})
                start = 0 if not index else (times[index - 1] + seconds) / 2
                end = duration if index + 1 == count else (seconds + times[index + 1]) / 2
                regions.extend({**box, "start": start, "end": end} for box in valid)
                if len(regions) > 128:
                    regions = regions[:128]
                    complete = False
                warnings.extend(found.get("warnings", []))
                if index == 0:
                    anchors = _tracks(samples, info["width"], info["height"])[:MAX_TRACKS]
                for anchor in anchors:
                    box = anchor["anchor"]
                    pad = max(6, min(20, box["h"] // 2))
                    x, y = max(0, box["x"] - pad), max(0, box["y"] - pad)
                    right, bottom = min(info["width"], box["x"] + box["w"] + pad), min(info["height"], box["y"] + box["h"] + pad)
                    scale = min(1, 192 / (right - x), 96 / (bottom - y))
                    size = (max(1, round((right - x) * scale)), max(1, round((bottom - y) * scale)))
                    patch = frame.crop((x, y, right, bottom)).convert("L").resize(size, Image.Resampling.BILINEAR)
                    entry = crops.setdefault(anchor["id"], {"patches": [], "interior": (
                        round((box["x"] - x) * scale), round((box["y"] - y) * scale),
                        max(1, round(box["w"] * scale)), max(1, round(box["h"] * scale)))})
                    entry["patches"].append(np.asarray(patch).copy())
                    patch.close()
            finally:
                frame.close()
    visual = {key: _visual(value["patches"], value["interior"]) for key, value in crops.items()}
    repair_supported = (not info["rotation"] and not info["has_alpha"] and info["sample_aspect_ratio"] == [1, 1]
                        and info["video_stream_count"] == 1 and info["width"] % 2 == 0 and info["height"] % 2 == 0
                        and info["pixel_format"] in {"yuv420p", "yuvj420p", "yuv422p", "yuvj422p", "yuv444p", "yuvj444p", "nv12", "rgb24", "bgr24", "gray"})
    detection = {"width": info["width"], "height": info["height"], "duration": duration,
                 "sample_count": count, "samples": samples, "regions": regions,
                 "scan_complete": complete, "repair_supported": repair_supported,
                 "tracking": "sampled_segments_only", "intervals": "half_open",
                 "warnings": list(dict.fromkeys(warnings)), "source_sha256": source_sha,
                 "has_audio": info["has_audio"]}
    video._stop(should_stop)
    if video._sha(source, should_stop) != source_sha:
        raise ValueError("原视频在检测期间被修改，已拒绝使用旧候选")
    result = classify_video_detection(detection, merchant_terms=merchant_terms, visual_evidence=visual)
    result["source_sha256"] = source_sha
    return result
