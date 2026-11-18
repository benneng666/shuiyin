"""CPU video-region interpolation. Public functions are synchronous worker APIs.

This is not original-pixel recovery or complete moving-watermark tracking.
No ffprobe, GPU, remote URL or model download is used by this module.
"""
from __future__ import annotations

import asyncio
import hashlib
import math
import os
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from uuid import uuid4

from PIL import Image

from .storage import atomic_json, now
from .video import MAX_VIDEO_BYTES, _InputMetadata, _container_kind, ffmpeg_executable, validate_video

MAX_PIXELS = 3840 * 2160
MAX_DURATION = 1800
MAX_DETECTION_SAMPLES = 24
MAX_MERGED_SECONDS = 30
WARNINGS = ["CPU 邻域插值仅适合小面积、简单背景，可能模糊、闪烁或损伤纹理，无法保证恢复被遮挡的原像素。",
            "仅抽样检测并给出分段候选框，不保证发现或跟踪所有移动水印；请预览并人工修正。",
            "视频画面将重新编码；原音轨复制保存，原文件不修改。"]


def _stop(should_stop):
    if should_stop():
        raise asyncio.CancelledError()


def _local_source(path):
    value = os.fspath(path)
    if value.startswith(("\\\\", "//")) or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value):
        raise ValueError("仅支持本机视频文件，不支持网址或网络共享")
    source = Path(value).expanduser().resolve(strict=True)
    if not source.is_file() or not 0 < source.stat().st_size <= MAX_VIDEO_BYTES:
        raise ValueError("视频必须为非空本地文件，最大 256 MiB")
    with source.open("rb") as handle:
        _container_kind(handle.read(4096))
    return source


def _input_args(source, *, noautorotate=False):
    args = ["-protocol_whitelist", "file", "-format_whitelist", "mov,matroska,webm", "-threads", "2"]
    if noautorotate:
        args += ["-noautorotate"]
    return args + ["-i", str(source)]


def _run(args, *, timeout=30, should_stop=lambda: False, stdout_limit=0, emit=lambda msg: None):
    """Drain both pipes with bounded buffers; kill/reap only our child on cancel."""
    _stop(should_stop)
    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    proc = subprocess.Popen([ffmpeg_executable(), "-nostdin", "-hide_banner", *args],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
    output, diagnostic = bytearray(), bytearray()
    oversized = threading.Event()

    def drain(pipe, target, limit, enforce=False):
        try:
            while data := pipe.read(65536):
                if limit:
                    available = max(0, limit - len(target))
                    target.extend(data[:available])
                    if enforce and len(data) > available:
                        oversized.set()
        finally:
            pipe.close()

    readers = [threading.Thread(target=drain, args=(proc.stdout, output, stdout_limit, bool(stdout_limit)), daemon=True),
               threading.Thread(target=drain, args=(proc.stderr, diagnostic, 256 * 1024), daemon=True)]
    for reader in readers:
        reader.start()
    started = last_update = time.monotonic()
    try:
        while proc.poll() is None:
            _stop(should_stop)
            if oversized.is_set():
                raise ValueError("解码帧超出缓冲上限")
            if time.monotonic() - started > timeout:
                raise TimeoutError("视频处理超时，未发布结果")
            if time.monotonic() - last_update >= 5:
                emit("视频处理中，可随时取消…")
                last_update = time.monotonic()
            time.sleep(0.05)
        _stop(should_stop)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
        for reader in readers:
            reader.join(timeout=5)
    if oversized.is_set():
        raise ValueError("解码帧超出缓冲上限")
    return proc.returncode, bytes(output), diagnostic.decode("utf-8", errors="replace")


def _probe(source, should_stop=lambda: False):
    code, _, text = _run(["-v", "info", *_input_args(source, noautorotate=True),
        "-map", "0:V:0", "-frames:v", "0", "-an", "-sn", "-dn", "-f", "null", "-"],
        timeout=15, should_stop=should_stop)
    meta = _InputMetadata()
    rotation, pixel_format, sar = 0.0, "", (1, 1)
    alpha_tag = False
    selected = False
    input_start = 0.0
    video_stream_count = 0
    for line in text.splitlines():
        meta.consume(line)
        if meta.done:
            break
        start = re.search(r"start: (-?[\d.]+)", line)
        if start:
            input_start = float(start[1])
        if re.search(r"Stream #0:\d+", line):
            is_video = "Video:" in line and "attached pic" not in line
            if is_video:
                video_stream_count += 1
            selected = is_video and video_stream_count == 1
            if selected:
                tail = line.split("Video:", 1)[1]
                fmt = re.search(r",\s*(yuv\w+|yuva\w+|nv12|rgba|argb|bgra|abgr|rgb\w+|bgr\w+|gbr\w+|gray\w*|ya\w+|pal8)\b", tail)
                pixel_format = fmt[1] if fmt else "unknown"
                ratio = re.search(r"SAR (\d+):(\d+)", tail)
                if ratio:
                    sar = (int(ratio[1]), int(ratio[2]))
        if selected:
            if re.search(r"alpha_mode\s*:\s*1\b", line, re.IGNORECASE):
                alpha_tag = True
            rotate = re.search(r"rotation of (-?[\d.]+) degrees", line)
            old_rotate = re.search(r"\brotate\s*:\s*(-?[\d.]+)", line)
            if rotate or old_rotate:
                rotation = float((rotate or old_rotate)[1]) % 360
    if code or not meta.video or not meta.duration or not math.isfinite(meta.duration):
        raise ValueError("无法读取有效视频轨道、尺寸或时长；封面图片不算视频")
    width, height = meta.video["width"], meta.video["height"]
    if not 0 < width * height <= MAX_PIXELS or meta.duration > MAX_DURATION:
        raise ValueError("当前 CPU 模式最多支持 4K 像素量、30 分钟视频")
    if not any(abs(rotation - x) < 0.1 for x in (0, 90, 180, 270, 360)):
        raise ValueError("不支持非直角旋转视频，请先另存为方向正常的视频")
    display_width, display_height = (height, width) if round(rotation) in {90, 270} else (width, height)
    alpha = alpha_tag or bool(re.match(r"(?:yuva|gbrap|rgba|argb|bgra|abgr|ya|pal8)", pixel_format))
    return {**meta.video, "width": display_width, "height": display_height,
        "encoded_width": width, "encoded_height": height, "duration": meta.duration,
        "has_audio": bool(meta.audio), "audio_streams": meta.audio, "audio_stream_count": len(meta.audio),
        "rotation": rotation, "pixel_format": pixel_format, "has_alpha": alpha,
        "sample_aspect_ratio": list(sar), "input_start_seconds": input_start,
        "video_stream_count": video_stream_count,
        "warnings": ["含旋转元数据：预览按显示方向，当前修复需先另存为方向正常的视频。"] if rotation else []}


def probe_video(path) -> dict:
    """Read headers only; width/height describe FFmpeg's displayed orientation."""
    return _probe(_local_source(path))


def _frame(source, seconds, info, should_stop=lambda: False):
    if not math.isfinite(float(seconds)) or not 0 <= seconds < info["duration"]:
        raise ValueError("预览时间必须在视频范围内")
    expected = info["width"] * info["height"] * 3
    code, payload, _ = _run(["-v", "error", "-ss", f"{float(seconds):.6f}",
        *_input_args(source), "-map", "0:V:0", "-frames:v", "1", "-an", "-sn", "-dn",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"], timeout=30,
        should_stop=should_stop, stdout_limit=expected)
    if code or len(payload) != expected:
        raise ValueError("无法提取该时间点的视频帧")
    return Image.frombytes("RGB", (info["width"], info["height"]), payload)


def extract_frame(path, seconds) -> Image.Image:
    source = _local_source(path)
    return _frame(source, float(seconds), _probe(source))


def _sample_count(duration):
    # Roughly one sample every four seconds, with a CPU work bound for long
    # clips. These samples still do not establish continuous watermark tracking.
    return min(MAX_DETECTION_SAMPLES, max(6, math.ceil(duration / 4)))


def _scene_signature(frame):
    return frame.convert("RGB").resize((32, 18), Image.Resampling.BILINEAR).tobytes()


def _scene_changed(previous, current):
    if previous is None:
        return False
    # A conservative cut heuristic. A cut between sampled frames cannot be
    # timestamped exactly; this only prevents joining their candidate ranges.
    return sum(abs(a - b) for a, b in zip(previous, current)) / (len(current) * 255) > 0.18


def _iou(a, b):
    intersection = max(0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])) * max(
        0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    return intersection / max(1, a["w"] * a["h"] + b["w"] * b["h"] - intersection)


def _union_box(a, b):
    x, y = min(a["x"], b["x"]), min(a["y"], b["y"])
    return dict(x=x, y=y, w=max(a["x"] + a["w"], b["x"] + b["w"]) - x,
                h=max(a["y"] + a["h"], b["y"] + b["h"]) - y)


def _jitter_match(state, candidate, merged, end):
    anchor, previous = state["anchor"], state["last_box"]
    if _iou(previous, candidate) < 0.65 or end - merged["start"] > MAX_MERGED_SECONDS:
        return False
    # Restrict cumulative drift, not only each pair of samples. A slowly moving
    # mark must not grow into one oversized static rectangle over the clip.
    for axis, extent in (("x", "w"), ("y", "h")):
        delta = abs(candidate[axis] + candidate[extent] / 2 - anchor[axis] - anchor[extent] / 2)
        if delta > max(3, anchor[extent] * 0.15) or not 0.75 <= candidate[extent] / anchor[extent] <= 1.34:
            return False
    union = _union_box(merged, candidate)
    return union["w"] * union["h"] <= anchor["w"] * anchor["h"] * 1.4


def _known_text_enabled(merchant_terms, detection_profile):
    from .watermark import _detection_options
    return detection_profile == "product" and bool(_detection_options(merchant_terms, detection_profile))


def _enhance_known_text(path, found, *, merchant_terms, detection_profile,
                        should_stop=lambda: False, emit=lambda msg: None):
    """Merge reviewable text only; this helper never approves video repair."""
    if not _known_text_enabled(merchant_terms, detection_profile):
        return found
    from .watermark_text_extra import detect_extra_text
    _stop(should_stop)
    extra = detect_extra_text(path, merchant_terms, base_detection=found,
                              should_stop=should_stop, emit=emit)
    _stop(should_stop)
    regions = [dict(row) for row in found.get("regions", [])]
    for row in extra.get("regions", []):
        candidate = dict(row)
        candidate["extra_text_kind"] = str(row.get("kind", "uncertain_brand"))
        candidate["extra_text_brand_only"] = row.get("brand_only_line") is True
        text = re.sub(r"[\W_]+", "", str(row.get("text") or row.get("label", "").split("：", 1)[-1]).casefold())
        duplicate = False
        for original in regions:
            original_text = re.sub(r"[\W_]+", "", str(original.get("text") or original.get("label", "").split("：", 1)[-1]).casefold())
            if text == original_text and _iou(original, candidate) >= .5:
                # Keep the existing footprint. An extra OCR pass must not turn
                # a small approved text box into a larger delogo rectangle.
                duplicate = True
                if candidate["extra_text_kind"] not in {"known_brand_text", "enhanced_overlay"} or not candidate["extra_text_brand_only"]:
                    original.update(extra_text_kind=candidate["extra_text_kind"],
                                    extra_text_brand_only=candidate["extra_text_brand_only"])
                break
        if not duplicate:
            regions.append(candidate)
    warnings = list(dict.fromkeys(list(found.get("warnings", [])) + list(extra.get("warnings", []))))
    return {**found, "regions": regions, "warnings": warnings,
            "scan_complete": found.get("scan_complete") is True and extra.get("scan_complete") is True
                             and not found.get("text_regions_truncated", False),
            "extra_text": {"regions": extra.get("regions", []),
                           "scan_complete": extra.get("scan_complete") is True,
                           "evidence": extra.get("evidence", {})}}


def detect_video(path, should_stop=lambda: False, emit=lambda msg: None, *,
                 include_corner_text=False, merchant_terms="", detection_profile="platform") -> dict:
    from .watermark import detect_image
    source = _local_source(path)
    info = _probe(source, should_stop)
    count = _sample_count(info["duration"])
    regions = []
    warnings = list(WARNINGS) + info["warnings"]
    active, previous_signature, scene_changes, capped = [], None, 0, False
    extra_enabled = _known_text_enabled(merchant_terms, detection_profile)
    sample_evidence, scan_complete = [], True
    interval = info["duration"] / count
    if interval > 4:
        warnings.append(f"长视频抽样上限为 {MAX_DETECTION_SAMPLES} 帧，当前相邻样本约隔 {interval:.1f} 秒；未采样区间仍需人工检查。")
    with tempfile.TemporaryDirectory(prefix="buyer_show_detect_") as directory:
        image_path = Path(directory) / "frame.png"
        for index in range(count):
            _stop(should_stop)
            start = index * info["duration"] / count
            # Keep the final boundary bit-identical to the probed duration.
            # Multiplication/division can otherwise produce duration + 1 ULP,
            # causing the strict repair-time validation to reject our own result.
            end = info["duration"] if index + 1 == count else (index + 1) * info["duration"] / count
            sample = min((start + end) / 2, max(0, info["duration"] - 0.1))
            emit(f"抽样检测视频帧 {index + 1}/{count}")
            frame = _frame(source, sample, info, should_stop)
            signature = _scene_signature(frame)
            changed = _scene_changed(previous_signature, signature)
            previous_signature = signature
            if changed:
                scene_changes += 1
            frame.save(image_path)
            found = detect_image(image_path, should_stop=should_stop, emit=emit,
                include_corner_text=include_corner_text, merchant_terms=merchant_terms,
                detection_profile=detection_profile,
                **({"return_text_regions": True} if extra_enabled else {}))
            found = _enhance_known_text(image_path, found, merchant_terms=merchant_terms,
                detection_profile=detection_profile, should_stop=should_stop, emit=emit)
            scan_complete = scan_complete and found.get("scan_complete") is True
            if extra_enabled:
                sample_evidence.append({"seconds": sample, **found.get("extra_text", {})})
            for warning in found.get("warnings", []):
                if warning not in warnings:
                    warnings.append(warning)
            seen, next_active, used = [], [], set()
            for region in found.get("regions", [])[:32]:
                try:
                    numbers = [float(region[key]) for key in ("x", "y", "w", "h")]
                    if not all(math.isfinite(n) and n == int(n) for n in numbers):
                        continue
                    box = dict(zip(("x", "y", "w", "h"), map(int, numbers)))
                except (KeyError, TypeError, ValueError):
                    continue
                if (box["x"] < 0 or box["y"] < 0 or box["w"] < 3 or box["h"] < 3
                    or box["x"] + box["w"] > info["width"] or box["y"] + box["h"] > info["height"]):
                    continue
                label = str(region.get("label", "待确认区域"))[:120]
                label_key = re.sub(r"\s+", "", label).casefold()
                if any(key == label_key and _iou(box, prior_box) >= 0.85 for key, prior_box in seen):
                    continue
                seen.append((label_key, box))
                try:
                    confidence = float(region.get("confidence", 0))
                    confidence = max(0, min(1, confidence)) if math.isfinite(confidence) else 0
                except (ValueError, TypeError):
                    confidence = 0
                matches = [state for state in active if not changed and interval <= 8
                    and state["index"] not in used and state["label_key"] == label_key
                    and _jitter_match(state, box, regions[state["index"]], end)]
                prior = max(matches, key=lambda state: _iou(state["last_box"], box), default=None)
                if prior is not None:
                    merged = regions[prior["index"]]
                    merged.update(_union_box(merged, box), end=end, confidence=min(merged["confidence"], confidence))
                    used.add(prior["index"])
                    next_active.append({**prior, "last_box": box})
                else:
                    if len(regions) >= 128:
                        capped = True
                        continue
                    regions.append({**box, "start": start, "end": end, "label": label,
                                    "confidence": confidence})
                    next_active.append({"index": len(regions) - 1, "last_box": box,
                                        "anchor": box, "label_key": label_key})
            # An absent detection cannot be bridged by a later matching label.
            active = next_active
    if capped:
        warnings.append("候选区域较多，已限制为 128 个；请人工确认所需范围。")
    if scene_changes:
        warnings.append("相邻抽样画面存在明显变化，已停止跨该区间合并；分段界限仍是抽样区间，不是精确场景切点。")
    _stop(should_stop)
    return {"width": info["width"], "height": info["height"], "duration": info["duration"],
            "regions": regions, "warnings": warnings, "sample_count": count,
            "sample_interval_seconds": interval, "scene_change_samples": scene_changes,
            "tracking": "sampled_segments_only", "intervals": "half_open",
            "scan_complete": scan_complete and not capped,
            "extra_text_samples": sample_evidence}


def _regions(values, info):
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= 128:
        raise ValueError("请提供 1 至 128 个待修复区域")
    width, height, duration = info["width"], info["height"], info["duration"]
    result, seen = [], set()
    for value in values:
        try:
            numbers = [float(value[k]) for k in ("x", "y", "w", "h")]
            start, end = float(value.get("start", 0)), float(value.get("end", duration))
        except (TypeError, KeyError, ValueError) as exc:
            raise ValueError("区域坐标或时间格式错误") from exc
        if not all(math.isfinite(n) for n in [*numbers, start, end]) or any(n != int(n) for n in numbers):
            raise ValueError("区域需要有限的整数像素坐标")
        x, y, w, h = map(int, numbers)
        if x < 0 or y < 0 or w < 3 or h < 3 or x + w > width or y + h > height:
            raise ValueError("修复框需要至少 3×3 像素，且完整位于画面内；可以贴边")
        if w * h > width * height * 0.12:
            raise ValueError("单个修复框超过画面 12%，不适合小水印插值修复")
        if not 0 <= start < end <= duration:
            raise ValueError("时间范围必须满足 0 ≤ 开始 < 结束 ≤ 视频时长")
        key = (x, y, w, h, start, end)
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(x=x, y=y, w=w, h=h, start=start, end=end,
                           label=str(value.get("label", "手动区域"))[:120]))
    for instant in {r["start"] for r in result}:
        area = sum(r["w"] * r["h"] for r in result if r["start"] <= instant < r["end"])
        if area > width * height * 0.2:
            raise ValueError("同一时刻修复范围超过画面 20%，请缩小框选")
    return result


def _sha(path, should_stop):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while data := handle.read(1024 * 1024):
            _stop(should_stop)
            digest.update(data)
    return digest.hexdigest()


def _repair_filters(boxes, info):
    """Supply delogo's missing border samples, then restore exact source size."""
    width, height = info["width"], info["height"]
    padding = 4 if any(r["x"] == 0 or r["y"] == 0 or r["x"] + r["w"] == width
                       or r["y"] + r["h"] == height for r in boxes) else 0
    filters = []
    if padding:
        # Even padding keeps chroma alignment. Mirror only the newly introduced
        # pixels rather than using a black artificial boundary for interpolation.
        filters += [f"pad=iw+{padding * 2}:ih+{padding * 2}:{padding}:{padding}:color=black",
                    f"fillborders=left={padding}:right={padding}:top={padding}:bottom={padding}:mode=mirror"]
    filters += [f"delogo=x={r['x'] + padding}:y={r['y'] + padding}:w={r['w']}:h={r['h']}:show=0:"
                f"enable='gte(t,{r['start']:.9f})*lt(t,{r['end']:.9f})'" for r in boxes]
    if padding:
        filters.append(f"crop={width}:{height}:{padding}:{padding}")
    return ",".join(filters), padding


def repair_video(source, regions, output_root=None, should_stop=lambda: False, emit=lambda msg: None,
                 *, method="delogo") -> dict:
    if method not in {"delogo", "smart", "lama_temporal"}:
        raise ValueError("未知视频修复方式。")
    source = _local_source(source)
    info = _probe(source, should_stop)
    if info["rotation"]:
        raise ValueError("当前修复不支持带旋转标记的视频，请先另存为方向正常的视频；原件未修改")
    if info["width"] % 2 or info["height"] % 2:
        raise ValueError("当前 MP4 修复仅支持偶数宽高，不会自动裁剪或补边")
    if info["has_alpha"]:
        raise ValueError("当前 MP4 修复不支持透明视频，不会自动丢弃透明通道")
    if info["sample_aspect_ratio"] != [1, 1]:
        raise ValueError("当前修复不支持非方形像素视频，不会静默改变显示比例")
    if info["video_stream_count"] != 1:
        raise ValueError("当前修复仅支持一个真实视频轨道")
    if info["pixel_format"] not in {"yuv420p", "yuvj420p", "yuv422p", "yuvj422p", "yuv444p", "yuvj444p", "nv12", "rgb24", "bgr24", "gray"}:
        raise ValueError("当前修复仅支持已验证的 8 位像素格式，不会静默降低位深")
    boxes = _regions(regions, info)
    filters, padding = _repair_filters(boxes, info)
    warnings = list(WARNINGS)
    source_sha = _sha(source, should_stop)
    base = Path(output_root).expanduser().resolve() if output_root is not None else source.parent / "processed"
    job = base / ("video-" + uuid4().hex)
    job.mkdir(parents=True, exist_ok=False)
    manifest_path, partial, output = job / "processing.json", job / "result.partial.mp4", job / "result.mp4"
    manifest = {"schema_version": 1, "created_at": now(), "status": "running", "source": str(source),
        "source_sha256": source_sha, "source_metadata": info, "boxes": boxes,
        "requested_method": method, "method": "delogo",
        "engine": "FFmpeg delogo CPU + libx264 CRF18", "warnings": warnings,
        "edge_strategy": "mirror_pad_then_delogo_crop" if padding else "interior_delogo",
        "internal_padding_pixels": padding,
        "timestamp_policy": "preserve_relative_pts_normalize_container_start_to_zero",
        "audio_policy": "copy_all_original_audio_streams", "intervals": "half_open",
        "output": None, "validation": None}
    atomic_json(manifest_path, manifest)
    script = job / "filters.txt"
    script.write_text(filters, encoding="ascii")
    try:
        enhanced = False
        if method != "delogo":
            from .watermark_video_quality import QualityFrameProcessor, PrecisionMaskUnavailable
            from .watermark_video_transport import QualityUnsupported, render_quality
            try:
                # Geometry has already passed the common validator. Any stricter
                # constructor rejection here is a quality-mode capability limit.
                try:
                    processor = QualityFrameProcessor(info["width"], info["height"], boxes,
                        should_stop=should_stop, emit=emit, use_ai=True)
                except ValueError as exc:
                    raise QualityUnsupported(str(exc)) from exc
                emit("开始视频纹理精修；简单背景快速处理，复杂画面使用 AI，CPU 可能需要数分钟，可取消。")
                evidence = render_quality(source, partial, boxes, info, processor,
                                          should_stop=should_stop, emit=emit)
                enhanced = True
                warnings.append("视频精修的纹理不等于真实细节；请检查物体边缘、转场及连续播放中的闪烁。")
                warnings.extend(evidence.get("algorithm", {}).get("warnings", []))
                manifest.update(method=method, engine="LaMa CPU adaptive keyframes + temporal propagation",
                                quality=evidence, edge_strategy="approved_regions_only",
                                internal_padding_pixels=0)
            except (QualityUnsupported, PrecisionMaskUnavailable) as exc:
                if method != "smart":
                    raise
                partial.unlink(missing_ok=True)
                reason = "当前视频未满足智能精修条件，已使用快速修补：" + str(exc)
                warnings.append(reason)
                manifest["quality_fallback_reason"] = str(exc)
                emit(reason)
        if not enhanced:
            if padding:
                warnings.append("贴边区域使用镜像延展的边界参与插值，输出恢复原尺寸；边缘缺少真实背景，可能残留或出现模糊，请逐段检查。")
            emit("开始 CPU 视频区域填补，音轨将原样复制；复杂背景可能出现模糊或闪烁。")
            code, _, _ = _run(["-v", "error", "-xerror", "-err_detect", "explode", "-copyts", "-start_at_zero",
                *_input_args(source, noautorotate=True), "-map", "0:V:0", "-map", "0:a?", "-sn", "-dn",
                "-filter_script:v", str(script), "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                "-pix_fmt", "yuv420p", "-threads", "2", "-fps_mode:v", "passthrough", "-enc_time_base:v", "demux",
                "-c:a", "copy", "-map_metadata", "0", "-map_chapters", "0", "-movflags", "+faststart",
                "-n", str(partial)], timeout=3600, should_stop=should_stop, emit=emit)
            if code:
                raise ValueError("FFmpeg 修复失败，可能是当前音轨不兼容 MP4 或滤镜不受支持；未发布结果")
        emit("正在完整解码验证修复结果…")
        validated = asyncio.run(validate_video(partial, should_stop))
        after = _probe(partial, should_stop)
        tolerance = max(0.12, 2 / max(float(info.get("fps") or 25), 1))
        if (after["width"], after["height"]) != (info["width"], info["height"]):
            raise ValueError("结果尺寸与原视频不同，未发布结果")
        if abs(after["duration"] - info["duration"]) > tolerance:
            raise ValueError("结果时长校验不通过，未发布结果")
        if after["audio_streams"] != info["audio_streams"]:
            raise ValueError("结果音轨信息与原视频不符，未发布结果")
        if _sha(source, should_stop) != source_sha:
            raise ValueError("处理期间原文件被其他程序修改，未发布结果")
        _stop(should_stop)
        partial.replace(output)
        manifest.update(status="completed", completed_at=now(), output=str(output),
                        validation=validated, output_metadata=after)
        atomic_json(manifest_path, manifest)
        return {"output": str(output), "manifest": str(manifest_path), "width": info["width"],
                "height": info["height"], "warnings": warnings, "method": manifest["method"]}
    except BaseException as exc:
        partial.unlink(missing_ok=True)
        manifest.update(status="cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
                        error=str(exc) or type(exc).__name__, completed_at=now())
        atomic_json(manifest_path, manifest)
        raise
