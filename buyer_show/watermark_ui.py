"""Local media mask editor. Tk stays on the main thread; work uses spawn."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import io
import json
import logging
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import unicodedata

LOGGER = logging.getLogger("buyer_show")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
DETECTION_PROFILES = {"商家水印（商品）": "product", "平台水印": "platform"}
REPAIR_METHODS = {"自动（按背景选择）": "auto", "Telea": "telea", "NS": "ns", "AI纹理修复（CPU）": "lama"}
BATCH_METHODS = {"智能修复（推荐）": "smart", "基础自动修复": "auto", "AI纹理修复（CPU）": "lama"}
VIDEO_METHODS = {"智能视频精修（推荐）": "smart", "快速视频修补": "delogo", "AI视频精修（较慢）": "lama_temporal"}
BATCH_STATES = {"queued": "待处理", "processing": "处理中", "processed": "已处理", "unchanged": "原样保留",
                "not_detected": "原样保留", "needs_review": "待确认", "failed": "失败", "cancelled": "已取消"}
QUEUE_FILTERS = {"全部": None, "待确认 / 失败": {"needs_review", "failed"}, "待确认": {"needs_review"},
                 "失败": {"failed"}, "已处理": {"processed"}, "原样保留": {"unchanged", "not_detected"},
                 "已取消": {"cancelled"}}
REPAIR_LIMIT = "请人工核对候选，商品正文或品牌字样可能误识别。CPU 基础修补可能模糊纹理；视频仅抽样，移动水印需分时段补框。原文件始终保留。"
LAMA_LIMIT = "AI 图片修复：首次约 208 MB 模型下载，之后可本机离线处理。CPU 加载和推理可能较慢，取消会等待当前步骤。生成纹理不等于真实原貌，请对比检查；原文件始终保留。"
VIDEO_QUALITY_LIMIT = "视频精修会按画面纹理修复关键帧，并检查相邻帧变化；CPU 处理可能需要数分钟。生成的细节不等于真实原貌，物体经过水印位置时仍需检查。原视频和音轨保留。"
VIDEO_SMART_LIMIT = "智能视频：简单背景快速处理，复杂纹理用 AI 关键帧精修。CPU 可能需要数分钟，首次模型约 208 MB；超出精修范围会使用快速修补并说明原因。原视频和音轨保留，仍可能有痕迹。"


def lama_model_notice():
    """Only inspect the cache; opening the GUI never downloads or loads a model."""
    try:
        from .watermark_ai import model_status
        state = model_status()
    except (OSError, RuntimeError, ValueError):
        return "AI 模型状态暂不可读，处理时检查。"
    if state.get("installed"):
        return "AI 模型：已缓存，处理前校验。"
    if state.get("size_bytes", 0):
        return "AI 模型：未就绪，处理时重新下载。"
    return "AI 模型：未下载，首次约 208 MB。"


def media_kind(path):
    suffix = Path(path).suffix.lower()
    return "image" if suffix in IMAGE_SUFFIXES else "video" if suffix in VIDEO_SUFFIXES else None


def load_task_catalog(directory, manifests=None, should_stop=lambda: False):
    """Select one task/current batch/latest batch, never classify-copy folders."""
    from .media_catalog import load_media_queue
    directory = Path(directory).resolve()
    if manifests is None:
        if (directory / "manifest.json").is_file():
            manifests = [str(directory / "manifest.json")]
        else:
            batches = sorted(directory.glob("batch-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            manifests = []
            if batches:
                batch = json.loads(batches[0].read_text(encoding="utf-8"))
                if isinstance(batch, list):
                    manifests = [r["manifest"] for r in batch if isinstance(r, dict) and r.get("manifest")]
    items, warnings, summaries, seen = [], [], [], set()
    for manifest_path in dict.fromkeys(manifests or []):
        if should_stop():
            raise asyncio.CancelledError()
        try:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            platform, product_id = str(manifest.get("platform") or "unknown"), str(manifest.get("product_id") or "")
            product_key = f"{platform}:{product_id}" if product_id else str(Path(manifest_path).resolve())
            valid = load_media_queue(manifest_path)
            for item in valid:
                if should_stop():
                    raise asyncio.CancelledError()
                if any(p.casefold() == "processed" for p in Path(item["path"]).parts):
                    continue
                key = (product_key, item["media_type"], item["sha256"])
                if key not in seen:
                    seen.add(key)
                    items.append({**item, "platform": platform, "product_id": product_id,
                                  "product_key": product_key, "manifest_path": str(Path(manifest_path).resolve())})
            stats = manifest.get("stats") or {}
            summaries.append({"product_id": manifest.get("product_id", ""), "complete": manifest.get("complete") is True,
                              "reviews_read": stats.get("reviews_read", 0), "reason": manifest.get("reason", "")})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            warnings.append("一个商品清单无法载入：" + type(exc).__name__)
    # Show images first so opening the window gives a useful still preview.
    items.sort(key=lambda item: item["media_type"] != "image")
    return {"items": items, "warnings": warnings, "summaries": summaries,
            "manifest_count": len(set(manifests or []))}


def is_text_editor(widget):
    try:
        return widget.winfo_class() in {"Entry", "TEntry", "Text", "Spinbox", "TSpinbox", "TCombobox"}
    except (AttributeError, tk.TclError):
        return False


def discover_media(directory, limit=1000, should_stop=lambda: False):
    """Do not feed processed outputs back into a folder import."""
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError("请选择存在的素材目录。")
    if any(part.casefold() == "processed" for part in root.parts):
        return []
    found = []
    for base, dirs, files in os.walk(root, followlinks=False):
        if should_stop():
            raise asyncio.CancelledError()
        dirs[:] = sorted(d for d in dirs if d.casefold() != "processed" and not Path(base, d).is_symlink())
        for name in sorted(files):
            path = Path(base, name)
            if media_kind(path) and not path.is_symlink():
                found.append(str(path.resolve()))
                if len(found) >= limit:
                    return found
    return found


@dataclass(frozen=True)
class CanvasTransform:
    width: int
    height: int
    scale: float
    left: float
    top: float

    @classmethod
    def fit(cls, width, height, canvas_width, canvas_height):
        if min(width, height, canvas_width, canvas_height) <= 0:
            raise ValueError("图片或画布尺寸无效。")
        scale = min(canvas_width / width, canvas_height / height)
        return cls(width, height, scale, (canvas_width - width * scale) / 2,
                   (canvas_height - height * scale) / 2)

    @classmethod
    def actual_size(cls, width, height, canvas_width, canvas_height):
        if min(width, height, canvas_width, canvas_height) <= 0:
            raise ValueError("图片或画布尺寸无效。")
        return cls(width, height, 1.0, max(0, (canvas_width - width) / 2),
                   max(0, (canvas_height - height) / 2))

    def point(self, x, y):
        return (min(self.width, max(0, (x - self.left) / self.scale)),
                min(self.height, max(0, (y - self.top) / self.scale)))

    def rectangle(self, start, end):
        a, b = self.point(*start), self.point(*end)
        x, y = math.floor(min(a[0], b[0])), math.floor(min(a[1], b[1]))
        right, bottom = math.ceil(max(a[0], b[0])), math.ceil(max(a[1], b[1]))
        if right <= x or bottom <= y:
            return None
        return {"x": x, "y": y, "w": right - x, "h": bottom - y, "label": "手动区域"}

    def box(self, region):
        x, y = self.left + region["x"] * self.scale, self.top + region["y"] * self.scale
        return x, y, x + region["w"] * self.scale, y + region["h"] * self.scale

    def visible_crop(self, x, y, viewport_width, viewport_height):
        """Original-pixel crop for a scrolled viewport, never a huge Tk bitmap."""
        a, b = self.point(x, y), self.point(x + viewport_width, y + viewport_height)
        box = (math.floor(a[0]), math.floor(a[1]), math.ceil(b[0]), math.ceil(b[1]))
        return box if box[2] > box[0] and box[3] > box[1] else None


def validated_region(values, width, height, duration=None):
    """Reject invalid edits rather than silently applying an unintended mask."""
    region = dict(values)
    try:
        for key in ("x", "y", "w", "h"):
            value = float(values[key])
            if not math.isfinite(value) or value != int(value):
                raise ValueError()
            region[key] = int(value)
        if (region["x"] < 0 or region["y"] < 0 or region["w"] <= 0 or region["h"] <= 0
                or region["x"] + region["w"] > width or region["y"] + region["h"] > height):
            raise ValueError()
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("坐标须为整数，宽高大于 0，区域不能超出原图范围。") from exc
    if duration is not None:
        try:
            start, end = float(values.get("start", 0)), float(values.get("end", duration))
            if not all(math.isfinite(v) for v in (start, end, duration)) or not 0 <= start < end <= duration:
                raise ValueError()
        except (TypeError, ValueError) as exc:
            raise ValueError("视频时间须满足：0 ≤ 开始秒 < 结束秒 ≤ 视频时长。") from exc
        region.update(start=start, end=end)
    else:
        region.pop("start", None)
        region.pop("end", None)
    return region


def region_active_at(region, seconds):
    """Video boxes use half-open time intervals; still-image boxes are timeless."""
    if "start" not in region and "end" not in region:
        return True
    try:
        return float(region["start"]) <= seconds < float(region["end"])
    except (KeyError, TypeError, ValueError):
        return False


def _thumbnail(image):
    image = image.convert("RGB")
    image.thumbnail((1400, 1000))
    stream = io.BytesIO()
    image.save(stream, "PNG")
    return stream.getvalue()


SOURCE_CHANGED = "源文件内容已变化，旧框不能用于新文件。请清空旧框并重新载入预览后再识别或框选。"


def _source_sha256(path, should_stop=lambda: False):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            if should_stop():
                raise asyncio.CancelledError()
            digest.update(chunk)
    return digest.hexdigest()


def _check_source(path, expected, should_stop=lambda: False):
    if not expected:
        raise ValueError("尚未绑定源文件，请重新载入预览后再处理。")
    if _source_sha256(path, should_stop) != expected:
        raise ValueError(SOURCE_CHANGED)


def _merge_extra_candidates(regions, additions):
    """Different detector prefixes do not create duplicate text masks."""
    from .watermark import MAX_REGIONS
    def text(row):
        value = row.get('text') or str(row.get('label', '')).split('：', 1)[-1]
        return re.sub(r'[^\w]', '', unicodedata.normalize('NFKC', value)).casefold()
    dropped = 0
    for candidate in additions:
        duplicate = False
        for old in regions:
            if old.get('detection_kind') == 'graphic' or candidate.get('detection_kind') == 'graphic':
                continue
            if not text(candidate) or text(candidate) != text(old):
                continue
            intersection = max(0, min(old['x']+old['w'], candidate['x']+candidate['w'])-max(old['x'], candidate['x'])) * max(
                0, min(old['y']+old['h'], candidate['y']+candidate['h'])-max(old['y'], candidate['y']))
            if intersection >= .7 * min(old['w']*old['h'], candidate['w']*candidate['h']):
                duplicate = True
                break  # Preserve the existing footprint, not a growing union.
        if not duplicate:
            if len(regions) < MAX_REGIONS:
                regions.append(dict(candidate))
            else:
                dropped += 1
    return dropped


def _accept_source_identity(item, source_sha, *, detection=False):
    """Never silently bind existing masks to replacement bytes."""
    if not source_sha:
        return True  # Older/in-memory result fixtures; real workers supply SHA.
    regions = item.get('regions') or []
    pending = item.get('batch_proposed_regions') or []
    bound = item.get('regions_source_sha256') or (item.get('preview_source_sha256') if regions else None)
    hashes = {row['_ui_source_sha256'] for row in regions if row.get('_ui_source_sha256')}
    if bound:
        hashes.add(bound)
    if pending and item.get('batch_source_sha256'):
        hashes.add(item['batch_source_sha256'])
    if ((regions or pending) and (not hashes or hashes != {source_sha})) or (
            detection and item.get('preview_source_sha256') and item['preview_source_sha256'] != source_sha):
        item.update(source_changed=True, result_stale=bool(item.get('result')), state='源文件已变化 · 清空旧框后重载')
        return False
    return True


def _read_preview(path, seconds=0, full_resolution=False, *, should_stop=lambda: False):
    from PIL import Image, ImageOps
    source_sha = _source_sha256(path, should_stop)
    if media_kind(path) == "video":
        from .watermark_video import probe_video, extract_frame
        info = probe_video(path)
        # A frame exactly at EOF may not exist.
        seconds = min(max(0, seconds), max(0, info["duration"] - 0.05))
        image = extract_frame(path, seconds)
    else:
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
        info = {"width": image.width, "height": image.height}
    try:
        # Bounded reads add no full-file buffer, including for long videos.
        _check_source(path, source_sha, should_stop)
        result = {"meta": info, "preview": _thumbnail(image), "seconds": seconds,
                  "source_sha256": source_sha}
        if full_resolution:
            stream = io.BytesIO()
            image.save(stream, "PNG", compress_level=1)
            result["full_preview"] = stream.getvalue()
        return result
    finally:
        image.close()


def worker_main(job, events, stop):
    """Importable multiprocessing target, including for a frozen executable."""
    emit = lambda message: events.put(("progress", str(message)))
    try:
        action, source = job["action"], job.get("source")
        if stop.is_set() and action != "batch":
            raise asyncio.CancelledError()
        if action == "catalog":
            emit("正在读取采集清单，检查已保存的原始素材…")
            result = load_task_catalog(job["directory"], job.get("manifests"), stop.is_set)
        elif action == "scan":
            directory = Path(job["directory"])
            manifests = [str(p) for p in directory.rglob("manifest.json")
                         if "processed" not in {part.casefold() for part in p.parts}]
            result = load_task_catalog(directory, manifests, stop.is_set) if manifests else {
                "paths": discover_media(directory, should_stop=stop.is_set)}
        elif action == "preview":
            result = _read_preview(source, job.get("seconds", 0), job.get("full_resolution", False), should_stop=stop.is_set)
            if job.get("output"):
                try:
                    after = _read_preview(job["output"], job.get("seconds", 0), job.get("full_resolution", False))
                    result["after"] = after["preview"]
                    if "full_preview" in after:
                        result["full_after"] = after["full_preview"]
                except Exception as exc:
                    result["warnings"] = ["处理后预览暂不可用：" + type(exc).__name__]
        elif action == "detect":
            expected_sha = job.get('expected_source_sha256')
            if 'expected_source_sha256' in job:
                _check_source(source, expected_sha, stop.is_set)
            options = {key: job[key] for key in ("include_corner_text", "merchant_terms", "detection_profile") if key in job}
            if media_kind(source) == "video":
                from .watermark_video import detect_video
                result = detect_video(source, should_stop=stop.is_set, emit=emit, **options)
            else:
                from .watermark import detect_image
                if options.get('detection_profile') == 'product':
                    options['return_text_regions'] = True
                result = detect_image(source, should_stop=stop.is_set, emit=emit, **options)
                if (options.get('detection_profile') == 'product' and result.get('source_sha256')
                        and options.get('merchant_terms')):
                    from .watermark_text_extra import detect_extra_text
                    extra = detect_extra_text(source, options['merchant_terms'], base_detection=result,
                                              should_stop=stop.is_set, emit=emit)
                    drops = _merge_extra_candidates(result['regions'], extra.get('regions', []))
                    result['warnings'] += extra.get('warnings', [])
                    result['enhanced_detection'] = extra.get('evidence', {})
                    if extra.get('scan_complete') is False or drops:
                        result['scan_complete'] = False
                        result['warnings'].append('增强检测未完整返回，仍需核对未覆盖区域。')
            if 'expected_source_sha256' in job:
                if result.get('source_sha256', expected_sha) != expected_sha:
                    raise ValueError(SOURCE_CHANGED)
                _check_source(source, expected_sha, stop.is_set)
                result['source_sha256'] = expected_sha
        elif action == "batch":
            from .watermark_batch import run_batch
            result = run_batch(job["items"], output_root=job.get("output_root"),
                include_products=job.get("include_products", True),
                include_reviews=job.get("include_reviews", False), include_local=job.get("include_local", False),
                image_method=job.get("image_method", "smart"), merchant_terms=job.get("merchant_terms", ""),
                video_method=job.get("video_method", "smart"),
                detection_profile="product", include_corner_text=job.get("include_corner_text", True),
                should_stop=stop.is_set, emit=emit,
                on_event=lambda event: events.put(("batch_progress", dict(event))))
        elif action == "repair":
            expected_sha = job.get('expected_source_sha256')
            if 'expected_source_sha256' in job:
                _check_source(source, expected_sha, stop.is_set)
            options = {}
            if media_kind(source) == "video":
                from .watermark_video import repair_video as repair
                options["method"] = job.get("video_method", "delogo")
            else:
                from .watermark import repair_image as repair
                options["method"] = job.get("method", "auto")
            result = repair(source, job["regions"], output_root=None, should_stop=stop.is_set, emit=emit, **options)
            if 'expected_source_sha256' in job:
                # Backend snapshots must agree too, closing a replacement
                # between the pre-check and the backend's own source read.
                try:
                    processing = json.loads(Path(result['manifest']).read_text(encoding='utf-8'))
                except (KeyError, OSError, ValueError, TypeError) as exc:
                    raise ValueError('无法核验处理记录的源文件，结果未作为本图成功结果发布。') from exc
                if processing.get('source_sha256') != expected_sha:
                    raise ValueError(SOURCE_CHANGED + ' 本次输出未作为当前素材结果发布。')
                _check_source(source, expected_sha)  # A committed result is verified even on late cancel.
            if result.get("manifest"):
                try:
                    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
                    warnings = manifest.get("warnings", [])
                    if isinstance(warnings, list):
                        result["warnings"] = list(dict.fromkeys(
                            str(value) for value in list(result.get("warnings") or []) + warnings))
                except (OSError, ValueError, TypeError):
                    result.setdefault("warnings", []).append("结果已另存，处理说明暂时无法读取。")
            # Once the backend publishes its result, do not hide its path if a
            # subsequent preview fails or the user requests cancellation.
            if not stop.is_set():
                try:
                    after = _read_preview(result["output"], job.get("seconds", 0), job.get("full_resolution", False))
                    result["after"] = after["preview"]
                    if "full_preview" in after:
                        result["full_after"] = after["full_preview"]
                except Exception as exc:
                    result.setdefault("warnings", []).append("结果已另存，预览失败：" + type(exc).__name__)
        else:
            raise ValueError("未知处理操作。")
        if stop.is_set() and action not in {"repair", "batch"}:
            raise asyncio.CancelledError()
        events.put(("result", result))
    except (asyncio.CancelledError, InterruptedError):
        events.put(("cancelled", None))
    except Exception as exc:
        LOGGER.warning("Watermark worker failure: %s", type(exc).__name__)
        message = str(exc)[:500] if isinstance(exc, (ValueError, RuntimeError)) else type(exc).__name__
        events.put(("error", message))


class WorkerController:
    """One worker at a time; cooperative cancel permits backend temp cleanup."""
    def __init__(self, context=None):
        self.context = context or mp.get_context("spawn")
        self.process = self.events = self.stop = None
        self.job = None

    @property
    def busy(self):
        return self.process is not None

    def start(self, job):
        if self.busy:
            raise RuntimeError("已有处理正在进行。")
        self.events, self.stop = self.context.Queue(), self.context.Event()
        self.job = dict(job)
        self.process = self.context.Process(target=worker_main, args=(self.job, self.events, self.stop), daemon=False)
        try:
            self.process.start()
        except Exception:
            self.events.close()
            self.events.join_thread()
            self.process = self.events = self.stop = self.job = None
            raise

    def cancel(self):
        if self.stop is not None:
            self.stop.set()

    def drain(self, limit=100):
        if self.events is None:
            return []
        result = []
        while limit is None or len(result) < limit:
            try:
                result.append(self.events.get_nowait())
            except queue.Empty:
                break
        return result

    def reap(self):
        if self.process is None or self.process.is_alive():
            return False
        self.process.join(timeout=0)
        self.process.close()
        self.events.close()
        self.events.join_thread()
        self.process = self.events = self.stop = self.job = None
        return True


class WatermarkWindow:
    def __init__(self, parent, default_dir, on_closed=None, manifests=None, autoload=False):
        self.window = tk.Toplevel(parent)
        self.window.title("去水印 · 批量处理与结果审核")
        self.window.geometry(f"{min(1380, self.window.winfo_screenwidth() - 60)}x{min(900, self.window.winfo_screenheight() - 100)}")
        self.window.minsize(1020, 700)
        self.default_dir = Path(default_dir)
        self.manifests = manifests
        self.on_closed = on_closed
        self.close_callbacks = []
        self.closing = self.closed = False
        self.controller = WorkerController()
        self.files, self.items = [], {}
        self.current = None
        self.selected_region = None
        self.transform = None
        self.drag = None
        self.photos = {}
        self.full_images = {}
        self.terminal_received = False
        self.status = tk.StringVar(value="导入图片或视频，选中单个文件后识别并审核区域。")
        self.warning = tk.StringVar(value="")
        self.seconds = tk.StringVar(value="0")
        self.edits = {key: tk.StringVar(value="0") for key in ("x", "y", "w", "h", "start", "end")}
        self.controls = []
        self.auto_detect = tk.BooleanVar(value=False)
        self.include_corner_text = tk.BooleanVar(value=True)
        self.detection_profile = tk.StringVar(value="商家水印（商品）")
        self.merchant_terms = tk.StringVar(value="")
        self.repair_method = tk.StringVar(value="自动（按背景选择）")
        self.batch_method = tk.StringVar(value="智能修复（推荐）")
        self.video_method = tk.StringVar(value="智能视频精修（推荐）")
        self.batch_products = tk.BooleanVar(value=True)
        self.batch_reviews = tk.BooleanVar(value=False)
        self.batch_local = tk.BooleanVar(value=False)
        self.batch_summary = tk.StringVar(value="整批识别并处理，疑似项集中待确认。")
        self.queue_filter = tk.StringVar(value="全部")
        self.batch_result = None
        self.visible_files = []
        self.model_notice = tk.StringVar(value="")
        self.repair_limit = tk.StringVar(value=REPAIR_LIMIT)
        self.zoom = tk.StringVar(value="适应窗口")
        self.readonly_controls = []
        self.counter = tk.StringVar(value="尚未载入素材")
        self.context_notice = tk.StringVar(value="可直接使用采集记录中的原图；处理后的副本不会重复进入队列。")
        self._build()
        self.repair_method.trace_add("write", self._update_repair_info)
        self.video_method.trace_add("write", self._update_repair_info)
        self._update_repair_info()
        for variable in (self.detection_profile, self.merchant_terms, self.include_corner_text):
            variable.trace_add("write", self._detection_settings_changed)
        self.window.protocol("WM_DELETE_WINDOW", self.request_close)
        self.poll_after = self.window.after(100, self._poll)
        self._bind_shortcuts()
        if autoload:
            self.window.after_idle(lambda: self._start({"action": "catalog", "directory": str(self.default_dir), "manifests": manifests}))

    @property
    def busy(self):
        return self.controller.busy

    def focus(self):
        self.window.deiconify()
        self.window.lift()
        self.window.focus_set()

    def _button(self, frame, text, command):
        button = ttk.Button(frame, text=text, command=command)
        button.pack(side="left", padx=(0, 5))
        self.controls.append(button)
        return button

    def _build(self):
        style = ttk.Style(self.window)
        style.configure("Watermark.Title.TLabel", font=("Microsoft YaHei UI", 18, "bold"), foreground="#17324d")
        style.configure("Watermark.Nav.TButton", font=("Microsoft YaHei UI", 11, "bold"), padding=(16, 9))
        style.configure("Watermark.Action.TButton", font=("Microsoft YaHei UI", 10, "bold"), padding=(12, 7))
        style.configure("Watermark.Batch.TButton", font=("Microsoft YaHei UI", 10, "bold"), padding=(10, 1))
        root = ttk.Frame(self.window, padding=10)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(3, weight=1)
        ttk.Label(root, text="素材处理", style="Watermark.Title.TLabel").grid(row=0, column=0, sticky="w")
        context_label = ttk.Label(root, textvariable=self.context_notice, foreground="#536579", wraplength=1250)
        context_label.grid(row=1, column=0, sticky="ew", pady=(2, 6))
        bar = ttk.Frame(root)
        bar.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self.batch_button = self._button(bar, "一键去水印", self._batch_start)
        self.batch_button.configure(style="Watermark.Batch.TButton")
        self._button(bar, "添加本地图片 / 视频…", self._import_files)
        self._button(bar, "重新读取采集素材", lambda: self._start({"action": "catalog", "directory": str(self.default_dir), "manifests": self.manifests}))
        self._button(bar, "添加文件夹…", self._choose_dir)
        self.cancel_button = ttk.Button(bar, text="取消当前操作", command=self._cancel, state="disabled")
        self.cancel_button.pack(side="right")
        body = ttk.Panedwindow(root, orient="horizontal")
        body.grid(row=3, column=0, sticky="nsew")
        left, right = ttk.Frame(body), ttk.Frame(body)
        body.add(left, weight=0)
        body.add(right, weight=5)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(3, weight=1)
        batch_options = ttk.LabelFrame(left, text="一键处理范围", padding=6)
        batch_options.pack(fill="x", pady=(0, 5))
        for text, variable in (("商品主图 / 详情图 / 商品视频", self.batch_products),
                               ("包含买家秀 / 追评", self.batch_reviews), ("包含本地未分类素材", self.batch_local)):
            control = ttk.Checkbutton(batch_options, text=text, variable=variable)
            control.pack(anchor="w")
            self.controls.append(control)
        self.batch_method_combo = ttk.Combobox(batch_options, textvariable=self.batch_method,
            values=list(BATCH_METHODS), width=25, state="readonly")
        self.batch_method_combo.pack(fill="x", pady=(4, 2))
        self.controls.append(self.batch_method_combo)
        self.readonly_controls.append(self.batch_method_combo)
        ttk.Label(batch_options, text="视频修复").pack(anchor="w", pady=(4, 0))
        self.batch_video_combo = ttk.Combobox(batch_options, textvariable=self.video_method,
            values=list(VIDEO_METHODS), width=25, state="readonly")
        self.batch_video_combo.pack(fill="x", pady=(2, 2))
        self.controls.append(self.batch_video_combo)
        self.readonly_controls.append(self.batch_video_combo)
        ttk.Label(batch_options, text="智能视频：按背景选择快速 / AI。\n复杂视频较慢，可能需要数分钟。\n首次 AI 模型约 208 MB，离线可复用。\n一键独立检测，不使用现有框选。",
                  wraplength=240, foreground="#536579").pack(fill="x")
        self.batch_progress = ttk.Progressbar(left, mode="determinate", maximum=1)
        self.batch_progress.pack(fill="x", pady=(2, 3))
        ttk.Label(left, textvariable=self.batch_summary, wraplength=250, justify="left").pack(fill="x")
        batch_open = ttk.Button(left, text="打开本批结果目录", command=self._open_batch)
        batch_open.pack(fill="x", pady=4)
        ttk.Label(left, text="素材队列", font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w", pady=(0, 6))
        self.filter_combo = ttk.Combobox(left, textvariable=self.queue_filter, values=list(QUEUE_FILTERS),
                                         state="readonly", width=25)
        self.filter_combo.pack(fill="x", pady=(0, 4))
        self.filter_combo.bind("<<ComboboxSelected>>", self._change_filter)
        self.controls.append(self.filter_combo)
        self.readonly_controls.append(self.filter_combo)
        queue_frame = ttk.Frame(left)
        queue_frame.pack(fill="both", expand=True)
        self.file_list = tk.Listbox(queue_frame, width=29, exportselection=False, activestyle="none",
            font=("Microsoft YaHei UI", 10), background="#f2f5fa", foreground="#223449",
            selectbackground="#d5e9ff", selectforeground="#12385d", relief="flat", highlightthickness=0)
        self.file_list.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(queue_frame, orient="vertical", command=self.file_list.yview)
        scroll.pack(side="right", fill="y")
        self.file_list.configure(yscrollcommand=scroll.set)
        self.file_list.bind("<<ListboxSelect>>", self._select_file)
        queue_actions = ttk.Frame(left)
        queue_actions.pack(fill="x", pady=6)
        self._button(queue_actions, "移出队列", self._remove_file)
        ttk.Label(left, text="每张图的框选和结果分别保留。\n源文件去重，处理结果不会重复载入。",
                  wraplength=230, justify="left", foreground="#536579").pack(fill="x", pady=3)
        navigation = ttk.Frame(right, padding=(10, 0, 0, 5))
        navigation.grid(row=0, column=0, sticky="ew")
        self.previous_button = ttk.Button(navigation, text="‹  上一张", command=lambda: self._navigate(-1), style="Watermark.Nav.TButton")
        self.previous_button.pack(side="left")
        ttk.Label(navigation, textvariable=self.counter, font=("Microsoft YaHei UI", 12, "bold"), anchor="center").pack(side="left", fill="x", expand=True)
        self.next_button = ttk.Button(navigation, text="下一张  ›", command=lambda: self._navigate(1), style="Watermark.Nav.TButton")
        self.next_button.pack(side="right")
        self.controls.extend([self.previous_button, self.next_button])
        self.file_detail = tk.StringVar(value="")
        detail_label = ttk.Label(right, textvariable=self.file_detail, wraplength=1000, foreground="#536579", anchor="center")
        detail_label.grid(row=1, column=0, sticky="ew", pady=(0, 4))
        auto_bar = ttk.Frame(right, padding=(8, 0))
        auto_bar.grid(row=2, column=0, sticky="ew", pady=(0, 3))
        auto_row = ttk.Frame(auto_bar)
        auto_row.pack(fill="x")
        auto_check = ttk.Checkbutton(auto_row, text="新图片自动识别（不自动擦除）", variable=self.auto_detect,
                                     command=self._maybe_auto_detect)
        auto_check.pack(side="left", padx=(0, 14))
        corner_check = ttk.Checkbutton(auto_row, text="包含边角文字候选 · 图片 / 视频", variable=self.include_corner_text)
        corner_check.pack(side="left")
        self.controls.extend([auto_check, corner_check])
        detection_row = ttk.Frame(auto_bar)
        detection_row.pack(fill="x", pady=(5, 0))
        ttk.Label(detection_row, text="识别模式").pack(side="left")
        self.profile_combo = ttk.Combobox(detection_row, textvariable=self.detection_profile,
            values=list(DETECTION_PROFILES), width=17, state="readonly")
        self.profile_combo.pack(side="left", padx=(5, 12))
        ttk.Label(detection_row, text="店名 / 品牌").pack(side="left")
        self.merchant_entry = ttk.Entry(detection_row, textvariable=self.merchant_terms, width=22)
        self.merchant_entry.pack(side="left", padx=5, fill="x", expand=True)
        ttk.Label(detection_row, text="逗号分隔", foreground="#536579").pack(side="left")
        self.controls.extend([self.profile_combo, self.merchant_entry])
        self.readonly_controls.append(self.profile_combo)
        workspace = ttk.Frame(right)
        workspace.grid(row=3, column=0, sticky="nsew")
        workspace.columnconfigure(0, weight=1)
        workspace.columnconfigure(1, weight=0)
        workspace.rowconfigure(0, weight=1)
        previews = ttk.Frame(workspace)
        previews.grid(row=0, column=0, sticky="nsew")
        inspector_holder = ttk.Frame(workspace, padding=(10, 0, 0, 0))
        inspector_holder.grid(row=0, column=1, sticky="ns")
        inspector_canvas = tk.Canvas(inspector_holder, width=242, highlightthickness=0, background="SystemButtonFace")
        inspector_canvas.pack(side="left", fill="both", expand=True)
        inspector_scroll = ttk.Scrollbar(inspector_holder, orient="vertical", command=inspector_canvas.yview)
        inspector_scroll.pack(side="right", fill="y")
        inspector_canvas.configure(yscrollcommand=inspector_scroll.set)
        inspector = ttk.Frame(inspector_canvas)
        inspector_id = inspector_canvas.create_window(0, 0, window=inspector, anchor="nw", width=242)
        inspector.bind("<Configure>", lambda _: inspector_canvas.configure(scrollregion=inspector_canvas.bbox("all")))
        inspector_canvas.bind("<Configure>", lambda e: inspector_canvas.itemconfigure(inspector_id, width=e.width))
        self.inspector_canvas = inspector_canvas
        previews.columnconfigure((0, 1), weight=1)
        previews.rowconfigure(2, weight=1)
        view_bar = ttk.Frame(previews)
        view_bar.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 4))
        ttk.Label(view_bar, text="预览").pack(side="left")
        self.zoom_combo = ttk.Combobox(view_bar, textvariable=self.zoom, values=("适应窗口", "100%"), width=10, state="readonly")
        self.zoom_combo.pack(side="left", padx=5)
        self.zoom_combo.bind("<<ComboboxSelected>>", self._change_zoom)
        ttk.Label(view_bar, text="100% 可滚动定位；两侧同步", foreground="#536579").pack(side="left")
        self.controls.append(self.zoom_combo)
        self.readonly_controls.append(self.zoom_combo)
        ttk.Label(previews, text="原图 · 拖动添加框").grid(row=1, column=0, sticky="w")
        ttk.Label(previews, text="处理后（原文件不变）").grid(row=1, column=1, sticky="w")
        self.before_canvas = tk.Canvas(previews, background="#e5eaf0", width=450, height=340, highlightthickness=0)
        self.after_canvas = tk.Canvas(previews, background="#e5eaf0", width=450, height=340, highlightthickness=0)
        self.before_canvas.grid(row=2, column=0, sticky="nsew", padx=(0, 3))
        self.after_canvas.grid(row=2, column=1, sticky="nsew", padx=(3, 0))
        self.preview_xscroll = ttk.Scrollbar(previews, orient="horizontal", command=lambda *args: self._scroll_preview("x", *args))
        self.preview_xscroll.grid(row=3, column=0, columnspan=2, sticky="ew")
        self.preview_yscroll = ttk.Scrollbar(previews, orient="vertical", command=lambda *args: self._scroll_preview("y", *args))
        self.preview_yscroll.grid(row=2, column=2, sticky="ns")
        self.before_canvas.configure(xscrollcommand=self.preview_xscroll.set, yscrollcommand=self.preview_yscroll.set)
        self.before_canvas.bind("<Configure>", lambda _: self._draw())
        self.after_canvas.bind("<Configure>", lambda _: self._draw())
        self.before_canvas.bind("<ButtonPress-1>", self._press)
        self.before_canvas.bind("<B1-Motion>", self._motion)
        self.before_canvas.bind("<ButtonRelease-1>", self._release)
        for canvas in (self.before_canvas, self.after_canvas):
            canvas.bind("<MouseWheel>", self._wheel_preview)
            canvas.bind("<Shift-MouseWheel>", lambda event: self._wheel_preview(event, horizontal=True))
        timeline = self.timeline = ttk.Frame(inspector)
        timeline.pack(fill="x", pady=5)
        ttk.Label(timeline, text="视频预览秒").pack(side="left")
        self.time_entry = ttk.Entry(timeline, textvariable=self.seconds, width=9)
        self.time_entry.pack(side="left", padx=5)
        self.controls.append(self.time_entry)
        self._button(timeline, "刷新帧对比", self._refresh_preview)
        method_bar = ttk.Frame(inspector)
        method_bar.pack(fill="x", pady=(0, 4))
        ttk.Label(method_bar, text="图片修复方式").pack(anchor="w")
        self.method_combo = ttk.Combobox(method_bar, textvariable=self.repair_method,
            values=list(REPAIR_METHODS), width=27, state="readonly")
        self.method_combo.pack(fill="x", pady=(3, 0))
        self.model_label = ttk.Label(method_bar, textvariable=self.model_notice, wraplength=235,
                                     foreground="#536579", justify="left")
        self.model_label.pack(fill="x", pady=(3, 0))
        self.controls.append(self.method_combo)
        self.readonly_controls.append(self.method_combo)
        ttk.Label(method_bar, text="视频修复方式").pack(anchor="w", pady=(7, 0))
        self.video_method_combo = ttk.Combobox(method_bar, textvariable=self.video_method,
            values=list(VIDEO_METHODS), width=27, state="readonly")
        self.video_method_combo.pack(fill="x", pady=(3, 0))
        self.controls.append(self.video_method_combo)
        self.readonly_controls.append(self.video_method_combo)
        actions = self.actions = ttk.Frame(inspector)
        actions.pack(fill="x", pady=3)
        self._button(actions, "重新识别候选", self._detect)
        self.repair_button = self._button(actions, "处理并预览 · 另存", self._repair)
        self.repair_button.configure(style="Watermark.Action.TButton")
        for button in actions.winfo_children():
            button.pack_configure(side="top", fill="x", padx=0, pady=3)
        columns = ("label", "box", "time", "confidence")
        ttk.Label(inspector, text="区域候选 · 选中后修改", font=("Microsoft YaHei UI", 10, "bold")).pack(anchor="w", pady=(7, 3))
        self.region_list = ttk.Treeview(inspector, columns=columns, displaycolumns=("label", "time"), show="headings", selectmode="browse", height=3)
        for name, title, width in (("label", "候选 / 手工区域", 125), ("box", "x / y / 宽 / 高", 110),
                                   ("time", "秒", 75), ("confidence", "识别置信度", 70)):
            self.region_list.heading(name, text=title)
            self.region_list.column(name, width=width, minwidth=65)
        self.region_list.pack(fill="x", pady=4)
        self.region_list.bind("<<TreeviewSelect>>", self._select_region)
        region_actions = ttk.Frame(inspector)
        region_actions.pack(fill="x", pady=(0, 5))
        self._button(region_actions, "删除选框", self._delete_region)
        self._button(region_actions, "清空", self._clear_regions)
        fields = ttk.Frame(inspector)
        fields.pack(fill="x")
        for index, (key, label) in enumerate((("x", "x"), ("y", "y"), ("w", "宽"), ("h", "高"), ("start", "起秒"), ("end", "止秒"))):
            row, col = divmod(index, 2)
            ttk.Label(fields, text=label).grid(row=row, column=col * 2, sticky="e", pady=3)
            entry = ttk.Entry(fields, width=6, textvariable=self.edits[key])
            entry.grid(row=row, column=col * 2 + 1, padx=(2, 6), pady=3)
            self.controls.append(entry)
        apply_bar = ttk.Frame(inspector)
        apply_bar.pack(fill="x", pady=4)
        self._button(apply_bar, "应用区域修改", self._apply_region)
        self.region_info = tk.StringVar(value="")
        ttk.Label(inspector, textvariable=self.region_info, wraplength=220, foreground="#536579").pack(fill="x")
        output_bar = ttk.Frame(right)
        output_bar.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        self._button(output_bar, "打开处理结果 / 视频", self._open_result)
        self._button(output_bar, "打开结果目录", lambda: self._open_result(folder=True))
        footer = ttk.Frame(root)
        footer.grid(row=4, column=0, sticky="ew")
        shortcut = ttk.Label(footer, text="切图：← / →、A / D、PgUp / PgDn    删除选框：Delete    处理另存：Ctrl + Enter（文字编辑时保留原快捷键）",
                            foreground="#536579", wraplength=1250)
        shortcut.pack(fill="x", pady=(7, 2))
        limit = ttk.Label(footer, textvariable=self.repair_limit,
                          foreground="#6b7280", wraplength=1250)
        limit.pack(fill="x")
        warning_label = ttk.Label(footer, textvariable=self.warning, foreground="#9c4a00", wraplength=1250, justify="left")
        warning_label.pack(fill="x", pady=3)
        status_label = self.status_label = ttk.Label(footer, textvariable=self.status, wraplength=1250, justify="left")
        status_label.pack(fill="x")
        def rewrap(event):
            for label in (context_label, shortcut, limit, warning_label, status_label):
                label.configure(wraplength=max(500, event.width - 10))
        root.bind("<Configure>", rewrap)
        right.bind("<Configure>", lambda e: detail_label.configure(wraplength=max(500, e.width - 10)))

    def _item(self):
        return self.items.get(self.current)

    def _bind_shortcuts(self):
        for key in ("<Left>", "<Right>", "<Key-a>", "<Key-A>", "<Key-d>", "<Key-D>", "<Prior>", "<Next>", "<Delete>", "<Control-Return>"):
            self.window.bind(key, self._shortcut, add="+")

    def _shortcut(self, event):
        if is_text_editor(event.widget):
            return None
        key = event.keysym.lower()
        if self.busy or self.closing:
            return "break"
        if key in {"left", "a", "prior"}:
            self._navigate(-1)
        elif key in {"right", "d", "next"}:
            self._navigate(1)
        elif key == "delete":
            self._delete_region()
        elif key == "return" and event.state & 0x4:
            self._repair()
        else:
            return None
        return "break"

    def _navigate(self, step):
        visible = self._visible_paths()
        if self.busy or self.closing or not visible:
            return
        index = visible.index(self.current) if self.current in visible else -1
        target = min(len(visible) - 1, max(0, index + step))
        if target == index:
            return
        self.file_list.selection_clear(0, "end")
        self.file_list.selection_set(target)
        self.file_list.see(target)
        self._select_file()

    def _visible_paths(self):
        variable = getattr(self, "queue_filter", None)
        accepted = QUEUE_FILTERS.get(variable.get() if variable else "全部")
        return [path for path in self.files if accepted is None or self.items[path].get("batch_status") in accepted]

    def _change_filter(self, *_):
        if self.busy or self.closing:
            return
        if self._item() and self.selected_region is not None and not self._apply_region():
            return
        self._refresh_queue()
        if self.current not in self.visible_files:
            if self.visible_files:
                self.file_list.selection_set(0)
                self._select_file()
            else:
                self._save_view()
                self.current, self.selected_region = None, None
                self.full_images.clear()
                self._sync_regions()
                self._draw()
                self.file_detail.set("当前筛选没有素材。")

    def _batch_items(self):
        values = []
        for path in self.files:
            item = self.items[path]
            sources = set(item.get("source_types") or [])
            product = bool(sources & {"product_main", "product_detail", "product_video"})
            review = bool(sources & {"review_initial", "review_append"})
            if not ((product and self.batch_products.get()) or (review and self.batch_reviews.get())
                    or (not product and not review and self.batch_local.get())):
                continue
            # UI masks and thumbnails must never become implicit batch approval.
            value = {key: item[key] for key in ("media_type", "sha256", "source_types", "categories",
                     "product_key", "platform", "product_id", "manifest_path") if key in item}
            value.update(path=path, media_type=media_kind(path))
            values.append(value)
        return values

    def _batch_start(self):
        if self.busy or self.closing:
            return
        if not self.files:
            self._start({"action": "catalog", "directory": str(self.default_dir), "manifests": self.manifests,
                         "start_batch": True})
            return
        items = self._batch_items()
        if not items:
            self._warn(["所选范围没有素材。默认处理商品主图、详情图和商品视频；买家秀或本地未分类素材需单独勾选。"])
            return
        self.batch_result = None
        for value in items:
            self.items[value["path"]].update(batch_status="queued", state="待批量处理")
        self.batch_summary.set(f"准备处理 0 / {len(items)} 项；无需逐张框选。")
        self.batch_progress.configure(maximum=len(items), value=0)
        self._start({"action": "batch", "items": items, "output_root": str(self.default_dir / "processed"),
                     "include_products": self.batch_products.get(),
                     "include_reviews": self.batch_reviews.get(), "include_local": self.batch_local.get(),
                     "image_method": BATCH_METHODS[self.batch_method.get()],
                     "video_method": VIDEO_METHODS[self.video_method.get()],
                     "merchant_terms": self.merchant_terms.get().strip(),
                     "include_corner_text": self.include_corner_text.get()})

    def _batch_update(self, payload):
        counts = payload.get("counts") or {}
        total = payload.get("total", sum(counts.get(key, 0) for key in BATCH_STATES if key != "not_detected"))
        index = payload.get("index", payload.get("current", sum(counts.get(key, 0) for key in
                         ("processed", "unchanged", "needs_review", "failed", "cancelled"))))
        completed = sum(counts.get(key, 0) for key in ("processed", "unchanged", "needs_review", "failed", "cancelled"))
        stage = {"detect": "检测", "repair": "处理"}.get(payload.get("stage"), "批次")
        self.batch_summary.set(f"{stage} {index} / {total} · 已处理 {counts.get('processed', 0)}，原样 {counts.get('unchanged', 0)}\n"
            f"待确认 {counts.get('needs_review', 0)}（含部分处理），失败 {counts.get('failed', 0)}，取消 {counts.get('cancelled', 0)}")
        self.batch_progress.configure(maximum=max(1, total), value=completed)
        path = payload.get("path") or payload.get("source")
        if path in self.items and payload.get("status") in BATCH_STATES:
            state = payload["status"]
            self.items[path].update(batch_status=state, state=BATCH_STATES[state])
            step = {"detect": "识别候选", "repair": "修复另存"}.get(payload.get("stage"), BATCH_STATES[state])
            self.status.set(f"一键处理 {index} / {total} · {step} · {Path(path).name[:60]}")
        self._refresh_queue()

    def _batch_candidates(self, item):
        candidates = item.pop("batch_proposed_regions", None)
        if candidates is None:
            return
        meta = item.get("meta") or {}
        if not meta:
            item["batch_proposed_regions"] = candidates
            return
        source_sha = item.get('batch_source_sha256') or item.get('preview_source_sha256')
        if source_sha and not _accept_source_identity(item, source_sha, detection=True):
            item['batch_proposed_regions'] = candidates
            return
        regions = [region for region in item["regions"] if region.get("_ui_origin") != "automatic"]
        for region in candidates:
            try:
                value = validated_region(region, meta["width"], meta["height"], meta.get("duration"))
                if not any(all(old.get(key) == value.get(key) for key in ("x", "y", "w", "h", "start", "end")) for old in regions):
                    regions.append({**value, "_ui_origin": "automatic", **({'_ui_source_sha256': source_sha} if source_sha else {})})
            except ValueError:
                pass
        item["regions"] = regions
        if regions and source_sha:
            item['regions_source_sha256'] = source_sha

    def _batch_finished(self, result):
        self.batch_result = result
        for value in result.get("items", []):
            path = value.get("path") or value.get("source")
            if path not in self.items:
                continue
            item = self.items[path]
            state = value.get("status", "failed")
            item.update(batch_status=state, state=BATCH_STATES.get(state, state), detect_attempted=True,
                        warnings=list(value.get("warnings") or []), partially_processed=bool(value.get("partially_processed")),
                        merchant_tokens=list(value.get('evidence', {}).get('merchant_tokens', [])))
            item['batch_source_sha256'] = value.get('source_sha256') or value.get('sha256')
            if item["partially_processed"]:
                item["state"] = "部分处理 · 待确认"
                item["warnings"].append("部分确认区域已处理，仍有候选待确认；左侧为原图，右侧为部分修复结果。")
            if value.get("reason"):
                item["warnings"].append(str(value["reason"]))
            if value.get("output"):
                item["result"] = {key: value[key] for key in ("output", "manifest") if value.get(key)}
                item["after"] = None
                item["preview"] = None  # Refresh just this item when selected.
                item["result_stale"] = False
            elif item.get("result"):
                item["warnings"].append("本次没有生成新结果，当前预览保留上次处理结果，不代表本次成功。")
            if state == "needs_review":
                item["batch_proposed_regions"] = value.get("proposed_regions", value.get("regions", []))
                self._batch_candidates(item)
            elif item["regions"]:
                item["warnings"].append("批次采用独立检测；已有手动框不表示本批修复范围，可另行逐张处理。")
        total = len(result.get("items", []))
        self._batch_update({"counts": result.get("counts", {}), "index": total, "total": total})
        if result.get("video_fallback_count"):
            self.batch_summary.set(self.batch_summary.get() + f"\n{result['video_fallback_count']} 个视频含快速回退，详见本批结果。")
        self.full_images.clear()
        self._sync_regions()
        self._draw()
        self._warn(result.get("warnings", []))
        self.status.set("批次已停止，已完成的结果保留。" if result.get("status") == "cancelled" else
                        "批次已结束。可打开结果目录；筛选“待确认 / 失败”集中审核，无需逐张确认全部素材。")
        if self.current and self.current in self._visible_paths():
            self._refresh_preview()
        else:
            self._change_filter()

    def _open_batch(self):
        path = (self.batch_result or {}).get("batch_dir")
        if not path:
            self._warn(["当前还没有已生成的批次目录。"])
            return
        try:
            os.startfile(str(path))
        except OSError as exc:
            self._warn(["无法打开批次目录：" + type(exc).__name__])

    def _refresh_queue(self):
        if not hasattr(self, "file_list"):
            return
        self.file_list.configure(state="normal")
        self.file_list.delete(0, "end")
        images = videos = 0
        self.visible_files = self._visible_paths()
        for index, path in enumerate(self.visible_files):
            item = self.items[path]
            kind = "图片" if media_kind(path) == "image" else "视频"
            images += kind == "图片"
            videos += kind == "视频"
            state = item.get("state", "待查看")
            self.file_list.insert("end", f"{index + 1:03d}  {kind} · {state}")
            if item.get("batch_status") in {"needs_review", "failed"}:
                self.file_list.itemconfigure(index, foreground="#9c4a00")
            elif item.get("result"):
                self.file_list.itemconfigure(index, foreground="#167344")
        if self.current in self.visible_files:
            index = self.visible_files.index(self.current)
            self.file_list.selection_set(index)
            self.file_list.see(index)
            self.counter.set(f"第 {index + 1} / {len(self.visible_files)} 项  ·  {images} 张图片 / {videos} 个视频")
        else:
            self.counter.set(f"{len(self.visible_files)} 项素材  ·  {images} 张图片 / {videos} 个视频")
        self.file_list.configure(state="disabled" if self.busy else "normal")

    def _maybe_auto_detect(self):
        item = self._item()
        if (not self.busy and not self.closing and item and item.get("preview")
                and media_kind(self.current) == "image" and self.auto_detect.get()
                and not item.get("detect_attempted")):
            self._detect()

    def _detection_options(self):
        return {"include_corner_text": self.include_corner_text.get(),
                "merchant_terms": self.merchant_terms.get().strip(),
                "detection_profile": DETECTION_PROFILES[self.detection_profile.get()]}

    def _detection_settings_changed(self, *_args):
        options = self._detection_options()
        for item in self.items.values():
            if item.get("detect_options") != options:
                item["detect_attempted"] = False
        if self._item():
            self.status.set("识别设置已更改。点击“重新识别候选”替换旧自动候选；手动修改的区域保留。")

    def _warn(self, warnings):
        if isinstance(warnings, str):
            warnings = [warnings]
        text = "；".join(str(value) for value in warnings)
        self.warning.set(text[:360] + ("…（完整说明见处理记录）" if len(text) > 360 else ""))
        if text:
            LOGGER.warning("去水印提示：%s", text[:1600])

    def _import_files(self):
        paths = filedialog.askopenfilenames(parent=self.window, title="选择本地图片 / 视频",
            initialdir=str(self.default_dir), filetypes=[("图片和视频", "*.jpg *.jpeg *.png *.webp *.bmp *.avif *.tif *.tiff *.mp4 *.mov *.webm *.mkv *.avi *.m4v")])
        self._add_files(paths)

    def _add_files(self, paths, preview_first=True):
        added = 0
        known_hashes = {(item.get("product_key"), item.get("media_type"), item.get("sha256")) for item in self.items.values() if item.get("sha256")}
        for entry in paths:
            metadata = entry if isinstance(entry, dict) else {}
            path = metadata.get("path") if metadata else entry
            path = str(Path(path).resolve())
            key = (metadata.get("product_key"), metadata.get("media_type"), metadata.get("sha256"))
            if path in self.items:
                self.items[path].update({key: metadata[key] for key in ("media_type", "sha256", "source_types",
                    "categories", "product_key", "platform", "product_id", "manifest_path") if key in metadata})
                continue
            if key[2] and key in known_hashes:
                continue
            if path not in self.items and media_kind(path) and Path(path).is_file() and "processed" not in {p.casefold() for p in Path(path).parts}:
                self.files.append(path)
                self.items[path] = {**metadata, "regions": [], "meta": {}, "preview": None, "after": None,
                                    "result": None, "warnings": [], "state": "待查看", "detect_attempted": False}
                if key[2]:
                    known_hashes.add(key)
                added += 1
        self._refresh_queue()
        self.status.set(f"新增 {added} 个文件；请选择文件并审核区域。")
        if preview_first and self.current is None and self.visible_files:
            self.file_list.selection_set(0)
            self._select_file()

    def _choose_dir(self):
        directory = filedialog.askdirectory(parent=self.window, initialdir=str(self.default_dir), title="选择素材目录（排除 processed）")
        if directory:
            self._scan(directory)

    def _scan(self, directory):
        self._start({"action": "scan", "directory": str(directory)})

    def _remove_file(self):
        if self.busy or self.current is None:
            return
        visible = self._visible_paths()
        index = visible.index(self.current) if self.current in visible else 0
        del self.items[self.current]
        self.files.remove(self.current)
        self.full_images.clear()
        self.current = None
        self.selected_region = None
        self._sync_regions()
        self._draw()
        self.file_detail.set("")
        self._refresh_queue()
        if self.visible_files:
            self.file_list.selection_set(min(index, len(self.visible_files) - 1))
            self._select_file()

    def _select_file(self, event=None):
        if self.busy or self.closing:
            return
        selected = self.file_list.curselection()
        if not selected:
            return
        visible = self._visible_paths()
        if selected[0] >= len(visible):
            return
        target = visible[selected[0]]
        if target == self.current:
            return
        if self._item() and self.selected_region is not None:
            if not self._apply_region():
                self._refresh_queue()
                return
        if self._item():
            self._item()["selected_region"] = self.selected_region
        self._save_view()
        self.full_images.clear()
        self.current = target
        self.selected_region = self._item().get("selected_region")
        self.seconds.set("0")
        item = self._item()
        self._sync_regions()
        self._warn(item["warnings"])
        self._refresh_queue()
        if item["preview"] is None or self.zoom.get() == "100%":
            self._refresh_preview()
        else:
            self.seconds.set(str(item.get("preview_seconds", 0)))
            self._show_dimensions()
            self._draw()
            self._maybe_auto_detect()

    def _preview_seconds(self):
        try:
            seconds = float(self.seconds.get())
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError()
            duration = self._item()["meta"].get("duration")
            if duration is not None and seconds > duration:
                raise ValueError()
            return seconds
        except (ValueError, TypeError) as exc:
            raise ValueError("预览时间须位于 0 到视频时长之间。") from exc

    def _refresh_preview(self):
        if self.busy or not self.current:
            return
        try:
            seconds = self._preview_seconds()
        except ValueError as exc:
            self._warn([str(exc)])
            return
        result = self._item()["result"] or {}
        self._start({"action": "preview", "source": self.current, "seconds": seconds,
                     "output": result.get("output"), "full_resolution": self.zoom.get() == "100%"})

    def _show_dimensions(self):
        meta = self._item()["meta"]
        tail = f" · 时长 {meta['duration']:.3f} 秒" if "duration" in meta else ""
        labels = {"product_main": "商品主图", "product_detail": "商品详情", "product_video": "商品视频", "review_initial": "买家评价", "review_append": "追评"}
        sources = " / ".join(labels.get(x, x) for x in self._item().get("source_types", []))
        self.file_detail.set(f"{sources or Path(self.current).name[:55]}  ·  {meta.get('width', '?')} × {meta.get('height', '?')}{tail}")
        if "duration" in meta:
            self.timeline.pack(fill="x", pady=5, before=self.actions)
            self.time_entry.configure(state="normal")
            self.method_combo.configure(state="disabled")
            self.video_method_combo.configure(state="readonly")
        else:
            self.timeline.pack_forget()
            self.time_entry.configure(state="disabled")
            self.method_combo.configure(state="readonly")
            self.video_method_combo.configure(state="disabled")
        self._update_repair_info()

    def _update_repair_info(self, *_):
        is_video = self.current is not None and media_kind(self.current) == "video"
        video_mode = VIDEO_METHODS.get(self.video_method.get()) if is_video else None
        self.model_notice.set("视频使用快速局部插值。" if video_mode == "delogo" else
            "视频使用纹理精修。" + lama_model_notice() if is_video else lama_model_notice())
        self.repair_limit.set(VIDEO_SMART_LIMIT if video_mode == "smart" else VIDEO_QUALITY_LIMIT
            if video_mode == "lama_temporal" else LAMA_LIMIT
            if not is_video and REPAIR_METHODS.get(self.repair_method.get()) == "lama" else REPAIR_LIMIT)

    def _start(self, job):
        if self.busy or self.closing:
            return
        try:
            self.controller.start(job)
        except Exception as exc:
            self._warn(["无法启动处理进程：" + type(exc).__name__])
            return
        self.terminal_received = False
        if job.get("source") in self.items:
            self.items[job["source"]]["state"] = {"preview": "加载中", "detect": "识别中", "repair": "处理中"}.get(job["action"], "处理中")
            self._refresh_queue()
        for widget in self.controls:
            widget.configure(state="disabled")
        self.file_list.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        if job.get("action") == "repair" and job.get("method") == "lama":
            self.status.set("正在准备 AI 图片修复；首次约 208 MB 下载，本地 CPU 处理可能较慢，可取消。")
        elif job.get("action") == "batch":
            self.status.set("一键处理中：整批共用一个后台进程，按范围自动识别；无候选原样保留，疑似项集中待确认。")
            self._refresh_queue()
        else:
            self.status.set("后台处理中；可取消。首次文字识别可能需要等待模型初始化。")

    def _idle(self):
        for widget in self.controls:
            widget.configure(state="normal")
        for widget in self.readonly_controls:
            widget.configure(state="readonly")
        self.file_list.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        if self._item() and self._item().get("meta"):
            self._show_dimensions()

    def _cancel(self):
        job = self.controller.job or {}
        self.controller.cancel()
        self.cancel_button.configure(state="disabled")
        if job.get("action") == "batch":
            self.status.set("已请求停止批次；等待当前步骤清理，已完成文件和批次记录会保留。")
        elif job.get("action") == "repair" and job.get("method") == "lama":
            self.status.set("已请求取消；模型下载可取消，CPU 加载 / 推理需等当前步骤结束后清理。原文件保留。")
        else:
            self.status.set("已请求取消，等待当前步骤退出并清理临时文件；原文件保留。")

    def _detect(self):
        if self.busy or not self.current or not self._item()["meta"]:
            return
        if self.selected_region is not None and not self._apply_region():
            return
        if self._item().get('source_changed'):
            self._warn([SOURCE_CHANGED])
            return
        self._item()["detect_attempted"] = True
        options = self._detection_options()
        if options['detection_profile'] == 'product' and not options['merchant_terms']:
            options['merchant_terms'] = ','.join(self._item().get('merchant_tokens', [])[:12])[:1024]
        self._start({"action": "detect", "source": self.current, **options,
                     'expected_source_sha256': self._item().get('preview_source_sha256')})

    def _repair(self):
        item = self._item()
        if self.busy or not item or not item["meta"]:
            return
        if not item["regions"]:
            self._warn(["请先自动识别或拖动添加区域，再审核并处理另存。"])
            return
        expected_sha = item.get('regions_source_sha256') or item.get('preview_source_sha256')
        region_hashes = {r['_ui_source_sha256'] for r in item['regions'] if r.get('_ui_source_sha256')}
        if item.get('source_changed') or region_hashes and region_hashes != {expected_sha}:
            self._warn([SOURCE_CHANGED])
            return
        # Applying editor contents explicitly avoids silently discarding edits.
        if self.selected_region is not None and not self._apply_region():
            return
        try:
            meta = item["meta"]
            regions = [validated_region(r, meta["width"], meta["height"], meta.get("duration")) for r in item["regions"]]
            for region in regions:
                region.pop("_ui_origin", None)
                region.pop('_ui_source_sha256', None)
            unrefreshed_seconds = False
            if "duration" in meta:
                # The editable seconds field does not change the displayed
                # source frame until Refresh is pressed. Compare the output at
                # that same displayed instant, not at an unapplied input value.
                seconds = float(item.get("preview_seconds", 0))
                if not math.isfinite(seconds) or not 0 <= seconds <= meta["duration"]:
                    raise ValueError("当前预览时间无效，请先刷新视频帧。")
                try:
                    unrefreshed_seconds = float(self.seconds.get()) != seconds
                except ValueError:
                    unrefreshed_seconds = True
                self.seconds.set(str(seconds))
            else:
                seconds = self._preview_seconds()
        except ValueError as exc:
            self._warn([str(exc)])
            return
        job = {"action": "repair", "source": self.current, "regions": regions, "seconds": seconds,
               "full_resolution": self.zoom.get() == "100%", 'expected_source_sha256': expected_sha}
        if media_kind(self.current) == "image":
            method = REPAIR_METHODS[self.repair_method.get()]
            if method == "auto":
                # "自动（按背景选择）" must really choose by background. A
                # textured box would otherwise be filled by Telea and leave a
                # visible smear; route it to the AI backend like the batch path.
                try:
                    from .watermark_batch import _smart_method
                    method = _smart_method(self.current, regions, lambda: False)
                except Exception:
                    method = "auto"
            job["method"] = method
        else:
            job["video_method"] = VIDEO_METHODS[self.video_method.get()]
        if unrefreshed_seconds:
            self._warn(list(item.get("warnings") or []) + [
                f"前后对比沿用当前显示的 {seconds:g} 秒。查看其他时刻请先输入秒数并点击“刷新帧对比”。"])
        self._start(job)

    def _sync_regions(self):
        self.region_list.delete(*self.region_list.get_children())
        item = self._item()
        if item:
            if type(self.selected_region) is not int or not 0 <= self.selected_region < len(item["regions"]):
                self.selected_region = None
            item["selected_region"] = self.selected_region
            for i, r in enumerate(item["regions"]):
                timing = f"{r['start']:.3f}—{r['end']:.3f}" if "start" in r else "—"
                confidence = r.get("confidence")
                confidence = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "—"
                self.region_list.insert("", "end", iid=str(i), values=(r.get("label", "文字候选"),
                    f"{r['x']} / {r['y']} / {r['w']} / {r['h']}", timing, confidence))
            if self.selected_region is not None:
                self.region_list.selection_set(str(self.selected_region))
                self._select_region()
            else:
                self.region_info.set("")
                for variable in self.edits.values():
                    variable.set("0")
        self._draw()

    def _mark_edited(self):
        item = self._item()
        if item:
            item["state"] = "已改框 · 待处理" if item.get("result") else f"待审核 {len(item['regions'])} 区"
            item["result_stale"] = bool(item.get("result"))
            self._refresh_queue()

    def _select_region(self, event=None):
        selected = self.region_list.selection()
        if not selected or not self._item():
            return
        self.selected_region = int(selected[0])
        if not 0 <= self.selected_region < len(self._item()["regions"]):
            self.selected_region = None
            self._sync_regions()
            return
        self._item()["selected_region"] = self.selected_region
        region = self._item()["regions"][self.selected_region]
        if hasattr(self, "region_info"):
            confidence = region.get("confidence")
            if region.get('detection_kind') == 'graphic':
                self.region_info.set('图形匹配候选，请核对轮廓位置。\n图形证据不是文字识别置信度。')
            else:
                self.region_info.set((f"文字识别置信度 {confidence:.2f}\n不代表水印可信度。" if isinstance(confidence, (int, float)) else "手动区域，可随时修改。"))
        for key, variable in self.edits.items():
            variable.set(str(region.get(key, 0)))
        self._draw()

    def _apply_region(self):
        if self.busy or self.selected_region is None or not self._item():
            return False
        item, meta = self._item(), self._item()["meta"]
        if type(self.selected_region) is not int or not 0 <= self.selected_region < len(item["regions"]):
            self.selected_region = None
            item["selected_region"] = None
            self._sync_regions()
            return True
        values = dict(item["regions"][self.selected_region])
        values.update({key: var.get() for key, var in self.edits.items()})
        try:
            region = validated_region(values, meta["width"], meta["height"], meta.get("duration"))
        except ValueError as exc:
            self._warn([str(exc)])
            return False
        if item["regions"][self.selected_region] != region:
            region["_ui_origin"] = "manual"
            item["regions"][self.selected_region] = region
            self._mark_edited()
        self._sync_regions()
        return True

    def _delete_region(self):
        if self.busy or self.selected_region is None or not self._item():
            return
        if type(self.selected_region) is not int or not 0 <= self.selected_region < len(self._item()["regions"]):
            self.selected_region = None
            self._sync_regions()
            return
        self._item()["regions"].pop(self.selected_region)
        if not self._item()['regions']:
            self._item().pop('regions_source_sha256', None)
        self._mark_edited()
        self.selected_region = None
        self._item()["selected_region"] = None
        self._sync_regions()
        if self._item().get('source_changed') and not self._item()['regions']:
            self._item().pop('batch_proposed_regions', None)
            self._refresh_preview()

    def _clear_regions(self):
        if self.busy or not self._item():
            return
        self._item()["regions"] = []
        self._item().pop('regions_source_sha256', None)
        self._item().pop('batch_proposed_regions', None)
        self._mark_edited()
        self.selected_region = None
        self._item()["selected_region"] = None
        self._sync_regions()
        if self._item().get('source_changed'):
            self._refresh_preview()

    def _save_view(self):
        item = self._item()
        if item:
            item.setdefault("views", {})[self.zoom.get()] = (self.before_canvas.xview()[0], self.before_canvas.yview()[0])

    def _scroll_preview(self, axis, *args):
        if self.closing:
            return
        getattr(self.before_canvas, axis + "view")(*args)
        self._save_view()
        self._draw()

    def _wheel_preview(self, event, horizontal=False):
        delta = getattr(event, "delta", 0)
        if delta:
            steps = -max(1, abs(int(delta / 120))) if delta > 0 else max(1, abs(int(delta / 120)))
            self._scroll_preview("x" if horizontal else "y", "scroll", steps * 60, "units")
        return "break"

    def _change_zoom(self, _event=None):
        if self.busy or not self._item():
            return
        if self.zoom.get() == "100%" and "preview" not in self.full_images:
            self._refresh_preview()
        self._draw()

    def _cache_full_previews(self, result):
        from PIL import Image
        for key, name in (("full_preview", "preview"), ("full_after", "after")):
            if result.get(key):
                with Image.open(io.BytesIO(result[key])) as image:
                    self.full_images[name] = image.convert("RGB")

    def _draw(self):
        from PIL import Image, ImageTk
        item = self._item()
        self.transform = None
        viewport_width = max(1, min(self.before_canvas.winfo_width(), self.after_canvas.winfo_width()))
        viewport_height = max(1, min(self.before_canvas.winfo_height(), self.after_canvas.winfo_height()))
        actual = self.zoom.get() == "100%"
        t = None
        if item and item.get("meta"):
            meta = item["meta"]
            transform_type = CanvasTransform.actual_size if actual else CanvasTransform.fit
            t = transform_type(meta["width"], meta["height"], viewport_width, viewport_height)
        scrollregion = (0, 0, max(viewport_width, t.width * t.scale) if t else viewport_width,
                        max(viewport_height, t.height * t.scale) if t else viewport_height)
        view = item.get("views", {}).get(self.zoom.get(), (0, 0)) if item else (0, 0)
        for name, canvas in (("preview", self.before_canvas), ("after", self.after_canvas)):
            canvas.delete("all")
            canvas.configure(scrollregion=scrollregion, xscrollincrement=1, yscrollincrement=1)
            canvas.xview_moveto(view[0])
            canvas.yview_moveto(view[1])
            vx, vy = canvas.canvasx(0), canvas.canvasy(0)
            if not item or not item.get(name) or not item["meta"]:
                text = "采集素材会自动载入\n也可添加本地图片 / 视频" if name == "preview" else "处理后对比\n审核框选后点击“处理并预览”"
                canvas.create_text(vx + viewport_width / 2, vy + viewport_height / 2,
                                   text=text, fill="#66788a", justify="center", font=("Microsoft YaHei UI", 12))
                continue
            x, y = t.left, t.top
            if actual:
                image = self.full_images.get(name)
                if image is None:
                    canvas.create_text(vx + viewport_width / 2, vy + viewport_height / 2,
                                       text="载入原尺寸预览后可定位\n可切回“适应窗口”", fill="#66788a", justify="center")
                    continue
                crop = t.visible_crop(vx, vy, viewport_width, viewport_height)
                if crop is None:
                    continue
                resized = image.crop(crop)
                x, y = t.left + crop[0], t.top + crop[1]
            else:
                with Image.open(io.BytesIO(item[name])) as image:
                    resized = image.resize((max(1, round(t.width * t.scale)), max(1, round(t.height * t.scale))), Image.Resampling.LANCZOS)
            self.photos[name] = ImageTk.PhotoImage(resized, master=self.window)
            canvas.create_image(x, y, image=self.photos[name], anchor="nw")
            if name == "after" and item.get("result_stale"):
                canvas.create_text(vx + 8, vy + 8, text="上次结果 · 框选已更改，需重新处理", fill="#9c4a00", anchor="nw")
            if name == "preview":
                self.transform = t
                for i, region in enumerate(item["regions"]):
                    if not region_active_at(region, item.get("preview_seconds", 0)):
                        continue
                    color = "#ffd24a" if i == self.selected_region else "#ff6262"
                    canvas.create_rectangle(*t.box(region), outline=color, width=2)
                    x, y, _, _ = t.box(region)
                    canvas.create_text(x + 3, y + 3, text=str(i + 1), fill=color, anchor="nw")

    def _press(self, event):
        if self.busy or self.transform is None:
            return
        if self._item().get('source_changed'):
            self._warn([SOURCE_CHANGED])
            return
        self.drag = (self.before_canvas.canvasx(event.x), self.before_canvas.canvasy(event.y))
        self.before_canvas.focus_set()
        x, y = self.transform.point(*self.drag)
        for i in reversed(range(len(self._item()["regions"]))):
            r = self._item()["regions"][i]
            if not region_active_at(r, self._item().get("preview_seconds", 0)):
                continue
            if r["x"] <= x <= r["x"] + r["w"] and r["y"] <= y <= r["y"] + r["h"]:
                self.selected_region = i
                self.region_list.selection_set(str(i))
                self._select_region()
                break

    def _motion(self, event):
        if self.drag and not self.busy:
            self.before_canvas.delete("draft")
            self.before_canvas.create_rectangle(*self.drag, self.before_canvas.canvasx(event.x), self.before_canvas.canvasy(event.y),
                                                outline="#6bdcff", dash=(4, 3), tags="draft")

    def _release(self, event):
        start, self.drag = self.drag, None
        self.before_canvas.delete("draft")
        end = (self.before_canvas.canvasx(event.x), self.before_canvas.canvasy(event.y))
        if not start or self.busy or self.transform is None or max(abs(start[0] - end[0]), abs(start[1] - end[1])) < 4:
            return
        region = self.transform.rectangle(start, end)
        if not region:
            return
        item = self._item()
        if item.get('source_changed'):
            self._warn([SOURCE_CHANGED])
            return
        region["_ui_origin"] = "manual"
        if item.get('preview_source_sha256'):
            region['_ui_source_sha256'] = item['preview_source_sha256']
            item['regions_source_sha256'] = item['preview_source_sha256']
        if "duration" in item["meta"]:
            region.update(start=0.0, end=item["meta"]["duration"])
        item["regions"].append(region)
        self._mark_edited()
        self.selected_region = len(item["regions"]) - 1
        self._sync_regions()
        self.status.set("已添加手动区域；视频默认全时长，可编辑开始与结束秒。")

    def _handle_result(self, job, result):
        if job["action"] == "batch":
            self._batch_finished(result)
            return
        if job["action"] in {"scan", "catalog"}:
            if "items" in result:
                summaries = result.get("summaries", [])
                images = sum(m["media_type"] == "image" for m in result["items"])
                videos = sum(m["media_type"] == "video" for m in result["items"])
                incomplete = sum(not m.get("complete") for m in summaries)
                reviews = sum(m.get("reviews_read", 0) for m in summaries if isinstance(m.get("reviews_read", 0), int))
                self.context_notice.set(f"采集记录：{len(summaries)} 个商品，已保存 {images} 张图片 / {videos} 个视频；读取评价 {reviews} 条。"
                                        + (f"仍有 {incomplete} 个商品未完成采集。" if incomplete else ""))
                self._warn(result.get("warnings", []) + ([] if result["items"] else ["没有可用的已保存素材；请先完成采集，或添加本地文件。未采到的评论或视频不会补入队列。"]))
            self._add_files(result.get("items", result.get("paths", [])), preview_first=not job.get("start_batch"))
            if len(result.get("paths", [])) >= 1000:
                self._warn(["本次目录导入达到 1000 个文件上限，可分目录导入。"])
            if job.get("start_batch") and self.files:
                self._batch_start()
            return
        item = self.items.get(job["source"])
        if item is None:
            return
        if getattr(self, "current", job["source"]) != job["source"]:
            LOGGER.warning("去水印忽略与当前文件不匹配的显示结果")
            return
        warnings = list(result.get("warnings") or [])
        if job['action'] in {'preview', 'detect'} and not _accept_source_identity(
                item, result.get('source_sha256'), detection=job['action'] == 'detect'):
            item['warnings'] = warnings + [SOURCE_CHANGED]
            self._warn(item['warnings'])
            self.status.set(SOURCE_CHANGED)
            return
        if job["action"] == "preview":
            seconds = result.get("seconds", job["seconds"])
            self.full_images.clear()
            self._cache_full_previews(result)
            source_sha = result.get('source_sha256')
            if source_sha and item.get('preview_source_sha256') not in {None, source_sha}:
                item.update(result=None, result_stale=False)
                result.pop('after', None)
                self.full_images.pop('after', None)
                warnings.append('源文件已更新，旧处理结果不再与新原图配对。')
            item.update(meta=result["meta"], preview=result["preview"], after=result.get("after"), preview_seconds=seconds)
            if source_sha:
                item.update(preview_source_sha256=source_sha, source_changed=False)
                if item['regions']:
                    item['regions_source_sha256'] = source_sha
            self._batch_candidates(item)
            self.seconds.set(str(seconds))
            self._show_dimensions()
            item["state"] = BATCH_STATES.get(item.get("batch_status")) or ("已另存" if item.get("result") else "待审核" if item["regions"] else "待识别")
            if item.get("partially_processed") and item.get("batch_status") == "needs_review":
                item["state"] = "部分处理 · 待确认"
            self.status.set("预览已就绪。可自动识别，或在左图拖动新增区域。")
        elif job["action"] == "detect":
            item["detect_attempted"] = True
            meta, accepted = item["meta"], []
            if result.get("width") != meta["width"] or result.get("height") != meta["height"]:
                self._warn(["识别尺寸与预览不同，未应用候选。请重新载入素材后手动画框。"])
                return
            for region in result.get("regions", []):
                try:
                    accepted.append({**validated_region(region, meta["width"], meta["height"], meta.get("duration")),
                                     "_ui_origin": "automatic", **({'_ui_source_sha256': result['source_sha256']} if result.get('source_sha256') else {})})
                except ValueError:
                    warnings.append("已忽略越界或时间段无效的候选。")
            # Replace only untouched automatic candidates. Legacy/manual masks
            # and automatically found masks subsequently edited by hand survive.
            previous = item["regions"]
            selected_index = getattr(self, "selected_region", None)
            selected = previous[selected_index] if selected_index is not None and selected_index < len(previous) else None
            existing = [region for region in previous if region.get("_ui_origin") != "automatic"]
            for region in accepted:
                if not any(all(old.get(k) == region.get(k) for k in ("x", "y", "w", "h", "start", "end")) for old in existing):
                    existing.append(region)
            item["regions"] = existing
            if existing and result.get('source_sha256'):
                item['regions_source_sha256'] = result['source_sha256']
            elif not existing:
                item.pop('regions_source_sha256', None)
            self.selected_region = next((index for index, region in enumerate(existing) if region is selected), None)
            item["detect_options"] = {key: job[key] for key in ("include_corner_text", "merchant_terms", "detection_profile") if key in job}
            item["state"] = f"待审核 {len(existing)} 区" if existing else "未发现候选"
            if previous != existing and item.get("result"):
                item.update(result_stale=True, state="已改框 · 待处理")
            self.status.set(f"更新 {len(accepted)} 个文字候选，保留手动区域；请逐项核对后处理，识别置信度不等于水印可信度。")
        elif job["action"] == "repair":
            item["result"] = {k: v for k, v in result.items() if k not in {"after", "full_after"}}
            item["after"] = result.get("after")
            self.full_images.pop("after", None)
            self._cache_full_previews(result)
            item.update(state="已另存", result_stale=False)
            if item.get("batch_status"):
                item["batch_status"] = "processed"
            self.status.set("已另存：" + str(result["output"]) + "。请对比结果；原文件保留。")
        item["warnings"] = warnings
        self._warn(warnings)
        self._sync_regions()
        self._refresh_queue()
        if job["action"] == "preview":
            self._maybe_auto_detect()

    def _poll(self):
        if self.closed:
            return
        if self.controller.busy:
            job = self.controller.job
            messages = self.controller.drain()
            finished = not self.controller.process.is_alive()
            if finished:
                # A normal child flushes its queue before exiting.
                messages += self.controller.drain(limit=None)
                exitcode = self.controller.process.exitcode
                self.controller.reap()
                self._idle()
            deferred_result = None
            for kind, payload in messages:
                if kind == "progress":
                    self.status.set(str(payload))
                elif kind == "batch_progress":
                    self._batch_update(payload)
                elif kind == "result":
                    self.terminal_received = True
                    deferred_result = payload
                elif kind == "cancelled":
                    self.terminal_received = True
                    if job.get("source") in self.items:
                        self.items[job["source"]]["state"] = "已取消"
                        self._refresh_queue()
                    self.status.set("当前操作已取消，原文件保留。")
                elif kind == "error":
                    self.terminal_received = True
                    if job.get("source") in self.items:
                        self.items[job["source"]].update(state="操作失败", warnings=[str(payload)])
                        self._refresh_queue()
                    self._warn(["操作未完成：" + str(payload)])
            completed_received = self.terminal_received
            if deferred_result is not None:
                # Queue result can precede process exit. Defer UI work that may
                # start another worker (directory import preview) until reaping.
                if not finished:
                    self.pending_result = (job, deferred_result)
                elif not self.closing:
                    self._handle_result(job, deferred_result)
            if finished:
                if hasattr(self, "pending_result"):
                    saved_job, result = self.pending_result
                    del self.pending_result
                    if not self.closing:
                        self._handle_result(saved_job, result)
                if not completed_received:
                    self._warn([f"处理进程意外结束（退出码 {exitcode}）；请检查结果目录，不能视为成功。"])
        if self.closing and not self.controller.busy:
            self._finish_close()
            return
        self.poll_after = self.window.after(100, self._poll)

    def _open_result(self, folder=False):
        item = self._item()
        result = item.get("result") if item else None
        if not result:
            self._warn(["当前文件还没有处理结果。"])
            return
        path = Path(result["output"])
        if folder:
            path = path.parent
        try:
            os.startfile(str(path))
        except OSError as exc:
            self._warn(["无法打开结果：" + type(exc).__name__])

    def request_close(self, on_closed=None):
        if on_closed and on_closed not in self.close_callbacks:
            self.close_callbacks.append(on_closed)
        if self.closed:
            if on_closed:
                on_closed()
            return
        self.closing = True
        if self.busy:
            self._cancel()
            self.window.title("正在取消处理并清理 · 完成后关闭")
        else:
            self._finish_close()

    def _finish_close(self):
        self.closed = True
        if getattr(self, "poll_after", None):
            self.window.after_cancel(self.poll_after)
            self.poll_after = None
        if hasattr(self, "full_images"):
            self.full_images.clear()
        self.window.destroy()
        if self.on_closed:
            self.on_closed()
        for callback in self.close_callbacks:
            callback()


def open_window(parent, default_dir, on_closed=None, manifests=None, autoload=False):
    return WatermarkWindow(parent, default_dir, on_closed, manifests, autoload)
