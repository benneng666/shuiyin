"""Pinned ONNX LaMa CPU inference, downloaded separately from the executable.

Only the fixed, SHA256-verified ONNX artifact is loaded. No remote Python,
pickle, model hub code, custom operators or GPU provider is used.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
import queue
import tempfile
import threading
import time

MODEL_REVISION = "c3c0c9e468934d62e79c329e35d82dd09ff8c444"
MODEL_SHA256 = "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6"
MODEL_SIZE = 208_044_816
MODEL_NAME = "lama_fp32.onnx"
MODEL_REPOSITORY = "https://huggingface.co/Carve/LaMa-ONNX"
MODEL_URL = f"{MODEL_REPOSITORY}/resolve/{MODEL_REVISION}/{MODEL_NAME}?download=true"
MODEL_MIRROR_URL = MODEL_URL.replace("https://huggingface.co/", "https://hf-mirror.com/")
MAX_ROIS = 32
# A 512x512 ONNX input does not mean the ROI may be downscaled. Tiles keep the
# original pixel grid; shrinking a large ROI to 512 and interpolating it back
# was the measured cause of soft/blurred texture after watermark removal.
MODEL_INPUT = 512
TILE_STRIDE = 384
# Mirrored context for masks touching the image edge, where the model otherwise
# sees no exterior background and invents dark blocks/horizontal streaks.
EDGE_CONTEXT = 96
MAX_TILES_PER_ROI = 96
_LOCK = threading.Lock()
_SESSION = None
_SESSION_KEY = None
_VERIFIED_KEY = None


class LamaError(RuntimeError):
    """Safe, user-facing failure; no signed download URL is included."""


def _stop(should_stop):
    if should_stop():
        raise asyncio.CancelledError()


def _emit(emit, message):
    try:
        emit(message)
    except Exception:
        pass


def model_path():
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".local" / "share")
    return base / "BuyerShowValidator" / "models" / MODEL_NAME


def _file_key(path):
    try:
        stat = path.stat()
        return str(path.resolve()), stat.st_size, stat.st_mtime_ns
    except OSError:
        return None


def model_status():
    """Read-only size check; integrity is always checked before model loading."""
    path = model_path()
    key = _file_key(path)
    return {"path": str(path), "installed": bool(key and key[1] == MODEL_SIZE),
            "size_bytes": key[1] if key else 0, "expected_size_bytes": MODEL_SIZE,
            "integrity_verified_in_this_process": bool(key and key == _VERIFIED_KEY),
            "sha256": MODEL_SHA256, "license": "Apache-2.0 (publisher model card)"}


def _verify(path, should_stop):
    global _VERIFIED_KEY
    key = _file_key(path)
    if not key or key[1] != MODEL_SIZE:
        return False
    if key == _VERIFIED_KEY:
        return True
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            _stop(should_stop)
            digest.update(chunk)
    if digest.hexdigest() != MODEL_SHA256:
        return False
    _VERIFIED_KEY = _file_key(path)
    return True


def _download_stream(url, offset, messages, aborted):
    """Only this daemon reads sockets; the cancellable caller owns disk writes."""
    def send(value):
        while not aborted.is_set():
            try:
                messages.put(value, timeout=.1)
                return
            except queue.Full:
                pass
    try:
        import httpx
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30, connect=15)) as client:
            with client.stream("GET", url, headers=headers) as response:
                response.raise_for_status()
                if response.url.scheme != "https":
                    raise LamaError("模型下载被重定向到非安全地址。")
                resumed = response.status_code == 206
                if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise LamaError("模型下载分段信息不匹配。")
                send(("start", resumed))
                for chunk in response.iter_bytes(64 * 1024):
                    if aborted.is_set():
                        return
                    send(("data", chunk))
        send(("done", None))
    except Exception as exc:
        send(("error", type(exc).__name__))


def ensure_model(should_stop=lambda: False, emit=lambda message: None):
    """Download lazily, cancel between socket chunks, then hash before publish."""
    _stop(should_stop)
    path = model_path()
    if _verify(path, should_stop):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".lama-download-", suffix=".part", dir=path.parent)
    os.close(descriptor)
    stage = Path(name)
    try:
        for attempt, url in enumerate((MODEL_URL, MODEL_MIRROR_URL, MODEL_MIRROR_URL)):
            _stop(should_stop)
            if attempt:
                _emit(emit, "官方模型传输未完成，尝试镜像续传；仍须通过同一 SHA256 校验。")
            offset = stage.stat().st_size
            messages, aborted = queue.Queue(maxsize=4), threading.Event()
            reader = threading.Thread(target=_download_stream, args=(url, offset, messages, aborted), daemon=True)
            reader.start()
            last_progress, complete = 0, False
            try:
                with stage.open("r+b") as out:
                    out.seek(offset)
                    while True:
                        _stop(should_stop)
                        if time.monotonic() - last_progress >= 1:
                            _emit(emit, f"下载 AI 模型：已下 {out.tell() / 1_000_000:.1f} / {MODEL_SIZE / 1_000_000:.1f} MB，可取消。")
                            last_progress = time.monotonic()
                        try:
                            kind, value = messages.get(timeout=.1)
                        except queue.Empty:
                            continue
                        if kind == "start" and not value:
                            out.seek(0)
                            out.truncate()
                        elif kind == "data":
                            if out.tell() + len(value) > MODEL_SIZE:
                                raise LamaError("模型下载大小不符合已核验版本，已拒绝加载。")
                            out.write(value)
                        elif kind == "done":
                            complete = True
                            break
                        elif kind == "error":
                            break
            finally:
                aborted.set()
                reader.join(timeout=.2)
            if complete:
                _emit(emit, "正在核对 AI 模型 SHA256…")
                if not _verify(stage, should_stop):
                    raise LamaError("AI 模型完整性校验失败，已拒绝加载；请重试下载。")
                _stop(should_stop)
                os.replace(stage, path)
                _emit(emit, "AI 模型下载并校验完成；之后可离线使用。")
                return path
        raise LamaError("AI 模型下载未完成，请检查网络后重试；也可先使用其他图片修复方式。")
    finally:
        stage.unlink(missing_ok=True)


def _get_session(should_stop, emit):
    global _SESSION, _SESSION_KEY
    path = ensure_model(should_stop, emit)
    key = _file_key(path)
    if _SESSION is None or _SESSION_KEY != key:
        _emit(emit, "正在载入 LaMa CPU 模型；本步骤结束后响应取消。")
        _stop(should_stop)
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = min(4, max(1, os.cpu_count() or 1))
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        try:
            session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        except Exception:
            raise LamaError("已校验的 LaMa 模型无法在当前 CPU 运行时载入。") from None
        inputs = {node.name: node for node in session.get_inputs()}
        if (set(inputs) != {"image", "mask"} or inputs["image"].shape[1:] != [3, 512, 512]
                or inputs["mask"].shape[1:] != [1, 512, 512]
                or any(node.type != "tensor(float)" for node in inputs.values())):
            raise LamaError("LaMa 模型输入契约不匹配，已停止。")
        _SESSION, _SESSION_KEY = session, key
        _stop(should_stop)
    return _SESSION


def _roi_groups(mask):
    import cv2
    import numpy as np
    height, width = mask.shape
    joined = cv2.dilate(mask.astype(np.uint8), np.ones((65, 65), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    if count - 1 > MAX_ROIS:
        raise LamaError(f"AI 修复区域过于分散（超过 {MAX_ROIS} 组），请分次处理。")
    groups = []
    for label in range(1, count):
        sx, sy, sw, sh = stats[label, :4]
        yy, xx = np.where(mask[sy:sy+sh, sx:sx+sw])
        if not len(xx):
            continue
        x, y = int(xx.min()+sx), int(yy.min()+sy)
        w, h = int(xx.max()-xx.min()+1), int(yy.max()-yy.min()+1)
        padding = max(64, min(256, max(w, h) // 3))
        side = max(512, max(w, h) + padding * 2)
        rw, rh = min(width, side), min(height, side)
        left = min(width-rw, max(0, round(x+w/2-rw/2)))
        top = min(height-rh, max(0, round(y+h/2-rh/2)))
        owned = (labels[top:top+rh, left:left+rw] == label) & mask[top:top+rh, left:left+rw]
        groups.append(((left, top, rw, rh), owned))
    return groups


def _tile_starts(length):
    """Overlapping 512 tiles covering *length* without resizing the image."""
    if length <= MODEL_INPUT:
        return [0]
    starts = list(range(0, length - MODEL_INPUT + 1, TILE_STRIDE))
    if starts[-1] != length - MODEL_INPUT:
        starts.append(length - MODEL_INPUT)
    return starts


def _predict_roi(session, rgb, mask, should_stop):
    """Full-resolution tiled inference; never downscale the published pixels.

    Tiles are blended only with each other (a raised-cosine-like ramp), never
    with the watermark-bearing original. A mask touching an image edge gets a
    mirrored context border so the model has exterior background to continue.
    """
    import cv2
    import numpy as np
    height, width = mask.shape
    binary = (mask != 0).astype(np.uint8)
    touches_edge = bool(binary[0].any() or binary[-1].any()
                        or binary[:, 0].any() or binary[:, -1].any())
    pad = EDGE_CONTEXT if touches_edge else 0
    if pad:
        # Mirroring the ROI copies the watermark colour into the invented
        # exterior context: a red corner badge or promo strip then gets
        # repainted as a large red/orange smear. Continue with the *clean*
        # background colour estimated from unmasked pixels instead.
        clean = rgb[binary == 0].reshape(-1, 3)
        background = (np.median(clean, axis=0) if len(clean)
                      else np.full(3, 255, dtype=np.float64))
        value = tuple(int(np.clip(round(v), 0, 255)) for v in background)
        source = cv2.copyMakeBorder(rgb, pad, pad, pad, pad,
                                    cv2.BORDER_CONSTANT, value=value)
        selected = cv2.copyMakeBorder(binary, pad, pad, pad, pad,
                                      cv2.BORDER_CONSTANT, value=0)
    else:
        source, selected = rgb, binary
    total_h, total_w = selected.shape
    starts_y, starts_x = _tile_starts(total_h), _tile_starts(total_w)
    if len(starts_y) * len(starts_x) > MAX_TILES_PER_ROI:
        raise LamaError("AI 修复区域过大，已拒绝执行以免生成低质量结果；请缩小选区或分次处理。")
    accumulator = np.zeros((total_h, total_w, 3), dtype=np.float32)
    weights = np.zeros((total_h, total_w, 1), dtype=np.float32)
    tiles = 0
    for top in starts_y:
        for left in starts_x:
            bottom, right = min(top + MODEL_INPUT, total_h), min(left + MODEL_INPUT, total_w)
            y0, x0 = max(0, bottom - MODEL_INPUT), max(0, right - MODEL_INPUT)
            patch_mask = selected[y0:bottom, x0:right]
            if not patch_mask.any():
                continue
            patch = source[y0:bottom, x0:right]
            ph, pw = patch.shape[:2]
            image = cv2.copyMakeBorder(patch, 0, MODEL_INPUT - ph, 0, MODEL_INPUT - pw,
                                       cv2.BORDER_REFLECT_101)
            tile_mask = cv2.copyMakeBorder(patch_mask.astype(np.float32),
                                           0, MODEL_INPUT - ph, 0, MODEL_INPUT - pw,
                                           cv2.BORDER_CONSTANT, value=0)
            inputs = {"image": np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32) / 255,
                      "mask": np.ascontiguousarray((tile_mask > 0)[None, None], dtype=np.float32)}
            _stop(should_stop)
            output = session.run(None, inputs)[0]
            _stop(should_stop)
            if output.shape != (1, 3, MODEL_INPUT, MODEL_INPUT) or not np.isfinite(output).all():
                raise LamaError("AI 推理输出格式或数值异常，未发布处理结果。")
            # The pinned Carve export returns RGB in 0..255 (not 0..1).
            predicted = np.clip(output[0].transpose(1, 2, 0)[:ph, :pw], 0, 255).astype(np.float32)
            ramp_y = np.minimum(np.linspace(.35, 1, ph, dtype=np.float32),
                                np.linspace(1, .35, ph, dtype=np.float32))[:, None]
            ramp_x = np.minimum(np.linspace(.35, 1, pw, dtype=np.float32),
                                np.linspace(1, .35, pw, dtype=np.float32))[None, :]
            tile_weight = np.clip(ramp_y * ramp_x, .1, 1)[:, :, None]
            accumulator[y0:bottom, x0:right] += predicted * tile_weight
            weights[y0:bottom, x0:right] += tile_weight
            tiles += 1
    if not tiles:
        raise LamaError("AI 修复未找到有效的局部推理区域。")
    # Untouched tiles keep the original RGB; only selected mask pixels are used
    # by the caller, so this never publishes an invented full-image result.
    filled = np.where(weights > 0,
                      accumulator / np.maximum(weights, 1e-6), source).clip(0, 255).astype(np.uint8)
    # Only approved pixels may carry generated content. Tiles overlap and a
    # tile also predicts its unmasked area; never let that leak to the caller.
    filled[selected == 0] = source[selected == 0]
    return filled[pad:pad + height, pad:pad + width]


def inpaint_rgb(rgb, mask, *, should_stop=lambda: False, emit=lambda message: None):
    """Return (RGB uint8, diagnostics), modifying only supplied nonzero mask.

    A process-local CPU session is reused. Native model calls are not forcibly
    interrupted; cancellation is checked before/after each ROI and tile.
    """
    import numpy as np
    _stop(should_stop)
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
            or not isinstance(mask, np.ndarray) or mask.shape != rgb.shape[:2]
            or mask.dtype.kind not in 'buif' or not np.isfinite(mask).all()
            or not rgb.shape[0] or not rgb.shape[1] or rgb.shape[0]*rgb.shape[1] > 40_000_000):
        raise ValueError("AI 修复需要 RGB uint8 图片和同尺寸单通道遮罩。")
    selected = mask != 0
    if not selected.any():
        return rgb.copy(), {"roi_count": 0, "inference_seconds": 0}
    groups = _roi_groups(selected)
    while not _LOCK.acquire(timeout=.1):
        _stop(should_stop)
    try:
        session = _get_session(should_stop, emit)
        result, rois, elapsed = rgb.copy(), [], 0.0
        for index, ((x, y, w, h), owned) in enumerate(groups):
            _stop(should_stop)
            _emit(emit, f"AI 修复局部区域 {index+1}/{len(groups)}；当前 CPU 推理结束后可取消。")
            local_mask = selected[y:y+h, x:x+w]
            started = time.perf_counter()
            predicted = _predict_roi(session, rgb[y:y+h, x:x+w], local_mask, should_stop)
            seconds = time.perf_counter() - started
            elapsed += seconds
            # A mask-edge feather against the *original* blended watermark
            # strokes back into the result (visible residual letter edges).
            # Use the prediction at 100% on every approved pixel; LaMa already
            # continues the surrounding texture, and mirror context handles
            # regions touching the image boundary.
            painted = np.clip(np.rint(predicted), 0, 255).astype(np.uint8)
            result[y:y+h, x:x+w][owned] = painted[owned]
            rois.append({"x": x, "y": y, "w": w, "h": h, "inference_seconds": round(seconds, 3)})
        _stop(should_stop)
        return result, {"model": "Carve/LaMa-ONNX/lama_fp32.onnx", "revision": MODEL_REVISION,
            "sha256": MODEL_SHA256, "size_bytes": MODEL_SIZE, "model_source": MODEL_REPOSITORY,
            "license": "Apache-2.0 (publisher model card)", "execution_provider": "CPUExecutionProvider",
            "roi_count": len(rois), "rois": rois, "inference_seconds": round(elapsed, 3),
            "input_size": [MODEL_INPUT, MODEL_INPUT], "tiled_full_resolution": True,
            "tile_stride": TILE_STRIDE, "edge_context_pixels": EDGE_CONTEXT,
            "mask_edge_composition": "prediction_only_no_watermark_mixed_back",
            "inside_mask_feather_pixels": 0,
            "outside_mask_pixels_preserved": True}
    finally:
        _LOCK.release()
