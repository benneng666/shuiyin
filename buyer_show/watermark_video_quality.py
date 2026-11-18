"""Bounded CPU key-frame inpainting for explicitly approved stationary masks.

The input is only the marked video. Neural pixels are predictions, never a
claim of recovering the original scene. The transport owns decoding/timestamps.
"""
from __future__ import annotations

import asyncio
import math
import time

import cv2
import numpy as np

from . import watermark_ai


class PrecisionMaskUnavailable(ValueError):
    """Automatic enhancement lacks reliable fine-mask evidence; use old delogo."""


def _stop(callback):
    if callback():
        raise asyncio.CancelledError()


def _border_fill(rgb, box):
    """Weighted four-border interpolation; copied image outside the box."""
    x, y, w, h = box
    height, width = rgb.shape[:2]
    left = rgb[y:y+h, max(0, x-1)].astype(np.float32)[:, None]
    right = rgb[y:y+h, min(width-1, x+w)].astype(np.float32)[:, None]
    top = rgb[max(0, y-1), x:x+w].astype(np.float32)[None]
    bottom = rgb[min(height-1, y+h), x:x+w].astype(np.float32)[None]
    xx = np.arange(1, w+1, dtype=np.float32)[None, :, None]
    yy = np.arange(1, h+1, dtype=np.float32)[:, None, None]
    weights = (1/xx, 1/(w+1-xx), 1/yy, 1/(h+1-yy))
    if x == 0 or y == 0 or x+w == width or y+h == height:
        # A clamped image boundary lies INSIDE the selected watermark. Never
        # blend it back as if it were an unmarked neighboring sample.
        available = [x > 0, x+w < width, y > 0, y+h < height]
        numerator = np.zeros((h, w, 3), np.float32)
        denominator = np.zeros((h, w, 1), np.float32)
        for samples, weight, valid in zip((left, right, top, bottom), weights, available):
            if valid:
                numerator += samples * weight
                denominator += weight
        if not np.all(denominator > 0):
            raise ValueError('修复框周围没有可用的背景像素。')
        value = numerator / denominator
    else:
        value = sum(a*b for a, b in zip((left, right, top, bottom), weights)) / sum(weights)
    result = rgb.copy()
    result[y:y+h, x:x+w] = np.clip(np.rint(value), 0, 255).astype(np.uint8)
    return result


def _crossing_structure(rgb, box):
    """Inspect real pixels just OUTSIDE the approved rectangle, never its text.

    A nearby object is insufficient: structured contrast must intersect at least
    two different sides of the rectangle. This also supplies a scene topology
    signature so changes cannot inherit the previous repaired object blindly.
    """
    x, y, w, h = box
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    profiles = [None] * 4
    if y >= 3:
        profiles[0] = np.median(gray[y-3:y, x:x+w], axis=0)
    if y+h+3 <= height:
        profiles[1] = np.median(gray[y+h:y+h+3, x:x+w], axis=0)
    if x >= 3:
        profiles[2] = np.median(gray[y:y+h, x-3:x], axis=1)
    if x+w+3 <= width:
        profiles[3] = np.median(gray[y:y+h, x+w:x+w+3], axis=1)
    scores = []
    for profile in profiles:
        if profile is None or len(profile) < 8:
            scores.append(0.)
            continue
        smooth = cv2.GaussianBlur(profile.reshape(1, -1), (5, 1), 0).ravel()
        scores.append(float(np.max(np.abs(smooth[6:]-smooth[:-6]))))
    topology = tuple(score > 18 for score in scores)
    return sum(topology) >= 2, topology


class QualityFrameProcessor:
    """A streaming callable with at most two bounded ROI histories.

    `boxes` must already have passed the application region checks. Neural
    inference runs only for local structure and is bounded by max_keyframes.
    A caller should report `diagnostics`, including budget fallback, to users.
    """

    def __init__(self, width, height, boxes, *, should_stop=lambda: False,
                 emit=lambda message: None, max_keyframes=120, keyframe_interval=.2,
                 pixel_masks=None, use_ai=True, temporal_propagation=False):
        self.width, self.height = int(width), int(height)
        if min(self.width, self.height) <= 0 or not self.width*self.height <= 3840*2160 or not 0 < len(boxes) <= 128:
            raise ValueError("视频精修支持最多 128 个分时区域和 4K 像素量。")
        self.boxes = []
        for b in boxes:
            values = [b[k] for k in ('x', 'y', 'w', 'h')]
            if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in values):
                raise ValueError("精修区域必须为整数像素坐标。")
            x, y, w, h = map(int, values)
            start, end = float(b.get('start', 0)), float(b.get('end', float('inf')))
            if (min(x, y) < 0 or min(w, h) < 3 or x+w > self.width or y+h > self.height
                    or w*h > self.width*self.height*.12 or not math.isfinite(start)
                    or math.isnan(end) or not 0 <= start < end):
                raise ValueError("视频精修区域或时间范围无效。")
            self.boxes.append((x, y, w, h, start, end))
        # Check half-open intervals before any expensive session is loaded.
        for t in sorted({b[4] for b in self.boxes}):
            if sum(b[4] <= t < b[5] for b in self.boxes) > 2:
                raise ValueError("视频精修同时最多支持两个局部区域。")
        if pixel_masks is None:
            pixel_masks = [None] * len(self.boxes)
        if len(pixel_masks) != len(self.boxes):
            raise ValueError("精细遮罩数量必须与分时区域一致。")
        self.pixel_masks = []
        for b, mask in zip(self.boxes, pixel_masks):
            if mask is not None:
                if (not isinstance(mask, np.ndarray) or mask.shape != (b[3], b[2])
                        or mask.dtype.kind not in 'buif' or not np.isfinite(mask).all()
                        or not np.any(mask)):
                    raise ValueError("精细遮罩必须为区域尺寸的非空单通道数组。")
                mask = (mask != 0).astype(np.uint8)
            self.pixel_masks.append(mask)
        if not isinstance(use_ai, bool):
            raise ValueError("视频精修 AI 开关必须是布尔值。")
        self.use_ai = use_ai
        if not isinstance(temporal_propagation, bool):
            raise ValueError("视频精修时序传播开关必须是布尔值。")
        self.temporal_propagation = temporal_propagation
        self._has_supplied_masks = any(m is not None for m in self.pixel_masks)
        if not isinstance(max_keyframes, int) or not 1 <= max_keyframes <= 240:
            raise ValueError("关键帧预算必须为 1–240。")
        if not math.isfinite(keyframe_interval) or not .1 <= keyframe_interval <= 1:
            raise ValueError("关键帧间隔超出范围。")
        self.should_stop, self.emit = should_stop, emit
        self.max_keyframes, self.keyframe_interval = max_keyframes, keyframe_interval
        self.session, self.states, self.last_timestamp = None, {}, None
        self.diagnostics = dict(engine='CPU LaMa keyframes + bounded optical-flow propagation',
            model_sha256=watermark_ai.MODEL_SHA256, keyframes=0, flow_frames=0,
            spatial_frames=0, propagation_frames=0, budget_fallback_frames=0, inference_seconds=0.,
            keyframe_interval=keyframe_interval, max_keyframes=max_keyframes,
            outside_mask_pixels_preserved=True, original_pixels_recovery_claimed=False,
            temporal_propagation=temporal_propagation, warnings=[])
        self.diagnostics['structure_policy'] = 'two_real_boundary_crossings_v2_clear_disappeared_object'
        self.diagnostics['precision_masks'] = sum(m is not None for m in self.pixel_masks)
        if not use_ai:
            self.diagnostics['engine'] = ('CPU precise-mask spatial reconstruction + temporal flow propagation'
                if temporal_propagation else 'CPU precise-mask spatial reconstruction')
        cv2.setNumThreads(2)

    def prepare(self, source, info, times, *, should_stop=None, emit=None):
        """Bounded pre-pass; the original video alone establishes a text mask."""
        stop = should_stop or self.should_stop
        output = emit or self.emit
        _stop(stop)
        if self.use_ai:
            # In controlled motion tests, fine glyph masks left colorful remnants
            # that the neural model extended into new strokes. Use the complete
            # approved box; never call a lower MAE a visual quality success.
            self.pixel_masks = [None] * len(self.boxes)
            self.diagnostics['precision_masks'] = 0
            self.diagnostics['mask_strategy'] = 'approved_rectangle_for_neural_consistency'
            return
        if not self._has_supplied_masks:
            from .watermark_video_precision import detect_precision_masks
            boxes = [dict(x=b[0], y=b[1], w=b[2], h=b[3], start=b[4],
                          end=min(b[5], float(info['duration']))) for b in self.boxes]
            detected = detect_precision_masks(source, boxes, source_info=info,
                                              should_stop=stop, emit=output)
            masks = detected['masks']
            if len(masks) != len(self.boxes):
                raise ValueError('精细遮罩分析返回的区域数量不匹配。')
            for b, m in zip(self.boxes, masks):
                if m is not None and (not isinstance(m, np.ndarray) or m.shape != (b[3], b[2])
                        or m.dtype.kind not in 'buif' or not np.isfinite(m).all() or not np.any(m)):
                    raise ValueError('精细遮罩分析返回了无效像素范围。')
            self.pixel_masks = [None if m is None else (m != 0).astype(np.uint8) for m in masks]
            self.diagnostics['precision_evidence'] = detected['evidence']
            self.diagnostics['precision_source_sha256'] = detected['source_sha256']
            self.diagnostics['precision_masks'] = sum(m is not None for m in masks)
        if not self.use_ai and any(m is None for m in self.pixel_masks):
            raise PrecisionMaskUnavailable('固定水印轮廓证据不足，使用快速修复。')
        _stop(stop)

    def _session(self):
        if self.session is None:
            import onnxruntime as ort
            path = watermark_ai.ensure_model(self.should_stop, self.emit)
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self.emit('正在加载视频精修 CPU 模型；加载结束后响应取消。')
            _stop(self.should_stop)
            session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
            inputs = {node.name: node for node in session.get_inputs()}
            if (set(inputs) != {'image', 'mask'} or inputs['image'].shape[1:] != [3, 512, 512]
                    or inputs['mask'].shape[1:] != [1, 512, 512]
                    or any(node.type != 'tensor(float)' for node in inputs.values())):
                raise watermark_ai.LamaError('视频精修模型输入契约不匹配。')
            self.session = session
            _stop(self.should_stop)
        return self.session

    def _predict(self, rgb, mask):
        started = time.perf_counter()
        predicted = watermark_ai._predict_roi(self._session(), rgb, mask, self.should_stop)
        self.diagnostics['inference_seconds'] += time.perf_counter() - started
        self.diagnostics['keyframes'] += 1
        return np.clip(np.rint(predicted), 0, 255).astype(np.uint8)

    def __call__(self, rgb, timestamp):
        _stop(self.should_stop)
        if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8
                or rgb.shape != (self.height, self.width, 3)):
            raise ValueError('视频精修需要原尺寸 RGB uint8 帧。')
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or timestamp < 0 or (self.last_timestamp is not None and timestamp < self.last_timestamp):
            raise ValueError('视频精修时间戳必须为有限递增值。')
        if self.last_timestamp is not None and timestamp-self.last_timestamp > 1:
            self.states.clear()
        self.last_timestamp = timestamp
        self.states = {i: state for i, state in self.states.items()
            if self.boxes[i][4] <= timestamp < self.boxes[i][5]}
        result = rgb.copy()
        for index, (x, y, w, h, start, end) in enumerate(self.boxes):
            _stop(self.should_stop)
            if not start <= timestamp < end:
                self.states.pop(index, None)
                continue
            # Fixed 512 context preserves the pinned model's original input scale.
            rw, rh = min(self.width, max(512, w+128)), min(self.height, max(512, h+128))
            rx = min(self.width-rw, max(0, round(x+w/2-rw/2)))
            ry = min(self.height-rh, max(0, round(y+h/2-rh/2)))
            local = rgb[ry:ry+rh, rx:rx+rw]
            lx, ly = x-rx, y-ry
            mask = np.zeros((rh, rw), np.uint8)
            mask[ly:ly+h, lx:lx+w] = 1 if self.pixel_masks[index] is None else self.pixel_masks[index]
            selected = mask[ly:ly+h, lx:lx+w] != 0
            ring = cv2.dilate(mask, np.ones((33, 33), np.uint8)) != 0
            ring &= mask == 0
            # Nearby objects outside the selected rectangle must not cause the
            # previous inpainted object to be carried into now-empty background.
            safe_ring = ring & (cv2.dilate(mask, np.ones((7, 7), np.uint8)) == 0)
            structured, topology = _crossing_structure(local, (lx, ly, w, h))
            spatial = _border_fill(local, (lx, ly, w, h))
            state = self.states.get(index)
            if not self.use_ai:
                # 锚定修复：与逐帧独立模式完全一致的 inpaint 结果，作为融合基与回退。
                anchored = cv2.inpaint(local, mask*255, 3, cv2.INPAINT_NS) if structured else spatial
                if not self.temporal_propagation or not structured:
                    self.states[index] = dict(source=local.copy(), output=anchored,
                        gray=cv2.cvtColor(spatial, cv2.COLOR_RGB2GRAY),
                        key_time=timestamp, topology=topology)
                    result[y:y+h, x:x+w][selected] = anchored[ly:ly+h, lx:lx+w][selected]
                    self.diagnostics['spatial_frames'] += 1
                    continue
                # 时序平滑（flow-guided EMA）：固定水印下从未暴露的像素无法"搬运真实值"，
                # 但逐帧 inpaint 各自猜一块会导致播放时补丁跳动。把上一帧"修复结果"沿
                # 光流 warp 到当前帧，与当前锚定结果融合，让补丁随运动连续演化、抑制闪烁。
                # 只在光流可靠的区域融合；转场/遮挡处退回锚定结果，避免鬼影。
                warped, reliable = None, False
                if state is not None:
                    motion_gray = cv2.cvtColor(spatial, cv2.COLOR_RGB2GRAY)
                    flow = cv2.calcOpticalFlowFarneback(motion_gray, state['gray'], None,
                        .5, 3, 23, 3, 5, 1.2, 0)
                    yy, xx = np.mgrid[:rh, :rw].astype(np.float32)
                    mx, my = xx+flow[..., 0], yy+flow[..., 1]
                    warped = cv2.remap(state['output'], mx, my, cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT_101)
                    observed = cv2.remap(state['source'], mx, my, cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REFLECT_101)
                    error = np.abs(observed.astype(np.float32)-local.astype(np.float32)).mean(axis=2)
                    reliable = bool(state.get('topology') == topology
                        and np.median(error[safe_ring]) < 6 and np.percentile(error[safe_ring], 90) < 20
                        and np.percentile(np.hypot(flow[..., 0], flow[..., 1])[mask != 0], 95) < 20)
                since_key = timestamp-state['key_time'] if state is not None else float('inf')
                if reliable and warped is not None and since_key < self.keyframe_interval:
                    # 权重偏向历史帧：0.4 锚定 + 0.6 传播，抑制逐帧猜测的抖动。
                    fused = np.clip(np.rint(.4*anchored.astype(np.float32)
                                            + .6*warped.astype(np.float32)), 0, 255).astype(np.uint8)
                    output = anchored.copy()
                    output[mask != 0] = fused[mask != 0]
                    key_time = state['key_time']
                    self.diagnostics['propagation_frames'] += 1
                else:
                    output = anchored
                    key_time = timestamp
                    self.diagnostics['spatial_frames'] += 1
                self.states[index] = dict(source=local.copy(), output=output,
                    gray=cv2.cvtColor(spatial, cv2.COLOR_RGB2GRAY), key_time=key_time, topology=topology)
                result[y:y+h, x:x+w][selected] = output[ly:ly+h, lx:lx+w][selected]
                continue
            if not structured:
                self.diagnostics['spatial_frames'] += 1
                self.states.pop(index, None)
                result[y:y+h, x:x+w][selected] = spatial[ly:ly+h, lx:lx+w][selected]
                continue
            warped, reliable = None, False
            if state is not None:
                # Hide the stationary watermark before estimating scene motion.
                motion_gray = cv2.cvtColor(spatial, cv2.COLOR_RGB2GRAY)
                flow = cv2.calcOpticalFlowFarneback(motion_gray, state['gray'], None,
                    .5, 3, 23, 3, 5, 1.2, 0)
                yy, xx = np.mgrid[:rh, :rw].astype(np.float32)
                mx, my = xx+flow[..., 0], yy+flow[..., 1]
                warped = cv2.remap(state['output'], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                observed = cv2.remap(state['source'], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                error = np.abs(observed.astype(np.float32)-local.astype(np.float32)).mean(axis=2)
                reliable = bool(state.get('topology') == topology
                    and np.median(error[safe_ring]) < 5 and np.percentile(error[safe_ring], 90) < 18
                    and np.percentile(np.hypot(flow[..., 0], flow[..., 1])[mask != 0], 95) < 18)
            refresh = state is None or not reliable or timestamp-state['key_time'] >= self.keyframe_interval
            if refresh and self.diagnostics['keyframes'] < self.max_keyframes:
                self.emit(f"视频精修：{timestamp:.1f} 秒局部结构修复，可取消。")
                repaired = self._predict(local, mask)
                # A small same-scene prediction blend reduces neural refresh popping.
                if reliable and warped is not None:
                    repaired = np.clip(np.rint(.85*repaired + .15*warped), 0, 255).astype(np.uint8)
                key_time = timestamp
            elif reliable and warped is not None and timestamp-state['key_time'] < 1:
                repaired, key_time = warped, state['key_time']
                self.diagnostics['flow_frames'] += 1
            else:
                repaired, key_time = spatial, timestamp
                if self.diagnostics['budget_fallback_frames'] == 0:
                    warning = '关键帧精修预算已用尽，剩余无法可靠传播的画面采用快速插值修复。'
                    self.diagnostics['warnings'].append(warning)
                    self.emit(warning)
                self.diagnostics['budget_fallback_frames'] += 1
            output = local.copy()
            output[mask != 0] = repaired[mask != 0]
            self.states[index] = dict(source=local.copy(), output=output,
                gray=cv2.cvtColor(spatial, cv2.COLOR_RGB2GRAY), key_time=key_time, topology=topology)
            result[y:y+h, x:x+w][selected] = output[ly:ly+h, lx:lx+w][selected]
        _stop(self.should_stop)
        return result
