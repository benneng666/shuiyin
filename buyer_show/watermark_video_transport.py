"""Bounded local RGB processing; the original stream remains the output clock.

Patch PNGs carry explicit per-frame durations, never an average frame rate.
The main input drives overlay and all original audio streams are stream copied.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
from fractions import Fraction
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time

MAX_SECONDS = 180
MAX_FRAMES = 5400
MAX_VERIFY_FRAMES = 216000
MAX_TEMP_BYTES = 8 * 1024 ** 3


class QualityUnsupported(ValueError):
    """A safe preflight limitation, before any repair output is rendered."""


def _stop(should_stop):
    if should_stop():
        raise asyncio.CancelledError()


def _spawn(args):
    from .video import ffmpeg_executable
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    return subprocess.Popen([ffmpeg_executable(), '-nostdin', '-hide_banner', *args],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **flags)


def _input(source):
    # Only paths already validated by the public video shell reach this helper.
    from .watermark_video import _local_source
    path = _local_source(source)
    return ['-protocol_whitelist', 'file', '-format_whitelist', 'mov,matroska,webm',
            '-threads', '2', '-noautorotate', '-i', str(path)]


def _command(args, *, should_stop, stdout_line=None, stderr_line=None, timeout=3600):
    """Consume both pipes continuously, bound diagnostic memory, reap on cancel."""
    _stop(should_stop)
    proc = _spawn(args)
    errors, tail = [], bytearray()

    def drain(pipe, handler, diagnostic=False):
        try:
            while raw := pipe.readline(32768):
                if diagnostic:
                    tail.extend(raw)
                    if len(tail) > 65536:
                        del tail[:-65536]
                if handler:
                    handler(raw.decode('utf-8', errors='replace'))
        except BaseException as exc:
            errors.append(exc)
        finally:
            pipe.close()

    readers = [threading.Thread(target=drain, args=(proc.stdout, stdout_line), daemon=True),
               threading.Thread(target=drain, args=(proc.stderr, stderr_line, True), daemon=True)]
    for reader in readers:
        reader.start()
    started = time.monotonic()
    try:
        while proc.poll() is None:
            _stop(should_stop)
            if errors:
                raise errors[0]
            if time.monotonic() - started > timeout:
                raise TimeoutError('视频帧处理超时')
            time.sleep(.03)
        for reader in readers:
            reader.join(timeout=5)
        _stop(should_stop)
        if errors:
            raise errors[0]
        if proc.returncode:
            raise ValueError('FFmpeg 视频处理失败：' + tail.decode('utf-8', errors='replace')[-1500:])
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
        for reader in readers:
            reader.join(timeout=5)


def _video_times(source, should_stop, *, limit=None, with_time_base=False):
    times, bases = [], []

    def consume(line):
        match = re.search(r'config in time_base:\s*(\d+)/(\d+)', line)
        if match:
            bases.append(Fraction(int(match[1]), int(match[2])))
        match = re.search(r'\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:', line)
        if match:
            if not bases:
                raise ValueError('视频缺少逐帧时间基准')
            if int(match[1]) != len(times):
                raise ValueError('视频帧序列不连续')
            if len(times) >= (limit or MAX_VERIFY_FRAMES):
                if limit:
                    raise QualityUnsupported(f'精修最多支持 {limit} 帧')
                raise ValueError('视频帧数超过时间轴校验上限')
            times.append(int(match[2]) * bases[-1])

    _command(['-v', 'info', '-copyts', *_input(source), '-map', '0:V:0',
        '-vf', 'showinfo=checksum=0', '-an', '-sn', '-dn', '-fps_mode', 'passthrough',
        '-f', 'null', '-'], should_stop=should_stop, stderr_line=consume)
    if not times or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('视频时间戳缺失或非递增，无法安全精修')
    return (times, bases[-1]) if with_time_base else times


def _audio_signatures(source, should_stop):
    """Hash every copied packet and its relative time, retaining stream offsets."""
    from .watermark_video import _probe
    if not _probe(source, should_stop)['audio_stream_count']:
        return []
    bases, rows = {}, {}

    def consume(line):
        match = re.match(r'#tb\s+(\d+):\s*(\d+)/(\d+)', line)
        if match:
            bases[int(match[1])] = Fraction(int(match[2]), int(match[3]))
            return
        if not line.strip() or line.startswith('#'):
            return
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 6:
            raise ValueError('音频验证记录不完整')
        stream, dts, pts, duration, size = map(int, parts[:5])
        base = bases[stream]
        row = rows.setdefault(stream, {'count': 0, 'bytes': 0, 'payload': hashlib.sha256(),
            'timing': hashlib.sha256(), 'first': pts * base, 'last_end': pts * base})
        row['payload'].update((str(size) + ':' + parts[5] + '\n').encode('ascii'))
        row['timing'].update(f'{pts * base - row["first"]}/{duration * base}\n'.encode('ascii'))
        row['count'] += 1
        row['bytes'] += size
        row['last_end'] = (pts + duration) * base

    _command(['-v', 'error', '-copyts', *_input(source), '-map', '0:a', '-vn', '-sn', '-dn',
        '-c:a', 'copy', '-f', 'framehash', '-hash', 'sha256', '-'],
        should_stop=should_stop, stdout_line=consume)
    return [{'stream': index, 'packet_count': r['count'], 'bytes': r['bytes'],
             'time_base': [bases[index].numerator, bases[index].denominator],
             'packet_sha256': r['payload'].hexdigest(), 'relative_timing_sha256': r['timing'].hexdigest(),
             'first_seconds': float(r['first']), 'last_end_seconds': float(r['last_end'])}
            for index, r in sorted(rows.items())]


def verify_timing_audio(source, output, should_stop=lambda: False):
    """Verify every video PTS and every audio payload before publishing a result."""
    from .watermark_video import _probe
    before, source_tb = _video_times(source, should_stop, with_time_base=True)
    after, output_tb = _video_times(output, should_stop, with_time_base=True)
    if len(before) != len(after):
        raise ValueError('结果帧数与原视频不一致，未发布结果')
    error = max(abs(float((b - before[0]) - (a - after[0]))) for b, a in zip(before, after))
    if error > .000002:
        raise ValueError('结果逐帧时间戳与原视频不一致，未发布结果')
    source_info, output_info = _probe(source, should_stop), _probe(output, should_stop)
    source_start, output_start = source_info['input_start_seconds'], output_info['input_start_seconds']
    # Resetting a nonzero container start can put the first video timestamp
    # between ticks of its encoder time base. Only this constant origin offset
    # permits quantization; all relative frame intervals above remain strict.
    origin_error = abs((float(before[0]) - source_start) - (float(after[0]) - output_start))
    origin_tolerance = float(max(source_tb, output_tb)) / 2 + .000001
    if origin_error > origin_tolerance:
        raise ValueError('视频起始偏移不一致，未发布结果')
    audio_before, audio_after = _audio_signatures(source, should_stop), _audio_signatures(output, should_stop)
    if len(audio_before) != len(audio_after):
        raise ValueError('音轨数量不一致，未发布结果')
    audio_offsets = []
    for a, b in zip(audio_before, audio_after):
        for key in ('packet_count', 'bytes', 'packet_sha256', 'relative_timing_sha256'):
            if a[key] != b[key]:
                raise ValueError('音频原始数据或时间轴改变，未发布结果')
        audio_tb = max(Fraction(*a['time_base']), Fraction(*b['time_base']))
        av_tolerance = float(max(source_tb, output_tb) + audio_tb) / 2 + .000001
        av_error = abs((a['first_seconds'] - float(before[0])) - (b['first_seconds'] - float(after[0])))
        if av_error > av_tolerance:
            raise ValueError('音视频相对起始偏移改变，未发布结果')
        audio_offsets.append({'stream': a['stream'], 'error_seconds': av_error,
                              'quantization_tolerance_seconds': av_tolerance})
    return {'video_frames': len(before), 'video_relative_pts_max_error_seconds': error,
            'video_time_bases': [[source_tb.numerator, source_tb.denominator],
                                [output_tb.numerator, output_tb.denominator]],
            'video_start_offset_error_seconds': origin_error,
            'video_start_quantization_tolerance_seconds': origin_tolerance,
            'audio_video_offset_checks': audio_offsets,
            'audio_streams': audio_after, 'all_audio_packets_preserved': True,
            'audio_video_offset_preserved': True}


def _spatial_groups(boxes):
    groups = []
    for box in boxes:
        x, y, w, h = (int(box[k]) for k in ('x', 'y', 'w', 'h'))
        rect = [x, y, x + w, y + h]
        changed = True
        while changed:
            changed = False
            for prior in list(groups):
                if rect[0] <= prior[2] and prior[0] <= rect[2] and rect[1] <= prior[3] and prior[1] <= rect[3]:
                    rect = [min(rect[0], prior[0]), min(rect[1], prior[1]),
                            max(rect[2], prior[2]), max(rect[3], prior[3])]
                    groups.remove(prior)
                    changed = True
        groups.append(rect)
    return [(x, y, right - x, bottom - y) for x, y, right, bottom in groups]


def _frames(source, shape, should_stop):
    """At most two full RGB frames live in the decoder queue."""
    import numpy as np
    size = shape[0] * shape[1] * 3
    proc = _spawn(['-v', 'error', '-xerror', *_input(source), '-map', '0:V:0',
        '-an', '-sn', '-dn', '-fps_mode', 'passthrough', '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-'])
    pending, aborted, errors = queue.Queue(maxsize=1), threading.Event(), []
    tail = bytearray()

    def put(value):
        while not aborted.is_set():
            try:
                pending.put(value, timeout=.05)
                return
            except queue.Full:
                pass

    def read():
        try:
            while not aborted.is_set():
                data = bytearray()
                while len(data) < size and not aborted.is_set():
                    part = proc.stdout.read(size - len(data))
                    if not part:
                        break
                    data.extend(part)
                if not data:
                    break
                if len(data) != size:
                    raise ValueError('视频解码帧不完整')
                put(data)
        except BaseException as exc:
            errors.append(exc)
        finally:
            proc.stdout.close()
            put(None)

    def diagnostics():
        try:
            while part := proc.stderr.read(4096):
                tail.extend(part)
                if len(tail) > 65536:
                    del tail[:-65536]
        finally:
            proc.stderr.close()

    readers = [threading.Thread(target=read, daemon=True), threading.Thread(target=diagnostics, daemon=True)]
    for reader in readers:
        reader.start()
    try:
        waiting_since = time.monotonic()
        while True:
            _stop(should_stop)
            try:
                data = pending.get(timeout=.05)
            except queue.Empty:
                if time.monotonic() - waiting_since > 60:
                    raise TimeoutError('视频解码帧等待超时')
                continue
            if data is None:
                break
            yield np.frombuffer(data, dtype=np.uint8).reshape(shape[0], shape[1], 3)
            waiting_since = time.monotonic()
        proc.wait(timeout=10)
        if errors:
            raise errors[0]
        if proc.returncode:
            raise ValueError('视频 RGB 解码失败：' + tail.decode('utf-8', errors='replace')[-1000:])
    finally:
        aborted.set()
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
        for reader in readers:
            reader.join(timeout=5)


def render_quality(source, partial, boxes, info, processor, should_stop=lambda: False, emit=lambda message: None):
    """Render a private partial MP4. The caller owns source hashing/publication."""
    import numpy as np
    from PIL import Image
    from .watermark_video import _regions
    _stop(should_stop)
    if info['duration'] > MAX_SECONDS:
        raise QualityUnsupported(f'精修最多支持 {MAX_SECONDS} 秒视频')
    boxes = _regions(boxes, info)
    groups = _spatial_groups(boxes)
    if not 1 <= len(groups) <= 2:
        raise QualityUnsupported('精修最多支持两个分离的空间区域')
    area = sum(w * h for _, _, w, h in groups)
    if area > info['width'] * info['height'] * .2:
        raise QualityUnsupported('精修局部区域合计超过画面 20%')
    emit('正在核对视频逐帧时间轴…')
    times = _video_times(source, should_stop, limit=MAX_FRAMES)
    if any(float(b - a) < .000002 for a, b in zip(times, times[1:])):
        raise QualityUnsupported('视频帧间隔小于精修时间精度')
    partial = Path(partial)
    if partial.exists():
        raise FileExistsError('视频输出路径已存在，拒绝覆盖')
    partial.parent.mkdir(parents=True, exist_ok=True)
    estimate = (area * 3 + 4096 * len(groups)) * len(times)
    if estimate > MAX_TEMP_BYTES or shutil.disk_usage(partial.parent).free < estimate + 512 * 1024 ** 2:
        raise QualityUnsupported('精修临时文件预算或磁盘空间不足')
    callback = getattr(processor, 'process', processor)
    source_offset = float(times[0]) - float(info.get('input_start_seconds', 0))
    # FFmpeg concat duration precision is a microsecond. Round absolute times
    # first, then difference them: rounding each 1/30 increment accumulates drift.
    patch_ticks = [round(float(t - times[0]) * 1_000_000) for t in times]
    if callable(prepare := getattr(processor, 'prepare', None)):
        prepare(source, info, [float(t) - float(info.get('input_start_seconds', 0)) for t in times],
                should_stop=should_stop, emit=emit)
        _stop(should_stop)
    temp_bytes, count = 0, 0
    with tempfile.TemporaryDirectory(prefix='video-patches-', dir=partial.parent) as temp:
        root = Path(temp)
        entries = [[] for _ in groups]
        frames = _frames(source, (info['height'], info['width']), should_stop)
        try:
            for index, decoded in enumerate(frames):
                if index >= len(times):
                    raise ValueError('视频在时间轴检查后增加了帧')
                _stop(should_stop)
                stamp = float(times[index]) - float(info.get('input_start_seconds', 0))
                original = decoded.copy()
                active = [r for r in boxes if float(r.get('start', 0)) <= stamp < float(r.get('end', info['duration']))]
                if active:
                    candidate = callback(decoded, stamp)
                    if not isinstance(candidate, np.ndarray) or candidate.dtype != np.uint8 or candidate.shape != original.shape:
                        raise ValueError('视频修复回调返回无效 RGB 帧')
                    result = original.copy()
                    for r in active:
                        x, y, w, h = (int(r[k]) for k in ('x', 'y', 'w', 'h'))
                        result[y:y+h, x:x+w] = candidate[y:y+h, x:x+w]
                else:
                    result = original
                _stop(should_stop)
                duration = (patch_ticks[index+1] - patch_ticks[index]) / 1_000_000 if index+1 < len(times) else max(.000002,
                    float(info['duration']) - stamp)
                for number, (x, y, w, h) in enumerate(groups):
                    name = f'p{number}-{index:06}.png'
                    path = root / name
                    Image.fromarray(result[y:y+h, x:x+w]).save(path, compress_level=1)
                    temp_bytes += path.stat().st_size
                    entries[number].append((name, duration))
                count += 1
                if index % 15 == 0 or count == len(times):
                    emit(f'视频精修 {count}/{len(times)} 帧（{stamp:.1f} 秒）')
        finally:
            frames.close()
        if count != len(times):
            raise ValueError('视频在时间轴检查后缺少帧')
        inputs, filters = [], []
        for number, ((x, y, w, h), rows) in enumerate(zip(groups, entries)):
            concat = root / f'patch-{number}.ffconcat'
            content = ['ffconcat version 1.0']
            for name, duration in rows:
                content += [f"file '{name}'", 'option framerate 1000000', f'duration {duration:.9f}']
            # Last repeated image gives concat a terminal duration; main input
            # remains authoritative and never emits this extra terminal frame.
            content += [f"file '{rows[-1][0]}'", 'option framerate 1000000']
            concat.write_text('\n'.join(content) + '\n', encoding='ascii')
            inputs += ['-protocol_whitelist', 'file', '-f', 'concat', '-safe', '0', '-i', str(concat)]
            filters.append(f'[{number+1}:v]setpts=PTS+{source_offset:.9f}/TB[p{number}]')
            main = '0:V:0' if number == 0 else f'v{number-1}'
            filters.append(f'[{main}][p{number}]overlay=x={x}:y={y}:eof_action=repeat:repeatlast=1:'
                           f'shortest=0:ts_sync_mode=nearest:format=rgb[v{number}]')
        script = root / 'overlay.txt'
        # framesync may expose a secondary terminal timestamp after main EOF.
        # Trim only the video filter output, not the muxing job: -frames:v or
        # -shortest would prematurely stop delayed/longer original audio.
        filters.append(f'[v{len(groups)-1}]trim=end_frame={len(times)}[final]')
        script.write_text(';'.join(filters), encoding='ascii')
        emit('正在按原始视频时间轴编码，保留全部原音轨…')
        encoded = root / 'encoded.mp4'
        _command(['-v', 'error', '-xerror', '-copyts', '-start_at_zero', *_input(source), *inputs,
                '-filter_complex_threads', '1', '-filter_complex_script', str(script),
                '-map', '[final]', '-map', '0:a?', '-sn', '-dn', '-c:v', 'libx264',
                '-crf', '18', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', '-threads', '2',
                '-fps_mode:v', 'passthrough', '-enc_time_base:v', 'demux', '-c:a', 'copy',
                '-map_metadata', '0', '-map_chapters', '0', '-movflags', '+faststart', '-n', str(encoded)],
                should_stop=should_stop)
        evidence = verify_timing_audio(source, encoded, should_stop)
        # Reserve the caller's new partial explicitly. Never remove another
        # process's file if it appeared between preflight and encode completion.
        owned = False
        try:
            with partial.open('xb') as target, encoded.open('rb') as source_file:
                owned = True
                while chunk := source_file.read(1024 * 1024):
                    _stop(should_stop)
                    target.write(chunk)
            _stop(should_stop)
        except BaseException:
            if owned:
                partial.unlink(missing_ok=True)
            raise
    return {'frames_processed': count, 'duration_seconds': info['duration'], 'spatial_rois': groups,
            'temporary_png_bytes': temp_bytes, 'outside_approved_rgb_preserved_before_encoding': True,
            'clock': 'original_video_master_overlay_nearest_patch_pts', 'timing_audio': evidence,
            'algorithm': getattr(processor, 'diagnostics', {})}
