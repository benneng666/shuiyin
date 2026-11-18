"""Small graphic companions of already approved, repeated brand text.

No brand/coordinate/color template is embedded. Only local neighborhoods of
approved brand anchors are examined. Graphic shape and anchor-relative layout
must repeat across three independent source hashes of the same product, with
changing immediate background. This is evidence, not universal logo recognition.
"""
from __future__ import annotations

import asyncio
import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping

from .watermark_batch_policy import normalize_text, text_is_product_copy

POLICY_VERSION = "anchored-repeated-graphics-v2"
MAX_ANCHORS = 6
MAX_COMPONENTS = 40
MAX_REGIONS = 96


def _stop(should_stop):
    if should_stop():
        raise asyncio.CancelledError()


def _box(row, width, height):
    try:
        values = [float(row[key]) for key in ("x", "y", "w", "h")]
        if not all(math.isfinite(v) and v == int(v) for v in values):
            return None
        x, y, w, h = map(int, values)
        if min(x, y) < 0 or min(w, h) < 3 or x + w > width or y + h > height:
            return None
        return {"x": x, "y": y, "w": w, "h": h}
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _anchors(decision, width, height):
    tokens = [normalize_text(t) for t in decision.get("evidence", {}).get("merchant_tokens", [])]
    tokens = [t for t in tokens if len(t) >= 3 and not text_is_product_copy(t)]
    result = []
    for row in decision.get("approved_regions", []):
        box = _box(row, width, height)
        text = normalize_text(str(row.get("text", row.get("label", ""))).split("：", 1)[-1])
        matches = [token for token in tokens if token in text]
        if box is None or not matches or text_is_product_copy(text):
            continue
        x, y, w, h = (box[key] for key in ("x", "y", "w", "h"))
        horizontal = "left" if x <= width * .22 else "right" if x + w >= width * .78 else ""
        vertical = "top" if y <= height * .24 else "bottom" if y + h >= height * .76 else ""
        if not horizontal or not vertical or w < width * .12 or h > height * .15:
            continue
        # Prefer the stable canonical token over OCR prefixes/suffixes.
        brand = min(matches, key=len)
        result.append({**box, "brand": brand, "corner": horizontal + "_" + vertical})
    return result[:MAX_ANCHORS]


def _rectangles(mask, left, top, scale, width, height, kind):
    """Cover only nearby component pixels; do not fill an entire curved bbox."""
    import numpy as np
    import cv2
    mask = cv2.dilate(mask.astype("uint8"), np.ones((3, 3), np.uint8)) > 0
    ys, xs = np.nonzero(mask)
    if not xs.size:
        return []
    if kind == "symbol":
        bounds = [(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)]
    else:
        # Fixed-size grid cells bound the damage of an automatic rectangle to
        # a few pixels around the actual line, even for a long curved stroke.
        cell = max(6, round(min(width, height) * .025 * scale))
        bounds = []
        for gy in range(int(ys.min()) // cell, int(ys.max()) // cell + 1):
            for gx in range(int(xs.min()) // cell, int(xs.max()) // cell + 1):
                y0, x0 = gy * cell, gx * cell
                yy, xx = np.nonzero(mask[y0:y0+cell, x0:x0+cell])
                if xx.size:
                    bounds.append((x0 + int(xx.min()), y0 + int(yy.min()),
                                   x0 + int(xx.max()) + 1, y0 + int(yy.max()) + 1))
    result = []
    for x0, y0, x1, y1 in bounds:
        x = max(0, math.floor(left + x0 / scale) - 1)
        y = max(0, math.floor(top + y0 / scale) - 1)
        right = min(width, math.ceil(left + x1 / scale) + 1)
        bottom = min(height, math.ceil(top + y1 / scale) + 1)
        if right - x >= 3 and bottom - y >= 3:
            result.append({"x": x, "y": y, "w": right - x, "h": bottom - y,
                           "label": "已批准品牌邻接重复图形", "detection_kind": "graphic"})
    return result


def _extract(image, anchor, text_boxes, should_stop):
    import cv2
    import numpy as np
    from PIL import Image
    width, height = image.size
    h = anchor["h"]
    margin_x, margin_y = max(8, round(h * .7)), max(10, round(h * 1.8))
    left, top = max(0, anchor["x"] - margin_x), max(0, anchor["y"] - margin_y)
    right = min(width, anchor["x"] + anchor["w"] + margin_x)
    bottom = min(height, anchor["y"] + h + margin_y)
    raw = image.crop((left, top, right, bottom)).convert("RGBA")
    matte = Image.new("RGBA", raw.size, "white")
    matte.alpha_composite(raw)
    raw.close()
    crop = matte.convert("RGB")
    matte.close()
    scale = min(1.0, 1400 / crop.width, 700 / crop.height)
    if scale < 1:
        crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))))
    rgb = np.asarray(crop).copy()
    crop.close()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    size = max(7, min(31, round(h * scale * .4))) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    white = (gray >= 205) & (cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel) >= 18)
    black = (gray <= 65) & (cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel) >= 25)
    colored = ((hsv[:, :, 1] >= 65) & (hsv[:, :, 2] >= 150)
               & (cv2.morphologyEx(hsv[:, :, 1], cv2.MORPH_TOPHAT, kernel) >= 25))
    text_mask = np.zeros(gray.shape, np.uint8)
    for box in text_boxes:
        x0, y0 = max(0, math.floor((box["x"] - left) * scale)), max(0, math.floor((box["y"] - top) * scale))
        x1, y1 = min(gray.shape[1], math.ceil((box["x"] + box["w"] - left) * scale)), min(gray.shape[0], math.ceil((box["y"] + box["h"] - top) * scale))
        if x1 > x0 and y1 > y0:
            text_mask[y0:y1, x0:x1] = 1
    candidates = []
    for channel, foreground in (("bright", white), ("dark", black), ("chromatic", colored)):
        _stop(should_stop)
        binary = cv2.morphologyEx(foreground.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        for label in range(1, count):
            _stop(should_stop)
            x, y, w, hh, area = map(int, stats[label])
            if area < 8 or w * hh > gray.size * .3:
                continue
            component = labels[y:y+hh, x:x+w] == label
            covered = int((component & (text_mask[y:y+hh, x:x+w] > 0)).sum())
            # Do not relabel known text as a newly recognized graphic.
            actual_w, actual_h = w / scale, hh / scale
            density = area / (w * hh)
            symbol = (4 <= actual_w <= h * .9 and 4 <= actual_h <= h * .9
                      and .4 <= actual_w / actual_h <= 2.5 and .06 <= density <= .75)
            line = (max(actual_w, actual_h) >= anchor["w"] * .18
                    and actual_h <= h * 1.2 and actual_w <= anchor["w"] * 1.35
                    and max(actual_w / max(1, actual_h), actual_h / max(1, actual_w)) >= 2.8
                    and density <= .26)
            actual_x, actual_y = left + x / scale, top + y / scale
            edge_line = (channel == "chromatic" and actual_w <= max(5, h * .14)
                         and max(10, h * .25) <= actual_h <= h * 1.8
                         and actual_h >= actual_w * 3
                         and min(actual_x, width - actual_x - actual_w) <= max(3, width * .02)
                         and actual_y + actual_h / 2 >= anchor["y"] + h * .5)
            if not symbol and not line and not edge_line:
                continue
            if covered / area >= (.98 if line or edge_line else .65):
                continue
            if edge_line:
                # Straight edge fragments never approve themselves. A later
                # pass must associate them with an independently proven curve.
                kind, eligible = "edge_line", False
            elif symbol:
                contours, hierarchy = cv2.findContours(component.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                # Closed compact symbols such as registered marks are useful;
                # random cloth speckle and detached letter strokes are not.
                has_hole = hierarchy is not None and any(row[3] >= 0 for row in hierarchy[0])
                if not has_hole:
                    continue
                cx, cy = left + (x + w / 2) / scale, top + (y + hh / 2) / scale
                ax, ay, aw, ah = (anchor[k] for k in ("x", "y", "w", "h"))
                near_corner = min(math.hypot(cx - px, cy - py) for px in (ax, ax + aw)
                                  for py in (ay, ay + ah)) <= ah * .8
                beside = (cx < ax or cx > ax + aw) and ay <= cy <= ay + ah
                if not near_corner and not beside:
                    continue
                kind, eligible = "symbol", True
            else:
                yy, xx = np.nonzero(component)
                points = np.stack((xx, yy), axis=1).astype(np.float32)
                centered = points - points.mean(axis=0)
                _, singular, _ = np.linalg.svd(centered, full_matrices=False)
                residual = float(singular[-1] / math.sqrt(len(points))) / scale
                # Repeated straight dividers/garment seams remain reviewable.
                kind, eligible = "line", residual >= 2.0
            full = np.zeros(gray.shape, np.uint8)
            full[y:y+hh, x:x+w] = component
            # Only previously unapproved pixels need a companion mask.
            full[text_mask > 0] = 0
            if full.sum() < 8:
                continue
            if kind == "edge_line":
                # A thin semitransparent edge stroke has a low-saturation
                # antialias halo beyond its connected high-contrast core. Keep
                # one small envelope, proportional to its measured ink width,
                # including portions already inside the approved text mask.
                padding = max(3, min(12, math.ceil(2 * area / max(w, hh) / scale)))
                ex, ey = max(0, math.floor(actual_x) - padding), max(0, math.floor(actual_y) - padding)
                er = min(width, math.ceil(actual_x + actual_w) + padding)
                eb = min(height, math.ceil(actual_y + actual_h) + padding)
                rectangles = [{"x": ex, "y": ey, "w": er - ex, "h": eb - ey,
                               "label": "已批准品牌邻接重复图形", "detection_kind": "graphic"}]
            else:
                rectangles = _rectangles(full, left, top, scale, width, height, kind)
            if not rectangles or len(rectangles) > 48:
                continue
            mask_area = sum(r["w"] * r["h"] for r in rectangles)
            if mask_area > width * height * .012:
                continue
            descriptor = cv2.resize(component.astype(np.uint8), (32, 32), interpolation=cv2.INTER_NEAREST)
            border = cv2.dilate(full, np.ones((9, 9), np.uint8)) > 0
            border &= full == 0
            context = np.median(rgb[border], axis=0) if border.any() else np.array([0, 0, 0])
            ink = rgb[y:y+hh, x:x+w][component]
            color = np.median(ink, axis=0)
            actual = {"x": left + x / scale, "y": top + y / scale,
                      "w": actual_w, "h": actual_h}
            candidates.append({"anchor": anchor, "kind": kind, "channel": channel,
                               "eligible": eligible, "rectangles": rectangles,
                               "relative": ((actual["x"] + actual_w / 2 - anchor["x"]) / anchor["w"],
                                            (actual["y"] + actual_h / 2 - anchor["y"]) / h,
                                            actual_w / anchor["w"], actual_h / h),
                               "component_box": actual, "descriptor": descriptor,
                               "context": context, "color": color,
                               "stroke_width": area / scale ** 2 / max(actual_w, actual_h),
                               "shape_sha256": hashlib.sha256(descriptor.tobytes()).hexdigest()})
            if len(candidates) >= MAX_COMPONENTS:
                return candidates, True
    return candidates, False


def _similar(a, b):
    import cv2
    import numpy as np
    if (a["kind"], a["channel"], a["anchor"]["corner"]) != (b["kind"], b["channel"], b["anchor"]["corner"]):
        return 0.0
    ax, ay, aw, ah = a["relative"]
    bx, by, bw, bh = b["relative"]
    low, high = (.4, 2.5) if a["kind"] == "edge_line" else (.65, 1.5)
    if (abs(ax - bx) > .12 or abs(ay - by) > .45 or not low <= aw / max(bw, 1e-5) <= high
            or not low <= ah / max(bh, 1e-5) <= high or float(np.linalg.norm(a["color"] - b["color"])) > 75):
        return 0.0
    aa, bb = a["descriptor"].astype(bool), b["descriptor"].astype(bool)
    da = cv2.dilate(aa.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    db = cv2.dilate(bb.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    score = min(float((aa & db).sum() / max(1, aa.sum())), float((bb & da).sum() / max(1, bb.sum())))
    return score if score >= .86 else 0.0


def _same_ink(a, b):
    """Relative hue/chroma and stroke width; no particular ink color is known."""
    import cv2
    import numpy as np
    aa = cv2.cvtColor(np.uint8([[a["color"]]]), cv2.COLOR_RGB2HSV)[0, 0].astype(float)
    bb = cv2.cvtColor(np.uint8([[b["color"]]]), cv2.COLOR_RGB2HSV)[0, 0].astype(float)
    hue_distance = min(abs(aa[0] - bb[0]), 180 - abs(aa[0] - bb[0]))
    return bool(hue_distance <= 12 and .4 <= aa[1] / max(1, bb[1]) <= 2.5
                and .55 <= aa[2] / max(1, bb[2]) <= 1.8
                and .4 <= a["stroke_width"] / max(.1, b["stroke_width"]) <= 2.5)


def analyze_graphics(scans, decisions, *, should_stop=lambda: False, emit=lambda message: None):
    """Return additive decisions; no mutation or repair of inputs.

    scans: item_id, path, product_key, source_types, detection (SHA and raw OCR).
    decisions: item_id, approved_regions, evidence.merchant_tokens. A graphics
    component cannot vote for its own brand, or vote more than once per SHA.
    """
    import numpy as np
    from .watermark import _load_image
    decisions = {str(d["item_id"]): d for d in decisions}
    prepared, grouped = [], defaultdict(list)
    for index, scan in enumerate(scans):
        _stop(should_stop)
        item_id = str(scan.get("item_id", index))
        result = {"item_id": item_id, "approved_regions": [], "proposed_regions": [], "review_regions": [],
                  "review_reasons": [], "warnings": [],
                  "evidence": {"policy": POLICY_VERSION, "graphics": []}}
        item = {"result": result, "candidates": [], "complete": True}
        prepared.append(item)
        decision = decisions.get(item_id, {})
        if not scan.get("path") or not scan.get("product_key") or not decision.get("approved_regions"):
            result["warnings"].append("没有已批准品牌文字锚点，未自动识别或擦除独立图案。")
            continue
        image = None
        try:
            emit(f"核对品牌邻接图形 {index + 1}/{len(scans)}")
            image, info, _ = _load_image(scan["path"])
            expected = scan.get("detection", {}).get("source_sha256") or scan.get("source_sha256") or scan.get("sha256")
            if expected and expected != info["sha256"]:
                raise ValueError("source_changed")
            item.update(source_sha256=info["sha256"], width=image.width, height=image.height)
            result["source_sha256"] = info["sha256"]
            anchors = _anchors(decision, image.width, image.height)
            boxes = [_box(row, image.width, image.height) for row in decision.get("approved_regions", [])]
            # Raw OCR boxes protect unrelated text that already has an identity.
            for row in scan.get("detection", {}).get("text_regions", []):
                if len(normalize_text(row.get("text", ""))) >= 2:
                    boxes.append(_box(row, image.width, image.height))
            boxes = [box for box in boxes if box is not None]
            for anchor in anchors:
                _stop(should_stop)
                candidates, truncated = _extract(image, anchor, boxes, should_stop)
                item["complete"] &= not truncated
                for candidate in candidates:
                    candidate.update(item_id=item_id, source_sha256=info["sha256"], product_key=scan["product_key"])
                    item["candidates"].append(candidate)
                    grouped[(scan["product_key"], anchor["brand"])].append(candidate)
            if not anchors:
                result["warnings"].append("现有批准文字未提供可核对的品牌锚点；独立图案保留原样。")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            item["complete"] = False
            result["review_reasons"].append("图形邻域读取或源文件校验未完成：" + type(exc).__name__)
        finally:
            if image is not None:
                image.close()
    # Compute independent repetition before linking edge fragments; this avoids
    # circular evidence in which two unproven lines would approve each other.
    for item in prepared:
        _stop(should_stop)
        for candidate in item["candidates"]:
            _stop(should_stop)
            votes = {}
            for other in grouped[(candidate["product_key"], candidate["anchor"]["brand"])]:
                score = _similar(candidate, other)
                if score and (other["source_sha256"] not in votes or score > votes[other["source_sha256"]][0]):
                    votes[other["source_sha256"]] = (score, other)
            backgrounds = [other["context"] for _, other in votes.values()]
            changed = any(float(np.linalg.norm(a - b)) >= 18 for a in backgrounds for b in backgrounds)
            candidate["proof"] = {"count": len(votes), "changed": bool(changed),
                                  "minimum_similarity": min((s for s, _ in votes.values()), default=0),
                                  "base_approved": bool(item["complete"] and candidate["eligible"] and len(votes) >= 3 and changed)}
    for item in prepared:
        result = item["result"]
        for candidate in item["candidates"]:
            _stop(should_stop)
            proof = candidate["proof"]
            linked_curve = next((other for other in item["candidates"]
                                 if candidate["kind"] == "edge_line" and other["kind"] == "line"
                                 and other["channel"] == "chromatic" and other["proof"]["base_approved"]
                                 and candidate["anchor"] == other["anchor"] and _same_ink(candidate, other)), None)
            approved = proof["base_approved"] or bool(item["complete"] and linked_curve is not None
                                                     and proof["count"] >= 3 and proof["changed"])
            # A ring-shaped buckle or cloth speckle near a logo is not enough
            # to burden every product photograph with a graphics warning.
            # Only independently repeated shapes become reviewable candidates.
            proposed = (linked_curve is not None if candidate["kind"] == "edge_line" else proof["count"] >= 3)
            if not proposed:
                continue
            regions = candidate["rectangles"]
            if len(result["proposed_regions"]) + len(regions) > MAX_REGIONS:
                result["review_reasons"].append("图形候选较多，已达到小遮罩数量上限，剩余内容未自动处理。")
                continue
            result["proposed_regions"].extend(regions)
            result["approved_regions" if approved else "review_regions"].extend(regions)
            evidence = {"kind": candidate["kind"], "channel": candidate["channel"],
                        "anchor_brand": candidate["anchor"]["brand"], "corner": candidate["anchor"]["corner"],
                        "component_box": candidate["component_box"], "regions": regions,
                        "independent_images": proof["count"], "changing_local_background": proof["changed"],
                        "minimum_shape_similarity": round(proof["minimum_similarity"], 6),
                        "shape_sha256": candidate["shape_sha256"], "approved": bool(approved),
                        "basis": "approved_brand_neighbor_repeated_shape_not_new_brand"}
            if candidate["kind"] == "edge_line":
                evidence.update(basis="repeated_edge_fragment_of_independently_approved_colored_curve",
                                linked_curve_sha256=linked_curve["shape_sha256"],
                                linked_curve_independent_images=linked_curve["proof"]["count"],
                                matching_relative_ink_and_stroke=True)
            result["evidence"]["graphics"].append(evidence)
            if not approved:
                result["review_reasons"].append("邻接图形缺少三张独立原图、变化背景或非装饰线条的充分证据，保留待确认。")
        # A component may lie in two neighboring brand-anchor search windows.
        for key in ("approved_regions", "proposed_regions", "review_regions"):
            result[key] = list({(r["x"], r["y"], r["w"], r["h"]): r for r in result[key]}.values())
            result[key] = [r for i, r in enumerate(result[key]) if not any(
                j != i and other["x"] <= r["x"] and other["y"] <= r["y"]
                and other["x"] + other["w"] >= r["x"] + r["w"]
                and other["y"] + other["h"] >= r["y"] + r["h"]
                for j, other in enumerate(result[key]))]
        # The disconnected inner letter of a registered-mark outline belongs
        # to that same graphic. Do not make its weaker independent descriptor
        # veto an already approved enclosing graphic rectangle.
        result["review_regions"] = [r for r in result["review_regions"] if not any(
            other["x"] <= r["x"] and other["y"] <= r["y"]
            and other["x"] + other["w"] >= r["x"] + r["w"]
            and other["y"] + other["h"] >= r["y"] + r["h"]
            for other in result["approved_regions"])]
        if not result["review_regions"]:
            result["review_reasons"] = [r for r in result["review_reasons"]
                if not r.startswith("邻接图形缺少")]
        result["review_reasons"] = list(dict.fromkeys(result["review_reasons"]))
        result["warnings"].append("图形补充只核对已批准品牌附近的重复细笔画；不能保证发现独立 Logo，也不能证明所有重复图案都是水印。")
    return [item["result"] for item in prepared]
