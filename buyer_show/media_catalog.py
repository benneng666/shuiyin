"""Safe material categories and a deduplicated queue of verified originals.

Category files are independent copies. They are never hard links, and an
existing file with different bytes is preserved rather than overwritten.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath

from PIL import Image

_ORIGINAL = re.compile(r"(images|videos)/([a-f0-9]{64})\.(jpg|png|webp|gif|bmp|avif|mp4|mov|webm|mkv)")
_IMAGE_EXTENSIONS = {"jpg", "png", "webp", "gif", "bmp", "avif"}
CATEGORY_ROOT = "素材分类"
RATIO_RELATIVE_TOLERANCE = 0.02


class CatalogError(ValueError):
    pass


class _Cancelled(Exception):
    pass


def _stop(should_stop):
    if should_stop():
        raise _Cancelled()


def _is_link(path):
    try:
        attrs = getattr(path.lstat(), "st_file_attributes", 0)
        return path.is_symlink() or bool(attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    except FileNotFoundError:
        return False


def safe_path(root, relative, *, create_parents=False):
    """No absolute paths, traversal, symlink/junction ancestors, or NTFS ADS."""
    if not isinstance(relative, str) or "\\" in relative or ":" in relative:
        raise CatalogError("unsafe_catalog_path")
    parts = PurePosixPath(relative).parts
    if not parts or relative.startswith("/") or any(x in {".", ".."} for x in parts):
        raise CatalogError("unsafe_catalog_path")
    root = Path(root).absolute()
    if _is_link(root):
        raise CatalogError("linked_product_directory")
    root_resolved = root.resolve(strict=True)
    current = root
    for index, part in enumerate(parts):
        current = current / part
        if _is_link(current):
            raise CatalogError("linked_catalog_path")
        if index < len(parts) - 1:
            if create_parents:
                current.mkdir(exist_ok=True)
            if current.exists() and not current.is_dir():
                raise CatalogError("catalog_parent_not_directory")
        if not current.resolve().is_relative_to(root_resolved):
            raise CatalogError("catalog_path_outside_product")
    return current


def sha256_file(path, should_stop=lambda: False):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            _stop(should_stop)
            digest.update(chunk)
    return digest.hexdigest()


def _source_path(root, item, media_type):
    relative, expected = item.get("file"), item.get("sha256")
    match = _ORIGINAL.fullmatch(relative) if isinstance(relative, str) else None
    if (not match or match[2] != expected or
            (match[1] == "images") != (media_type == "image") or
            (match[3] in _IMAGE_EXTENSIONS) != (media_type == "image")):
        raise CatalogError("invalid_original_path_or_hash")
    path = safe_path(root, relative)
    if not path.is_file():
        raise CatalogError("original_missing")
    if not 0 < path.stat().st_size <= (32 if media_type == "image" else 256) * 1024 * 1024:
        raise CatalogError("original_size_invalid")
    return path


def _associations(data):
    lookup = {}

    def add(kind, url, source, review_id=None):
        if not isinstance(url, str):
            return
        ref = {"source_type": source}
        if review_id is not None:
            ref["review_id"] = str(review_id)
        refs = lookup.setdefault((kind, url), [])
        if ref not in refs:
            refs.append(ref)

    for media in data.get("product_media", []):
        if media.get("media_type") in {"image", "video"} and media.get("source_type") in {"product_main", "product_detail", "product_video"}:
            add(media["media_type"], media.get("url"), media["source_type"])
    for row in data.get("records", []):
        if row.get("review_type") not in {"initial", "append"}:
            continue
        for kind, field in (("image", "images"), ("video", "videos")):
            for url in row.get(field, []):
                add(kind, url, "review_" + row["review_type"], row.get("review_id"))
    return lookup


def _display_size(path):
    # EXIF transforms 5/6/7/8 swap axes. Read-only: copied files retain every byte.
    with Image.open(path) as image:
        width, height = image.size
        orientation = image.getexif().get(274, 1)
    return (height, width) if orientation in {5, 6, 7, 8} else (width, height)


def _ratio(width, height):
    value = width / height
    if abs(value - 1) <= RATIO_RELATIVE_TOLERANCE:
        return "1x1"
    if abs(value / 0.75 - 1) <= RATIO_RELATIVE_TOLERANCE:
        return "3x4"
    return "其他比例"


def _category(source, kind, ratio):
    if source.startswith("review_"):
        part = "初评" if source == "review_initial" else "追评"
        return f"{CATEGORY_ROOT}/买家秀/{part}/{'图片' if kind == 'image' else '视频'}"
    if kind == "video" and source in {"product_main", "product_detail", "product_video"}:
        return f"{CATEGORY_ROOT}/商品视频"
    if source == "product_main" and kind == "image":
        return f"{CATEGORY_ROOT}/商品主图/{ratio}"
    if source == "product_detail" and kind == "image":
        return f"{CATEGORY_ROOT}/详情图"
    return f"{CATEGORY_ROOT}/未分类/{'图片' if kind == 'image' else '视频'}"


def _successful(data):
    for kind, field in (("image", "images"), ("video", "videos")):
        for url, item in data.get(field, {}).items():
            if item.get("status") == "downloaded":
                yield kind, url, item


def build_media_queue(root, data, media_type=None):
    """Return verified originals only; URLs/cookies/review text never leave here."""
    if media_type not in {None, "image", "video"}:
        raise ValueError("invalid_media_type")
    associations = _associations(data)
    queue, verified = {}, {}
    for kind, url, item in _successful(data):
        if media_type is not None and kind != media_type:
            continue
        try:
            path = _source_path(root, item, kind)
            key = (str(path), item["sha256"])
            if key not in verified:
                verified[key] = sha256_file(path) == item["sha256"]
            if not verified[key]:
                continue
        except (OSError, ValueError):
            continue
        key = (kind, item["sha256"])
        result = queue.setdefault(key, {"path": str(path.resolve()), "media_type": kind,
            "sha256": item["sha256"], "categories": [], "source_types": []})
        for ref in associations.get((kind, url), [{"source_type": "unclassified"}]):
            if ref["source_type"] not in result["source_types"]:
                result["source_types"].append(ref["source_type"])
    for entry in data.get("classification", {}).get("entries", []):
        result = queue.get((entry.get("media_type"), entry.get("source_sha256")))
        if result and entry.get("status") == "ready" and entry.get("file"):
            try:
                safe_path(root, entry["file"])
            except (OSError, ValueError):
                continue
            if entry["file"] not in result["categories"]:
                result["categories"].append(entry["file"])
    return list(queue.values())


def load_media_queue(manifest_path, media_type=None):
    path = Path(manifest_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("invalid_media_manifest")
    return build_media_queue(path.parent, data, media_type)


def _copy_independent(source, target, expected, should_stop):
    created = False
    try:
        # Exclusive creation protects existing user files, including races.
        with target.open("xb") as destination:
            created = True
            digest = hashlib.sha256()
            with source.open("rb") as original:
                while chunk := original.read(1024 * 1024):
                    _stop(should_stop)
                    destination.write(chunk)
                    digest.update(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        if digest.hexdigest() != expected:
            raise CatalogError("original_changed_during_copy")
    except BaseException:
        if created:
            target.unlink(missing_ok=True)
        raise


def refresh_categories(root, data, should_stop=lambda: False, emit=lambda msg: None):
    """Update the in-memory manifest; caller checkpoints it after this returns."""
    previous = {e.get("key"): e for e in data.get("classification", {}).get("entries", [])}
    associations, desired, errors, verified = _associations(data), {}, [], {}
    summary = dict(created=0, reused=0, failed=0, pending=0, preserved_user_modified=0, cancelled=False, complete=False)
    try:
        for kind, url, item in _successful(data):
            _stop(should_stop)
            try:
                source = _source_path(root, item, kind)
                identity = (item["file"], item["sha256"])
                if identity not in verified:
                    if sha256_file(source, should_stop) != item["sha256"]:
                        raise CatalogError("original_hash_mismatch")
                    dimensions = _display_size(source) if kind == "image" else (None, None)
                    verified[identity] = dimensions
                width, height = verified[identity]
                ratio = _ratio(width, height) if kind == "image" else None
                for ref in associations.get((kind, url), [{"source_type": "unclassified"}]):
                    category = _category(ref["source_type"], kind, ratio)
                    key = "|".join((kind, item["sha256"], category))
                    entry = desired.setdefault(key, {"key": key, "media_type": kind,
                        "source_sha256": item["sha256"], "source_files": [], "category": category,
                        "associations": [], "source_types": [], "display_width": width,
                        "display_height": height, "ratio": ratio, "status": "pending", "file": None,
                        "preserved_files": list(previous.get(key, {}).get("preserved_files", []))})
                    if item["file"] not in entry["source_files"]:
                        entry["source_files"].append(item["file"])
                    if ref not in entry["associations"]:
                        entry["associations"].append(ref)
                    if ref["source_type"] not in entry["source_types"]:
                        entry["source_types"].append(ref["source_type"])
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                errors.append({"media_type": kind, "source_sha256": item.get("sha256"),
                               "error": str(exc) if isinstance(exc, CatalogError) else type(exc).__name__})
        for key, entry in desired.items():
            _stop(should_stop)
            try:
                source = safe_path(root, entry["source_files"][0])
                ext = source.suffix
                expected = entry["source_sha256"]
                category = entry["category"]
                prior_file = previous.get(key, {}).get("file")
                candidates = []
                if (isinstance(prior_file, str) and PurePosixPath(prior_file).parent.as_posix() == category
                        and re.fullmatch(re.escape(expected) + r"(?:-原件-\d+)?" + re.escape(ext), PurePosixPath(prior_file).name)):
                    candidates.append(prior_file)
                canonical = f"{category}/{expected}{ext}"
                if canonical not in candidates:
                    candidates.append(canonical)
                serial = 2
                while True:
                    _stop(should_stop)
                    relative = candidates.pop(0) if candidates else f"{category}/{expected}-原件-{serial}{ext}"
                    target = safe_path(root, relative, create_parents=True)
                    if target.exists():
                        if not target.is_file():
                            raise CatalogError("category_target_not_file")
                        same_source = os.path.samefile(source, target)
                        if not same_source and sha256_file(target, should_stop) == expected:
                            entry.update(file=relative, status="ready", action="reused")
                            summary["reused"] += 1
                            break
                        if relative not in entry["preserved_files"]:
                            entry["preserved_files"].append(relative)
                            summary["preserved_user_modified"] += 1
                        serial += 1
                        if serial > 1000:
                            raise CatalogError("too_many_preserved_category_files")
                        continue
                    try:
                        _copy_independent(source, target, expected, should_stop)
                    except FileExistsError:
                        continue  # A concurrent creator wins; never overwrite it.
                    entry.update(file=relative, status="ready", action="created")
                    summary["created"] += 1
                    break
            except (OSError, ValueError) as exc:
                entry.update(status="failed", error=str(exc) if isinstance(exc, CatalogError) else type(exc).__name__)
                summary["failed"] += 1
    except _Cancelled:
        summary["cancelled"] = True
    summary["failed"] += len(errors)
    summary["pending"] = sum(e["status"] == "pending" for e in desired.values())
    summary["complete"] = not summary["cancelled"] and not summary["failed"] and not summary["pending"]
    # Historical category files are never removed. Keep a record of categories
    # absent from the current manifest without treating them as current success.
    retired = [e for key, e in previous.items() if key not in desired]
    data["classification"] = {"schema_version": 1, "root": CATEGORY_ROOT,
        "ratio_relative_tolerance": RATIO_RELATIVE_TOLERANCE, "summary": summary,
        "complete": summary["complete"], "entries": list(desired.values()), "errors": errors,
        "historical_entries": data.get("classification", {}).get("historical_entries", []) + retired}
    emit(f"素材分类：新建 {summary['created']}，复用 {summary['reused']}，失败 {summary['failed']}")
    return dict(summary)
