"""Independent material copies, display ratios and manifest-backed queues."""
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from buyer_show.media_catalog import load_media_queue, safe_path
from buyer_show.storage import ProductStore, validate_image

SOURCE = "https://item.taobao.com/item.htm?id=123"


class CategoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="buyer_show_categories_")
        self.addCleanup(self.temp.cleanup)
        self.store = ProductStore(Path(self.temp.name), "taobao", "123", SOURCE)

    def image(self, key, size, *, orientation=None, sources=()):
        buffer = io.BytesIO()
        image = Image.new("RGB", size, "coral")
        if orientation:
            exif = Image.Exif()
            exif[274] = orientation
            image.save(buffer, "JPEG", exif=exif)
        else:
            image.save(buffer, "PNG")
        payload = buffer.getvalue()
        info = validate_image(payload)
        relative = "images/" + info["sha256"] + info.pop("extension")
        target = self.store.root / relative
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(payload)
        url = "https://img.alicdn.com/" + key
        self.store.data["images"][url] = dict(status="downloaded", file=relative, **info)
        self.store.merge_product_media([dict(url=url, media_type="image", source_type=s) for s in sources])
        return url, target

    def review(self, url, kind="initial", rid="r1", media="image"):
        self.store.merge([dict(platform="taobao", product_id="123", review_id=rid,
            review_type=kind, content="", images=[url] if media == "image" else [],
            videos=[url] if media == "video" else [])])

    def entries(self):
        return self.store.data["classification"]["entries"]

    def test_main_ratio_exif_and_details_use_source_not_shape(self):
        self.image("square.png", (100, 100), sources=["product_main"])
        self.image("near-square.png", (100, 101), sources=["product_main"])
        self.image("portrait.jpg", (400, 300), orientation=6, sources=["product_main"])
        self.image("wide.png", (120, 60), sources=["product_main"])
        self.image("detail.png", (101, 101), sources=["product_detail"])
        result = self.store.refresh_categories()
        self.assertTrue(result["complete"])
        self.assertEqual(result["created"], 5)
        categories = [e["category"] for e in self.entries()]
        self.assertEqual(categories.count("素材分类/商品主图/1x1"), 2)
        self.assertIn("素材分类/商品主图/3x4", categories)
        self.assertIn("素材分类/商品主图/其他比例", categories)
        self.assertIn("素材分类/详情图", categories)
        portrait = next(e for e in self.entries() if e["ratio"] == "3x4")
        self.assertEqual((portrait["display_width"], portrait["display_height"]), (300, 400))
        for entry in self.entries():
            original, copied = self.store.root / entry["source_files"][0], self.store.root / entry["file"]
            self.assertEqual(original.read_bytes(), copied.read_bytes())
            self.assertFalse(os.path.samefile(original, copied))

    def test_multiple_sources_duplicate_urls_and_queue_deduplicate_hash(self):
        url, original = self.image("same.png", (80, 80), sources=["product_main", "product_detail"])
        alias, _ = self.image("alias.png", (80, 80), sources=["product_main"])
        self.review(url)
        self.review(alias, rid="r2")
        self.review(url, kind="append")
        result = self.store.refresh_categories()
        self.assertEqual(result["created"], 4)
        initial = next(e for e in self.entries() if e["category"] == "素材分类/买家秀/初评/图片")
        self.assertEqual(len(initial["associations"]), 2)
        queue = load_media_queue(self.store.path)
        self.assertEqual(len(queue), 1)
        self.assertEqual(Path(queue[0]["path"]), original.resolve())
        self.assertEqual(len(queue[0]["categories"]), 4)
        self.assertEqual(set(queue[0]["source_types"]), {"product_main", "product_detail", "review_initial", "review_append"})
        self.assertNotIn("https://", json.dumps(queue))
        again = self.store.refresh_categories()
        self.assertEqual((again["created"], again["reused"]), (0, 4))

    def test_user_edited_copy_is_preserved_and_original_copy_gets_new_name(self):
        _, original = self.image("main.png", (80, 80), sources=["product_main"])
        self.store.refresh_categories()
        old = self.store.root / self.entries()[0]["file"]
        original_bytes = original.read_bytes()
        old.write_bytes(b"user edited version")
        result = self.store.refresh_categories()
        new = self.store.root / self.entries()[0]["file"]
        self.assertNotEqual(old, new)
        self.assertEqual(old.read_bytes(), b"user edited version")
        self.assertEqual(new.read_bytes(), original_bytes)
        self.assertEqual(original.read_bytes(), original_bytes)
        self.assertEqual(result["preserved_user_modified"], 1)
        self.assertIn(old.relative_to(self.store.root).as_posix(), self.entries()[0]["preserved_files"])
        self.assertEqual(self.store.refresh_categories()["created"], 0)

    def test_video_sources_copy_to_product_and_review_directories(self):
        payload = b"synthetic fixture; video decoding tested in test_video.py"
        digest = hashlib.sha256(payload).hexdigest()
        relative = f"videos/{digest}.mp4"
        original = self.store.root / relative
        original.parent.mkdir()
        original.write_bytes(payload)
        url = "https://cloud.video.taobao.com/a.mp4"
        self.store.data["videos"][url] = dict(status="downloaded", file=relative, sha256=digest)
        self.store.merge_product_media([dict(url=url, media_type="video", source_type="product_detail")])
        self.review(url, media="video")
        self.review(url, media="video", kind="append")
        self.assertEqual(self.store.refresh_categories()["created"], 3)
        self.assertEqual({e["category"] for e in self.entries()},
            {"素材分类/商品视频", "素材分类/买家秀/初评/视频", "素材分类/买家秀/追评/视频"})
        self.assertEqual(len(load_media_queue(self.store.path, "video")), 1)
        self.assertEqual(load_media_queue(self.store.path, "image"), [])

    def test_legacy_records_missing_classification_and_video_fields(self):
        url, _ = self.image("legacy.png", (70, 100))
        self.review(url)
        self.store.data.pop("videos")
        self.store.data.pop("product_media")
        self.store.data["records"][0].pop("videos")
        self.store.path.write_text(json.dumps(self.store.data), encoding="utf-8")
        self.assertEqual(len(load_media_queue(self.store.path)), 1)
        restored = ProductStore(Path(self.temp.name), "taobao", "123", SOURCE)
        self.assertTrue(restored.refresh_categories()["complete"])
        self.assertEqual(restored.data["schema_version"], 1)

    def test_missing_corrupt_or_unsafe_original_does_not_enter_queue(self):
        url, original = self.image("bad.png", (70, 100), sources=["product_main"])
        original.write_bytes(b"corrupt")
        result = self.store.refresh_categories()
        self.assertEqual(result["failed"], 1)
        self.assertFalse(result["complete"])
        self.assertEqual(load_media_queue(self.store.path), [])
        self.assertFalse(self.store.finish(dict(status="completed", complete=True))["complete"])
        self.store.data["images"][url]["file"] = "../outside.png"
        self.store.checkpoint()
        self.assertEqual(load_media_queue(self.store.path), [])
        for path in ("../escape", "/absolute", "素材分类/file:stream", "素材分类\\escape"):
            with self.assertRaises(ValueError):
                safe_path(self.store.root, path, create_parents=True)

    def test_copy_failure_and_cancel_never_claim_completed(self):
        self.image("a.png", (70, 100), sources=["product_main"])
        self.image("b.png", (71, 100), sources=["product_main"])
        with patch("buyer_show.media_catalog._copy_independent", side_effect=PermissionError()):
            result = self.store.refresh_categories()
        self.assertEqual(result["failed"], 2)
        self.assertFalse(self.store.finish(dict(status="completed", complete=True))["complete"])
        stopped = False
        from buyer_show.media_catalog import _copy_independent

        def copy_then_stop(*args):
            nonlocal stopped
            _copy_independent(*args)
            stopped = True

        with patch("buyer_show.media_catalog._copy_independent", side_effect=copy_then_stop):
            result = self.store.refresh_categories(should_stop=lambda: stopped)
        self.assertEqual(result["created"], 1)
        self.assertEqual(result["pending"], 1)
        self.assertTrue(result["cancelled"])
        self.assertFalse(result["complete"])
        resumed = self.store.refresh_categories()
        self.assertTrue(resumed["complete"])
        self.assertEqual((resumed["created"], resumed["reused"]), (1, 1))

    def test_category_junction_or_symlink_is_not_followed(self):
        self.image("a.png", (70, 100), sources=["product_main"])
        from buyer_show.media_catalog import _is_link

        def is_link(path):
            return path.name == "素材分类" or _is_link(path)

        with patch("buyer_show.media_catalog._is_link", side_effect=is_link):
            result = self.store.refresh_categories()
        self.assertEqual(result["failed"], 1)
        self.assertFalse((self.store.root / "素材分类").exists())

    def test_new_association_invalidates_old_classification_until_refreshed(self):
        url, _ = self.image("main.png", (70, 100), sources=["product_main"])
        self.assertTrue(self.store.refresh_categories()["complete"])
        self.review(url)
        self.assertFalse(self.store.finish(dict(status="completed", complete=True))["complete"])
        self.assertTrue(self.store.refresh_categories()["complete"])
        self.assertTrue(self.store.finish(dict(status="completed", complete=True))["complete"])


if __name__ == "__main__":
    unittest.main()
