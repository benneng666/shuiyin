"""Graphic companions need brand, independent images and shape evidence."""
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageDraw, ImageFont

from buyer_show.watermark_graphics import analyze_graphics


class GraphicsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="buyer_show_graphics_")
        self.root = Path(self.temp.name)
        self.font = ImageFont.load_default(size=16)

    def tearDown(self):
        self.temp.cleanup()

    def inputs(self, *, same_bytes=False, static_plate=False, straight=False, far=False,
               anchors=True, products=None, changing_shape=False):
        scans, decisions = [], []
        for index in range(3):
            level = 80 if same_bytes else 80 + index * 40
            image = Image.new("RGB", (640, 400), (level, level, level))
            draw = ImageDraw.Draw(image)
            if static_plate:
                draw.rectangle((0, 10, 245, 165), fill=(120, 120, 120))
            draw.text((10, 74), "NORTHSHOP", font=self.font, fill="white")
            draw.ellipse((185, 40, 205, 60), outline="white", width=2)
            draw.text((190, 43), "R", font=ImageFont.load_default(size=11), fill="white")
            if far:
                draw.arc((20, 290, 190, 330), 0, 180, fill=(30, 220, 240), width=2)
            elif straight:
                draw.line((20, 120, 190, 120), fill=(30, 220, 240), width=2)
            elif changing_shape:
                points = [(20, 110), (60 + index * 30, 125 + index * 3), (190, 110)]
                draw.line(points, fill=(30, 220, 240), width=2)
            else:
                draw.arc((20, 95, 190, 135), 0, 180, fill=(30, 220, 240), width=2)
            path = self.root / f"{index}.png"
            image.save(path)
            image.close()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            scans.append({"item_id": str(index), "path": str(path),
                          "product_key": products[index] if products else "taobao:fixture",
                          "source_types": ["product_main"], "detection": {"source_sha256": digest, "text_regions": []}})
            decisions.append({"item_id": str(index), "approved_regions": ([
                {"x": 10, "y": 70, "w": 180, "h": 30, "label": "批量商家水印：NORTHSHOP"}] if anchors else []),
                "evidence": {"merchant_tokens": ["northshop"]}})
        return scans, decisions

    def test_repeated_registered_symbol_and_colored_curve_have_small_independent_masks(self):
        scans, decisions = self.inputs()
        original = copy.deepcopy(decisions)
        result = analyze_graphics(scans, decisions)
        self.assertEqual(decisions, original)
        json.dumps(result, allow_nan=False)
        for row in result:
            self.assertTrue(all(region.get("detection_kind") == "graphic" for region in row["proposed_regions"]))
            accepted = [g for g in row["evidence"]["graphics"] if g["approved"]]
            self.assertTrue(any(g["kind"] == "symbol" for g in accepted), row)
            self.assertTrue(any(g["kind"] == "line" and g["channel"] == "chromatic" for g in accepted), row)
            self.assertTrue(all(g["independent_images"] == 3 for g in accepted))
            lines = [r for g in accepted if g["kind"] == "line" for r in g["regions"]]
            self.assertTrue(lines)
            self.assertTrue(all(r["w"] <= 16 and r["h"] <= 16 for r in lines))
            self.assertFalse(any(r["x"] <= 105 < r["x"]+r["w"] and r["y"] <= 118 < r["y"]+r["h"] for r in lines))
            self.assertEqual(row["source_sha256"], scans[int(row["item_id"])]["detection"]["source_sha256"])

    def test_same_bytes_cannot_vote_three_times(self):
        scans, decisions = self.inputs(same_bytes=True)
        self.assertTrue(all(not row["approved_regions"] for row in analyze_graphics(scans, decisions)))

    def test_static_physical_plate_cannot_borrow_changed_remote_background(self):
        scans, decisions = self.inputs(static_plate=True)
        result = analyze_graphics(scans, decisions)
        self.assertTrue(all(not row["approved_regions"] for row in result))
        self.assertTrue(all(not g["changing_local_background"] for row in result for g in row["evidence"]["graphics"]))

    def test_graphics_do_not_invent_anchors_or_cross_product_evidence(self):
        scans, decisions = self.inputs(anchors=False)
        self.assertTrue(all(not row["proposed_regions"] for row in analyze_graphics(scans, decisions)))
        scans, decisions = self.inputs(products=["taobao:1", "taobao:2", "taobao:3"])
        self.assertTrue(all(not row["approved_regions"] for row in analyze_graphics(scans, decisions)))

    def test_straight_decorative_divider_is_not_automatically_removed(self):
        scans, decisions = self.inputs(straight=True)
        result = analyze_graphics(scans, decisions)
        self.assertFalse(any(g["kind"] == "line" and g["approved"] for row in result for g in row["evidence"]["graphics"]))

    def test_remote_colored_product_line_outside_brand_neighborhood_is_ignored(self):
        scans, decisions = self.inputs(far=True)
        result = analyze_graphics(scans, decisions)
        self.assertFalse(any(g["kind"] == "line" for row in result for g in row["evidence"]["graphics"]))

    def test_transparent_hidden_graphics_are_not_detected_as_visible_marks(self):
        scans, decisions = self.inputs()
        for scan in scans:
            path = Path(scan["path"])
            with Image.open(path) as image:
                rgba = image.convert("RGBA")
            rgba.putalpha(0)
            rgba.save(path)
            rgba.close()
            scan["detection"]["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertTrue(all(not row["proposed_regions"] for row in analyze_graphics(scans, decisions)))

    def test_changed_source_is_reviewable_and_cannot_vote(self):
        scans, decisions = self.inputs()
        scans[0]["detection"]["source_sha256"] = "0" * 64
        result = analyze_graphics(scans, decisions)
        self.assertTrue(result[0]["review_reasons"])
        self.assertTrue(all(not row["approved_regions"] for row in result))

    def test_cancel_has_no_file_mutation(self):
        scans, decisions = self.inputs()
        before = [Path(scan["path"]).read_bytes() for scan in scans]
        with self.assertRaises(asyncio.CancelledError):
            analyze_graphics(scans, decisions, should_stop=lambda: True)
        self.assertEqual(before, [Path(scan["path"]).read_bytes() for scan in scans])

    def edge_inputs(self, *, curve=True, wrong_color=False, one_off=False, static_edge=False):
        scans, decisions = [], []
        for index in range(3):
            level = 80 + index * 40
            image = Image.new("RGB", (640, 400), (level, level, level))
            draw = ImageDraw.Draw(image)
            draw.text((430, 284), "NORTHSHOP", font=self.font, fill="white")
            # Deliberately magenta, not the real sample's cyan.
            ink = (230, 40, 180)
            if curve:
                draw.arc((440, 310, 610, 350), 0, 180, fill=ink, width=2)
            if static_edge:
                draw.rectangle((625, 335, 639, 390), fill=(120, 120, 120))
            if not one_off or index == 0:
                draw.line((637, 345, 637, 382), fill=(40, 220, 100) if wrong_color else ink, width=3)
            path = self.root / f"edge-{index}.png"
            image.save(path)
            image.close()
            scans.append({"item_id": str(index), "path": str(path), "product_key": "taobao:edge-fixture",
                          "source_types": ["product_main"], "detection": {
                              "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}})
            decisions.append({"item_id": str(index), "approved_regions": [{
                "x": 430, "y": 280, "w": 200, "h": 40, "label": "批量商家水印：NORTHSHOP"}],
                "evidence": {"merchant_tokens": ["northshop"]}})
        return scans, decisions

    def test_short_edge_line_requires_independently_approved_curve_same_ink_and_repeated_layout(self):
        result = analyze_graphics(*self.edge_inputs())
        for row in result:
            edges = [g for g in row["evidence"]["graphics"] if g["kind"] == "edge_line"]
            self.assertTrue(edges, row)
            self.assertTrue(all(g["approved"] and g["independent_images"] == 3 for g in edges))
            self.assertTrue(all(g["linked_curve_independent_images"] == 3 for g in edges))
            self.assertTrue(all(g["matching_relative_ink_and_stroke"] for g in edges))

    def test_standalone_or_differently_colored_edge_divider_is_not_approved(self):
        for kwargs in ({"curve": False}, {"wrong_color": True}):
            with self.subTest(kwargs=kwargs):
                result = analyze_graphics(*self.edge_inputs(**kwargs))
                self.assertFalse(any(g["kind"] == "edge_line" and g["approved"]
                                     for row in result for g in row["evidence"]["graphics"]))

    def test_linked_edge_without_independent_repetition_or_changed_local_background_gets_review_box(self):
        for kwargs in ({"one_off": True}, {"static_edge": True}):
            with self.subTest(kwargs=kwargs):
                result = analyze_graphics(*self.edge_inputs(**kwargs))
                edges = [g for g in result[0]["evidence"]["graphics"] if g["kind"] == "edge_line"]
                self.assertTrue(edges)
                self.assertTrue(all(not g["approved"] for g in edges))
                self.assertTrue(result[0]["review_regions"])


if __name__ == "__main__":
    unittest.main()
