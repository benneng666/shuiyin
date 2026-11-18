"""Execute the DOM observer in Node against a small DOM fixture, no browser."""
import importlib.util
import json
from pathlib import Path
import subprocess
import unittest

from buyer_show.adapters.product_media import _PRODUCT_DOM, _PRODUCT_SCROLL, _taobao_completion_observation


_HARNESS = r"""
const fs = require('fs');
const job = JSON.parse(fs.readFileSync(0, 'utf8'));
class Element {
  constructor(tag, attrs={}, rect={top:0,bottom:100,height:100}) {
    this.tagName=tag.toUpperCase(); this.attrs=attrs; this.id=attrs.id||''; this.className=attrs.class||'';
    this.children=[]; this.parentElement=null; this.rect=rect; this.visible=true; this.tabIndex=-1;
    this.textContent=''; this.complete=true; this.naturalWidth=800; this.naturalHeight=1000;
    this.currentSrc=attrs.src||''; this.scrollTop=0; this.scrollHeight=0; this.clientHeight=0; this.overflowY='visible';
  }
  add(child) {child.parentElement=this; this.children.push(child); return child;}
  all() {return this.children.flatMap(e=>[e,...e.all()]);}
  getAttribute(key) {return this.attrs[key]??null;}
  getClientRects() {return this.visible?[this.rect]:[];}
  getBoundingClientRect() {
    const delta=this.geometryScroll?this.geometryScroll.scrollTop-this.initialScroll:0;
    return {...this.rect,top:this.rect.top-delta,bottom:this.rect.bottom-delta};
  }
  contains(e) {while(e) {if(e===this)return true; e=e.parentElement;}return false;}
  querySelectorAll(selector) {
    return this.all().filter(e=>{
      if(selector==='*')return true;
      if(['img','video','source','iframe'].includes(selector))return e.tagName===selector.toUpperCase();
      if(selector.includes('mainPicVideo--'))return /mainPicVideo--|videoPicWrap--/.test(e.className);
      if(selector.includes('aria-busy'))return e.attrs['aria-busy']==='true'||e.attrs.role==='progressbar';
      return false;
    });
  }
  closest(selector) {
    for(let e=this;e;e=e.parentElement) {
      if(selector==='video'&&e.tagName==='VIDEO')return e;
      if(selector==='[class*="video"], [class*="Video"]'&&/video/i.test(e.className))return e;
      if(selector.includes('recommend')&&/recommend/i.test(e.className))return e;
    }
    return null;
  }
  scrollBy(_x,y) {this.scrollTop=Math.min(this.scrollHeight-this.clientHeight,Math.max(0,this.scrollTop+y));}
}
const body=new Element('body',{}, {top:0,bottom:800,height:800});
body.scrollTop=4200;body.clientHeight=800;body.scrollHeight=5000;
const gallery=body.add(new Element('div',{id:'picGalleryEle'}, {top:-4200,bottom:-3600,height:600}));
gallery.add(new Element('img',{src:'https://img.alicdn.com/main.jpg?sig=a%2Bb'}));
const detail=body.add(new Element('div',{class:'desc-root'}, {top:-3500,bottom:800,height:4300}));
const image=detail.add(new Element('img',{src:'https://img.alicdn.com/detail.jpg'}));
let scrollHost=null;
if(job.case==='lazy') {
  image.attrs['data-src']='https://img.alicdn.com/full.jpg?sig=a%2Bb';
  image.currentSrc=image.attrs.src='https://img.alicdn.com/loading.png';
  image.complete=false;image.naturalWidth=0;image.naturalHeight=0;
}
if(job.case==='broken') {image.complete=true;image.naturalWidth=0;image.naturalHeight=0;}
if(job.case==='small_content') {image.naturalWidth=32;image.naturalHeight=32;}
if(['video_pending','video_loaded','unbound_gallery_entry'].includes(job.case)) {
  const slot=gallery.add(new Element('div',{class:'mainPicVideo--fixture'}));
  const inner=slot.add(new Element('div',{class:'videoPicWrap--fixture'}));
  inner.add(new Element('img',{src:'https://img.alicdn.com/cover.jpg'}));
  if(job.case!=='video_pending')inner.add(new Element('video',{src:'https://cloud.video.taobao.com/product.mp4?sig=a%2Bb'}));
  if(job.case==='unbound_gallery_entry')gallery.add(new Element('div',{class:'videoPicWrap--separate-entry'}))
    .add(new Element('img',{src:'https://img.alicdn.com/cover.jpg'}));
}
if(['hidden_cover_layer','hidden_cover_without_player'].includes(job.case)) {
  if(job.case==='hidden_cover_layer') {
    const player=gallery.add(new Element('div',{class:'mainPicVideo--fixture'}));
    player.add(new Element('video',{src:'https://cloud.video.taobao.com/product.mp4'}));
  }
  const cover=gallery.add(new Element('div',{class:'videoPicWrap--fixture',opacity:'0','z-index':'-1'}));
  cover.add(new Element('img',{src:'https://img.alicdn.com/cover.jpg'}));
}
if(job.case==='iframe')detail.add(new Element('iframe'));
if(job.case==='expand') {const b=detail.add(new Element('button'));b.textContent='展开全部';}
if(job.case==='nested_scroll') {
  body.children=body.children.filter(e=>e!==detail);
  scrollHost=body.add(new Element('section',{id:'detail-scroll'}, {top:100,bottom:700,height:600}));
  scrollHost.overflowY='auto';scrollHost.scrollTop=300;scrollHost.clientHeight=600;scrollHost.scrollHeight=6000;
  scrollHost.add(detail);
  detail.rect={top:-200,bottom:5800,height:6000};
}
if(['recommendations','near_end','detail_grows','restored_in_recommendations'].includes(job.case)) {
  body.scrollHeight=64381;
  const top=job.case==='near_end'?19000:job.case==='restored_in_recommendations'?63000:19378;
  body.scrollTop=top;
  gallery.rect={top:-top,bottom:600-top,height:600};
  detail.rect={top:700-top,bottom:20178-top,height:19478};
}
if(job.case==='missing_detail') {body.children=body.children.filter(e=>e!==detail);detail.parentElement=null;}
let fakeDetail=null;
if(job.case==='recommendations') {
  const recommendation=body.add(new Element('section',{class:'recommend-list'}));
  fakeDetail=recommendation.add(new Element('div',{class:'desc-root'},{top:30000,bottom:50000,height:20000}));
  fakeDetail.add(new Element('img',{src:'https://img.alicdn.com/another-product.jpg'}));
}
for(const element of body.all()) {
  if(element===scrollHost)continue;
  element.geometryScroll=scrollHost&&scrollHost.contains(element)?scrollHost:body;
  element.initialScroll=element.geometryScroll.scrollTop;
}
global.document={body,documentElement:body,scrollingElement:body,
  querySelector:selector=>selector==='.desc-root'&&body.contains(detail)?detail:null,
  querySelectorAll:selector=>selector==='.desc-root'?(body.contains(detail)?[detail,...(fakeDetail?[fakeDetail]:[])]:[]):selector.includes('#picGalleryEle')?[gallery]:[],
  createTreeWalker:()=>({nextNode:()=>false})};
global.NodeFilter={SHOW_TEXT:4};
global.location={href:'https://detail.tmall.com/item.htm?id=950216355977'};
global.getComputedStyle=e=>({visibility:e.visible?'visible':'hidden',overflowY:e.overflowY,cursor:'auto',backgroundImage:'none',
  opacity:e.attrs.opacity??'1',zIndex:e.attrs['z-index']??'0',display:e.attrs.display??'flex'});
const observation=eval('('+job.dom+')')('taobao');
const scroll=eval('('+job.scroll+')')('taobao');
let grown=null;
if(job.case==='detail_grows') {
  detail.rect.bottom+=1200;detail.rect.height+=1200;
  const pending=eval('('+job.dom+')')('taobao');
  const steps=[eval('('+job.scroll+')')('taobao'),eval('('+job.scroll+')')('taobao')];
  grown={pending,steps,terminal:eval('('+job.dom+')')('taobao')};
}
console.log(JSON.stringify({observation,scroll,grown,document_scroll:body.scrollTop,nested_scroll:scrollHost&&scrollHost.scrollTop}));
"""


class ProductDOMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.find_spec("playwright")
        if spec is None:
            raise unittest.SkipTest("Playwright's bundled Node runtime is unavailable")
        cls.node = Path(next(iter(spec.submodule_search_locations))) / "driver" / "node.exe"
        if not cls.node.is_file():
            raise unittest.SkipTest("This fixture uses the existing Windows Node runtime")

    def observe(self, case):
        result = subprocess.run([str(self.node), "-e", _HARNESS], input=json.dumps(
            {"case": case, "dom": _PRODUCT_DOM, "scroll": _PRODUCT_SCROLL}),
            encoding="utf-8", capture_output=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_real_observer_output_can_prove_a_loaded_bottom_fixture(self):
        result = self.observe("loaded")["observation"]
        self.assertTrue(_taobao_completion_observation(result)[0])
        self.assertEqual(result["completion"]["scopes"]["product_detail"]["root_bounds"][0]["bottom"], 5000)
        self.assertTrue(any(row["url"].endswith("sig=a%2Bb") for row in result["media"]))

    def test_lazy_placeholder_and_broken_node_prevent_completion_without_url_guessing(self):
        result = self.observe("lazy")["observation"]
        detail = result["completion"]["scopes"]["product_detail"]
        self.assertEqual((detail["loading_images"], detail["lazy_pending"]), (1, 1))
        self.assertFalse(_taobao_completion_observation(result)[0])
        self.assertFalse(any("full.jpg" in row["url"] for row in result["media"]))
        broken = self.observe("broken")["observation"]
        self.assertEqual(broken["completion"]["scopes"]["product_detail"]["broken_images"], 1)
        self.assertFalse(_taobao_completion_observation(broken)[0])

    def test_small_loaded_content_is_not_guessed_to_be_a_placeholder(self):
        result = self.observe("small_content")["observation"]
        self.assertTrue(_taobao_completion_observation(result)[0])
        self.assertTrue(any(row["source_type"] == "product_detail" for row in result["media"]))

    def test_empty_video_slot_is_pending_and_nested_slot_is_counted_once(self):
        pending = self.observe("video_pending")["observation"]
        self.assertEqual(pending["unresolved_videos"], 1)
        self.assertEqual(pending["completion"]["scopes"]["product_main"]["video_slots"], 1)
        self.assertFalse(_taobao_completion_observation(pending)[0])
        loaded = self.observe("video_loaded")["observation"]
        self.assertEqual(loaded["unresolved_videos"], 0)
        self.assertTrue(_taobao_completion_observation(loaded)[0])
        self.assertFalse(any("cover.jpg" in row["url"] for row in loaded["media"]))

    def test_sibling_video_entry_stays_unresolved_without_an_explicit_binding(self):
        result = self.observe("unbound_gallery_entry")["observation"]
        scope = result["completion"]["scopes"]["product_main"]
        self.assertEqual((scope["video_slots"], scope["unresolved_video_slots"]), (2, 1))
        self.assertEqual(len([r for r in result["media"] if r["media_type"] == "video"]), 1)
        self.assertEqual([r["kind"] for r in scope["video_slot_details"]], ["main_player_container", "video_entry_unbound"])
        self.assertNotIn("sig=", json.dumps(scope["video_slot_details"]))
        self.assertFalse(_taobao_completion_observation(result)[0])

    def test_hidden_cover_layer_is_not_counted_as_unbound_video_entry(self):
        # Live Tmall 2026-11-09: opacity:0 / z-index:-1 cover beside a resolved player.
        result = self.observe("hidden_cover_layer")["observation"]
        scope = result["completion"]["scopes"]["product_main"]
        self.assertEqual((scope["video_slots"], scope["unresolved_video_slots"]), (1, 0))
        self.assertEqual(scope["hidden_cover_slots"], 1)
        self.assertEqual(result["unresolved_videos"], 0)
        self.assertTrue(_taobao_completion_observation(result)[0])
        self.assertEqual(len([r for r in result["media"] if r["media_type"] == "video"]), 1)

    def test_hidden_cover_without_resolved_player_still_blocks_completion(self):
        # The same hidden layer must not be used to explain away a missing video.
        result = self.observe("hidden_cover_without_player")["observation"]
        scope = result["completion"]["scopes"]["product_main"]
        self.assertEqual((scope["video_slots"], scope["unresolved_video_slots"]), (1, 1))
        self.assertEqual(scope["hidden_cover_slots"], 0)
        self.assertEqual(result["unresolved_videos"], 1)
        self.assertFalse(_taobao_completion_observation(result)[0])

    def test_embedded_or_expandable_detail_is_not_declared_complete(self):
        for case in ("iframe", "expand"):
            with self.subTest(case=case):
                self.assertFalse(_taobao_completion_observation(self.observe(case)["observation"])[0])

    def test_scroll_action_targets_observed_detail_ancestor_instead_of_window(self):
        result = self.observe("nested_scroll")
        self.assertEqual(result["observation"]["completion"]["scroll_context"], "SECTION:detail-scroll:")
        self.assertEqual(result["scroll"]["before"], 300)
        self.assertEqual(result["nested_scroll"], 780)
        self.assertEqual(result["document_scroll"], 4200)

    def test_recommendation_tail_is_outside_scroll_and_completion_scope(self):
        result = self.observe("recommendations")
        observation = result["observation"]
        self.assertEqual(observation["completion"]["scroll_height"], 64381)
        self.assertEqual(observation["completion"]["scope_bottom"], 20178)
        self.assertTrue(_taobao_completion_observation(observation)[0])
        self.assertEqual(result["document_scroll"], 19378)
        self.assertFalse(result["scroll"]["moved"])
        self.assertFalse(any("another-product" in row["url"] for row in observation["media"]))

    def test_last_scroll_clamps_and_restored_recommendation_position_moves_back(self):
        for case in ("near_end", "restored_in_recommendations"):
            with self.subTest(case=case):
                result = self.observe(case)
                self.assertEqual(result["document_scroll"], 19378)
                self.assertEqual(result["scroll"]["scope_bottom"], 20178)

    def test_newly_materialized_detail_extends_only_product_scroll_boundary(self):
        result = self.observe("detail_grows")
        self.assertTrue(_taobao_completion_observation(result["observation"])[0])
        self.assertFalse(_taobao_completion_observation(result["grown"]["pending"])[0])
        self.assertEqual(result["grown"]["pending"]["completion"]["scope_bottom"], 21378)
        self.assertEqual(result["document_scroll"], 20578)
        self.assertTrue(_taobao_completion_observation(result["grown"]["terminal"])[0])

    def test_missing_detail_does_not_authorize_document_scrolling(self):
        result = self.observe("missing_detail")
        self.assertFalse(_taobao_completion_observation(result["observation"])[0])
        self.assertEqual(result["scroll"]["reason"], "missing_detail_root")
        self.assertEqual(result["document_scroll"], 4200)


if __name__ == "__main__":
    unittest.main()
