"""Exercise our DOM JavaScript against a small in-memory tree, no browser."""

import json
import shutil
import subprocess
import unittest

from buyer_show.adapters.douyin import _SNAPSHOT_JS
from buyer_show.adapters.product_media import _PRODUCT_DOM
from buyer_show.adapters.taobao import _DOM, _SCROLL_REVIEWS


TREE = r"""
class El {
  constructor(tag, cls='', attrs={}, text='') {
    this.tagName=tag.toUpperCase(); this.className=cls; this.attrs=attrs; this.ownText=text;
    this.id=attrs.id || ''; this.tabIndex=-1;
    this.children=[]; this.parentElement=null; this.naturalWidth=400; this.naturalHeight=400; this.complete=true;
    this.scrollTop=0; this.clientHeight=200; this.scrollHeight=200;
    this.currentSrc=attrs.currentSrc || ''; this.bg=attrs.bg || 'none';
  }
  add(e) { this.children.push(e); e.parentElement=this; return e; }
  descendants() { return this.children.flatMap(e => [e,...e.descendants()]); }
  matches(selector) {
    selector=selector.trim();
    if (selector===':not(*)') return false;
    if (selector==='*') return true;
    if (selector.includes(' ')) {
      const parts=selector.split(/\s+/); const tail=parts.pop();
      return this.matches(tail) && !!this.parentElement?.closest(parts.join(' '));
    }
    const attr=/\[class\*=["']([^"']+)["']\]/.exec(selector);
    if (attr && !this.className.includes(attr[1])) return false;
    selector=selector.replace(/\[class\*=["'][^"']+["']\]/,'');
    const attributes=[...selector.matchAll(/\[([\w-]+)(?:=["']?([^"'\]]+)["']?)?\]/g)];
    if (attributes.some(m=>m[2]===undefined ? !(m[1] in this.attrs) : this.attrs[m[1]]!==m[2])) return false;
    selector=selector.replace(/\[[^\]]+\]/g,'');
    const id=/#([\w-]+)/.exec(selector);
    if (id && this.attrs.id!==id[1]) return false;
    const classes=[...selector.matchAll(/\.([\w-]+)/g)].map(m=>m[1]);
    if (!classes.every(c=>this.className.split(' ').includes(c))) return false;
    const tag=/^[a-z]+/.exec(selector);
    return !tag || this.tagName.toLowerCase()===tag[0];
  }
  closest(s) { for(let e=this;e;e=e.parentElement) if(s.split(',').some(v=>e.matches(v))) return e; return null; }
  querySelectorAll(s) { return this.descendants().filter(e=>s.split(',').some(v=>e.matches(v))); }
  querySelector(s) { return this.querySelectorAll(s)[0] || null; }
  getAttribute(k) { return this.attrs[k] || ''; }
  getClientRects() { return [{}]; }
  getBoundingClientRect() { return {top:0,bottom:200,left:0,right:200,width:200,height:200}; }
  contains(e) { for(;e;e=e.parentElement) if(e===this)return true;return false; }
  scrollTo(x,y) { this.scrollTop=Math.max(0,Math.min(y,this.scrollHeight-this.clientHeight)); }
  scrollBy(x,y) { this.scrollTo(x,this.scrollTop+y); }
  get innerText() { return [this.ownText,...this.children.map(c=>c.innerText)].filter(Boolean).join('\n'); }
  get textContent() { return this.innerText; }
}
const body=new El('body');
global.location={href:'https://example.test/product'};
global.NodeFilter={SHOW_TEXT:4};
global.getComputedStyle=e=>({visibility:'visible',overflowY:'auto',backgroundImage:e.bg,cursor:'auto'});
global.document={body,documentElement:new El('html'),scrollingElement:body,
 querySelector:s=>body.querySelector(s),querySelectorAll:s=>body.querySelectorAll(s),
 createTreeWalker(scope) {
   const texts=[scope,...scope.descendants()].filter(e=>e.ownText).map(e=>({parentElement:e,textContent:e.ownText}));
   let i=-1;
   return {currentNode:null,nextNode(){this.currentNode=texts[++i]||null;return !!this.currentNode;}};
 }};
"""


@unittest.skipUnless(shutil.which("node"), "Node needed only for synthetic DOM regression")
class MediaDomTests(unittest.TestCase):
    def test_current_taobao_drawer_album_excludes_summary_avatar_and_video_cover(self):
        fixture = r"""
const summary=body.add(new El('div','Comment--H5QmJwe9',{'data-rate-id':'summary'}));
summary.add(new El('div','album--sq8vrGV3')).add(new El('div','photo--ZUITAPZq')).add(new El('img','',{src:'https://img.alicdn.com/summary.jpg'}));
const drawer=body.add(new El('div','Drawer--nzZd5HyY'));
const root=drawer.add(new El('div','comments--ChxC7GEN')); root.scrollHeight=5366;root.clientHeight=704;
const card=root.add(new El('div','Comment--H5QmJwe9',{'data-rate-id':'actual'}));
card.add(new El('div','avatar--lblMaOem')).add(new El('img','',{src:'https://img.alicdn.com/avatar.jpg'}));
const album=card.add(new El('div','album--sq8vrGV3'));
album.add(new El('div','photo--ZUITAPZq')).add(new El('img','',{src:'https://img.alicdn.com/buyer.jpg'}));
const cover=album.add(new El('div','photo--ZUITAPZq'));
cover.add(new El('img','',{src:'https://img.alicdn.com/video-cover.jpg'}));
cover.add(new El('span','playerIcon--wiAgap5w'));
"""
        rows = self.run_dom(_DOM, fixture)
        self.assertEqual([r["id"] for r in rows], ["actual"])
        self.assertEqual(rows[0]["pics"], ["https://img.alicdn.com/buyer.jpg"])
        self.assertEqual(rows[0]["unresolved_videos"], 1)
        scroll = self.run_dom(_SCROLL_REVIEWS, fixture)
        self.assertEqual(scroll["after"], 4662)
        self.assertEqual(scroll["scope"], "review_container")
        self.assertTrue(scroll["moved"])
        self.assertTrue(scroll["at_bottom"])

    def run_dom(self, function, fixture, argument=""):
        script = TREE + fixture + "\nconst collect = " + function + ";\nprocess.stdout.write(JSON.stringify(collect(" + argument + ")));"
        result = subprocess.run([shutil.which("node"), "-e", script], capture_output=True,
                                text=True, encoding="utf-8", timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_product_douyin_excludes_buyer_show_recommendations_and_cover(self):
        fixture = r"""
const main=body.add(new El('div','head-figure__media-view'));
main.add(new El('div','head-figure__media-view__content',{bg:'url("https://p3-item.ecombdimg.com/main.jpg")'}));
main.add(new El('div','head-figure__buyershow')).add(new El('div','head-figure__media-view__content',{bg:'url("https://p3-item.ecombdimg.com/buyer.jpg")'}));
main.add(new El('div','head-figure__threeD')).add(new El('div','head-figure__media-view__content',{bg:'url("https://p3-item.ecombdimg.com/3d.jpg")'}));
main.add(new El('img','head-figure__media-view__video__main-pic',{src:'https://p3-item.ecombdimg.com/poster.jpg'}));
const detail=body.add(new El('div','product-big-img-list'));
detail.add(new El('img','product-big-img-list__every-img',{src:'https://p3-item.ecombdimg.com/detail.jpg'}));
body.add(new El('div','shop-recommend')).add(new El('div','product-big-img-list')).add(new El('img','product-big-img-list__every-img',{src:'https://p3-item.ecombdimg.com/recommend.jpg'}));
body.add(new El('video','',{src:'https://v3.douyinvod.com/unrelated.mp4'}));
"""
        data = self.run_dom(_PRODUCT_DOM, fixture, "'douyin'")
        self.assertEqual([m["url"].rsplit("/", 1)[1] for m in data["media"]], ["main.jpg", "detail.jpg"])
        self.assertEqual(data["unresolved_videos"], 1)

    def test_product_taobao_native_video_sources_and_main_detail_are_separate(self):
        fixture = r"""
const main=body.add(new El('div','',{id:'picGalleryEle'}));
main.add(new El('img','',{src:'https://img.alicdn.com/main.jpg'}));
main.add(new El('img','',{src:'https://img.alicdn.com/poster.jpg'}));
const video=main.add(new El('video','',{currentSrc:'https://cloud.video.taobao.com/a.mp4?sig=a%2Bb',poster:'//img.alicdn.com/poster.jpg'}));
video.add(new El('source','',{src:'https://cloud.video.taobao.com/b.m3u8'}));
const detail=body.add(new El('div','desc-root'));
detail.add(new El('img','',{src:'https://img.alicdn.com/detail.jpg'}));
detail.add(new El('div','Recommend--x')).add(new El('img','',{src:'https://img.alicdn.com/recommended.jpg'}));
body.add(new El('img','',{src:'https://img.alicdn.com/avatar.jpg'}));
"""
        data = self.run_dom(_PRODUCT_DOM, fixture, "'taobao'")
        self.assertEqual({(m["source_type"], m["url"].split("/")[-1]) for m in data["media"]}, {
            ("product_main", "main.jpg"), ("product_detail", "detail.jpg"),
            ("product_video", "a.mp4?sig=a%2Bb"), ("product_video", "b.m3u8")})

    def test_taobao_dom_initial_append_video_boundaries_and_poster_exclusion(self):
        fixture = r"""
const card=body.add(new El('div','Comment--H5QmJwe9',{'data-rate-id':'r'}));
const media=card.add(new El('div','imageWrapper--x'));
media.add(new El('img','',{currentSrc:'https://img.alicdn.com/poster.jpg'}));
media.add(new El('video','',{currentSrc:'https://cloud.video.taobao.com/first.mp4',poster:'//img.alicdn.com/poster.jpg'}));
const append=card.add(new El('div','append--x'));
append.add(new El('video','',{src:'blob:https://detail.tmall.com/not-downloadable'}));
append.add(new El('video','',{})).add(new El('source','',{src:'https://cloud.video.taobao.com/append.mp4'}));
body.add(new El('video','',{src:'https://cloud.video.taobao.com/advert.mp4'}));
"""
        rows = self.run_dom(_DOM, fixture)
        self.assertEqual(rows[0]["videos"], ["https://cloud.video.taobao.com/first.mp4"])
        self.assertEqual(rows[0]["pics"], [])
        self.assertEqual(rows[0]["appendComment"]["videos"], ["https://cloud.video.taobao.com/append.mp4"])
        self.assertEqual(rows[0]["appendComment"]["unresolved_videos"], 1)

    def test_douyin_dom_video_only_more_than_five_and_ambiguous_append(self):
        fixture = r"""
const root=body.add(new El('div','eRfaxg7R'));
const card=root.add(new El('div','Ez3MC6d7',{'data-review-id':'r'}));
const media=card.add(new El('div','qZmIVPuL'));
for(let i=0;i<8;i++) media.add(new El('video','',{src:'https://v3.douyinvod.com/'+i+'.mp4'}));
media.children[0].attrs.poster='//p3-item.ecombdimg.com/cover.jpg';
media.add(new El('img','',{currentSrc:'https://p3-item.ecombdimg.com/cover.jpg'}));
const append=root.add(new El('div','Ez3MC6d7',{'data-review-id':'unclear'}));
append.add(new El('span','','','追加评价'));
append.add(new El('video','',{src:'https://v3.douyinvod.com/unclear.mp4'}));
"""
        data = self.run_dom(_SNAPSHOT_JS, fixture)
        self.assertEqual(len(data["reviews"][0]["videos"]), 8)
        self.assertEqual(data["reviews"][0]["imgs"], [])
        self.assertEqual(data["reviews"][1]["videos"], [])
        self.assertEqual(data["ambiguous_appends"], 1)


if __name__ == "__main__":
    unittest.main()
