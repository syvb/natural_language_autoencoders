"""Generate embed.html — the fully static, client-side 27B explorer for posts.

Same pattern as the v3 (Qwen2.5-7B) space's build_embed.py: plain HTML/CSS/JS,
no server. The default blackmail-honeypot sample is baked into the page
(gzip+base64, decoded with the browser's DecompressionStream) so first paint is
instant; the Space's full precache.json (~34MB, all 39 samples) streams in
behind it, so redeploying the Space (deploy.sh) refreshes the other samples
with no HTML rebuild. Baking reads the LOCAL precache.json — keep it in sync
with the Space (deploy.sh does) or the baked honeypot could drift.

The eval-awareness heatmap and leave-one-out ablation overlays were dropped:
their JSONs only cover the 5 curated samples and went stale (and unused, per
the entry-count guard) when the 34 WeirdChat samples landed in precache.json.

Usage:
    python3 build_embed.py                # → embed.html (fetches from the Space repo)
    python3 build_embed.py --base-url URL # fetch precache.json from elsewhere
    python3 build_embed.py --inline       # bake ALL data in (~34MB, no CORS/network)
    python3 build_embed.py --widget       # → embed_widget.html: document-shell-free
                                          #   fragment for sandboxed-iframe embeds
                                          #   (e.g. LessWrong post widgets)
    python3 build_embed.py --widget-page  # → embed_widget_page.html: the widget
                                          #   fragment wrapped in a minimal page
                                          #   shell, for hosting on syvb.ca and
                                          #   iframing FROM the LW post (keeps the
                                          #   2.6MB payload out of the editor)

Rebuild only when embed_template.html changes (or to switch data source).
"""
import argparse
import base64
import gzip
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_BASE = "https://huggingface.co/spaces/syvb/nla-qwen36-27b-explorer/resolve/main"
HONEYPOT_PREFIX = "<|im_start|>system"  # must match the template's constant


def baked_honeypot_b64() -> str:
    pre = json.loads((HERE / "precache.json").read_text())
    idx = next(i for i, e in enumerate(pre["entries"])
               if e["text"].startswith(HONEYPOT_PREFIX))
    baked = {"i": idx, "n": len(pre["entries"]), "meta": pre["meta"],
             "entry": pre["entries"][idx]}
    raw = json.dumps(baked, ensure_ascii=False, separators=(",", ":")).encode()
    return base64.b64encode(gzip.compress(raw, 9)).decode()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=DEFAULT_BASE,
                    help="directory URL holding precache/eval_awareness/loo .json")
    ap.add_argument("--inline", action="store_true",
                    help="bake the data files into the page instead of fetching")
    ap.add_argument("--widget", action="store_true",
                    help="emit a <style>+body fragment for sandboxed-iframe embeds")
    ap.add_argument("--widget-page", action="store_true",
                    help="emit the widget fragment wrapped in a minimal page shell")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.widget_page:
        args.widget = True
    out = Path(args.out or (HERE / ("embed_widget_page.html" if args.widget_page
                                    else "embed_widget.html" if args.widget
                                    else "embed.html")))

    if args.inline:
        pre = json.loads((HERE / "precache.json").read_text())
        assert all("pieces" in e for e in pre["entries"]), \
            "precache.json has no 'pieces' — regenerate (precompute_cache.py)"
        # </script> inside a JSON string would end the script block early
        source = json.dumps({"inline": pre}).replace("</", "<\\/")
        baked = ""  # the inline path never touches the baked blob
    else:
        source = json.dumps({"url": f"{args.base_url}/precache.json"})
        baked = baked_honeypot_b64()

    template = (HERE / "embed_template.html").read_text()
    assert template.count("__DATA_SOURCE__") == 1
    assert template.count("__IS_WIDGET__") == 1
    assert template.count("__BAKED_B64__") == 1
    page = (template.replace("__DATA_SOURCE__", source)
            .replace("__IS_WIDGET__", "true" if args.widget else "false")
            .replace("__BAKED_B64__", baked))
    if args.widget:
        style = re.search(r"<style>.*?</style>", page, re.DOTALL).group(0)
        body = re.search(r"<body>(.*)</body>", page, re.DOTALL).group(1)
        # widget-only overrides: fixed-px token panel (vh is circular inside
        # auto-height iframes — and the honeypot transcript NEEDS the scroll),
        # side-by-side panels at LW desktop width (the page's 880px breakpoint
        # never fires in a 340-700px iframe), and a denser results card. The
        # two control bars stay in the right column's flow (unlike the v3
        # widget's single absolute bar — two stacked bars would collide with
        # the sample-picker row).
        overrides = """<style>
.wrap{padding:4px 2px 8px;}
.tokscroll{max-height:320px;font-size:13px;line-height:1.85;}
.tokhead{padding:6px 12px;}
.picker{margin-bottom:6px;}
.picker select{padding:4px 8px;font-size:11.5px;}
.bars{gap:4px;margin-bottom:6px;}
.modebar button{padding:4px 10px;font-size:11px;}
.nlaviz{padding:10px 10px 10px 8px;}
.nlaviz .sub{margin-bottom:8px;}
.nlaviz .chips{gap:16px;margin-bottom:10px;}
.nlaviz .chip .v{font-size:16px;}
.nlaviz .row,.nlaviz .axisrow{grid-template-columns:10px minmax(0,1fr) 68px 40px;gap:6px;}
.nlaviz .row{padding:3px 2px;}
.nlaviz .line{font-size:10px;line-height:1.3;-webkit-line-clamp:3;line-clamp:3;}
.nlaviz .idx{font-size:9.5px;}
.nlaviz .val{font-size:10.5px;}
.nlaviz .note{margin-top:6px;font-size:10.5px;}
/* the 68px axis track can't fit lo + zero labels without collision */
.nlaviz .axislab span:first-child:nth-last-child(3){display:none;}
/* sandbox can't open links; attribution lives in the post body instead */
footer{display:none;}
/* two-pane once the column is wide enough (560px was fine). The bug was vertical:
   the token box was a short 320px panel next to the taller results card, so the
   left column had blank space below it. Give the token box ~the results-card
   height in two-pane so it fills the column (transcript scrolls inside). Can't
   auto-stretch to match: the iframe is auto-height and an `auto` grid row would
   size to the full 4.7k-token transcript and blow up. Narrow/stacked keeps 320px. */
@media (min-width:560px){
  .cols{grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:10px;}
  .side{position:static;}
  .tokscroll{max-height:500px;}
}
</style>"""
        page = style + overrides + body
        # LessWrong's publish pipeline entity-decodes widget source once; the
        # fragment must be a fixed point of that transform or publishing
        # breaks the JS (learned the hard way with "&quot;" in the escaper)
        import html as html_lib
        assert html_lib.unescape(page) == page, (
            "widget fragment contains decodable HTML entities — publish-unsafe")
        if args.widget_page:
            # height reporter: the LW post embeds this page via a tiny relay
            # widget (see repo docstring) — post our content height up so the
            # nested iframe can be sized exactly, no internal scrollbars
            reporter = """<script>
(function () {
  function send() {
    parent.postMessage({ type: "nla-embed-height",
                         height: document.documentElement.scrollHeight }, "*");
  }
  new ResizeObserver(send).observe(document.body);
  addEventListener("load", send);
})();
</script>"""
            page = ('<!doctype html>\n<html lang="en">\n<head>\n'
                    '<meta charset="utf-8">\n'
                    '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
                    '<title>NLA Qwen3.6-27B — activation explorer (embed)</title>\n'
                    '</head>\n<body>\n' + page + reporter + '\n</body>\n</html>\n')
    out.write_text(page)
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB, "
          f"{'widget fragment, ' if args.widget else ''}"
          f"{'inline data' if args.inline else 'fetches from ' + args.base_url})")


if __name__ == "__main__":
    main()
